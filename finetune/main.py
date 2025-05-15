import os
import json
import sys
import math
import random
import argparse
import torch
import numpy as np
import csv
from tqdm import tqdm
from pathlib import Path
from nltk.tokenize import sent_tokenize
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.util import cos_sim
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from rank_bm25 import BM25Okapi

from huggingface_hub import login
from utils import extract_metrics_from_log

# ------------------------
# Command-line arguments
# ------------------------
parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer with either MultipleNegativesRankingLoss or TripletLoss")
parser.add_argument(
    "--loss_type",
    choices=["multiple_negatives", "triplet"],
    default="multiple_negatives",
    help="Which loss function to use",
)
args = parser.parse_args()
LOSS_TYPE = args.loss_type

# ------------------------
# HuggingFace login (token via env or hardcode)
# ------------------------
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
login(HUGGINGFACE_TOKEN)

# ------------------------
# Configurations
# ------------------------
JSON_FILES = [
    "/home/renyang/jadhav/LegalBench-RAG/combined/maud.json",
]
CORPUS_PATHS = [
    "/home/renyang/jadhav/LegalBench-RAG/corpus/maud",
]
MODEL_NAME = "BAAI/bge-small-en-v1.5"
OUTPUT_DIR = "maud-fine-tuned-bge-small-1000-epochs"
BATCH_SIZE = 32
EPOCHS = 1000
LR = 5e-6
VALID_SPLIT_RATIO = 0.1
SEED = 42
MAX_NEGATIVES = 5    # for triplet negative sampling

# ------------------------
# Setup device & seeds
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ------------------------
# Initialize model & tokenizer
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentenceTransformer(MODEL_NAME, device=device)

# ------------------------
# Prepare output logging
# ------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
METRICS_FILE = os.path.join(OUTPUT_DIR, "training_metrics.csv")
LOG_FILE     = os.path.join(OUTPUT_DIR, "training_log.txt")

# create empty files if missing
for p in (METRICS_FILE, LOG_FILE):
    Path(p).touch()

sys.stdout = open(LOG_FILE, "w", encoding="utf-8")

# ------------------------
# Metric logging callback
# ------------------------
def log_metrics(epoch, steps, scores):
    scores = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
              for k, v in scores.items()}
    file_exists = os.path.isfile(METRICS_FILE)
    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        headers = ["epoch", "steps"] + list(scores.keys())
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([epoch, steps] + list(scores.values()))

def evaluation_callback(score, epoch, steps):
    if isinstance(score, (int, float, np.floating)):
        score = {"metric": score}
    log_metrics(epoch, steps, score)
    scheduler.step()

# ------------------------
# Helper functions
# ------------------------
def find_text_file(file_name):
    for corpus_path in CORPUS_PATHS:
        candidate = os.path.join(corpus_path, file_name)
        if os.path.exists(candidate):
            return candidate
    return None

def extract_text_snippet(file_name, start, end):
    path = find_text_file(file_name)
    if not path:
        print(f"Warning: {file_name} not found!")
        return ""
    return Path(path).read_text(encoding="utf-8")[start:end]

def truncate_text(text, tokenizer, max_length=512):
    truncated = ""
    for sent in sent_tokenize(text):
        if len(tokenizer.tokenize(truncated + " " + sent)) < max_length:
            truncated += " " + sent
        else:
            break
    return truncated.strip()

def load_corpus_examples(json_file):
    examples = []
    data = json.load(open(json_file, encoding="utf-8"))

    # Build BM25 index
    all_docs = []
    for cp in CORPUS_PATHS:
        for txt in Path(cp).rglob("*.txt"):
            all_docs.append(txt.read_text(encoding="utf-8"))
    bm25 = BM25Okapi([doc.split() for doc in all_docs])

    for sample in tqdm(data["tests"], desc=f"Loading {os.path.basename(json_file)}"):
        raw_query = sample["query"].strip()
        query = truncate_text(f"Represent this sentence for searching relevant passages: {raw_query}", tokenizer)

        positives = [
            truncate_text(sn["answer"].strip(), tokenizer)
            for sn in sample["snippets"]
            if sn.get("answer", "").strip()
        ]
        if not positives:
            continue

        # retrieve and pick negatives
        retrieved = bm25.get_top_n(query.split(), all_docs, n=100)
        negatives = [
            truncate_text(doc, tokenizer)
            for doc in retrieved
            if doc not in positives
        ][:MAX_NEGATIVES]

        if LOSS_TYPE == "multiple_negatives":
            for pos in positives:
                examples.append(InputExample(texts=[query, pos]))
        else:  # triplet
            for pos in positives:
                for neg in negatives:
                    examples.append(InputExample(texts=[query, pos, neg]))

    return examples

# ------------------------
# Load & split data
# ------------------------
all_examples = []
for jf in JSON_FILES:
    all_examples += load_corpus_examples(jf)

random.shuffle(all_examples)
val_size = int(len(all_examples) * VALID_SPLIT_RATIO)
train_size = len(all_examples) - val_size
train_dataset, val_dataset = random_split(all_examples, [train_size, val_size])

print(f"Total examples: {len(all_examples)}")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

# Prepare evaluator on validation set
queries = {i: truncate_text(ex.texts[0], tokenizer) for i, ex in enumerate(val_dataset)}
corpus  = {i: truncate_text(ex.texts[1], tokenizer) for i, ex in enumerate(val_dataset)}
relevant = {i: [i] for i in queries}

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant,
    show_progress_bar=True,
)

# ------------------------
# Select loss and optimizer
# ------------------------
if LOSS_TYPE == "multiple_negatives":
    loss_func = losses.MultipleNegativesRankingLoss(model=model)
elif LOSS_TYPE == "triplet":
    loss_func = losses.TripletLoss(model=model)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

# ------------------------
# Train!
# ------------------------
warmup_steps     = math.ceil(len(train_loader) * EPOCHS * 0.1)
eval_steps       = max(1, len(train_loader) // 2)

model.fit(
    train_objectives=[(train_loader, loss_func)],
    evaluator=evaluator,
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    evaluation_steps=eval_steps,
    optimizer_class=AdamW,
    optimizer_params={"lr": LR, "weight_decay": 1e-4},
    callback=evaluation_callback,
    output_path=OUTPUT_DIR,
    save_best_model=True,
    show_progress_bar=True,
)

# Cleanup
sys.stdout.close()
sys.stdout = sys.__stdout__

# Extract metrics from log into a summary
extract_metrics_from_log(LOG_FILE)
