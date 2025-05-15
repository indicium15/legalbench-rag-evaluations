# Evaluating Retrieval Precision with Different RAG Methods

This repository extends the code provided in the [LegalBench-RAG](https://github.com/zeroentropy-ai/legalbenchrag) dataset to add support for:

1. Using advanced chunking techniques, such as semantic chunking.
2. Using open-source embedding and re-ranking models available on Hugging-Face
3. Evaluating advanced RAG methods.

## Development Environment
1. Create a virtual environment and download the requirements listed in the `requirements.txt` package.
2. Download [Ollama](https://ollama.com/) for LLM support in the Self-RAG method. The Ollama application also supports embedding models, which can be used in retrieval.
## Instructions
1. Download and extract the original LegalBench-RAG corpus provided at [this](https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=0) link.
2. Extract all the .json files into a folder and update the folder path in the `BENCHMARK_DIR` variable in the `.env` file.
3. Similarly, update the `CORPUS_DIR` variable to the path of the folder containing the four dataset sub-folders with the text files for each dataset.

### Specifying RAG Method
In the .env file, update the `RAG_METHOD` variable to toggle which file will be run. The following methods are currently supported:
1. baseline - baseline and re-ranking RAG implementation.
2. selfrag - Self-RAG implementation.
3. metadata - RAG with metadata implementation

### Specifying Embedding and Re-Ranking Model
Similarly, by specifying the values of `OLLAMA_EMBEDDING_MODEL`, `BGE_EMBEDDING_MODEL` or `HF_EMBEDDING_MODEL`, the open-source embedding model to be used can be defined.

Similarly, the re-ranking model to be used can be specified uysing the `BGE_RERANKING_MODEL` variable. Since only models from BAAI were tested for re-ranking, only BAAI-based models are currently supported.

### Specifying LLM to use for Metadata Extraction
To change the LLM used for metadata extraction, first ensure that it has been downloaded via `ollama pull model-name`. Subsequently, change the variable `OLLAMA_METADATA_MODEL` to the model you would like to use.

## Fine-Tuning Embedding Model
Refer to the script in `finetune/main.py` to fine-tune sentence embedding models (such as BGE) on one or more LegalBench datasets using contrastive learning. This helps adapt general-purpose embedding models to the domain of legal QA and improve retrieval quality in downstream RAG pipelines.

This script fine-tunes a SentenceTransformer model using one of two contrastive learning objectives:

1. MultipleNegativesRankingLoss (default): Uses (query, positive) pairs only. All other positives in the batch act as implicit negatives.
2. TripletLoss: Uses (query, positive, negative) triplets. Negatives are mined using BM25 from the same corpus.

### Configuration
To configure this script, follow the below instructions:
Open finetune/main.py and update the following paths:
```
JSON_FILES = [
    "/absolute/path/to/LegalBench-RAG/combined/maud.json"
]
CORPUS_PATHS = [
    "/absolute/path/to/LegalBench-RAG/corpus/maud"
]
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # or any HuggingFace model
OUTPUT_DIR = "maud-fine-tuned-bge-small-1000-epochs"
BATCH_SIZE = 32
EPOCHS = 1000
LR = 5e-6
VALID_SPLIT_RATIO = 0.1
SEED = 42
```

Additionally, make sure to export your HuggingFace access token (if needed for model download).

### Running
To run fine-tuning with MultipleNegativesRankingLoss (default):

```
python finetune/main.py
```

To run fine-tuning with TripletLoss:

```
python finetune/main.py --loss_type triplet
```

### Outputs
After training, the following files will be created in the specified OUTPUT_DIR:
1. training_log.txt – Full training logs (stdout)
2. training_metrics.csv – Epoch-wise evaluation metrics (e.g., MRR, Recall@K)
3. checkpoint-best/ – Best model weights saved by evaluation score