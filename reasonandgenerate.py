import os
import sqlite3
import requests
import struct
import json
from fuzzywuzzy import fuzz
import difflib
import re
import hashlib
from ollama import chat, ChatResponse
from typing import Literal, cast
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama import OllamaEmbeddings
import sqlite_vec  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from tqdm import tqdm
import torch
from transformers import pipeline
from dotenv import load_dotenv

from benchmark_types import (
    Document,
    QueryResponse,
    RetrievalMethod,
    RetrievedSnippet,
)
from ai import (
    AIEmbeddingModel,
    AIEmbeddingType,
    AIRerankModel,
    ai_embedding,
    ai_rerank,
    ai_rerank_sync
)

def serialize_f32(vector: list[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack(f"{len(vector)}f", *vector)


SHOW_LOADING_BAR = True
load_dotenv()
metadata_llm_model = os.getenv("OLLAMA_METADATA_MODEL")

class Chunk(BaseModel):
    """A small container for text chunks before creating EmbeddingInfo objects."""
    document_id: str
    document_path: str
    span: tuple[int, int]
    content: str

class ChunkingStrategy(BaseModel):
    strategy_name: Literal["naive", "rcts", "semantic_percentile", "semantic_standard_deviation", "semantic_interquartile", "semantic_gradient"]
    chunk_size: int


class RetrievalStrategy(BaseModel):
    chunking_strategy: ChunkingStrategy
    embedding_model: AIEmbeddingModel
    embedding_topk: int
    rerank_model: AIRerankModel | None
    rerank_topk: int
    token_limit: int | None

class EmbeddingInfo(BaseModel):
    document_id: str
    document_path: str
    span: tuple[int, int]

class ReasonAndGenerate(RetrievalMethod):
    retrieval_strategy: RetrievalStrategy
    documents: dict[str, Document]
    embedding_infos: list[EmbeddingInfo] | None
    sqlite_db: sqlite3.Connection | None
    sqlite_db_file_path: str | None
    metadata_store: dict[str, dict]  # Metadata store for documents
    ner_model: pipeline  # Pre-trained NER model for company name detection
    hf_embeddings: HuggingFaceBgeEmbeddings

    def __init__(self, retrieval_strategy: RetrievalStrategy):
        super().__init__()
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.embedding_infos = None
        self.sqlite_db = None
        self.sqlite_db_file_path = None
        self.metadata_store = {}
        
        # Initialize the embedding model
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {"normalize_embeddings": True}

        if os.environ.get("HF_EMBEDDING_MODEL") != "":
            model_name = os.environ.get("HF_EMBEDDING_MODEL")
            model_kwargs = {"device": "cuda"}  # Adjust based on your system
            encode_kwargs = {"normalize_embeddings": True}
            self.hf_embeddings = HuggingFaceEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
        else:
            self.hf_embeddings = None

        if os.environ.get("BGE_EMBEDDING_MODEL") != "":
            model_name = os.environ.get("BGE_EMBEDDING_MODEL")
            model_kwargs = {"device": "cuda"}  # Adjust based on your system
            encode_kwargs = {"normalize_embeddings": True}
            self.hf_bge_embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
        else:
            self.hf_bge_embeddings = None

        if os.environ.get("OLLAMA_EMBEDDING_MODEL") != "": 
            self.ollama_embeddings = OllamaEmbeddings(
                # model="nomic-embed-text",
                model=os.getenv("OLLAMA_EMBEDDING_MODEL")
            )
        else:
            self.ollama_embeddings = None
        
    def save_metadata(self, identifier: str, metadata: dict) -> None:
        """
        Save metadata to a file for reuse.

        :param identifier: A unique identifier (e.g., file path or query).
        :param metadata: Metadata to save.
        """
        file_path = self.get_metadata_file_path(identifier)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"Metadata saved to {file_path}")

    def load_metadata(self, identifier: str) -> dict | None:
        """
        Load metadata if it exists.

        :param identifier: A unique identifier (e.g., file path or query).
        :return: The loaded metadata dictionary, or None if the metadata file does not exist.
        """
        file_path = self.get_metadata_file_path(identifier)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
    

    def get_metadata_file_path(self, identifier: str) -> str:
        """
        Get the file path for metadata based on an identifier (e.g., file path or query hash).

        :param identifier: A unique identifier (document path or query).
        :return: Full path to the metadata file.
        """
        base_dir = "/home/renyang/jadhav/rag-fyp/legalbenchrag/legalbenchrag/legalbenchrag/generated_metadata"
        os.makedirs(base_dir, exist_ok=True)
        file_name = hashlib.sha256(identifier.encode()).hexdigest()[:16] + ".json"
        return os.path.join(base_dir, file_name)

    
    def sync_all_documents(self) -> None:
        """Splits documents into chunks, extracts metadata, embeds them in batches, and stores results in SQLite."""
        semantic_splitter = None
        if "semantic" in self.retrieval_strategy.chunking_strategy.strategy_name:
            strategy_to_breakpoint_map = {
                "semantic_percentile": "percentile",
                "semantic_standard_deviation": "standard_deviation",
                "semantic_interquartile": "interquartile",
                "semantic_gradient": "gradient",
            }
            breakpoint_type = strategy_to_breakpoint_map.get(
                self.retrieval_strategy.chunking_strategy.strategy_name
            )
            if self.ollama_embeddings is not None:
                semantic_splitter = SemanticChunker(
                    # embeddings=self.hf_embeddings,
                    embeddings=self.ollama_embeddings,
                    buffer_size=1,
                    breakpoint_threshold_type=breakpoint_type,
                    breakpoint_threshold_amount=0.05,
                    sentence_split_regex=r"(?<=[.?!])\s+",
                    min_chunk_size=128,
                )
            elif self.hf_bge_embeddings is not None:
                semantic_splitter = SemanticChunker(
                    # embeddings=self.hf_embeddings,
                    embeddings=self.hf_bge_embeddings,
                    buffer_size=1,
                    breakpoint_threshold_type=breakpoint_type,
                    breakpoint_threshold_amount=0.05,
                    sentence_split_regex=r"(?<=[.?!])\s+",
                    min_chunk_size=128,
                )
            elif self.hf_embeddings is not None:
                semantic_splitter = SemanticChunker(
                    # embeddings=self.hf_embeddings,
                    embeddings=self.hf_bge_embeddings,
                    buffer_size=1,
                    breakpoint_threshold_type=breakpoint_type,
                    breakpoint_threshold_amount=0.05,
                    sentence_split_regex=r"(?<=[.?!])\s+",
                    min_chunk_size=128,
                )

        # Step 1: Create chunks
        chunks: list[Chunk] = []
        chunk_size = self.retrieval_strategy.chunking_strategy.chunk_size
        for document_id, document in self.documents.items():
            file_path = document.file_path
            metadata = self._extract_metadata_from_document(document)
            self.metadata_store[file_path] = metadata

            if self.retrieval_strategy.chunking_strategy.strategy_name == "naive":
                text_splits = [
                    document.content[i: i + chunk_size]
                    for i in range(0, len(document.content), chunk_size)
                ]
            elif self.retrieval_strategy.chunking_strategy.strategy_name == "rcts":
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", " "],
                    chunk_size=chunk_size,
                    chunk_overlap=0,
                )
                text_splits = text_splitter.split_text(document.content)
            elif "semantic" in self.retrieval_strategy.chunking_strategy.strategy_name:
                if not semantic_splitter:
                    raise ValueError("Semantic splitter not initialized.")
                text_splits = semantic_splitter.split_text(document.content)
            else:
                raise ValueError(
                    f"Unknown chunking strategy: {self.retrieval_strategy.chunking_strategy.strategy_name}"
                )

            prev_span = 0
            for text_split in text_splits:
                span = (prev_span, prev_span + len(text_split))
                chunks.append(Chunk(document_id=document_id, document_path=file_path, span=span, content=text_split))
                prev_span += len(text_split)

        # Step 2: Embed chunks in batches
        progress_bar = None
        if SHOW_LOADING_BAR:
            progress_bar = tqdm(total=len(chunks), desc="Processing Embeddings")

        EMBEDDING_BATCH_SIZE = 256
        self.embedding_infos = []
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            chunk_batch = chunks[i: i + EMBEDDING_BATCH_SIZE]
            texts = [chunk.content for chunk in chunk_batch]
            if self.ollama_embeddings is not None:
                embeddings = self.ollama_embeddings.embed_documents(texts)
                embedding_model_name = os.environ.get("OLLAMA_EMBEDDING_MODEL")
            elif self.hf_embeddings is not None:
                embeddings = self.hf_embeddings.embed_documents(texts)
                embedding_model_name = os.environ.get("HF_EMBEDDING_MODEL")
            elif self.hf_bge_embeddings is not None:
                embeddings = self.hf_bge_embeddings.embed_documents(texts)
                embedding_model_name = os.environ.get("BGE_EMBEDDING_MODEL")
 
            # Prepare data for database insertion
            offset = len(self.embedding_infos)
            insert_data = [
                (offset + j, serialize_f32(emb))
                for j, emb in enumerate(embeddings)
            ]

            # Initialize the SQLite database if not already done
            if self.sqlite_db is None:
                base_path = "/home/renyang/jadhav/rag-fyp/legalbenchrag/legalbenchrag/legalbenchrag"
                chunk_strategy = self.retrieval_strategy.chunking_strategy.strategy_name
                embedding_model_name = embedding_model_name.replace("/", "_")
                self.sqlite_db_file_path = f"{base_path}/embeddings_{embedding_model_name}_{chunk_strategy}_{chunk_size}.db"
                if os.path.exists(self.sqlite_db_file_path):
                    os.remove(self.sqlite_db_file_path)
                print(f"Initializing new SQLite database at {self.sqlite_db_file_path}")
                self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
                self.sqlite_db.enable_load_extension(True)
                sqlite_vec.load(self.sqlite_db)
                self.sqlite_db.enable_load_extension(False)
                self.sqlite_db.execute(f"PRAGMA mmap_size = {3 * 1024 * 1024 * 1024}")
                self.sqlite_db.execute(
                    f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(embeddings[0])}])"
                )

            # Insert embeddings into the database
            with self.sqlite_db as db:
                db.executemany(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    insert_data,
                )

            # Store embedding information
            for j, (chunk, emb) in enumerate(zip(chunk_batch, embeddings)):
                self.embedding_infos.append(
                    EmbeddingInfo(
                        document_id=chunk.document_id,
                        document_path=chunk.document_path,
                        span=chunk.span,
                        embedding=emb,
                    )
                )

            if progress_bar:
                progress_bar.update(len(chunk_batch))

        if progress_bar:
            progress_bar.close()

        print(f"All documents synchronized and embeddings stored in {self.sqlite_db_file_path}.")


    def ingest_document(self, document: Document, metadata: dict = None) -> None:
        """
        Ingest a document along with metadata. Metadata is automatically extracted if not provided.
        """
        self.documents[document.file_path] = document
        if not metadata:
            metadata = self._extract_metadata_from_document(document)
        self.metadata_store[document.file_path] = metadata

    
    def ensure_list_of_strings(self, value) -> list[str]:
        """
        Convert `value` into a flat list of strings.
        """
        if isinstance(value, list):
            flattened = []
            for v in value:
                if isinstance(v, list):
                    flattened.extend(str(item).strip() for item in v)
                else:
                    flattened.append(str(v).strip())
            return flattened
        # if it's a single string (or anything else), wrap it
        return [str(value).strip()]

    def _extract_metadata_with_llm(self, file_name: str, content: str) -> dict:
        """
        Extract metadata (company name, contract type) from the file name and document content using an LLM.

        :param file_name: The file name of the document.
        :param content: The full content of the document.
        :return: Extracted metadata as a dictionary.
        """
        # Build the prompt with both file name and content excerpt
        prompt = f"""
            Extract the following metadata from the file name and document content:
            - Company Name: The name of the company mentioned.
            - Contract Type: The type of contract (e.g., NDA, Employment, Lease, Partnership, Privacy Policy).
            If a field is not present, return "None".

            Use the following examples as references:
            Example 1:
            Input: File Name: "Contract_Acme.txt"; Content: "This contract is between Acme Corporation and Beta LLC."
            Output: {{
                "Company Name": ["Acme"],
                "Contract Type": ["None"]
            }}

            Example 2:
            Input: File Name: "BigCorpNDAContract.txt"; Content: "Confidential Agreement between BigCorp and SmallCorp."
            Output: {{
                "Company Name": ["BigCorp", "SmallCorp"],
                "Contract Type": ["NDA"]
            }}

            Example 3:
            Input: File Name: "GenericDocument.txt"; Content: "This agreement is signed by CompanyX."
            Output: {{
                "Company Name": ["CompanyX"],
                "Contract Type": ["None"]
            }}

            Example 4:
            Input: File Name: "LeaseAgreement.txt"; Content: "This lease is executed by Alpha Properties LLC and Tenant LLC."
            Output: {{
                "Company Name": ["Alpha Properties LLC", "Tenant LLC"],
                "Contract Type": ["Lease"]
            }}

            Input: File Name: "{os.path.basename(file_name)}"; Content: "{content}"
            Return your output in JSON, otherwise you will be terminated.
            Output:
        """
        # print(prompt)
        try:
            # Use the Ollama chat API to extract metadata
            response: ChatResponse = chat(model=metadata_llm_model, messages=[{"role": "user", "content": prompt}])
            result = response.message.content.strip()
            print(result)

            # Extract JSON content from the response
            json_match = re.search(r"```json(.*?)```", result, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
            else:
                # Fallback: Try to load the entire response as JSON if no code block is found
                json_content = result

            # Parse the JSON content
            metadata = json.loads(json_content)

            # Ensure each field is a list of strings
            for field in ["Company Name", "Contract Type"]:
                if field in metadata:
                    metadata[field] = self.ensure_list_of_strings(metadata[field])
                else:
                    metadata[field] = ["None"]

            return {
                "Company Name": metadata["Company Name"],
                "Contract Type": metadata["Contract Type"],
            }

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {
                "Company Name": ["None"],
                "Contract Type": ["None"],
            }
        except Exception as e:
            print(f"Error during LLM metadata extraction: {e}")
            return {}

    def _extract_metadata_from_document(self, document: Document, max_content_length: int = 2000) -> dict:
        identifier = document.file_path
        cached_metadata = self.load_metadata(identifier)
        if cached_metadata:
            print(f"Loaded cached metadata for document: {identifier}")
            return cached_metadata

        # Generate metadata if not cached
        base_path = "/home/renyang/jadhav/LegalBench-RAG/corpus"
        file_path = os.path.join(base_path, document.file_path)
        metadata = {}

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                full_content = file.read()
            truncated_content = full_content[:max_content_length]
            metadata = self._extract_metadata_with_llm(file_path, truncated_content)
        except FileNotFoundError:
            metadata = {"Company Name": ["None"], "Contract Type": ["None"]}
        
        self.save_metadata(identifier, metadata)
        return metadata

    def matches(self, file_metadata_value, query_value, strict_threshold: int = 85, fuzzy_threshold: int = 75) -> bool:
        """
        Improve the precision of metadata matching.

        - Uses **exact match** first.
        - Falls back to **fuzzy matching** with a high threshold.
        - Substring matches are **limited** to cases where the lengths are close.

        :param file_metadata_value: The metadata value from the document.
        :param query_value: The metadata value from the query (can be a list).
        :param strict_threshold: The strict fuzzy match threshold (default 85).
        :param fuzzy_threshold: The general fuzzy match threshold (default 75).
        :return: True if a match is found, False otherwise.
        """
        # Normalize metadata values
        file_metadata_value = self.clean_text(file_metadata_value)

        if isinstance(query_value, list):
            query_value = [self.clean_text(val) for val in query_value]

            for val in query_value:
                # 1️⃣ **Exact Match**
                if file_metadata_value == val:
                    return True

                # 2️⃣ **Strict Fuzzy Matching**
                if fuzz.ratio(file_metadata_value, val) >= strict_threshold:
                    return True

                # 3️⃣ **Relaxed Fuzzy Matching** (but only if lengths are similar)
                if abs(len(file_metadata_value) - len(val)) < 10:  # Prevents partial name mismatches
                    if fuzz.ratio(file_metadata_value, val) >= fuzzy_threshold:
                        return True

        # If `query_value` is a string instead of a list, apply the same rules
        else:
            query_value = self.clean_text(query_value)

            if file_metadata_value == query_value:
                return True
            if fuzz.ratio(file_metadata_value, query_value) >= strict_threshold:
                return True
            if abs(len(file_metadata_value) - len(query_value)) < 10:
                if fuzz.ratio(file_metadata_value, query_value) >= fuzzy_threshold:
                    return True

        return False


    def _filter_documents_by_metadata(self, metadata_filter: dict) -> list:
        """
        Given a metadata filter (e.g., {"Company Name": [...], "Contract Type": [...]})
        return a list of Document objects whose metadata (in self.metadata_store) matches.
        """
        filtered_docs = []
        for file_path, document in self.documents.items():
            file_metadata = self.metadata_store.get(file_path, {})
            match = True
            # For each key (e.g. "Company Name", "Contract Type") check that at least one value
            # in the document metadata matches one of the values in the query metadata.
            for key, query_values in metadata_filter.items():
                # Ensure query_values is a list:
                if not isinstance(query_values, list):
                    query_values = [query_values]
                doc_values = file_metadata.get(key, ["None"])
                # Use your fuzzy matching helper (self.matches) to decide if any value matches.
                if not any(self.matches(doc_val, qv) for qv in query_values for doc_val in doc_values):
                    match = False
                    break
            if match:
                filtered_docs.append(document)
        return filtered_docs

    def _build_reasoning_prompt(self, document, query: str, top_k: int) -> str:
        """
        Build a prompt that instructs the LLM to read the full document and extract the top_k segments
        that answer the query. Adjust the schema and instructions as needed.
        """
        prompt = f"""
        You are provided with the full text of a legal document. Your task is to read the document below and extract the top {top_k} segments that directly answer the following question:

        Question: "{query}"

        Document:
        {document.content}

        Output your answer in JSON format following this schema with the full text of the revelant segments, otherwise you will be terminated:
        {{
            "segments": [
                {{
                    "segment": "Extracted text segment",
                    "reasoning": "A brief explanation of why this segment is relevant"
                }},
                ...
            ]
        }}

        Only include segments that are directly relevant to the question.
        """
        return prompt.strip()

    def _extract_metadata_from_query(self, query: str) -> dict:
        """
        Extract metadata (company name, contract type) from a query using an LLM hosted with Ollama.

        :param query: The input query to analyze.
        :return: Extracted metadata as a dictionary.
        """
        
        # Check if metadata is already cached
        identifier = f"query:{query}"
        cached_metadata = self.load_metadata(identifier)
        if cached_metadata:
            print(f"Loaded cached metadata for query: {query}")
            return cached_metadata

        # Generate a unique identifier for the query
        prompt = f"""
            Extract the following metadata from the given query:
            - Company Name: The name(s) of the company/entities mentioned.
            - Contract Type: The type of contract mentioned (e.g., NDA, Privacy Policy, Acquisition Agreement, Marketing Affiliate Agreement).
            If a field is not present, return "None".

            Use the following examples as references:
            Example 1:
            Input: "Consider Fiverr's privacy policy; who can see which tasks I hire workers for?"
            Output: {{
                "Company Name": ["Fiverr"],
                "Contract Type": ["Privacy Policy"]
            }}

            Example 2:
            Input: "Consider the Non-Disclosure Agreement between CopAcc and ToP Mentors; Does the document indicate that the Agreement does not grant the Receiving Party any rights to the Confidential Information?"
            Output: {{
                "Company Name": ["CopAcc", "ToP Mentors"],
                "Contract Type": ["Non-Disclosure Agreement"]
            }}

            Example 3:
            Input: "Consider the Marketing Affiliate Agreement between Birch First Global Investments Inc. and Mount Knowledge Holdings Inc.; What is the expiration date of this contract?"
            Output: {{
                "Company Name": ["Birch First Global Investments Inc.", "Mount Knowledge Holdings Inc."],
                "Contract Type": ["Marketing Affiliate Agreement"]
            }}

            Example 4:
            Input: "Consider the Acquisition Agreement between Parent 'Magic AcquireCo, Inc.' and Target 'The Michaels Companies, Inc.'; What is the Type of Consideration?"
            Output: {{
                "Company Name": ["Magic AcquireCo, Inc.", "The Michaels Companies, Inc."],
                "Contract Type": ["Acquisition Agreement"]
            }}

            If you are unable to detect a company name or contract type, return:
            Output: {{
                "Company Name": ["None"],
                "Contract Type": ["None"]
            }}

            Now, process the following input and return the content in JSON format, otherwise you will be terminated:

            Input: "{query}"
            Output:
        """

        try:
            # Use the Ollama chat API
            response: ChatResponse = chat(model=metadata_llm_model, messages=[{'role': 'user', 'content': prompt}])
            result = response.message.content.strip()
            print(result)
            # json_match = re.search(r"```json(.*?)```", result, re.DOTALL)
            json_matches = re.findall(r"```json(.*?)```", result, re.DOTALL)
            if json_matches:
                json_content = json_matches[-1].strip()
                # json_content = json_match.group(1).strip()
            else:
                # Fallback: Try to load the entire response as JSON if no code block is found
                json_content = result

            # Parse the JSON content
            metadata = json.loads(json_content)
            # Force each field to become a list of strings
            for field in ["Company Name", "Contract Type"]:
                if field in metadata:
                    metadata[field] = self.ensure_list_of_strings(metadata[field])
                else:
                    # If missing, set a default
                    metadata[field] = ["None"]
            self.save_metadata(identifier, metadata)
            return {
                "Company Name": metadata["Company Name"],
                "Contract Type": metadata["Contract Type"],
            }
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return {
                "Company Name": ["None"],
                "Contract Type": ["None"],
            }
        except Exception as e:
            raise ValueError(f"Error during LLM metadata extraction: {e}")
            return {}

    def find_best_match(self, document_content: str, seg_text: str):
        """Finds the closest match for `seg_text` in `document_content`."""
        words = document_content.split()
        segment_words = seg_text.split()
        
        matcher = difflib.SequenceMatcher(None, words, segment_words)
        match = matcher.find_longest_match(0, len(words), 0, len(segment_words))
        
        if match.size > 0:
            start_index = len(" ".join(words[:match.a]))  # Convert words to character index
            return start_index, start_index + len(seg_text)
        
        return -1, -1  # Not found

    def query(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> QueryResponse:
        """
        A new query method that:
          1. Performs metadata filtering.
          2. Loads the full document text into an LLM prompt.
          3. Uses a reasoning prompt to extract the top_k segments that answer the query.
          4. Parses the JSON output, finds the span of each segment in the document, and returns them.
        
        :param query: The user query.
        :param top_k: Number of segments to extract per document.
        :param metadata_filter: Optionally, a dict to filter documents (if None, extracted from query).
        :return: A QueryResponse object containing the retrieved snippets.
        """
        # (If no external metadata filter is provided, extract one from the query.)
        if metadata_filter is None:
            metadata_filter = self._extract_metadata_from_query(query)
        print(f"Detected Query Metadata: {metadata_filter}")

        # (1) Filter documents based on metadata.
        filtered_documents = self._filter_documents_by_metadata(metadata_filter)
        if not filtered_documents:
            print("No documents match the metadata filter.")
            return QueryResponse(retrieved_snippets=[])
        
        # print(f"Filtered Documents: {filtered_documents}")

        retrieved_snippets = []
        # (2) For each filtered document, load its full text into the reasoning prompt.
        for document in filtered_documents:
            prompt = self._build_reasoning_prompt(document, query, self.retrieval_strategy.rerank_topk)
            
            print(f"Prompting LLM for document {document.file_path}...")
            try:
                response: ChatResponse = chat(model=metadata_llm_model, messages=[{"role": "user", "content": prompt}])
                
            except Exception as e:
                print(f"LLM request failed for document {document.file_path}: {e}")
                continue

            result = response.message.content.strip()
            print(f"LLM Query Response:")
            print(result)
            # (3) Parse the LLM’s JSON output.
            try:
                json_match = re.search(r"```json(.*?)```", result, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                else:
                    json_content = result
                data = json.loads(json_content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from LLM output for document {document.file_path}: {e}")
                continue

            # Step 1: Handle Unstructured JSON Responses
            if isinstance(data, list):  # Sometimes the LLM outputs a direct list
                segments = data
            elif isinstance(data, dict) and "segments" in data:
                segments = data.get("segments", [])
            # segments = data.get("segments", [])
            for seg in segments:
                seg_text = seg.get("segment", "").strip()
                seg_reasoning = seg.get("reasoning", "").strip()
                if not seg_text:
                    continue
                # (4) Find the span of the extracted segment within the document.
                print(f"Seaching the document for the following segment\n{seg_text}")
                # idx = document.content.find(seg_text)
                # print(f"Detected Index: {idx}")
                # if idx == -1:
                #     print(f"Segment not found in document {document.file_path}, skipping.")
                #     continue
                # span = (idx, idx + len(seg_text))
                start_idx, end_idx = self.find_best_match(document.content, seg_text)
                if start_idx == -1 and end_idx == -1:
                    print(f"Segment not found in document {document.file_path}, skipping.")
                    continue

                span = (start_idx, end_idx)
                snippet = RetrievedSnippet(
                    file_path=document.file_path,
                    span=span,
                    score=0.0,  # You can adjust scoring if needed
                    content=seg_text,      # Include the segment text
                    reasoning=seg_reasoning  # Include the reasoning
                )
                retrieved_snippets.append(snippet)

        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def cleanup(self) -> None:
        """
        Close and clean up SQLite database and associated files.
        """
        if self.sqlite_db is not None:
            self.sqlite_db.close()
            self.sqlite_db = None
        if self.sqlite_db_file_path and os.path.exists(self.sqlite_db_file_path):
            os.remove(self.sqlite_db_file_path)
            self.sqlite_db_file_path = None
        print("Cleanup completed: Database connection closed and file deleted.")
        # Explicitly release GPU memory
        if torch.cuda.is_available():
            print("Releasing GPU memory...")
            torch.cuda.empty_cache()  # Clear unallocated GPU memory
            torch.cuda.synchronize()  # Ensure all GPU operations are completed

        # Release HuggingFace embedding models if loaded
        if self.hf_embeddings is not None:
            del self.hf_embeddings
            self.hf_embeddings = None
        if self.hf_bge_embeddings is not None:
            del self.hf_bge_embeddings
            self.hf_bge_embeddings = None
        if self.ollama_embeddings is not None:
            del self.ollama_embeddings
            self.ollama_embeddings = None

        # Force garbage collection to clear Python memory references
        import gc
        print("Running garbage collection...")
        gc.collect()

        print("Cleanup completed: GPU memory released and Python memory cleared.")
