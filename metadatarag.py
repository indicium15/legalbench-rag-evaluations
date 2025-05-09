import os
import sqlite3
import requests
import struct
import json
from fuzzywuzzy import fuzz
import datetime
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
from langchain_unstructured import UnstructuredLoader

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

class RetrievalWithMetadata(RetrievalMethod):
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
        self.total_query_count = 0
        self.successful_metadata_query_count = 0
        self.failed_metadata_queries = []  # store query texts (or IDs) that had no metadata match

        # This attribute will be set for each query:
        self.last_query_metadata_match = False
            
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
        print(f"Saving metadata for {identifier} to {file_path}")

    def load_metadata(self, identifier: str) -> dict | None:
        """
        Load metadata if it exists.

        :param identifier: A unique identifier (e.g., file path or query).
        :return: The loaded metadata dictionary, or None if the metadata file does not exist.
        """
        file_path = self.get_metadata_file_path(identifier)
        
        if os.path.exists(file_path):
            print(f"Loading metadata for {identifier} from {file_path}")
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

    # def sync_all_documents(self) -> None:
    #     """
    #     For each document in self.documents, extract its metadata
    #     and store it (along with the full document content) for later use.
    #     No chunking or embedding is performed at this stage.
    #     """
    #     for document_id, document in self.documents.items():
    #         # Extract metadata (and cache it) for later filtering.
    #         metadata = self._extract_metadata_from_document(document)
    #         self.metadata_store[document.file_path] = metadata
    #     print("All documents synchronized (metadata and full content stored).")

    
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
            elif "unstructured" in self.retrieval_strategy.chunking_strategy.strategy_name:
                corpus_path = "/home/renyang/jadhav/LegalBench-RAG/corpus"
                document_path = os.path.join(corpus_path, file_path)
                text_splitter = UnstructuredLoader(
                    document_path,
                    chunking_strategy="by_title",
                    max_characters=chunk_size,
                    include_orig_elements=False,
                )
                text_docs = text_splitter.load()
                text_splits = [text.page_content for text in text_docs]
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
                (
                    offset + j,
                    serialize_f32(emb),
                    os.path.basename(chunk_batch[j].document_path)  # file_name
                )
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
                # self.sqlite_db.execute(
                #     f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(embeddings[0])}])"
                # )
                self.sqlite_db.execute(
                    f"""CREATE VIRTUAL TABLE vec_items USING vec0(
                        embedding float[{len(embeddings[0])}],
                        file_name text,
                    )"""
                )  

            # Insert embeddings into the database
            with self.sqlite_db as db:
                db.executemany(
                    # "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    "INSERT INTO vec_items(rowid, embedding, file_name) VALUES (?, ?, ?)",
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

    # def query(self, query: str) -> QueryResponse:
    #     if not self.documents:
    #         raise ValueError("No documents available!")
        
    #     # Increment query count for tracking
    #     self.total_query_count += 1

    #     # 1. Extract metadata from the query.
    #     query_metadata = self._extract_metadata_from_query(query)
    #     print(f"Detected query metadata: {query_metadata}")

    #     # 2. Filter documents by metadata.
    #     # Use your existing matches() function to compare each document’s metadata.
    #     matching_docs = [
    #         doc for doc in self.documents.values()
    #         if any(
    #             self.matches(doc_metadata_value, query_metadata.get("Company Name", []))
    #             for doc_metadata_value in self.metadata_store.get(doc.file_path, {}).get("Company Name", [])
    #         )
    #     ]
    #     if not matching_docs:
    #         print("No documents passed the metadata filter!")
    #         self.failed_metadata_queries.append(query)
    #         return QueryResponse(retrieved_snippets=[])
        
    #     # Record successful metadata match.
    #     self.successful_metadata_query_count += 1

    #     # 3. Now, for each matching document, perform chunking and compute embeddings.
    #     semantic_splitter = None
    #     # Use the same chunking logic as before
    #     if "semantic" in self.retrieval_strategy.chunking_strategy.strategy_name:
    #         strategy_to_breakpoint_map = {
    #             "semantic_percentile": "percentile",
    #             "semantic_standard_deviation": "standard_deviation",
    #             "semantic_interquartile": "interquartile",
    #             "semantic_gradient": "gradient",
    #         }
    #         breakpoint_type = strategy_to_breakpoint_map.get(
    #             self.retrieval_strategy.chunking_strategy.strategy_name
    #         )
    #         if self.ollama_embeddings is not None:
    #             semantic_splitter = SemanticChunker(
    #                 embeddings=self.ollama_embeddings,
    #                 buffer_size=1,
    #                 breakpoint_threshold_type=breakpoint_type,
    #                 breakpoint_threshold_amount=0.05,
    #                 sentence_split_regex=r"(?<=[.?!])\s+",
    #                 min_chunk_size=128,
    #             )
    #         elif self.hf_bge_embeddings is not None:
    #             semantic_splitter = SemanticChunker(
    #                 embeddings=self.hf_bge_embeddings,
    #                 buffer_size=1,
    #                 breakpoint_threshold_type=breakpoint_type,
    #                 breakpoint_threshold_amount=0.05,
    #                 sentence_split_regex=r"(?<=[.?!])\s+",
    #                 min_chunk_size=128,
    #             )
    #         elif self.hf_embeddings is not None:
    #             semantic_splitter = SemanticChunker(
    #                 embeddings=self.hf_bge_embeddings,  # (Note: here, you appear to use hf_bge_embeddings as fallback)
    #                 buffer_size=1,
    #                 breakpoint_threshold_type=breakpoint_type,
    #                 breakpoint_threshold_amount=0.05,
    #                 sentence_split_regex=r"(?<=[.?!])\s+",
    #                 min_chunk_size=128,
    #             )

    #     chunk_size = self.retrieval_strategy.chunking_strategy.chunk_size
    #     candidate_snippets = []

    #     for doc in matching_docs:
    #         # Decide on the chunking method based on your configuration.
    #         if self.retrieval_strategy.chunking_strategy.strategy_name == "naive":
    #             text_splits = [
    #                 doc.content[i: i + chunk_size]
    #                 for i in range(0, len(doc.content), chunk_size)
    #             ]
    #         elif self.retrieval_strategy.chunking_strategy.strategy_name == "rcts":
    #             from langchain_text_splitters import RecursiveCharacterTextSplitter
    #             text_splitter = RecursiveCharacterTextSplitter(
    #                 separators=["\n\n", "\n", ".", " "],
    #                 chunk_size=chunk_size,
    #                 chunk_overlap=0,
    #             )
    #             text_splits = text_splitter.split_text(doc.content)
    #         elif "semantic" in self.retrieval_strategy.chunking_strategy.strategy_name:
    #             if not semantic_splitter:
    #                 raise ValueError("Semantic splitter not initialized.")
    #             text_splits = semantic_splitter.split_text(doc.content)
    #         else:
    #             raise ValueError(f"Unknown chunking strategy: {self.retrieval_strategy.chunking_strategy.strategy_name}")

    #         # Create Chunk objects for the document.
    #         chunks = []
    #         prev_span = 0
    #         for text_split in text_splits:
    #             span = (prev_span, prev_span + len(text_split))
    #             chunks.append(Chunk(document_id=doc.file_path, document_path=doc.file_path, span=span, content=text_split))
    #             prev_span += len(text_split)
            
    #         # Embed the chunks.
    #         texts = [chunk.content for chunk in chunks]
    #         if self.ollama_embeddings is not None:
    #             chunk_embeddings = self.ollama_embeddings.embed_documents(texts)
    #         elif self.hf_embeddings is not None:
    #             chunk_embeddings = self.hf_embeddings.embed_documents(texts)
    #         elif self.hf_bge_embeddings is not None:
    #             chunk_embeddings = self.hf_bge_embeddings.embed_documents(texts)
    #         else:
    #             raise ValueError("No embedding model available!")
            
    #         # 4. Get the query embedding (using the same logic as before).
    #         if self.ollama_embeddings is not None:
    #             query_embedding = self.ollama_embeddings.embed_query(query)
    #         elif self.hf_embeddings is not None:
    #             query_embedding = self.hf_embeddings.embed_query(query)
    #         elif self.hf_bge_embeddings is not None:
    #             query_embedding = self.hf_bge_embeddings.embed_query(query)
    #         else:
    #             raise ValueError("No embedding model available for query embedding!")
            
    #         # 5. Compute similarity scores between query_embedding and each chunk embedding.
    #         # (Here you can use cosine similarity since embeddings are normalized.)
    #         for chunk, emb in zip(chunks, chunk_embeddings):
    #             # Example: cosine similarity = dot(query_embedding, emb)
    #             # Since the embeddings are normalized, higher dot product means higher similarity.
    #             score = sum(q * v for q, v in zip(query_embedding, emb))
    #             candidate_snippets.append(
    #                 RetrievedSnippet(
    #                     file_path=doc.file_path,
    #                     span=chunk.span,
    #                     score=score,
    #                     content=chunk.content,
    #                     reasoning=""
    #                 )
    #             )
        
    #     # 6. Sort candidate snippets by score descending (higher is better).
    #     top_k = self.retrieval_strategy.embedding_topk
    #     candidate_snippets.sort(key=lambda s: s.score, reverse=True)
    #     top_snippets = candidate_snippets[:top_k]

    #     # Optionally, rerank using your rerank model.
    #     if self.retrieval_strategy.rerank_model is not None:
    #         print(f"Reranking {len(top_snippets)} snippets...")
    #         reranked_indices = ai_rerank_sync(
    #             self.retrieval_strategy.rerank_model,
    #             query,
    #             texts=[snippet.content for snippet in top_snippets]
    #         )
    #         top_snippets = [top_snippets[i] for i in reranked_indices[:top_k]]
        
    #     return QueryResponse(retrieved_snippets=top_snippets)


    def query(self, query: str) -> QueryResponse:
        """
        Process a query by detecting metadata, performing retrieval, and dynamically expanding results if needed.

        :param query: The user query in natural language.
        :return: A QueryResponse object containing filtered results.
        """
        if not self.sqlite_db or not self.embedding_infos:
            raise ValueError("Documents not synchronized!")

        # Extract metadata from the query
        self.total_query_count += 1
        query_metadata = self._extract_metadata_from_query(query)
        print(f"Detected query metadata: {query_metadata}")

        # Perform retrieval and filtering
        if self.ollama_embeddings is not None:
            query_embedding = self.ollama_embeddings.embed_query(query)
        elif self.hf_embeddings is not None:
            query_embedding = self.hf_embeddings.embed_query(query)
        elif self.hf_bge_embeddings is not None:
            query_embedding = self.hf_bge_embeddings.embed_query(query)
        if self.retrieval_strategy.rerank_model is not None:
            retrieved_snippets = self._retrieve_and_filter(query, query_embedding, query_metadata, self.retrieval_strategy.rerank_topk)
        else:
            retrieved_snippets = self._retrieve_and_filter(query, query_embedding, query_metadata, self.retrieval_strategy.embedding_topk)
        # Update counters based on whether metadata matching was successful.
        if self.last_query_metadata_match:
            self.successful_metadata_query_count += 1
        else:
            self.failed_metadata_queries.append(query)
            self.successful_metadata_query_count = max(0, self.successful_metadata_query_count - 1)

        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def _extract_metadata_from_document(self, document: Document, max_content_length: int = 2500) -> dict:
        identifier = document.file_path
        cached_metadata = self.load_metadata(identifier)
        if cached_metadata and not (cached_metadata.get("Company Name") == ["None"] and cached_metadata.get("Contract Type") == ["None"]):
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


    def _extract_metadata_from_query(self, query: str) -> dict:
        """
        Extract metadata (company name, contract type) from a query using an LLM hosted with Ollama.

        :param query: The input query to analyze.
        :return: Extracted metadata as a dictionary.
        """
        
        # Check if metadata is already cached
        identifier = f"query:{query}"
        cached_metadata = self.load_metadata(identifier)
        if cached_metadata and not (cached_metadata.get("Company Name") == ["None"] and cached_metadata.get("Contract Type") == ["None"]):
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

            Now, process the following input and return the content in a JSON code block. If you do not return the output in a code block, you will be terminated:

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

            Example 5:
            Input: File Name: "A.txt"; Content: "This agreement is signed by CompanyX."
            Output: {{
                "Company Name": ["A", "CompanyX"],
                "Contract Type": ["None"]
            }}

            It is critical that you provide the output in a JSON code block format. If you do not, you will be terminated.
            Input: File Name: "{os.path.basename(file_name)}"; Content: "{content}"
            Output:
        """

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

    def get_embedding_info_text(self, snippet_or_info) -> str:
        """
        Retrieve text for a given snippet or embedding information.
            
        :param snippet_or_info: Either a RetrievedSnippet or EmbeddingInfo object.
        :return: The corresponding text from the document.
        """
        if isinstance(snippet_or_info, RetrievedSnippet):
            document = self.documents.get(snippet_or_info.file_path)
            if not document:
                raise ValueError(f"Document not found for file path: {snippet_or_info.file_path}")
            return document.content[
                snippet_or_info.span[0]: snippet_or_info.span[1]
            ]
        elif isinstance(snippet_or_info, EmbeddingInfo):
            document = self.documents.get(snippet_or_info.document_id)
            if not document:
                raise ValueError(f"Document not found for document ID: {snippet_or_info.document_id}")
            return document.content[
                snippet_or_info.span[0]: snippet_or_info.span[1]
            ]
        else:
            raise TypeError("Expected RetrievedSnippet or EmbeddingInfo.")


    def _retrieve_and_filter(
        self,
        query,
        query_embedding: list[float],
        metadata_filter: dict,
        initial_top_k: int = 10,
        expand_factor: int = 2,
        max_tries: int = 5,
        embedding_top_k: int = 50  # Initial embedding filtering size
    ) -> list[RetrievedSnippet]:
        """
        Retrieve and filter snippets based on metadata, with fallback to similarity-based results and reranking.
        """
        current_top_k = embedding_top_k
        filtered_snippets: list[RetrievedSnippet] = []  # Snippets filtered by metadata
        fallback_snippets: list[RetrievedSnippet] = []  # Fallback similarity-based snippets
        excluded_row_ids = set()  # Keep track of already-processed rows
        remaining_tokens = self.retrieval_strategy.token_limit
        tries = 0

        matching_file_names = set()
        for file_path, metadata in self.metadata_store.items():
            company_names = metadata.get("Company Name", [])
            query_company_names = metadata_filter.get("Company Name", [])
            print(f"Query Company Name: {query_company_names}")

            if any(self.matches(doc_name, query_company_names) for doc_name in company_names):
                matching_file_names.add(os.path.basename(file_path))
                self.last_query_metadata_match = True
                print(f"Identified {os.path.basename(file_path)} as a matching file")

        if not matching_file_names:
            print("No matching file names found in metadata filtering.")
            print("Falling back to similarity based searching.")
            matching_file_names = {os.path.basename(file_path) for file_path in self.metadata_store.keys()} 
            self.last_query_metadata_match = False
            # return []
        
        while len(filtered_snippets) < initial_top_k and tries < max_tries:
            # Increment the number of tries
            tries += 1

            # Generate SQL query dynamically based on excluded_row_ids
            if excluded_row_ids:
                file_placeholders = ",".join(["?"] * len(matching_file_names))
                rowid_placeholders = ",".join("?" for _ in excluded_row_ids)
                query_sql = f"""
                    SELECT rowid, distance
                    FROM vec_items
                    WHERE embedding MATCH ?
                    AND file_name IN ({placeholders})
                    AND rowid NOT IN ({rowid_placeholders})
                    AND k = ?
                    ORDER BY distance ASC
                """
                # params = [serialize_f32(query_embedding), *excluded_row_ids, current_top_k]
                # params = [serialize_f32(query_embedding), current_top_k]
                params = [serialize_f32(query_embedding)] + list(matching_file_names) + list(excluded_row_ids) + [current_top_k]
            else:
                placeholders = ",".join(["?"] * len(matching_file_names))
                query_sql = f"""
                    SELECT rowid, distance
                    FROM vec_items
                    WHERE embedding MATCH ?
                    AND file_name IN ({placeholders})
                    AND k = ?
                    ORDER BY distance ASC
                """
                # params = [serialize_f32(query_embedding), *matching_file_names, current_top_k]
                params = [serialize_f32(query_embedding)] + list(matching_file_names) + [current_top_k]

            rows = self.sqlite_db.execute(query_sql, params).fetchall()
            # print(f"Attempt {tries}: Number of rows retrieved: {len(rows)}")

            if not rows:
                break  # Exit if no more results are found

            # Extract indices and corresponding embedding infos
            indices = [row[0] for row in rows]
            excluded_row_ids.update(indices)  # Mark these rows as processed
            retrieved_embedding_infos = [self.embedding_infos[i] for i in indices]

            # Collect similarity-based snippets for fallback
            fallback_snippets.extend([
                RetrievedSnippet(
                    file_path=embedding_info.document_path,
                    span=embedding_info.span,
                    score=1.0 / (i + 1),  # Positional scoring
                )
                for i, embedding_info in enumerate(retrieved_embedding_infos)
            ])

            # Step 1: Filter by metadata
            for i, embedding_info in enumerate(retrieved_embedding_infos):
                # Because we already know the file_name matched (and thus the metadata matched),
                # you can directly add the snippet without checking the same “Company Name” again.
                span = embedding_info.span
                if remaining_tokens is not None and remaining_tokens <= 0:
                    break
                if remaining_tokens is not None:
                    span = (span[0], min(span[1], span[0] + remaining_tokens))

                snippet = RetrievedSnippet(
                    file_path=embedding_info.document_path,
                    span=span,
                    score=1.0 / (i + 1),
                )
                filtered_snippets.append(snippet)
                if remaining_tokens is not None:
                    remaining_tokens -= span[1] - span[0]


            # Stop if we have collected enough filtered snippets
            if len(filtered_snippets) >= initial_top_k:
                break

            # Expand the search window and continue searching
            current_top_k *= expand_factor
        
        # Fallback if not enough matching snippets are found
        if len(filtered_snippets) < initial_top_k:
            print("Falling back to similarity-based snippets.")
            fallback_snippets.sort(key=lambda snippet: snippet.score, reverse=True)
            filtered_snippets.extend(fallback_snippets[:initial_top_k - len(filtered_snippets)])

        # Step 2: Rerank the filtered snippets
        if self.retrieval_strategy.rerank_model is not None:
            print(f"Reranking {len(filtered_snippets)} snippets...")
            reranked_indices = ai_rerank_sync(
                self.retrieval_strategy.rerank_model,
                query,
                texts=[
                    self.get_embedding_info_text(snippet)
                    for snippet in filtered_snippets
                ],
            )
            # Reorder the filtered snippets based on reranked indices
            filtered_snippets = [filtered_snippets[i] for i in reranked_indices[:initial_top_k]]
            rerank_topk = self.retrieval_strategy.rerank_topk
            return filtered_snippets[:rerank_topk]

        # Return the top_k matching and reranked snippets
        return filtered_snippets[:initial_top_k]

    def _retrieve_from_db(self, query_embedding: list[float], top_k: int, expand_factor: int = 2) -> list[tuple]:
        """
        Retrieve matching rows from the database based on the query embedding with over-fetching.

        :param query_embedding: The embedding for the query.
        :param top_k: The desired number of results.
        :param expand_factor: Factor to expand the search window for over-fetching.
        :return: List of rows from the database.
        """
        print(f"Querying for top k: {top_k}")
        print(f"Expand factor: {expand_factor}")
        rows = self.sqlite_db.execute(
            """
            SELECT rowid, distance
            FROM vec_items
            WHERE embedding MATCH ?
            AND k = ?
            ORDER BY distance ASC
            """,
            (serialize_f32(query_embedding), min(expand_factor * top_k, 4096)),
        ).fetchall()
        print(f"Database returned {len(rows)} rows.")
        return rows

    def clean_text(self, text):
        """Remove special characters and extra spaces, making comparison stricter."""
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
        return " ".join(text.split())  # Normalize spaces

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

        return False  # If none of the conditions are met, it's not a match

    def _filter_snippets_by_metadata(self, snippets: list[RetrievedSnippet], metadata_filter: dict) -> list[RetrievedSnippet]:
        """
        Filter retrieved snippets based on metadata.
        Supports multiple values for metadata fields with relaxed matching.

        :param snippets: The retrieved snippets to filter.
        :param metadata_filter: The metadata conditions to apply.
        :return: A list of snippets that match the metadata filter.
        """
        filtered_snippets = []
        for snippet in snippets:
            file_metadata = self.metadata_store.get(snippet.file_path, {})
            print(f"File Metadata in filter snippets: {file_metadata}")
            # Check relaxed match for each metadata key
            if all(
                matches(self, file_metadata.get(key, ""), value)
                for key, value in metadata_filter.items()
            ):
                filtered_snippets.append(snippet)
        print(f"Filtered Snippets: {filtered_snippets}")
        return filtered_snippets

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
         # Print out query metadata matching statistics.
        print(f"Total queries processed: {self.total_query_count}")
        print(f"Queries with successful metadata matching: {self.successful_metadata_query_count}")
        print("Queries without metadata matching:")
        for query in self.failed_metadata_queries:
            print(f"  - {query}")
        if self.total_query_count > 0:
            success_percent = (self.successful_metadata_query_count / self.total_query_count) * 100
        else:
            success_percent = 0.0

        # Get the metadata directory (the same base_dir used in get_metadata_file_path)
        metadata_dir = os.path.join("/home/renyang/jadhav/rag-fyp/legalbenchrag/legalbenchrag/legalbenchrag", "generated_metadata")
        metadata_files = []
        if os.path.exists(metadata_dir):
            metadata_files = [os.path.join(metadata_dir, f) for f in os.listdir(metadata_dir) if f.endswith(".json")]

        # Write the statistics and file paths to a text file
        # Get the current timestamp
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Generate the filename with timestamp
        stats_file = os.path.join(metadata_dir, f"cleanup_stats_{timestamp}.txt")

        with open(stats_file, "w", encoding="utf-8") as f:
            f.write("Query Metadata Matching Statistics\n")
            f.write(f"Total queries processed: {self.total_query_count}\n")
            f.write(f"Successful metadata queries: {self.successful_metadata_query_count}\n")
            f.write(f"Failed metadata queries: {len(self.failed_metadata_queries)}\n")
            f.write(f"Percentage successful: {success_percent:.2f}%\n\n")
            f.write("Failed Queries:\n")
            for q in self.failed_metadata_queries:
                f.write(f"{q}\n")
            f.write("\nMetadata File Paths:\n")
            for file_path in metadata_files:
                f.write(f"{file_path}\n")
        print(f"Cleanup statistics saved in: {stats_file}")
