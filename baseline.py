import os
import sqlite3
import requests
import struct
import json
from fuzzywuzzy import fuzz
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

class ChunkingStrategy(BaseModel):
    strategy_name: Literal["naive", "rcts", "semantic_percentile", "semantic_standard_deviation", "semantic_interquartile", "semantic_gradient", "unstructured"]
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


class BaselineRetrievalMethod(RetrievalMethod):
    retrieval_strategy: RetrievalStrategy
    documents: dict[str, Document]
    embedding_infos: list[EmbeddingInfo] | None
    sqlite_db: sqlite3.Connection | None
    sqlite_db_file_path: str | None

    def __init__(self, retrieval_strategy: RetrievalStrategy):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.embedding_infos = None
        self.sqlite_db = None
        self.sqlite_db_file_path = None

    async def cleanup(self) -> None:
        if self.sqlite_db is not None:
            self.sqlite_db.close()
            self.sqlite_db = None
        if self.sqlite_db_file_path is not None and os.path.exists(
            self.sqlite_db_file_path
        ):
            os.remove(self.sqlite_db_file_path)
            self.sqlite_db_file_path = None

    async def ingest_document(self, document: Document) -> None:
        self.documents[document.file_path] = document

    async def sync_all_documents(self) -> None:
        class Chunk(BaseModel):
            document_id: str
            span: tuple[int, int]
            content: str

        # Semantic splitter initialization
        semantic_splitter = None
        if "semantic" in self.retrieval_strategy.chunking_strategy.strategy_name:
            strategy_to_breakpoint_map = {
                "semantic_percentile": "percentile",
                "semantic_standard_deviation": "standard_deviation",
                "semantic_interquartile": "interquartile",
                "semantic_gradient": "gradient",
            }
            breakpoint_type = strategy_to_breakpoint_map.get(chunk_strategy_name)
            semantic_splitter = SemanticChunker(
                embeddings=self.hf_embeddings,
                buffer_size=1,  # Default buffer size
                breakpoint_threshold_type=breakpoint_type,
                breakpoint_threshold_amount=0.05,  # Default threshold; adjust as needed
                sentence_split_regex=r"(?<=[.?!])\s+",
                # sentence_split_regex = r"(?<=[.?!])",
                min_chunk_size=128,  # Minimum chunk size; adjust as needed
            )

        # Calculate chunks
        chunks: list[Chunk] = []
        for document_id, document in self.documents.items():
            # Get chunks
            chunk_size = self.retrieval_strategy.chunking_strategy.chunk_size
            match self.retrieval_strategy.chunking_strategy.strategy_name:
                case "naive":
                    text_splits: list[str] = []
                    for i in range(0, len(document.content), chunk_size):
                        text_splits.append(document.content[i : i + chunk_size])
                case "rcts":
                    synthetic_data_splitter = RecursiveCharacterTextSplitter(
                        separators=[
                            "\n\n",
                            "\n",
                            "!",
                            "?",
                            ".",
                            ":",
                            ";",
                            ",",
                            " ",
                            "",
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=0,
                        length_function=len,
                        is_separator_regex=False,
                        strip_whitespace=False,
                    )
                    text_splits = synthetic_data_splitter.split_text(document.content)
                case strategy if "semantic" in strategy:
                    if semantic_splitter is None:
                        raise ValueError("Semantic splitter not initialized.")
                    text_splits = semantic_splitter.split_text(document.content)
                    print(f"Original content length: {len(document.content)}")
                    print(f"Sum of text splits length: {sum(len(text_split) for text_split in text_splits)}")
                    # print(f"Text splits: {text_splits}")

            # TODO: removed these assertions after adding new precision and recall calculations downstream
            # new calculations account for overlapping chunks so this should work now?
            # assert sum(len(text_split) for text_split in text_splits) == len(
            #     document.content
            # )
            # assert "".join(text_splits) == document.content

            # Get spans from chunks
            prev_span: tuple[int, int] | None = None
            for text_split in text_splits:
                prev_index = prev_span[1] if prev_span is not None else 0
                span = (prev_index, prev_index + len(text_split))
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        span=span,
                        content=text_split,
                    )
                )
                prev_span = span

        # Calculate embeddings
        progress_bar: tqdm | None = None
        if SHOW_LOADING_BAR:
            progress_bar = tqdm(
                total=len(chunks), desc="Processing Embeddings", ncols=100
            )

        EMBEDDING_BATCH_SIZE = 2048
        self.embedding_infos = []
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            chunk_batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
            assert len(chunk_batch) > 0
            embeddings = await ai_embedding(
                self.retrieval_strategy.embedding_model,
                [chunk.content for chunk in chunk_batch],
                AIEmbeddingType.DOCUMENT,
                callback=lambda: (progress_bar.update(1), None)[1]
                if progress_bar
                else None,
            )
            assert len(chunk_batch) == len(embeddings)
            # Save the Info
            if self.sqlite_db is None:
                # random_id = str(uuid4())
                self.sqlite_db_file_path = "./data/cache/baseline.db"
                if os.path.exists(self.sqlite_db_file_path):
                    os.remove(self.sqlite_db_file_path)
                self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
                self.sqlite_db.enable_load_extension(True)
                sqlite_vec.load(self.sqlite_db)
                self.sqlite_db.enable_load_extension(False)
                # Set RAM Usage and create vector table
                self.sqlite_db.execute(f"PRAGMA mmap_size = {3*1024*1024*1024}")
                self.sqlite_db.execute(
                    f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(embeddings[0])}])"
                )

            with self.sqlite_db as db:
                insert_data = [
                    (len(self.embedding_infos) + i, serialize_f32(embedding))
                    for i, embedding in enumerate(embeddings)
                ]
                db.executemany(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    insert_data,
                )
                for chunk, embedding in zip(chunk_batch, embeddings):
                    self.embedding_infos.append(
                        EmbeddingInfo(
                            document_id=chunk.document_id,
                            span=chunk.span,
                            embedding=embedding,
                        )
                    )
        if progress_bar:
            progress_bar.close()

    async def query(self, query: str) -> QueryResponse:
        if self.sqlite_db is None or self.embedding_infos is None:
            raise ValueError("Sync documents before querying!")
        # Get TopK Embedding results
        query_embedding = (
            await ai_embedding(
                self.retrieval_strategy.embedding_model, [query], AIEmbeddingType.QUERY
            )
        )[0]
        rows = self.sqlite_db.execute(
            """
            SELECT
                rowid,
                distance
            FROM vec_items
            WHERE embedding MATCH ?
            AND k = ?
            ORDER BY distance ASC
            """,
            [serialize_f32(query_embedding), self.retrieval_strategy.embedding_topk]
        ).fetchall()

        indices = [cast(int, row[0]) for row in rows]
        retrieved_embedding_infos = [self.embedding_infos[i] for i in indices]

        # Rerank
        if self.retrieval_strategy.rerank_model is not None:
            reranked_indices = await ai_rerank(
                self.retrieval_strategy.rerank_model,
                query,
                texts=[
                    self.get_embedding_info_text(embedding_info)
                    for embedding_info in retrieved_embedding_infos
                ],
            )
            retrieved_embedding_infos = [
                retrieved_embedding_infos[i]
                for i in reranked_indices[: self.retrieval_strategy.rerank_topk]
            ]

        # Get the top retrieval snippets, up until the token limit
        remaining_tokens = self.retrieval_strategy.token_limit
        retrieved_snippets: list[RetrievedSnippet] = []
        for i, embedding_info in enumerate(retrieved_embedding_infos):
            if remaining_tokens is not None and remaining_tokens <= 0:
                break
            span = embedding_info.span
            if remaining_tokens is not None:
                span = (span[0], min(span[1], span[0] + remaining_tokens))
            retrieved_snippets.append(
                RetrievedSnippet(
                    file_path=embedding_info.document_id,
                    span=span,
                    score=1.0 / (i + 1),
                )
            )
            if remaining_tokens is not None:
                remaining_tokens -= span[1] - span[0]
        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def get_embedding_info_text(self, embedding_info: EmbeddingInfo) -> str:
        return self.documents[embedding_info.document_id].content[
            embedding_info.span[0] : embedding_info.span[1]
        ] 

class Chunk(BaseModel):
    """A small container for text chunks before creating EmbeddingInfo objects."""
    document_id: str
    document_path: str
    span: tuple[int, int]
    content: str


class NonSynchronousBaselineRetrievalMethod(RetrievalMethod):
    retrieval_strategy: RetrievalStrategy
    documents: dict[str, Document]
    embedding_infos: list[EmbeddingInfo] | None
    sqlite_db: sqlite3.Connection | None
    sqlite_db_file_path: str | None
    hf_embeddings: HuggingFaceBgeEmbeddings

    def __init__(self, retrieval_strategy: RetrievalStrategy):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.embedding_infos = None
        self.sqlite_db = None
        self.sqlite_db_file_path = None
        # Initialize a single instance of HuggingFaceBgeEmbeddings
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
        

    def cleanup(self) -> None:
        """Close and clean up SQLite database and associated files."""
        if self.sqlite_db is not None:
            self.sqlite_db.close()
            self.sqlite_db = None
        if self.sqlite_db_file_path and os.path.exists(self.sqlite_db_file_path):
            os.remove(self.sqlite_db_file_path)
            self.sqlite_db_file_path = None

    def ingest_document(self, document: Document) -> None:
        """Add a document to the in-memory document store."""
        self.documents[document.file_path] = document

    def sync_all_documents(self) -> None:
        """Splits documents into chunks, embeds them in batches, and stores results in SQLite."""

        # Initialize semantic splitter if needed
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
                    breakpoint_threshold_amount=None,
                    sentence_split_regex=r"(?<=[.?!])\s+",
                    min_chunk_size=128,
                )
            elif self.hf_bge_embeddings is not None:
                semantic_splitter = SemanticChunker(
                    # embeddings=self.hf_embeddings,
                    embeddings=self.hf_bge_embeddings,
                    buffer_size=1,
                    breakpoint_threshold_type=breakpoint_type,
                    breakpoint_threshold_amount=None,
                    sentence_split_regex=r"(?<=[.?!])\s+",
                    min_chunk_size=128,
                )
            elif self.hf_embeddings is not None:
                semantic_splitter = SemanticChunker(
                    # embeddings=self.hf_embeddings,
                    embeddings=self.hf_embeddings,
                    buffer_size=1,
                    breakpoint_threshold_type=breakpoint_type,
                    breakpoint_threshold_amount=None,
                    sentence_split_regex=r"(?<=[.?!])\s+",
                    min_chunk_size=128,
                )

        # 1) Create chunks
        chunks: list[Chunk] = []
        for document_id, document in self.documents.items():
            file_path = document.file_path
            chunk_size = self.retrieval_strategy.chunking_strategy.chunk_size
            if self.retrieval_strategy.chunking_strategy.strategy_name == "naive":
                text_splits = [
                    document.content[i : i + chunk_size]
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
                    kwargs={
                        "combine_text_under_n_chars": 100
                    }
                )
                text_docs = text_splitter.load()
                text_splits = [text.page_content for text in text_docs]
            else:
                raise ValueError(f"Unknown chunking strategy: {self.retrieval_strategy.chunking_strategy.strategy_name}")

            # Collect spans
            prev_span = 0
            for text_split in text_splits:
                span = (prev_span, prev_span + len(text_split))
                chunks.append(Chunk(document_id=document_id, document_path=file_path,span=span, content=text_split))
                prev_span += len(text_split)

        # 2) Embed the chunks in batches
        progress_bar = None
        if SHOW_LOADING_BAR:
            progress_bar = tqdm(total=len(chunks), desc="Processing Embeddings")

        EMBEDDING_BATCH_SIZE = 64  # Adjust based on GPU memory and model performance
        self.embedding_infos = []
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            chunk_batch = chunks[i : i + EMBEDDING_BATCH_SIZE]
            texts = [chunk.content for chunk in chunk_batch]
            # embeddings = self.hf_embeddings.embed_documents(texts)
            if self.ollama_embeddings is not None:
                embeddings = self.ollama_embeddings.embed_documents(texts)
                embedding_model_name = os.environ.get("OLLAMA_EMBEDDING_MODEL")
            elif self.hf_embeddings is not None:
                embeddings = self.hf_embeddings.embed_documents(texts)
                embedding_model_name = os.environ.get("HF_EMBEDDING_MODEL")
            elif self.hf_bge_embeddings is not None:
                embeddings = self.hf_bge_embeddings.embed_documents(texts)
                embedding_model_name = os.environ.get("BGE_EMBEDDING_MODEL")
 
            # Prepare insert_data matching rowid -> embedding
            # so rowid = (current length of embedding_infos) + i
            offset = len(self.embedding_infos)
            insert_data = [
                (offset + j, serialize_f32(emb))
                for j, emb in enumerate(embeddings)
            ]

            # If db is None, initialize as before
            if self.sqlite_db is None:
                base_path = "/home/renyang/jadhav/rag-fyp/legalbenchrag/legalbenchrag/legalbenchrag"
                chunk_strategy = self.retrieval_strategy.chunking_strategy.strategy_name
                embedding_model_name = embedding_model_name.replace("/", "_")
                self.sqlite_db_file_path = f"{base_path}/embeddings_{embedding_model_name}_{chunk_strategy}_{chunk_size}.db"

                # Check if the database already exists
                if not os.path.exists(self.sqlite_db_file_path):
                    print(f"Initializing new SQLite database at {self.sqlite_db_file_path}")
                    self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
                    self.sqlite_db.enable_load_extension(True)
                    sqlite_vec.load(self.sqlite_db)
                    self.sqlite_db.enable_load_extension(False)
                    self.sqlite_db.execute(f"PRAGMA mmap_size = {3 * 1024 * 1024 * 1024}")
                    self.sqlite_db.execute(
                        f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(embeddings[0])}])"
                    )
                else:
                    print(f"Reusing existing SQLite database at {self.sqlite_db_file_path}")
                    self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
            # Old Code
            # if self.sqlite_db is None:
            #     self.sqlite_db_file_path = "./data/cache/baseline.db"
            #     if os.path.exists(self.sqlite_db_file_path):
            #         os.remove(self.sqlite_db_file_path)
            #     self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
            #     self.sqlite_db.enable_load_extension(True)
            #     sqlite_vec.load(self.sqlite_db)
            #     self.sqlite_db.enable_load_extension(False)
            #     self.sqlite_db.execute(f"PRAGMA mmap_size = {3 * 1024 * 1024 * 1024}")
            #     self.sqlite_db.execute(
            #         f"CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[{len(embeddings[0])}])"
            #     )

            # Insert all embeddings at once
            with self.sqlite_db as db:
                db.executemany(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    insert_data,
                )

            # Update self.embedding_infos in parallel
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
        
    
    def query(self, query: str) -> QueryResponse:
        """Process a query and return the top results."""
        if not self.sqlite_db or not self.embedding_infos:
            raise ValueError("Documents not synchronized!")

        if self.ollama_embeddings is not None:
            query_embedding = self.ollama_embeddings.embed_query(query)
        elif self.hf_embeddings is not None:
            query_embedding = self.hf_embeddings.embed_query(query)
        elif self.hf_bge_embeddings is not None:
            query_embedding = self.hf_bge_embeddings.embed_query(query)

        # print(self.retrieval_strategy.embedding_topk)
        rows = self.sqlite_db.execute(
            """
            SELECT rowid, distance
            FROM vec_items
            WHERE embedding MATCH ?
            AND k = ?
            ORDER BY distance ASC;
            """,
            (serialize_f32(query_embedding), self.retrieval_strategy.embedding_topk),
        ).fetchall()

        indices = [cast(int, row[0]) for row in rows]
        # print(f"indices from query: {indices}")
        # print(f"Length of embedding_infos: {len(self.embedding_infos)}")
        retrieved_embedding_infos = [self.embedding_infos[i] for i in indices]

        if self.retrieval_strategy.rerank_model is not None:
            reranked_indices = ai_rerank_sync(
                self.retrieval_strategy.rerank_model,
                query,
                texts=[
                    self.get_embedding_info_text(embedding_info)
                    for embedding_info in retrieved_embedding_infos
                ],
            )
            retrieved_embedding_infos = [
                retrieved_embedding_infos[i]
                for i in reranked_indices[: self.retrieval_strategy.rerank_topk]
            ]

        # Get the top retrieval snippets, up until the token limit
        remaining_tokens = self.retrieval_strategy.token_limit
        retrieved_snippets: list[RetrievedSnippet] = []
        for i, embedding_info in enumerate(retrieved_embedding_infos):
            if remaining_tokens is not None and remaining_tokens <= 0:
                break
            span = embedding_info.span
            if remaining_tokens is not None:
                span = (span[0], min(span[1], span[0] + remaining_tokens))
            retrieved_snippets.append(
                RetrievedSnippet(
                    file_path=embedding_info.document_id,
                    span=span,
                    score=1.0 / (i + 1),
                )
            )
            if remaining_tokens is not None:
                remaining_tokens -= span[1] - span[0]
        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def get_embedding_info_text(self, embedding_info: EmbeddingInfo) -> str:
        """Retrieve text for a given embedding."""
        return self.documents[embedding_info.document_id].content[
            embedding_info.span[0] : embedding_info.span[1]
        ]

