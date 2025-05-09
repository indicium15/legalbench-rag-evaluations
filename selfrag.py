import os
import sqlite3
import requests
import struct
import json
from fuzzywuzzy import fuzz
import re
import hashlib
from typing import Literal, cast
import concurrent.futures
from transformers import AutoTokenizer
import time
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langgraph.graph import END, StateGraph, START
from langgraph.errors import GraphRecursionError
from langchain_ollama import OllamaEmbeddings, ChatOllama
import sqlite_vec  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import torch
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pprint import pprint
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
from typing import List
from typing_extensions import TypedDict


def serialize_f32(vector: list[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack(f"{len(vector)}f", *vector)


SHOW_LOADING_BAR = True
load_dotenv()

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

class Chunk(BaseModel):
    """A small container for text chunks before creating EmbeddingInfo objects."""
    document_id: str
    document_path: str
    span: tuple[int, int]
    content: str

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

# Document Grader Chain
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# LLM with function call
llm = ChatOllama(model="llama3.1", format="json", temperature=0)
# llm = ChatOllama(model="deepseek-r1:14b", format="json", temperature=0)
prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

# RAG Chain
prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = prompt | llm | StrOutputParser()

# Hallucination Grader Chain
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm | JsonOutputParser()

# Answer Grader
prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n 
    Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()

# Question Re-Writer
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}.\n
     The improved query must start with the below string: Given a web search query, retrieve relevant passages that answer the query: \n 
     Improved question: \n """,

    input_variables=["generation", "question"],
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

class SelfRAGRetrievalMethod(RetrievalMethod):
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

        # Embedding model initialization
        if os.environ.get("HF_EMBEDDING_MODEL") != "":
            model_name = os.environ.get("HF_EMBEDDING_MODEL")
            model_kwargs = {"device": "cuda"}
            encode_kwargs = {"normalize_embeddings": True}
            self.hf_embeddings = HuggingFaceEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
        else:
            self.hf_embeddings = None

        if os.environ.get("BGE_EMBEDDING_MODEL") != "":
            model_name = os.environ.get("BGE_EMBEDDING_MODEL")
            model_kwargs = {"device": "cuda"}
            encode_kwargs = {"normalize_embeddings": True}
            self.hf_bge_embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
        else:
            self.hf_bge_embeddings = None

        if os.environ.get("OLLAMA_EMBEDDING_MODEL") != "":
            self.ollama_embeddings = OllamaEmbeddings(
                model=os.getenv("OLLAMA_EMBEDDING_MODEL")
            )
        else:
            self.ollama_embeddings = None

        # Initialize LangGraph workflow
        self.workflow = self._initialize_graph_workflow()

    def ingest_document(self, document: Document) -> None:
        """Ingest a document into the retrieval method."""
        self.documents[document.file_path] = document

    def sync_all_documents(self) -> None:
        """Synchronize all ingested documents into embeddings and store in SQLite."""
        # Initialize semantic splitter if needed
        semantic_splitter = self._initialize_semantic_splitter()

        # Create document chunks
        chunks = []
        for document_id, document in self.documents.items():
            file_path = document.file_path
            chunk_size = self.retrieval_strategy.chunking_strategy.chunk_size
            text_splits = self._split_document(document, chunk_size, semantic_splitter)

            # Collect spans
            prev_span = 0
            for text_split in text_splits:
                span = (prev_span, prev_span + len(text_split))
                chunks.append(Chunk(document_id=document_id, document_path=file_path, span=span, content=text_split))
                prev_span += len(text_split)

        # Embed chunks and store in SQLite
        self._embed_and_store_chunks(chunks)

    def query(self, question: str) -> QueryResponse:
        """Run the retrieval method on the given dataset using the graph workflow."""
        state = {"question": question}

        try:
            # Initialize the last valid state to None
            last_valid_state = None

            # Run the workflow
            for output in self.workflow.stream(state):
                # Print the state of each node
                print("Intermediate State:")
                for key, value in output.items():
                    print(f"Node '{key}':")
                    print(value)

                # Save the last valid state in case of recursion errors
                last_valid_state = output

            final_state = output
        except GraphRecursionError:
            print("Recursion limit reached. Returning the last valid state.")
            final_state = last_valid_state

        print(f"Final State: {final_state}")

        # Extract 'generation' and 'documents' from the 'generate' key in final_state
        # generation = final_state.get("generate", {}).get("generation", "")
        # documents = final_state.get("generate", {}).get("documents", [])
        if "retrieve" in final_state:
            documents = final_state["retrieve"].get("documents", [])
        elif "grade_documents" in final_state:
            documents = final_state["grade_documents"].get("documents", [])

        # Convert retrieved documents back into RetrievedSnippet objects
        retrieved_snippets = [
            RetrievedSnippet(
                file_path=doc["document_id"],  # Use dictionary key access
                span=doc["span"],             # Use dictionary key access
                answer=doc["content"],
                score=1.0 / (i + 1),
            )
            for i, doc in enumerate(documents)
        ]

        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def cleanup(self) -> None:
        """Cleanup resources such as SQLite connections and temporary files."""
        if self.sqlite_db is not None:
            self.sqlite_db.close()
            self.sqlite_db = None

        if self.sqlite_db_file_path and os.path.exists(self.sqlite_db_file_path):
            os.remove(self.sqlite_db_file_path)
            self.sqlite_db_file_path = None

    # Helper methods
    def get_embedding_info_text(self, embedding_info: EmbeddingInfo) -> str:
        """
        Retrieve the text corresponding to an EmbeddingInfo object.

        Args:
            embedding_info (EmbeddingInfo): The embedding information object.

        Returns:
            str: The text corresponding to the span in the original document.
        """
        # Get the document based on document_id
        document = self.documents.get(embedding_info.document_id)

        if not document:
            raise ValueError(f"Document with ID {embedding_info.document_id} not found.")

        # Extract the text based on the span
        start, end = embedding_info.span
        return document.content[start:end]

    def _retrieve_node(self, state):
        """
        Retrieve documents using SQL-based retrieval.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Updated state with retrieved documents.
        """
        # New logic
        print("---RETRIEVE---")
        preamble = "Given a web search query, retrieve relevant passages that answer the query: "
        question_text = state["question"]
        full_question = preamble + question_text

        # Load Hugging Face SentenceTransformer model
        model_name = os.environ.get("HF_EMBEDDING_MODEL")
        if not model_name:
            raise ValueError("HF_EMBEDDING_MODEL is not set in environment variables.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_tokens = 512  # Adjust as per model token limit

        # Tokenize full question
        tokenized_question = tokenizer(full_question, truncation=False)["input_ids"]

        # If exceeds max_tokens, truncate from the START of question_text
        if len(tokenized_question) > max_tokens:
            excess_tokens = len(tokenized_question) - max_tokens
            # Tokenize the question_text separately
            question_tokens = tokenizer(question_text, truncation=False)["input_ids"]
            truncated_question_tokens = question_tokens[excess_tokens:]  # Remove excess from start
            truncated_question_text = tokenizer.decode(truncated_question_tokens, skip_special_tokens=True)

            # Reconstruct the final query with the preamble
            full_question = preamble + truncated_question_text

        # print(f"Final query (tokens: {len(tokenizer(full_question)['input_ids'])}): {full_question}")

        # Perform retrieval and filtering
        if self.ollama_embeddings is not None:
            query_embedding = self.ollama_embeddings.embed_query(full_question)
        elif self.hf_embeddings is not None:
            query_embedding = self.hf_embeddings.embed_query(full_question)
        elif self.hf_bge_embeddings is not None:
            query_embedding = self.hf_bge_embeddings.embed_query(full_question)

        # print("---RETRIEVE---")
        # question = state["question"]
        # question = f"Given a web search query, retrieve relevant passages that answer the query: {state['question']}"

        # # Perform retrieval and filtering
        # if self.ollama_embeddings is not None:
        #     query_embedding = self.ollama_embeddings.embed_query(question)
        # elif self.hf_embeddings is not None:
        #     query_embedding = self.hf_embeddings.embed_query(question)
        # elif self.hf_bge_embeddings is not None:
        #     query_embedding = self.hf_bge_embeddings.embed_query(question)

        # Perform SQL-based embedding search
        rows = self.sqlite_db.execute(
            """
            SELECT rowid, distance
            FROM vec_items
            WHERE embedding MATCH ?
            AND k = ?
            ORDER BY distance ASC
            """,
            (serialize_f32(query_embedding), self.retrieval_strategy.embedding_topk),
        ).fetchall()

        # Retrieve document embedding info
        indices = [row[0] for row in rows]
        retrieved_docs = [self.embedding_infos[i] for i in indices]
        if self.retrieval_strategy.rerank_model is not None:
            reranked_indices = ai_rerank_sync(
                self.retrieval_strategy.rerank_model,
                full_question,
                texts = [
                    self.get_embedding_info_text(embedding_info)
                    for embedding_info in retrieved_docs
                ]
            )
            retrieved_docs = [
                retrieved_docs[i]
                for i in reranked_indices[:self.retrieval_strategy.rerank_topk] 
            ]
        # Convert retrieved documents into the expected format
        print(retrieved_docs)
        documents = [
            {
                "document_id": doc.document_id,
                "content": self.get_embedding_info_text(doc),
                "span": doc.span,  # Preserve the chunk's span
            }
            for doc in retrieved_docs
        ]
        print(f"Retrieved Documents: {documents}")

        return {"documents": documents, "question": full_question}

    # def _generate_node(self, state):
    #     """
    #     Generate an answer using retrieved documents.

    #     Args:
    #         state (dict): The current graph state.

    #     Returns:
    #         state (dict): Updated state with the generated answer.
    #     """
    #     print("---GENERATE---")
    #     question = state["question"]
    #     documents = state["documents"]

    #     # Combine document contents for RAG generation
    #     context = "\n\n".join(doc["content"] for doc in documents)

    #     # Generate answer using the RAG chain
    #     generation = rag_chain.invoke({"context": context, "question": question})
    #     print(f"Generation: {generation}")

    #     return {"documents": documents, "question": question, "generation": generation}


    def _generate_node(self, state):
        """
        Generate an answer using retrieved documents with a hard timeout.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Updated state with the generated answer.
        """
        TIMEOUT_SECONDS = 15  # Max time to wait before failing
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # Combine document contents for RAG generation
        context = "\n\n".join(doc["content"] for doc in documents)

        def generate_response():
            return rag_chain.invoke({"context": context, "question": question})

        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(generate_response)
                generation = future.result(timeout=TIMEOUT_SECONDS)  # Enforce timeout
                print(f"Generation Successful: {generation}")
                return {"documents": documents, "question": question, "generation": generation}

        except concurrent.futures.TimeoutError:
            print(f"Timeout: Generation failed after {TIMEOUT_SECONDS} seconds.")
        except Exception as e:
            print(f"Error: {e}")

        print("Returning failure response.")
        return {"documents": documents, "question": question, "generation": "Failed to generate response."}

    def _grade_documents_node(self, state):
        """
        Grade retrieved documents for relevance to the question.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Updated state with only relevant documents.
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        print(f"Question: {question}")
        documents = state["documents"]
        print(f"Documents: {documents}")

        # Score each document
        filtered_docs = []
        for doc in documents:
            score = retrieval_grader.invoke({"question": question, "document": doc["content"]})
            print(score)
            grade = score['score']
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue

        return {"documents": filtered_docs, "question": question}

    def _transform_query_node(self, state):
        """
        Transform the query to optimize for retrieval.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Updated state with the rewritten query.
        """
        print("---TRANSFORM QUERY---")
        question = state["question"]

        # Rewrite the question
        better_question = question_rewriter.invoke({"question": question})

        return {"documents": state["documents"], "question": better_question}


    def _decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    def _grade_generation_v_documents_and_question(self, state):
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """

        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        score = hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score['score']

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = answer_grader.invoke({"question": question, "generation": generation})
            grade = score['score']
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"

    def _initialize_graph_workflow(self):
        """Initialize the graph workflow for RAG-based retrieval."""
        # New version without generate
        workflow = StateGraph(GraphState)
        # Add nodes (excluding 'generate')
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("transform_query", self._transform_query_node)

        # Add edges (modified)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "transform_query": "transform_query",
                # Since "generate" is removed, route successful paths to END
                "generate": END,  # Or, you could route it to another useful node
            },
        )
        workflow.add_edge("transform_query", "retrieve")

        # Compile the workflow
        return workflow.compile()


        # Old version with generate
        # workflow = StateGraph(GraphState)

        # # Add nodes
        # workflow.add_node("retrieve", self._retrieve_node)
        # workflow.add_node("grade_documents", self._grade_documents_node)
        # workflow.add_node("generate", self._generate_node)
        # workflow.add_node("transform_query", self._transform_query_node)

        # # Add edges
        # workflow.add_edge(START, "retrieve")
        # workflow.add_edge("retrieve", "grade_documents")
        # workflow.add_conditional_edges(
        #     "grade_documents",
        #     self._decide_to_generate,
        #     {
        #         "transform_query": "transform_query",
        #         "generate": "generate",
        #     },
        # )
        # workflow.add_edge("transform_query", "retrieve")
        # workflow.add_conditional_edges(
        #     "generate",
        #     self._grade_generation_v_documents_and_question,
        #     {
        #         "not supported": "generate",
        #         "useful": END,
        #         "not useful": "transform_query",
        #     },
        # )

        # # Compile the workflow
        # return workflow.compile()

    def _initialize_semantic_splitter(self):
        """Initialize a semantic splitter based on the retrieval strategy."""
        if "semantic" not in self.retrieval_strategy.chunking_strategy.strategy_name:
            return None

        breakpoint_type = {
            "semantic_percentile": "percentile",
            "semantic_standard_deviation": "standard_deviation",
            "semantic_interquartile": "interquartile",
            "semantic_gradient": "gradient",
        }.get(self.retrieval_strategy.chunking_strategy.strategy_name)

        embeddings = (
            self.ollama_embeddings
            or self.hf_bge_embeddings
            or self.hf_embeddings
        )

        return SemanticChunker(
            embeddings=embeddings,
            buffer_size=1,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=0.05,
            sentence_split_regex=r"(?<=[.?!])\s+",
            min_chunk_size=128,
        )

    def _split_document(self, document: Document, chunk_size: int, semantic_splitter):
        """Split a document into chunks."""
        # print(f"SELECTED STRATEGY: {self.retrieval_strategy.chunking_strategy.strategy_name}")
        if self.retrieval_strategy.chunking_strategy.strategy_name == "naive":
            return [
                document.content[i : i + chunk_size]
                for i in range(0, len(document.content), chunk_size)
            ]

        if self.retrieval_strategy.chunking_strategy.strategy_name == "rcts":
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ".", " "],
                chunk_size=chunk_size,
                chunk_overlap=0,
            )
            return text_splitter.split_text(document.content)

        if "semantic" in self.retrieval_strategy.chunking_strategy.strategy_name:
            if not semantic_splitter:
                raise ValueError("Semantic splitter not initialized.")
            return semantic_splitter.split_text(document.content)
        
        if self.retrieval_strategy.chunking_strategy.strategy_name == "unstructured":
            corpus_path = "/home/renyang/jadhav/LegalBench-RAG/corpus"
            file_path = document.file_path
            document_path = os.path.join(corpus_path, file_path)
            text_splitter = UnstructuredLoader(
                document_path,
                chunking_strategy="by_title",
                max_characters=chunk_size,
                include_orig_elements=False,
            )
            text_docs = text_splitter.load()
            text_splits = [text.page_content for text in text_docs]
            return text_splits

        raise ValueError(f"Unknown chunking strategy: {self.retrieval_strategy.chunking_strategy.strategy_name}")

    def _embed_and_store_chunks(self, chunks: list[Chunk]) -> None:
        """Embed document chunks and store them in SQLite."""
        if self.sqlite_db is None:
            self._initialize_sqlite_db()

        progress_bar = tqdm(total=len(chunks), desc="Processing Embeddings")
        batch_size = 64
        self.embedding_infos = []

        for i in range(0, len(chunks), batch_size):
            chunk_batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in chunk_batch]
            embeddings = self._embed_texts(texts)

            # Prepare data for SQLite
            offset = len(self.embedding_infos)
            insert_data = [
                (offset + j, serialize_f32(emb))
                for j, emb in enumerate(embeddings)
            ]

            # Store embeddings
            with self.sqlite_db as db:
                db.executemany(
                    "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    insert_data,
                )

            # Update embedding infos
            for j, (chunk, emb) in enumerate(zip(chunk_batch, embeddings)):
                self.embedding_infos.append(
                    EmbeddingInfo(
                        document_id=chunk.document_id,
                        document_path=chunk.document_path,
                        span=chunk.span,
                        embedding=emb,
                    )
                )

            progress_bar.update(len(chunk_batch))

        progress_bar.close()

    def _initialize_sqlite_db(self):
        """Initialize the SQLite database with dynamic embedding size detection."""
        base_path = "./data"
        embedding_model_name = os.environ.get("HF_EMBEDDING_MODEL", "default").replace("/", "_")
        chunk_strategy = self.retrieval_strategy.chunking_strategy.strategy_name
        self.sqlite_db_file_path = f"{base_path}/embeddings_{embedding_model_name}_{chunk_strategy}.db"

        # Dynamically determine the embedding size by embedding a sample text
        sample_text = "This is a test sentence."
        if self.ollama_embeddings:
            embeddings = self.ollama_embeddings.embed_documents([sample_text])
            embedding_size = len(embeddings[0])
        elif self.hf_embeddings:
            embeddings = self.hf_embeddings.embed_documents([sample_text])
            embedding_size = len(embeddings[0])
        elif self.hf_bge_embeddings:
            embeddings = self.hf_bge_embeddings.embed_documents([sample_text])
            embedding_size = len(embeddings[0])
        else:
            raise ValueError("No embedding model is initialized to determine embedding size.")

        # Initialize SQLite database
        # Check if the database file already exists and delete it if it does
        if os.path.exists(self.sqlite_db_file_path):
            os.remove(self.sqlite_db_file_path)
        self.sqlite_db = sqlite3.connect(self.sqlite_db_file_path)
        self.sqlite_db.enable_load_extension(True)
        sqlite_vec.load(self.sqlite_db)
        self.sqlite_db.enable_load_extension(False)
        self.sqlite_db.execute(f"PRAGMA mmap_size = {3 * 1024 * 1024 * 1024}")

        # Create virtual table with the detected embedding size
        self.sqlite_db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{embedding_size}])"
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the appropriate embedding model."""
        if self.ollama_embeddings:
            return self.ollama_embeddings.embed_documents(texts)
        if self.hf_embeddings:
            return self.hf_embeddings.embed_documents(texts)
        if self.hf_bge_embeddings:
            return self.hf_bge_embeddings.embed_documents(texts)
        raise ValueError("No embedding model initialized.")
