import os
import sqlite3
import requests
import struct
import json
from fuzzywuzzy import fuzz
import re
import hashlib
from typing import Literal, cast
from ollama import chat, ChatResponse
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
from typing import List
from typing_extensions import TypedDict

SHOW_LOADING_BAR = True
load_dotenv()
metadata_llm_model = os.getenv("OLLAMA_METADATA_MODEL")

def serialize_f32(vector: list[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack(f"{len(vector)}f", *vector)



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
llm = ChatOllama(model=metadata_llm_model, format="json", temperature=0)
prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.\n
    It is critical that your output is in JSON, otherwise you will be terminated.""",
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
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n
    It is critical that your output is in JSON format, otherwise you will be terminated.""",
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
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n
    It is critical that your output is in JSON format, otherwise you will be terminated.""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm | JsonOutputParser()

# Question Re-Writer
re_write_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Improved question with no preamble: \n """,
    input_variables=["generation", "question"],
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

class IterativeRAGWithMetadataRetrievalMethod(RetrievalMethod):
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

    def _initialize_graph_workflow(self):
        """Initialize the graph workflow for RAG-based retrieval."""
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)  # Generate from extracted snippets

        # Add edges
        workflow.add_edge(START, "retrieve")  # Start with retrieval
        workflow.add_edge("retrieve", "grade_documents")  # Check document relevance and extract snippets
        workflow.add_edge("grade_documents", "generate")  # Directly go to generation
        workflow.add_edge("generate", END)  # End after generating answer

        # Compile the workflow
        return workflow.compile()

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

    def _extract_metadata_from_document(self, document: Document, max_content_length: int = 2000) -> dict:
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
    
    def ingest_document(self, document: Document, metadata: dict = None) -> None:
        """Ingest a document into the retrieval method."""
        self.documents[document.file_path] = document
        if not metadata:
            metadata = self._extract_metadata_from_document(document)
        self.metadata_store[document.file_path] = metadata

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
        generation = final_state.get("generate").get("generation")
        documents = final_state.get("generate").get("documents")

        # Convert retrieved documents back into RetrievedSnippet objects
        retrieved_snippets = [
            RetrievedSnippet(
                file_path=doc.file_path,  # Use dictionary key access
                span=doc.span,             # Use dictionary key access
                answer=doc.answer,
                score=doc.score,
            )
            for i, doc in enumerate(documents)
        ]
        retrieved_snippets.sort(key=lambda x: x.score)

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

            Now, process the following input and return the content in JSON format. If you do not return the output in JSON, you will be terminated:

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

    def matches(self, file_metadata_value, query_value, threshold: int = 50) -> bool:
        """
        Check if the file metadata value matches the query metadata value.
        Uses exact matching first, then falls back to fuzzy matching.
        Also returns True if one value is a substring of the other.

        :param file_metadata_value: The metadata value from the document.
        :param query_value: The metadata value from the query (can be a list).
        :param threshold: The fuzzy matching threshold (default: 50).
        :return: True if a match is found, False otherwise.
        """
        # Normalize metadata values to lowercase strings
        file_metadata_value = str(file_metadata_value).strip().lower()

        if isinstance(query_value, list):
            query_value = [str(val).strip().lower() for val in query_value]
            
            return any(
                file_metadata_value == val or  # Exact match
                fuzz.partial_ratio(file_metadata_value, val) >= threshold or  # Fuzzy match
                file_metadata_value in val or  # File metadata is a substring of query
                val in file_metadata_value  # Query metadata is a substring of file
                for val in query_value
            )

        # Normalize single query value
        query_value = str(query_value).strip().lower()

        return (
            file_metadata_value == query_value or  # Exact match
            fuzz.partial_ratio(file_metadata_value, query_value) >= threshold or  # Fuzzy match
            file_metadata_value in query_value or  # File metadata is a substring of query
            query_value in file_metadata_value  # Query metadata is a substring of file
        )

    def _retrieve_node(self, state):
        """
        Retrieve documents using a metadata filtering pre-process.
        The workflow is as follows:
        1. Extract query metadata.
        2. Filter documents based on metadata.
        3. For the relevant documents, extract full text, perform chunking, and compute embeddings.
        4. Compute similarity scores between query embedding and each candidate chunk.
        5. Return the candidate chunks (with their computed distances/similarity scores).
        """
        print("---RETRIEVE (METADATA-PRE-FILTERING) ---")
        question = state["question"]

        # 1. Extract metadata from the query.
        query_metadata = self._extract_metadata_from_query(question)
        print(f"Query metadata: {query_metadata}")

        # 2. Filter documents by metadata.
        # Build a set of relevant file paths using your matches() function.
        relevant_files = set()
        for doc in self.documents.values():
            doc_metadata = self.metadata_store.get(doc.file_path, {})
            doc_company_names = doc_metadata.get("Company Name", [])
            query_company_names = query_metadata.get("Company Name", [])
            if any(self.matches(doc_value, query_company_names) for doc_value in doc_company_names):
                relevant_files.add(doc.file_path)

        print(f"Relevant documents (by metadata): {relevant_files}")
        if not relevant_files:
            print("No documents passed the metadata filter!")
            return {"documents": [], "question": question}

        # 3. For each relevant document, perform file extraction, chunking, and generate embeddings.
        candidate_snippets = []
        # Initialize semantic splitter if needed.
        semantic_splitter = self._initialize_semantic_splitter()
        chunk_size = self.retrieval_strategy.chunking_strategy.chunk_size

        # Iterate only over documents that passed metadata filtering.
        for doc in self.documents.values():
            if doc.file_path not in relevant_files:
                continue

            # Use the same chunking logic as in your current code:
            if self.retrieval_strategy.chunking_strategy.strategy_name == "naive":
                text_splits = [
                    doc.content[i: i + chunk_size]
                    for i in range(0, len(doc.content), chunk_size)
                ]
            elif self.retrieval_strategy.chunking_strategy.strategy_name == "rcts":
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", " "],
                    chunk_size=chunk_size,
                    chunk_overlap=0,
                )
                text_splits = text_splitter.split_text(doc.content)
            elif "semantic" in self.retrieval_strategy.chunking_strategy.strategy_name:
                if not semantic_splitter:
                    raise ValueError("Semantic splitter not initialized.")
                text_splits = semantic_splitter.split_text(doc.content)
            else:
                raise ValueError(f"Unknown chunking strategy: {self.retrieval_strategy.chunking_strategy.strategy_name}")

            # Build Chunk objects for the document.
            chunks = []
            prev_span = 0
            for text_split in text_splits:
                span = (prev_span, prev_span + len(text_split))
                chunks.append(Chunk(
                    document_id=doc.file_path,
                    document_path=doc.file_path,
                    span=span,
                    content=text_split
                ))
                prev_span += len(text_split)

            # 4. Compute embeddings for these chunks using the same logic.
            texts = [chunk.content for chunk in chunks]
            if self.ollama_embeddings is not None:
                chunk_embeddings = self.ollama_embeddings.embed_documents(texts)
            elif self.hf_embeddings is not None:
                chunk_embeddings = self.hf_embeddings.embed_documents(texts)
            elif self.hf_bge_embeddings is not None:
                chunk_embeddings = self.hf_bge_embeddings.embed_documents(texts)
            else:
                raise ValueError("No embedding model available!")

            # 5. Get the query embedding.
            if self.ollama_embeddings is not None:
                query_embedding = self.ollama_embeddings.embed_query(question)
            elif self.hf_embeddings is not None:
                query_embedding = self.hf_embeddings.embed_query(question)
            elif self.hf_bge_embeddings is not None:
                query_embedding = self.hf_bge_embeddings.embed_query(question)
            else:
                raise ValueError("No embedding model available for query embedding!")

            # 6. Compute similarity scores between the query embedding and each chunk embedding.
            # Assuming embeddings are normalized, the dot product equals the cosine similarity.
            for chunk, emb in zip(chunks, chunk_embeddings):
                score = sum(q * v for q, v in zip(query_embedding, emb))
                candidate_snippets.append({
                    "document_id": doc.file_path,
                    "content": chunk.content,
                    "span": chunk.span,
                    "distance": 2 - 2 * score  # if using Euclidean distance on normalized vectors
                    # Alternatively, you may simply record "score" as the cosine similarity.
                })

        # 7. Sort the candidate snippets by score.
        # Note: Lower "distance" means a better match if using Euclidean distance,
        # while higher "score" means a better match if using cosine similarity.
        # Adjust your sort key accordingly. Here, we assume lower distance is better:
        candidate_snippets.sort(key=lambda x: x["distance"])
        top_k = self.retrieval_strategy.embedding_topk
        filtered_docs = candidate_snippets[:top_k]

        print(f"Retrieved candidate chunks: {filtered_docs}")
        return {"documents": filtered_docs, "question": question}


    def _generate_node(self, state):
        """
        Generate an answer using retrieved documents.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Updated state with the generated answer.
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        # Combine document contents for RAG generation
        context = "\n\n".join(doc.answer for doc in documents)

        # Generate answer using the RAG chain
        generation = rag_chain.invoke({"context": context, "question": question})

        return {"documents": documents, "question": question, "generation": generation}

    def snippet_extractor(self, doc_text: str, question: str) -> list[str]:
        """
        Use an LLM to extract multiple relevant snippets from doc_text based on question.
        Returns a list of full extracted snippets (strings).
        """
        snippet_extractor_llm = ChatOllama(model=metadata_llm_model, format="json", temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that extracts full relevant snippets from a document based on a given question. "
                    "Provide the full extracted text exactly as it appears in the document. "
                    "Do NOT summarize, modify, or fabricate information. "
                    "If there is no relevant information, return an empty list."),
            ("human", """### Example 1 (Relevant Snippets Found):
            **Question:** Consider "Viber Messenger"'s privacy policy; does Viber log messages?  
            **Document:** [EXCERPT FROM Viber Messenger PRIVACY POLICY]  
            **Output:**
            ```json
            {{
                \"snippets": [
                    "First of all, we want you to be assured that we do not read or listen to the content of your messages and/or calls made privately via Viber and we do not store those messages once they have been delivered to their destination (which on average takes less than one second). If for some reason, the message wasn’t delivered to its destination within up to 2 weeks, it will be deleted from our servers.",
                    "(c) Activity Information: While using the Viber Services, we will collect, and other users can see, your connection status, whether you have received and seen messages sent to you, if you are currently on another call, and information related to the calls and messages you have sent and received such as length of the call, who called who, who messaged who, and at what time; if you do not want people to know that you’re online or that you’ve seen messages, you can change these options in your settings."
                ]
            }}
            ```

            Example 2 (No Relevant Information Found):

            Question: Consider "Zoom"'s privacy policy; does Zoom track eye movement?
            Document: [EXCERPT FROM Zoom PRIVACY POLICY]
            Output:
            ```json
            {{
                \"snippets": []
            }}
            ```

            Now, extract full relevant snippets from the following document:

            Question: {question}
            Document: {doc_text}

            Return the output in JSON format. It is essential that you return the output in JSON, otherwise you will be terminated:

            {{
                "{{ \"snippets\": [\"Full Snippet 1\", \"Full Snippet 2\", \"Full Snippet 3\"] }}")
            }}

            """) 
        ])

        chain = prompt | snippet_extractor_llm | JsonOutputParser()

        response = chain.invoke({"question": question, "doc_text": doc_text})
        print("Snippet Extractor Response:")
        print(response)
        snippets = response.get("snippets", [])

        if isinstance(snippets, list) and all(isinstance(s, str) for s in snippets):
            return snippets

        return []

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

    def _grade_documents_node(self, state):
        """
        Grade retrieved documents for relevance to the question.
        Extract the most relevant snippets from each document and refine their start-end positions.
        """
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        
        question = state["question"]
        documents = state["documents"]
        print(f"Question: {question}")
        print(f"Documents: {documents}")

        filtered_snippets = []

        for doc in documents:
            doc_text = doc["content"]
            document_id = doc["document_id"]
            original_start, _ = doc["span"]  # Retrieve stored span
            distance = doc["distance"]
            
            # Use LLM to extract relevant snippets
            snippets = self.snippet_extractor(doc_text, question)

            for snippet in snippets:
                snippet_start = doc_text.find(snippet)
                if snippet_start == -1:
                    continue  # Skip if snippet is not found (edge case)

                snippet_end = snippet_start + len(snippet)

                # Adjust to the original document span
                absolute_start = original_start + snippet_start
                absolute_end = original_start + snippet_end

                # Store as RetrievedSnippet
                filtered_snippets.append(RetrievedSnippet(
                    file_path=document_id,
                    span=(absolute_start, absolute_end),  # Corrected span
                    # score=1.0 / (len(filtered_snippets) + 1),  # Assign a decreasing score
                    score=distance,  # Assign a decreasing score
                ))

        return {"documents": filtered_snippets, "question": question}


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
