import os
import sqlite3
import requests
import struct
import json
from fuzzywuzzy import fuzz
import re
import hashlib
from typing import Literal, cast
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langgraph.graph import END, StateGraph, START
from langgraph.errors import GraphRecursionError
from langchain_ollama import OllamaEmbeddings, ChatOllama
import sqlite_vec  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from ollama import chat, ChatResponse
from tqdm import tqdm
import torch
from transformers import pipeline
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pprint import pprint

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
metadata_llm_model = os.getenv("OLLAMA_METADATA_MODEL")

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
     Here is the initial question: \n\n {question}. Output only the improved question with no preamble. Keep any references to companies or parties that allow this query to be separated from others.: \n """,
    input_variables=["generation", "question"],
)
question_rewriter = re_write_prompt | llm | StrOutputParser()

class SelfRAGWithMetadataRetrievalMethod(RetrievalMethod):
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
        self.metadata_store = {}

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
        
        if "retrieve" in final_state:
            documents = final_state["retrieve"].get("documents", [])
        elif "grade_documents" in final_state:
            documents = final_state["grade_documents"].get("documents", [])
        elif "generate" in final_state:
            documents = final_state["generate"].get("documents", [])
        

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
    
    def clean_text(self, text):
        """Remove special characters and extra spaces, making comparison stricter."""
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
        return " ".join(text.split()) 

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
                print(f"Identified {os.path.basename(file_path)} as a matching file")

        if not matching_file_names:
            print("No matching file names found in metadata filtering.")
            matching_file_names = {os.path.basename(file_path) for file_path in self.metadata_store.keys()} 
        
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
        
        self.last_query_metadata_match = len(filtered_snippets) > 0
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

    def _retrieve_node(self, state):
        """
        Retrieve documents using SQL-based retrieval.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Updated state with retrieved documents.
        """
        print("---RETRIEVE---")
        question = state["question"]
        # question_metadata = self._extract_metadata_from_query(question)
        state["attempt_count"] = state.get("attempt_count", 0) + 1
        if state["attempt_count"] > 3:
            print("Exceeded maximum retrieval attempts; using baseline search.")
            # Force a baseline retrieval by ignoring metadata filters.
            state["metadata_filter"] = {} 
        else:
            if "metadata_filter" not in state:
                state["metadata_filter"] = self._extract_metadata_from_query(question)
    
        question_metadata = state["metadata_filter"]

        # Perform retrieval and filtering
        if self.ollama_embeddings is not None:
            query_embedding = self.ollama_embeddings.embed_query(question)
        elif self.hf_embeddings is not None:
            query_embedding = self.hf_embeddings.embed_query(question)
        elif self.hf_bge_embeddings is not None:
            query_embedding = self.hf_bge_embeddings.embed_query(question)

        retrieved_snippets = self._retrieve_and_filter(question, query_embedding, question_metadata, self.retrieval_strategy.embedding_topk)
        print(retrieved_snippets)
        documents = [
            {
                "document_id": doc.file_path,
                "content": self.get_embedding_info_text(doc),
                "span": doc.span,  # Preserve the chunk's span
            }
            for doc in retrieved_snippets
        ]

        print(f"Retrieved Documents: {documents}")

        return {"documents": documents, "question": question}

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
        context = "\n\n".join(doc["content"] for doc in documents)

        # Generate answer using the RAG chain
        generation = rag_chain.invoke({"context": context, "question": question})

        return {"documents": documents, "question": question, "generation": generation}

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
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("transform_query", self._transform_query_node)

        # Add edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self._grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        # Compile the workflow
        return workflow.compile()

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
                (offset + j, 
                serialize_f32(emb),
                os.path.basename(chunk_batch[j].document_path))  # file_name)
                for j, emb in enumerate(embeddings)
            ]

            # Store embeddings
            with self.sqlite_db as db:
                db.executemany(
                    # "INSERT INTO vec_items(rowid, embedding) VALUES (?, ?)",
                    "INSERT INTO vec_items(rowid, embedding, file_name) VALUES (?, ?, ?)",
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
        # self.sqlite_db.execute(
        #     f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(embedding float[{embedding_size}])"
        # )

        self.sqlite_db.execute(
            f"""CREATE VIRTUAL TABLE vec_items USING vec0(
                embedding float[{embedding_size}],
                file_name text,
            )"""
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
