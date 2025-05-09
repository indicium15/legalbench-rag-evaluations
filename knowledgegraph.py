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
from pyvis.network import Network
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
import networkx as nx

# Load environment variables
load_dotenv()
metadata_llm_model = os.getenv("OLLAMA_METADATA_MODEL")

def serialize_f32(vector: list[float]) -> bytes:
    """Serializes a list of floats into a compact 'raw bytes' format."""
    return struct.pack(f"{len(vector)}f", *vector)

SHOW_LOADING_BAR = True

###############################################################################
# New Knowledge Graph RetrievalMethod using LLM-based prompting for legal KG
###############################################################################

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

class LegalGraphRetrievalMethod(RetrievalMethod):
    retrieval_strategy: RetrievalStrategy
    documents: dict[str, Document]
    # We will store our combined knowledge graph here
    knowledge_graph: nx.DiGraph | None
    # We keep the SQLite-related attributes in case you want to fallback to embeddings.
    embedding_infos: list[EmbeddingInfo] | None
    sqlite_db: sqlite3.Connection | None
    sqlite_db_file_path: str | None

    def __init__(self, retrieval_strategy: RetrievalStrategy):
        self.retrieval_strategy = retrieval_strategy
        self.documents = {}
        self.knowledge_graph = None
        self.embedding_infos = None
        self.sqlite_db = None
        self.sqlite_db_file_path = None

        # Initialize embedding models as needed.
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

    def ingest_document(self, document: Document) -> None:
        """Ingest a document into the in-memory store."""
        self.documents[document.file_path] = document

    def sync_all_documents(self) -> None:
        """
        For each ingested document, use an LLM prompt fine-tuned for legal texts
        (privacy_qa, cuad, maud, contractnli) to extract a knowledge graph.
        All individual graphs are then merged into a single NetworkX directed graph.
        """
        combined_graph = nx.DiGraph()
        stubs = []  # optionally, keep a list of per-document outputs
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

        for chunk in chunks:
            # print(chunk)
            kg_data = self._extract_legal_knowledge_graph(chunk.content)
            # Add entities as nodes (with an optional attribute "type")
            for entity in kg_data.get("entities", []):
                combined_graph.add_node(entity["name"], type=entity.get("type", "Unknown"))
            # Add relationships as directed edges
            for rel in kg_data.get("relations", []):
                if "source" not in rel or "target" not in rel or "relationship" not in rel:
                    print(f"Skipping malformed relation: {rel}")  # Debugging output
                    continue 
                source = rel["source"]
                target = rel["target"]
                relationship = rel.get("relationship", "related_to")
                # Optionally, you can accumulate weights if the same edge appears more than once.
                if combined_graph.has_edge(source, target):
                    combined_graph[source][target]["weight"] += 1
                else:
                    combined_graph.add_edge(source, target, relationship=relationship, weight=1)
            stubs.append(kg_data)
        
        self.knowledge_graph = combined_graph
        self.visualize_graph(output_file="test_graph.html")
        print("Knowledge Graph built with {} nodes and {} edges.".format(
            combined_graph.number_of_nodes(), combined_graph.number_of_edges()))

    def _extract_legal_knowledge_graph(self, doc_text: str) -> dict:
        """
        Use an LLM to extract a legal knowledge graph from the document text.
        The prompt is fine-tuned for legal texts from privacy_qa, cuad, maud, and contractnli.
        Returns a dict with keys "entities" and "relations".
        """
        prompt = f"""
            You are a legal domain expert tasked with constructing a knowledge graph from a legal document.
            The document may come from datasets such as privacy_qa, CUAD, MAUD, or ContractNLI.
            Extract the key legal entities (such as parties, contract types, obligations, clauses, rights, penalties) 
            and the relationships between them (e.g. "agrees to", "is subject to", "acquired by", "governed by").
            Return your output in the following JSON format exactly:

            {{
            "entities": [
                {{"name": "Entity1", "type": "Party"}},
                {{"name": "Entity2", "type": "Contract Type"}},
                ...
            ],
            "relations": [
                {{"source": "Entity1", "target": "Entity2", "relationship": "agrees to"}},
                ...
            ]
            }}

            Ensure that:
            - Only include important legal entities.
            - Use clear relationship labels.
            - Your output is valid JSON and contained within a JSON code block.
            - The JSON keys correspond to the provided examples.

            Document:
            \"\"\"{doc_text}\"\"\"

            Output:
        """
        try:
            response: ChatResponse = chat(model=metadata_llm_model, messages=[{"role": "user", "content": prompt}])
            result = response.message.content.strip()
            print(f"Response from LLM: {result}")
            # Try to extract JSON from a code block
            json_match = re.search(r"```json(.*?)```", result, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
            else:
                json_content = result
            kg = json.loads(json_content)
            return kg
        except Exception as e:
            print(f"Error extracting legal knowledge graph: {e}")
            return {"entities": [], "relations": []}

    def query(self, query: str) -> QueryResponse:
        """
        Process a query by querying the knowledge graph.
        This method uses simple graph-based retrieval:
          1. Tokenize the query.
          2. Identify graph nodes matching query terms.
          3. Retrieve neighboring nodes as candidate legal concepts.
          4. Use these candidate nodes to identify relevant documents.
          5. Return retrieved snippets (e.g., document page contents that mention these entities).
        """
        if self.knowledge_graph is None:
            raise ValueError("Documents have not been synchronized into a knowledge graph!")

         # 1. Compute semantic embeddings for all nodes in the graph.
        entity_embeddings = {}
        for node in self.knowledge_graph.nodes:
            # Here we assume embed_query returns a list of floats.
            entity_embeddings[node] = self.ollama_embeddings.embed_query(node)
        
        # 2. Compute embedding for the query.
        query_embedding = self.ollama_embeddings.embed_query(query)
        
        # 3. Use cosine similarity to semantically match nodes.
        threshold = 0.7  # Adjust threshold as needed.
        matched_nodes = []
        for node, emb in entity_embeddings.items():
            sim = cosine_similarity(query_embedding, emb)
            if sim > threshold:
                matched_nodes.append(node)
        print("Semantic matched nodes:", matched_nodes)
        
        # 4. Multi-hop retrieval: include immediate neighbors (successors and predecessors).
        related_entities = set(matched_nodes)
        for node in matched_nodes:
            related_entities.update(self.knowledge_graph.neighbors(node))
            related_entities.update(self.knowledge_graph.predecessors(node))
        print("Related entities after multi-hop retrieval:", related_entities)
        
        # 5. Identify documents that mention at least one of these entities.
        candidate_docs = []
        for doc in self.documents.values():
            if any(entity.lower() in doc.content.lower() for entity in related_entities):
                candidate_docs.append(doc)
        
        # 6. Build retrieved snippets (here, returning the full content as a simple example).
        retrieved_snippets = []
        score = 1.0
        for doc in candidate_docs:
            retrieved_snippets.append(RetrievedSnippet(
                file_path=doc.file_path,
                span=(0, len(doc.content)),
                score=score,
                answer=doc.content
            ))
            score *= 0.9  # Decrease score for subsequent documents.
        
        return QueryResponse(retrieved_snippets=retrieved_snippets)

    def cleanup(self) -> None:
        """Cleanup resources, such as closing the SQLite database if used."""
        if self.sqlite_db is not None:
            self.sqlite_db.close()
            self.sqlite_db = None
        if self.sqlite_db_file_path is not None and os.path.exists(self.sqlite_db_file_path):
            os.remove(self.sqlite_db_file_path)
            self.sqlite_db_file_path = None
        # Clear the knowledge graph if desired
        self.knowledge_graph = None

    def visualize_graph(self, output_file: str = "knowledge_graph.html"):
        """
        Generate an interactive HTML visualization of the knowledge graph using Pyvis.

        :param output_file: Path to save the HTML file.
        """
        if self.knowledge_graph is None:
            print("No knowledge graph available to visualize.")
            return

        # Create a Pyvis network
        net = Network(height="800px", width="100%", directed=True, notebook=False)

        # Add nodes with labels
        for node, attrs in self.knowledge_graph.nodes(data=True):
            label = f"{node} ({attrs.get('type', 'Unknown')})"
            net.add_node(node, label=label, title=label)

        # Add edges with relationship labels
        for source, target, edge_attrs in self.knowledge_graph.edges(data=True):
            relationship = edge_attrs.get("relationship", "related_to")
            net.add_edge(source, target, title=relationship, label=relationship)

        # Save as an HTML file
        net.show(output_file)
        print(f"Knowledge graph visualization saved to {output_file}")

    # (Optional) You may add additional helper methods (e.g., for visualization)
