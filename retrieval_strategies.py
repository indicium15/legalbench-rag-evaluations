from typing import Literal
from baseline import ChunkingStrategy, RetrievalStrategy
from ai import AIEmbeddingModel, AIRerankModel

chunk_strategy_names: list[Literal["naive", "rcts"]] = ["naive", "rcts"]
chunk_strategy_names: list[Literal["unstructured", "naive", "rcts", "semantic_percentile", "semantic_standard_deviation", "semantic_interquartile", "semantic_gradient"]] = [
    "naive", 
    # "rcts", 
    # "semantic_percentile", 
    # "semantic_standard_deviation", 
    # "semantic_interquartile", 
    # "semantic_gradient",
    # "unstructured"
]
# chunk_strategy_names: list[Literal["semantic_percentile"]] = ["semantic_percentile"]

rerank_models: list[AIRerankModel | None] = [
    None,
    # AIRerankModel(company="huggingface-bge", model="BAAI/bge-reranker-v2-m3")
    # AIRerankModel(company="cohere", model="rerank-english-v3.0"),
    # AIRerankModel(company="huggingface-bge", model="truthsystems/legal-bge-reranker-large")
    # AIRerankModel(company="huggingface", model="phannhat/Legal-BGE-Reranker-Finetune")
]
# chunk_sizes: list[int] = [128, 256, 512]
# chunk_sizes: list[int] = [128, 256, 512]
# chunk_sizes: list[int] = [512, 1024, 2048]
chunk_sizes: list[int] = [512]
# top_ks: list[int] = [1, 2, 4, 8, 16]
# top_ks: list[int] = [2, 4, 8, 16]
# top_ks: list[int] = [16, 32]
top_ks: list[int] = [1]

RETRIEVAL_STRATEGIES: list[RetrievalStrategy] = []
# rerank_model = IRerankModel(company="huggingface", model="BAAI/bge-reranker-v2-m3")
# rerank_model = None
top_k = 1
# RETRIEVAL_STRATEGIES.append(
#     RetrievalStrategy(
#         chunking_strategy=ChunkingStrategy(strategy_name = "rcts", chunk_size = 2000),
#         # embedding_model=AIEmbeddingModel(company="huggingface", model="intfloat/multilingual-e5-large-instruct"),
#         embedding_model=AIEmbeddingModel(company="huggingface-bge", model="BAAI/bge-m3"),
#         embedding_topk=10,
#         rerank_model=None,
#         rerank_topk=10,
#         token_limit=None,
#     ),
# )

# RETRIEVAL_STRATEGIES.append(
#     RetrievalStrategy(
#         chunking_strategy=ChunkingStrategy(strategy_name = "naive", chunk_size = 500),
#         embedding_model=AIEmbeddingModel(company="ollama", model="nomic-embed-text"),
#         embedding_topk=10,
#         rerank_model=rerank_model,
#         rerank_topk=1,
#         token_limit=None,
#     ),
# )

for chunk_strategy_name in chunk_strategy_names:
    for chunk_size in chunk_sizes:
        chunking_strategy = ChunkingStrategy(
            strategy_name=chunk_strategy_name,
            chunk_size=chunk_size,
        )
        for rerank_model in rerank_models:
            for top_k in top_ks:
                RETRIEVAL_STRATEGIES.append(
                    RetrievalStrategy(
                        chunking_strategy=chunking_strategy,
                        # embedding_model=AIEmbeddingModel(company="huggingface", model="BAAI/bge-large-en-v1.5"),
                        # embedding_model=AIEmbeddingModel(company="ollama", model="nomic-embed-text"),
                        embedding_model=AIEmbeddingModel(company="huggingface", model="intfloat/multilingual-e5-large-instruct"),
                        embedding_topk=300 if rerank_model is not None else top_k,
                        rerank_model=rerank_model,
                        rerank_topk=top_k,
                        token_limit=None,
                    ),
                )
