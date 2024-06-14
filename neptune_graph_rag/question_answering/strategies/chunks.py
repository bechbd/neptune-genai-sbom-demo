from neptune_graph_rag.common.neptune_utils import node_result
from neptune_graph_rag.question_answering.strategies.base_strategy import BaseStrategy
from neptune_graph_rag.question_answering.strategies.facts import (
    FactSimilaritySearch,
    FactExpansion,
)


class ChunkSimilaritySearch(BaseStrategy):
    def __init__(self, neptune_client, top_k=5):
        self.neptune_client = neptune_client
        self.top_k = top_k

    @classmethod
    def strategy_name(cls):
        return "chunk-similarity-search"

    def accept(self, context):

        embedding = context.user_context.question_embedding

        results = None

        for i in [2, 10, 30]:

            top_k = self.top_k * i

            cypher = f"""
            CALL neptune.algo.vectors.topKByEmbedding(
                {embedding},
                {{   
                    topK: {top_k},
                    concurrency: 4
                }}
            )
            YIELD node, score
            WITH node as chunk, score WHERE 'Chunk' in labels(chunk)
            MATCH (chunk)-[:SOURCE]->(source:Source)
            RETURN {{
                score: score,
                {node_result('source')},
                {node_result('chunk', ['chunkId'])}
            }} AS result ORDER BY result.score ASC LIMIT {self.top_k}
            """

            query_results = self.neptune_client.execute_query(cypher)
            results = [query_result["result"] for query_result in query_results]

            if len(results) >= self.top_k:
                break

        self.add_results(context, results)

        return context


class GetChunks(BaseStrategy):
    def __init__(
        self, neptune_client, results_accessors=[ChunkSimilaritySearch], top_n=5
    ):
        self.neptune_client = neptune_client
        self.top_n = top_n
        self.results_accessors = results_accessors

    @classmethod
    def strategy_name(cls):
        return "chunks"

    def accept(self, context):

        chunk_ids = []

        for results_accessor in self.results_accessors:
            chunk_ids.extend(
                results_accessor.get_results(context, selector="chunk/chunkId")
            )

        cypher = f"""
        MATCH (chunk:Chunk)-[:SOURCE]->(source) WHERE chunk.chunkId in $chunk_ids
        RETURN {{
            {node_result('chunk')},
            {node_result('source')}
        }} AS result
        """

        params = {"chunk_ids": chunk_ids[: self.top_n]}

        results = self.neptune_client.execute_query(cypher, params)

        self.add_results(context, [result["result"] for result in results])

        return context


class RerankChunks(BaseStrategy):
    def __init__(
        self,
        results_accessors=[ChunkSimilaritySearch, FactSimilaritySearch, FactExpansion],
    ):
        self.results_accessors = results_accessors

    @classmethod
    def strategy_name(cls):
        return "reranked-chunks"

    def accept(self, context):

        def rescore(r, max_score):
            score = r["score"]
            divisor = score / max_score
            r["score"] = round((1.0 / divisor), 2)
            return r

        all_results = []
        for results_accessor in self.results_accessors:
            all_results.extend(results_accessor.get_results(context))

        all_scores = [r["score"] for r in all_results]

        max_score = max(all_scores) if all_scores else 1
        rescored_results = [rescore(r, max_score) for r in all_results]

        reranked_results = {}

        for rescored_result in rescored_results:
            chunk = rescored_result["chunk"]
            if chunk["chunkId"] not in reranked_results:
                reranked_results[chunk["chunkId"]] = rescored_result
            else:
                new_score = (
                    reranked_results[chunk["chunkId"]]["score"]
                    + rescored_result["score"]
                )
                reranked_results[chunk["chunkId"]]["score"] = new_score

        self.add_results(
            context,
            sorted(reranked_results.values(), key=lambda c: c["score"], reverse=True),
        )

        return context
