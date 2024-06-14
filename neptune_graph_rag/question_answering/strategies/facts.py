import boto3
import numpy as np
from numpy.linalg import norm
import statistics
import time
from opensearchpy import AWSV4SignerAsyncAuth, AsyncHttpConnection
from neptune_graph_rag.common.neptune_utils import node_result
from neptune_graph_rag.question_answering.strategies.base_strategy import BaseStrategy
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.vector_stores.opensearch import (
    OpensearchVectorStore,
    OpensearchVectorClient,
)


class FactSimilaritySearch(BaseStrategy):
    @classmethod
    def strategy_name(cls):
        return "fact-similarity-search"


class OpenSearchFactSimilaritySearch(FactSimilaritySearch):
    def __init__(self, endpoint, top_k=50):
        session = boto3.Session()
        region = session.region_name
        credentials = session.get_credentials()
        service = "aoss"
        idx = "fact"

        auth = AWSV4SignerAsyncAuth(credentials, region, service)

        text_field = "value"
        embedding_field = "embedding"

        count = 0
        index = None
        while not index:
            try:
                client = OpensearchVectorClient(
                    endpoint,
                    idx,
                    1536,
                    embedding_field=embedding_field,
                    text_field=text_field,
                    use_ssl=True,
                    verify_certs=True,
                    http_auth=auth,
                    connection_class=AsyncHttpConnection,
                )
                vector_store = OpensearchVectorStore(client)
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store
                )
                index = VectorStoreIndex.from_vector_store(
                    vector_store,
                    storage_context=storage_context,
                    embed_model=BedrockEmbedding(),
                )
                self.indexes[idx] = index
            except Exception as err:
                count += 1
                if count >= 3:
                    raise
                else:
                    time.sleep(2)

        self.retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=top_k,
        )

    def accept(self, context):

        nodes = self.retriever.retrieve(context.user_context.question[0])

        results = []

        for node in nodes:
            result = node.metadata
            result["fact"]["value"] = node.text
            result["score"] = node.score
            results.append(result)

        self.add_results(context, [result for result in results])

        return context


class NeptuneFactSimilaritySearch(FactSimilaritySearch):
    def __init__(self, neptune_client, top_k=50):
        self.neptune_client = neptune_client
        self.top_k = top_k

    def accept(self, context):

        embedding = context.user_context.question_embedding
        top_k = self.top_k * 3

        cypher = f"""
        CALL neptune.algo.vectors.topKByEmbedding(
            {embedding},
            {{   
                topK: {top_k},
                concurrency: 4
            }}
        )
        YIELD node, score
        WITH node as fact, score WHERE 'Fact' in labels(fact)
        MATCH (fact)-[:CHUNK]->(chunk:Chunk)-[:SOURCE]->(source:Source)
         RETURN {{
            score: score,
            {node_result('fact')},
            {node_result('source')},
            {node_result('chunk', ['chunkId'])}
        }} AS result ORDER BY result.score ASC LIMIT {self.top_k}
        """

        results = self.neptune_client.execute_query(cypher)

        self.add_results(context, [result["result"] for result in results])

        return context


class FactExpansion(BaseStrategy):

    def __init__(
        self,
        neptune_client,
        top_k=5,
        frontier_size=20,
        max_depth=3,
        results_accessors=[FactSimilaritySearch],
    ):
        self.neptune_client = neptune_client
        self.top_k = top_k
        self.frontier_size = frontier_size
        self.max_depth = max_depth
        self.results_accessors = results_accessors

    @classmethod
    def strategy_name(cls):
        return "expanded-facts"

    def l2_norm(self, embedding1, embedding2):

        A = np.array(embedding1)
        B = np.array(embedding2)

        v = norm(A - B)

        return v * v

    def top_k_facts(self, embeddings, node_ids):

        top_k_facts = []

        if not node_ids:
            return top_k_facts

        cypher = f"""
        MATCH (sourceFact:Fact)<-[:OBJECT]-(e1:Entity)-[:SUBJECT]->(fact:Fact)
        <-[:OBJECT]-(e2:Entity)<-[r:RELATION]-(e1), 
        (fact)-[:CHUNK]->(chunk:Chunk)-[:SOURCE]->(source:Source) 
        WHERE sourceFact.factId IN $node_ids AND sourceFact <> fact
        WITH DISTINCT sourceFact, fact, chunk, source, r.score AS score ORDER BY score DESC LIMIT {self.frontier_size}
        CALL neptune.algo.vectors.get(fact)
        YIELD embedding
        RETURN {{
            {node_result('sourceFact')}, 
            {node_result('fact')},
            {node_result('chunk', ['chunkId'])},
            {node_result('source')},
            embedding: embedding, 
            score: score
        }} AS result
        """

        params = {"node_ids": list(node_ids)}

        results = self.neptune_client.execute_query(cypher, params)

        for result in results:
            score = statistics.fmean(
                [
                    self.l2_norm(embedding, result["result"]["embedding"])
                    for embedding in embeddings
                ]
            )

            top_k_facts.append(
                {
                    "fact": result["result"]["fact"],
                    "sourceFact": result["result"]["sourceFact"],
                    "source": result["result"]["source"],
                    "chunk": result["result"]["chunk"],
                    "score": score,
                }
            )

        return sorted(top_k_facts, key=lambda c: c["score"], reverse=True)[: self.top_k]

    def accept(self, context):

        embedding = context.user_context.question_embedding

        results = []

        def is_new_fact(new_result):
            return new_result["fact"]["factId"] not in [
                result["fact"]["factId"] for result in results
            ]

        fact_ids = []
        for results_accessor in self.results_accessors:
            fact_ids.extend(
                results_accessor.get_results(context, selector="fact/factId")
            )

        for i in range(0, self.max_depth):

            new_facts = list(
                filter(is_new_fact, self.top_k_facts([embedding], fact_ids))
            )

            if not new_facts:
                break

            results.extend(new_facts)
            fact_ids = set([fact["fact"]["factId"] for fact in new_facts])

        self.add_results(context, results)

        return context

    @classmethod
    def format_results(cls, context):

        expanded_facts = cls.get_results(context, FactExpansion)

        results = {}
        for expanded_fact in expanded_facts:
            results[expanded_fact["sourceFact"]["value"]] = None
            results[expanded_fact["fact"]["value"]] = None

        return "\n".join(results.keys())
