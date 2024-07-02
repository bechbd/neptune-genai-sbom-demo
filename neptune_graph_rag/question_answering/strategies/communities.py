"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""

from neptune_graph_rag.common.neptune_utils import node_result
from neptune_graph_rag.question_answering.strategies.base_strategy import BaseStrategy
from neptune_graph_rag.question_answering.strategies.facts import FactSimilaritySearch
from neptune_graph_rag.question_answering.strategies.keywords import KeywordSearch


class GetCommunities(BaseStrategy):
    def __init__(
        self,
        neptune_client,
        top_n=5,
        results_accessors=[FactSimilaritySearch, KeywordSearch],
    ):
        self.neptune_client = neptune_client
        self.top_n = top_n
        self.results_accessors = results_accessors

    @classmethod
    def strategy_name(cls):
        return "communities"

    def get_communities(self, ids):
        cypher = f"""
        MATCH (fact:Fact)-[:MEMBER_OF]->(community) WHERE fact.factId in $ids
        WITH DISTINCT community
        RETURN {{
            {node_result('community')}
        }} AS result LIMIT {self.top_n}
        UNION
        MATCH (entity:Entity)-->(:Fact)-[:MEMBER_OF]->(community) WHERE entity.entityId in $ids
        WITH DISTINCT community
        RETURN {{
            {node_result('community')}
        }} AS result LIMIT {self.top_n}
        """

        params = {"ids": ids}

        results = self.neptune_client.execute_query(cypher, params)

        return [result["result"] for result in results][: self.top_n]

    def accept(self, context):

        ids = []

        for results_accessor in self.results_accessors:
            ids.extend(
                results_accessor.get_results(
                    context, selector="fact|entity/factId|entityId"
                )
            )

        communities = self.get_communities(ids)

        results = {}

        for c in communities:
            community_id = c["community"]["communityId"]
            if community_id not in results:
                results[community_id] = c

        self.add_results(context, list(results.values()))

        return context
