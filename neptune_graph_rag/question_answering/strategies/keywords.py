import os
from pathlib import Path
from neptune_graph_rag.common.neptune_utils import node_id, node_result
from neptune_graph_rag.question_answering.strategies.base_strategy import BaseStrategy
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.bedrock import Bedrock


class ExtractKeywords(BaseStrategy):

    def __init__(
        self,
        llm_model=os.environ["EXTRACTION_MODEL"],
        prompt_path="./neptune_graph_rag/prompts/extract-keywords-prompt.txt",
        max_keywords=5,
    ):

        self.llm = Bedrock(model=llm_model, temperature=0.0, streaming=True)
        self.prompt_template = Path(prompt_path).read_text()
        self.max_keywords = max_keywords

    @classmethod
    def strategy_name(cls):
        return "extract-keywords"

    def accept(self, context):

        results = self.llm.predict(
            PromptTemplate(template=self.prompt_template),
            text=context.user_context.question,
            max_keywords=self.max_keywords,
        )

        keywords = results.split("|")

        self.add_results(context, [keyword.strip() for keyword in keywords])

        return context


class KeywordSearch(BaseStrategy):
    def __init__(self, neptune_client, results_accessors=[ExtractKeywords]):
        self.neptune_client = neptune_client
        self.results_accessors = results_accessors

    @classmethod
    def strategy_name(cls):
        return "keyword-search"

    def get_keywords(self, context):

        keywords = []

        for results_accessor in self.results_accessors:
            keywords.extend(results_accessor.get_results(context))

        return set(keywords)

    def accept(self, context):

        keywords = self.get_keywords(context)

        entities = []
        entity_ids = []

        cypher = f"""MATCH (entity:Entity) 
        WHERE entity.entityId STARTS WITH $keyword 
        RETURN {{
            {node_result('entity')}
        }} AS result"""

        for keyword in keywords:

            if keyword:

                params = {"keyword": node_id(keyword)}

                results = self.neptune_client.execute_query(cypher, params)

                for result in results:
                    entity = result["result"]
                    entity_id = entity["entity"]["entityId"]
                    if entity_id not in entity_ids:
                        entity_ids.append(entity_id)
                        entities.append(entity)

        self.add_results(context, entities)

        return context
