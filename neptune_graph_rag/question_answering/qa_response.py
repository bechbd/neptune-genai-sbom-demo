import os
import re
from pathlib import Path
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.bedrock import Bedrock
from neptune_graph_rag.common.neptune_utils import get_source_value
from neptune_graph_rag.question_answering.strategies.base_strategy import BaseStrategy
from neptune_graph_rag.question_answering.strategies.facts import (
    FactSimilaritySearch,
    FactExpansion,
)
from neptune_graph_rag.question_answering.strategies.chunks import GetChunks
from neptune_graph_rag.question_answering.strategies.communities import GetCommunities


class Response:
    def __init__(self, question, answer, context):
        self.question = question
        self.answer = answer
        self.context = context

    def __str__(self):
        return self.answer

    def __repr__(self):
        return f"Question: {self.question}\n\nAnswer: {self.answer}\n\nContext: {self.context}"

    def get_answer_without_references(self):
        answer = self.answer
        index = answer.find("\nAnswer: ")
        answer = answer[(index + 9) :] if index >= 0 else answer
        return re.sub("\[[,\s,0-9]*\]", "", answer)


class QAResponse:

    def __init__(
        self,
        llm_model=os.environ["EXTRACTION_MODEL"],
        prompt_path="./neptune_graph_rag/prompts/answer-question-prompt.txt",
    ):

        self.llm = Bedrock(
            model=llm_model, temperature=0.0, max_tokens=4096, streaming=True
        )
        self.prompt_template = Path(prompt_path).read_text()

    def generate_response(self, context):

        def source(result):
            properties = result["source"]
            source_id = properties["sourceId"]
            return get_source_value(properties, source_id)

        def format_chunk(result):
            return "{} [{}]".format(result["chunk"]["value"], source(result))

        def format_fact(result):
            return "{} [{}]".format(result["fact"]["value"], source(result))

        chunks = "\n\n".join(
            BaseStrategy.get_results(context, GetChunks, selector_func=format_chunk)
        )

        communities = "\n\n".join(
            BaseStrategy.get_results(
                context, GetCommunities, selector="community/value"
            )
        )

        facts = "\n\n".join(
            BaseStrategy.get_results(
                context,
                [FactSimilaritySearch, FactExpansion],
                selector_func=format_fact,
            )
        )

        sources = "{}\n\n{}".format(communities, chunks)

        llm_context = {"sources": sources, "facts": facts}

        answer = self.llm.predict(
            PromptTemplate(template=self.prompt_template),
            text=context.user_context.question,
            sources=sources,
            facts=facts,
        )

        return Response(context.user_context.question, answer, llm_context)
