import os
import re
from pathlib import Path
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.bedrock import Bedrock


class EvaluateCompleteness:
    def __init__(
        self,
        llm_model=os.environ["EVALUATION_MODEL"],
        prompt_path="./prompts/evaluate-completeness-prompt.txt",
    ):
        self.llm = Bedrock(
            model=llm_model, temperature=0.0, max_tokens=4096, streaming=True
        )
        self.prompt_template = Path(prompt_path).read_text()

    def evaluate(self, response):
        answer_without_references = response.get_answer_without_references()

        evaluation = self.llm.predict(
            PromptTemplate(template=self.prompt_template),
            question=response.question,
            answer=answer_without_references,
            searchResults="\n\n".join(response.context.values()),
        )

        def first_or_default(a, default=""):
            return a[0] if a else None

        answer_satisfies_question = first_or_default(
            re.findall(
                "<answerSatisfiesQuestion>(.*)</answerSatisfiesQuestion>", evaluation
            )
        )
        reason = first_or_default(
            re.findall("<reason>(.*)</reason>", evaluation, re.DOTALL)
        )
        statements = first_or_default(
            re.findall("<statements>(.*)</statements>", evaluation, re.DOTALL)
        )
        additional_statements = first_or_default(
            re.findall(
                "<additionalStatements>(.*)</additionalStatements>",
                evaluation,
                re.DOTALL,
            )
        )
        improved_answer = first_or_default(
            re.findall("<improvedAnswer>(.*)</improvedAnswer>", evaluation, re.DOTALL)
        )

        num_statements = len(statements.split("\n"))
        num_additional_statements = len(additional_statements.split("\n"))

        return {
            "result": answer_satisfies_question,
            "reason": reason,
            "score": (
                0.0
                if answer_satisfies_question == "no"
                else num_statements / (num_statements + num_additional_statements)
            ),
            "num_statements": (
                0 if answer_satisfies_question == "no" else num_statements
            ),
            "num_additonal_statements": (
                0 if answer_satisfies_question == "no" else num_additional_statements
            ),
            "statements": "" if answer_satisfies_question == "no" else statements,
            "additional_statements": (
                "" if answer_satisfies_question == "no" else additional_statements
            ),
            "improved_answer": (
                "" if answer_satisfies_question == "no" else improved_answer
            ),
        }


class EvaluateFaithfulness:
    def __init__(
        self,
        llm_model=os.environ["EVALUATION_MODEL"],
        prompt_path="./prompts/evaluate-faithfulness-prompt.txt",
    ):
        self.llm = Bedrock(
            model=llm_model, temperature=0.0, max_tokens=4096, streaming=True
        )
        self.prompt_template = Path(prompt_path).read_text()

    def evaluate(self, response):
        answer_without_references = response.get_answer_without_references()

        evaluation = self.llm.predict(
            PromptTemplate(template=self.prompt_template),
            question=response.question,
            answer=answer_without_references,
            searchResults="\n\n".join(response.context.values()),
        )

        evaluations = re.findall("<evaluation>(.*)</evaluation>", evaluation)

        count = 0
        fully_supported_count = 0
        partially_supported_count = 0
        unsupported_count = 0

        for e in evaluations:
            if e.lower().startswith("fully"):
                fully_supported_count += 1
            if e.lower().startswith("partially"):
                partially_supported_count += 1
            if e.lower().startswith("unsupported"):
                unsupported_count += 1.0
            count += 1

        score = (fully_supported_count + (partially_supported_count / 2)) / count

        return {
            "score": round(score, 2),
            "full_supported": fully_supported_count,
            "partially_supported": partially_supported_count,
            "unsupported": unsupported_count,
            "evaluation": [
                e.replace("<evaluation>", "[").replace("</evaluation>", "]")
                for e in evaluation.split("\n")
            ],
        }
