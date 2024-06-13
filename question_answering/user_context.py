import json
import common.bedrock_retry_utils
from llama_index.embeddings.bedrock import BedrockEmbedding


class UserContext:
    def __init__(self, question, embed_model=None):
        self.embed_model = embed_model if embed_model else BedrockEmbedding()
        self.question = (question,)
        self.question_embedding = self.embed_model._get_embedding(question, "text")

    def to_dict(self):
        return {
            "question": self.question,
            "question_embedding": self.question_embedding,
        }

    def to_json(self):
        return json.dumps(self_to_dict(), indent=2)
