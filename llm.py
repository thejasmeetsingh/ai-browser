from enum import Enum
from ollama import AsyncClient


PARAPHRASE_PROMPT = "You are a linguistic compression expert. Your task is to analyze user-submitted queries and\
distill them to their most essential meaning. Begin by interpreting the user's intent—consider\
the semantics, implied context, and practical objective of the query. Then, paraphrase the core\
idea clearly and efficiently, using no more than 50 characters, including spaces and punctuation.\
Focus on what the user is trying to say, not just the words they used. Remove any filler, redundancy,\
or unnecessary specificity unless it is critical to the meaning. Use commonly understood language and\
ensure the output is natural, clear, and fulfills the user's intent. Return only the optimized query with\
no explanations."

SUMMARY_PROMPT = "You are a summarization expert. Your task is to carefully read the given text and produce a clear, concise\
summary that retains all the key points, important topics, and essential information. Do not omit any critical\
details, and ensure that the meaning and intent of the original content are preserved. Avoid unnecessary repetition\
or overly general statements."


class SystemPromptType(Enum):
    PARAPHRASE = "P"
    SUMMARY = "S"


class CustomOllama:
    def __init__(self):
        self.client = AsyncClient()
        self.optons = {"num_ctx": 17000}

        self.model = None

    @staticmethod
    def format_response(content: str) -> str:
        return content.split("</think>")[-1].strip()
    
    @staticmethod
    def get_sys_prompts() -> dict:
        return {
            SystemPromptType.PARAPHRASE.value: PARAPHRASE_PROMPT,
            SystemPromptType.SUMMARY.value: SUMMARY_PROMPT
        }

    def set_model(self, model: str) -> None:
        self.model = model

    async def get_models(self) -> list:
        response = await self.client.list()
        return response.models

    async def generate(self, sys_prompt_type: str, user_prompt: str) -> str:
        sys_prompts = self.get_sys_prompts()

        response = await self.client.generate(
            model=self.model,
            prompt=user_prompt,
            system=sys_prompts.get(sys_prompt_type, ""),
            options=self.optons
        )

        return self.format_response(response.response)
