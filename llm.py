import datetime
from enum import Enum

from ollama import AsyncClient


current_dt = datetime.datetime.now().strftime("%A, %d-%b-%Y, %H:%I %p")


PARAPHRASE_PROMPT = f"System DateTime: {current_dt}\nYou analyze user queries and transform them into optimized search" \
"queries under 50 characters. Preserve the original meaning and intent while making queries search-engine friendly." \
"Use keywords, remove filler words, and apply SEO best practices." \
"Consider multiple interpretations for ambiguous queries." \
"Return only the optimized query with no explanations."

SUMMARY_PROMPT = f"System DateTime: {current_dt}\nYou process web content and create detailed summaries without losing key"\
"information. Extract main points, preserve all links and media references, maintain factual accuracy, and note important"\
"dates/statistics. Address user follow-up questions directly from the analyzed content." \
"Structure summaries with clear sections and identify actionable takeaways." \
"Preserve author perspective while ensuring clarity."


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
