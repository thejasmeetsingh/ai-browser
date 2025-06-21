import datetime
from enum import Enum

from ollama import AsyncClient


current_dt = datetime.datetime.now().strftime("%A, %d-%b-%Y, %H:%I %p")


PARAPHRASE_PROMPT = "*User Query: {}*\n\nAnalyze user queries and transform them into optimized search" \
"queries under 50 characters. Preserve the original meaning and intent while making queries search-engine friendly." \
"Use keywords, remove filler words, and apply SEO best practices." \
"Consider multiple interpretations for ambiguous queries." \
"Return only the optimized query with no explanations."

SUMMARY_PROMPT = f"System DateTime: {current_dt}\nYou process web content and create detailed summaries without losing key"\
"information. Extract main points, preserve all links and media references, maintain factual accuracy, and note important"\
"dates/statistics. Address user follow-up questions directly from the analyzed content." \
"Structure summaries with clear sections and identify actionable takeaways." \
"Preserve author perspective while ensuring clarity."

TOP_LINKS_PROMPT = "**User Query:** {}\n**Web Search Results:** {}\n\nBased on the above, return a string" \
"containing the **top link** that is the most relevant and likely to effectively address or answer " \
"the user's query."


class PromptType(Enum):
    PARAPHRASE = "P"
    SUMMARY = "S"
    NEW_SEARCH = "NS"
    TOP_LINKS = "TL"


class CustomOllama:
    def __init__(self):
        self.client = AsyncClient()
        self.optons = {"num_ctx": 17000}

        self.model = None

    @staticmethod
    def format_response(content: str) -> str:
        return content.replace("<think>", "*").replace("</think>", "*")

    @staticmethod
    def get_prompts() -> dict[str, str]:
        return {
            PromptType.PARAPHRASE.value: PARAPHRASE_PROMPT,
            PromptType.SUMMARY.value: SUMMARY_PROMPT,
            PromptType.TOP_LINKS.value: TOP_LINKS_PROMPT
        }

    def set_model(self, model: str) -> None:
        self.model = model

    async def get_models(self) -> list:
        response = await self.client.list()
        return response.models
    
    async def chat(self, messages: list[dict], format: dict | None = None) -> str:
        response = await self.client.chat(model=self.model, messages=messages, format=format)
        return self.format_response(response.message.content)

    async def generate(self, sys_prompt_type: str, user_prompt: str, format: dict | None = None) -> str:
        sys_prompts = self.get_prompts()

        response = await self.client.generate(
            model=self.model,
            prompt=user_prompt,
            system=sys_prompts.get(sys_prompt_type, ""),
            options=self.optons,
            format=format
        )

        return self.format_response(response.response)
