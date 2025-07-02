import datetime
from enum import Enum
from typing import Optional, Dict, List, Any, Union

from ollama import AsyncClient
from rich.console import Console
from ollama._types import ResponseError


console = Console()

def get_current_datetime() -> str:
    """
    Get the current datetime in a formatted string.
    
    Returns:
        Formatted datetime string (e.g., "Monday, 01-Jan-2024, 14:02 PM")
    """
    return datetime.datetime.now().strftime("%A, %d-%b-%Y, %H:%M %p")


# Prompt templates
PARAPHRASE_PROMPT = (
    "*User Query: {}*\n\n"
    "Analyze user queries and transform them into optimized search queries under 50 characters. "
    "Preserve the original meaning and intent while making queries search-engine friendly. "
    "Use keywords, remove filler words, and apply SEO best practices. "
    "Consider multiple interpretations for ambiguous queries. "
    "Return only the optimized query with no explanations."
)

SUMMARY_PROMPT = (
    f"System DateTime: {get_current_datetime()}\n"
    "You process web content and create detailed summaries without losing key information. "
    "Extract main points, preserve all links and media references, maintain factual accuracy, "
    "and note important dates/statistics. Address user follow-up questions directly from the "
    "analyzed content. Structure summaries with clear sections and identify actionable takeaways. "
    "Preserve author perspective while ensuring clarity."
)

TOP_LINKS_PROMPT = (
    "**User Query:** {}\n"
    "**Web Search Results:**\n{}\n\n"
    "Based on the above, return a string containing the **top link** that is the most relevant "
    "and likely to effectively address or answer the user's query."
)

# System prompts for different use cases
SYSTEM_PROMPTS = {
    "CREATIVE": "You are a creative assistant that helps with brainstorming and creative writing.",
    "ANALYTICAL": "You are an analytical assistant that provides detailed analysis and insights.",
    "TECHNICAL": "You are a technical assistant that helps with programming and technical questions.",
    "GENERAL": "You are a helpful assistant that provides accurate and informative responses."
}


class PromptType(Enum):
    """
    Enumeration of available prompt types for different LLM operations.
    
    Attributes:
        PARAPHRASE: For query optimization and paraphrasing
        SUMMARY: For content summarization
        TOP_LINKS: For link relevance analysis
        NEW_SEARCH: For new search queries (placeholder)
    """
    PARAPHRASE = "PARAPHRASE"
    SUMMARY = "SUMMARY"
    TOP_LINKS = "TOP_LINKS"
    NEW_SEARCH = "NEW_SEARCH"


class OllamaClientError(Exception):
    """Custom exception for Ollama client errors."""
    pass


class OllamaClient:
    """
    Enhanced wrapper for Ollama AsyncClient with error handling and convenience methods.
    
    This class provides a high-level interface for interacting with Ollama models,
    including predefined prompt templates and robust error handling.
    
    Attributes:
        client (AsyncClient): The underlying Ollama client
        model (str): Currently selected model name
        default_options (dict): Default options for model interactions
    """
    
    def __init__(self, host: Optional[str] = None, timeout: Optional[float] = None):
        """
        Initialize the Ollama client wrapper.
        
        Args:
            host: Optional Ollama server host (default: None uses Ollama default)
            timeout: Optional timeout for requests in seconds
        """
        try:
            if host:
                self.client = AsyncClient(host=host, timeout=timeout)
            else:
                self.client = AsyncClient(timeout=timeout)
        except Exception as e:
            raise OllamaClientError(f"Failed to initialize Ollama client: {e}")
        
        self.model: Optional[str] = None
        self.default_options: Dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 2048
        }
        
        console.log("[blue]Ollama client initialized successfully[/blue]")

    @staticmethod
    def format_response(content: str) -> str:
        """
        Format response content by replacing think tags with emphasis.
        
        Args:
            content: Raw response content
            
        Returns:
            Formatted content with think tags replaced
        """
        if not isinstance(content, str):
            return str(content)
            
        return content.replace("<think>", "*").replace("</think>", "*")

    @staticmethod
    def get_prompt_templates() -> Dict[str, str]:
        """
        Get all available prompt templates.
        
        Returns:
            Dictionary mapping prompt types to template strings
        """
        return {
            PromptType.PARAPHRASE.value: PARAPHRASE_PROMPT,
            PromptType.SUMMARY.value: SUMMARY_PROMPT,
            PromptType.TOP_LINKS.value: TOP_LINKS_PROMPT
        }

    def set_model(self, model: str) -> None:
        """
        Set the model to use for subsequent operations.
        
        Args:
            model: Name of the Ollama model to use
            
        Raises:
            ValueError: If model name is empty or invalid
        """
        if not model or not isinstance(model, str):
            raise ValueError("Model name must be a non-empty string")
            
        self.model = model.strip()
        console.log(f"[blue]Model set to: {self.model}[/blue]")

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available Ollama models.
        
        Returns:
            List of available model information
            
        Raises:
            OllamaClientError: If unable to retrieve models
        """
        try:
            response = await self.client.list()
            models = response.get('models', []) if hasattr(response, 'get') else response.models
            console.log(f"[blue]Retrieved {len(models)} available models[/blue]")
            return models
        except Exception as e:
            raise OllamaClientError(f"Failed to retrieve models: {e}")

    async def chat(
        self, 
        messages: List[Dict[str, str]], 
        options: Optional[Dict[str, Any]] = None,
        format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a chat request to the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            options: Optional model options (overrides defaults)
            format_type: Optional response format ('json', etc.)
            
        Returns:
            Formatted response content
            
        Raises:
            OllamaClientError: If model not set or request fails
        """
        if not self.model:
            raise OllamaClientError("No model set. Use set_model() first.")
        
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Merge options with defaults
        final_options = {**self.default_options, **(options or {})}
        
        try:
            response = await self.client.chat(
                model=self.model,
                messages=messages,
                options=final_options,
                format=format
            )
            
            content = response.message.content if hasattr(response, 'message') else str(response)
            return self.format_response(content)
            
        except ResponseError as e:
            raise OllamaClientError(f"Chat request failed: {e}")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error during chat: {e}")

    async def generate(
        self, 
        prompt_type: Union[PromptType, str], 
        user_input: str,
        options: Optional[Dict[str, Any]] = None,
        format: Optional[dict[str, str]] = None,
        custom_system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using a predefined prompt template.
        
        Args:
            prompt_type: Type of prompt to use (PromptType enum or string)
            user_input: User input to process
            options: Optional model options (overrides defaults)
            format_type: Optional response format ('json', etc.)
            custom_system_prompt: Optional custom system prompt to override template
            
        Returns:
            Formatted response content
            
        Raises:
            OllamaClientError: If model not set or request fails
            ValueError: If prompt type is invalid
        """
        if not self.model:
            raise OllamaClientError("No model set. Use set_model() first.")
        
        if not user_input.strip():
            raise ValueError("User input cannot be empty")
        
        # Convert enum to string if needed
        if isinstance(prompt_type, PromptType):
            prompt_type_str = prompt_type.value
        else:
            prompt_type_str = str(prompt_type)
        
        # Get system prompt
        if custom_system_prompt:
            system_prompt = custom_system_prompt
        else:
            prompt_templates = self.get_prompt_templates()
            system_prompt = prompt_templates.get(prompt_type_str, "")
            
            if not system_prompt:
                available_types = list(prompt_templates.keys())
                raise ValueError(
                    f"Invalid prompt type: {prompt_type_str}. "
                    f"Available types: {available_types}"
                )
        
        # Format system prompt if it contains placeholders
        if '{}' in system_prompt:
            try:
                system_prompt = system_prompt.format(user_input)
            except (ValueError, IndexError) as e:
                console.log(f"[yellow]Could not format system prompt: {e}[/yellow]")
        
        # Merge options with defaults
        final_options = {**self.default_options, **(options or {})}
        
        try:
            response = await self.client.generate(
                model=self.model,
                prompt=user_input,
                system=system_prompt,
                options=final_options,
                format=format
            )
            
            content = response.response if hasattr(response, 'response') else str(response)
            return self.format_response(content)
            
        except ResponseError as e:
            raise OllamaClientError(f"Generate request failed: {e}")
        except Exception as e:
            raise OllamaClientError(f"Unexpected error during generation: {e}")

    async def quick_paraphrase(self, messages: List[Dict[str, str]], query: str) -> str:
        """
        Convenience method for quick query paraphrasing.
        
        Args:
            messages: List of messages
            query: Query to paraphrase
            
        Returns:
            Paraphrased query
        """

        prompt_templates = self.get_prompt_templates()
        prompt = prompt_templates.get(PromptType.PARAPHRASE.value, "")
        prompt = prompt.format(query)

        return await self.chat(
            messages=messages + [{
                "role": "user",
                "content": prompt
            }],
            format={
                "type": "object",
                "properties": {
                    "improved_query": {
                        "type": "string"
                    }
                },
                "required": ["improved_query"]
            }
        )

    async def health_check(self) -> bool:
        """
        Check if the Ollama server is responsive.
        
        Returns:
            True if server is responsive, False otherwise
        """
        try:
            await self.get_available_models()
            return True
        except Exception:
            return False

    def __repr__(self) -> str:
        """String representation of the client."""
        return f"OllamaClient(model={self.model}, options={self.default_options})"
