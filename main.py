import sys
import time
import json
import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Any

from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv
from rich.console import Console

from utils import convert_dict_to_markdown
from markdown_typer import MarkdownTyper, CustomIntPrompt
from llm_client import OllamaClient, PromptType, OllamaClientError
from web_search_client import WebSearchClient, SearchResult, WebSearchError, ContentExtractionError


console = Console()


# Configuration constants
class Config:
    """Application configuration constants."""
    NUM_CTX = 50000
    DEFAULT_SEARCH_COUNT = 10
    MAX_CONTENT_LENGTH = 100000
    MIN_QUERY_LENGTH = 2
    MAX_QUERY_LENGTH = 500
    DEFAULT_MODEL_TIMEOUT = 60.0
    DEFAULT_SEARCH_TIMEOUT = 30.0


@dataclass
class AppState:
    """Application state container."""
    ollama_client: OllamaClient
    search_client: WebSearchClient
    typer: MarkdownTyper
    console: Console
    selected_model: Optional[str] = None
    conversation_history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        """Initialize conversation history if not provided."""
        if self.conversation_history is None:
            self.conversation_history = []


class SearchResultProcessor:
    """Handles processing and formatting of search results."""
    
    @staticmethod
    def convert_results_to_markdown(results: List[SearchResult]) -> str:
        """
        Convert search results to formatted markdown.
        
        Args:
            results: List of SearchResult objects
            
        Returns:
            Formatted markdown string
        """
        if not results:
            return "No search results found."
        
        search_results_dict = [search_result.to_dict() for search_result in results]
        
        return convert_dict_to_markdown({"results": search_results_dict})

    @staticmethod
    def create_content_summary(results: List[SearchResult], extracted_content: Optional[str] = None) -> str:
        """
        Create a comprehensive content summary from search results and extracted content.
        
        Args:
            results: List of SearchResult objects
            extracted_content: Optional extracted web page content
            
        Returns:
            Formatted content summary
        """
        content_parts = ["# Summary\n"]
        
        # Add search result snippets
        if results:
            snippets = [result.snippet for result in results if result.snippet.strip()]
            if snippets:
                content_parts.append("## Search Results Overview")
                content_parts.append(". ".join(snippets) + ".")
                content_parts.append("")
        
        # Add extracted content if available
        if extracted_content and extracted_content.strip():
            # Truncate if too long
            if len(extracted_content) > Config.MAX_CONTENT_LENGTH:
                extracted_content = extracted_content[:Config.MAX_CONTENT_LENGTH] + "... [Content truncated]"
            
            content_parts.extend([
                "## Detailed Information",
                extracted_content,
                ""
            ])
        
        return "\n".join(content_parts)


class InputValidator:
    """Handles input validation and sanitization."""
    
    @staticmethod
    def validate_query(query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user query input.
        
        Args:
            query: User input query
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or not query.strip():
            return False, "Query cannot be empty"
        
        query = query.strip()
        
        if len(query) < Config.MIN_QUERY_LENGTH:
            return False, f"Query must be at least {Config.MIN_QUERY_LENGTH} characters"
        
        if len(query) > Config.MAX_QUERY_LENGTH:
            return False, f"Query must be less than {Config.MAX_QUERY_LENGTH} characters"
        
        return True, None
    
    @staticmethod
    def is_exit_command(query: str) -> bool:
        """Check if query is an exit command."""
        if not query:
            return False
        return query.lower().strip() in {"exit", "quit", "q", "bye"}


class ModelManager:
    """Handles model selection and management."""
    
    @staticmethod
    async def get_available_models(ollama_client: OllamaClient) -> List[Dict[str, Any]]:
        """
        Get available Ollama models with error handling.
        
        Args:
            ollama_client: Ollama client instance
            
        Returns:
            List of available models
            
        Raises:
            OllamaClientError: If unable to retrieve models
        """
        try:
            return await ollama_client.get_available_models()
        except Exception as e:
            raise OllamaClientError(f"Failed to retrieve available models: {e}")
    
    @staticmethod
    def display_model_selection(console: Console, models: List[Dict[str, Any]]) -> str:
        """
        Display model selection interface and get user choice.
        
        Args:
            console: Rich console instance
            models: List of available models
            
        Returns:
            Selected model name
            
        Raises:
            ValueError: If no models available or invalid selection
        """
        if not models:
            raise ValueError("No models available")
        
        # Create model choices
        choices = []
        table = Table(title="🤖 Available Ollama Models", show_header=True, header_style="bold blue")
        table.add_column("Index", style="dim", width=6)
        table.add_column("Model Name", style="green")
        table.add_column("Size", style="yellow")
        
        for idx, model in enumerate(models, start=1):
            model_name = model.get('model', 'Unknown').split(':')[0].title()
            model_size = model.get('size', 'Unknown')
            
            # Format size if it's a number
            if isinstance(model_size, (int, float)):
                model_size = f"{model_size / (1024**3):.1f} GB"
            
            choices.append(f"{idx}. {model_name}")
            table.add_row(str(idx), model_name, str(model_size))
        
        console.print(table)
        
        try:
            choice = CustomIntPrompt.ask(
                "\n[bold blue]Select a model[/bold blue]",
                choices=choices,
                show_choices=False
            )
            
            if 1 <= choice <= len(models):
                selected_model = models[choice - 1]
                return selected_model.get('name', selected_model.get('model', 'unknown'))
            else:
                raise ValueError("Invalid model selection")
                
        except (ValueError, KeyboardInterrupt) as e:
            raise ValueError(f"Model selection failed: {e}")


class ConversationManager:
    """Manages conversation flow and history."""
    
    def __init__(self, app_state: AppState):
        """Initialize conversation manager with app state."""
        self.app_state = app_state
        self.setup_system_message()
    
    def setup_system_message(self) -> None:
        """Setup initial system message for the conversation."""
        system_prompt = self.app_state.ollama_client.get_prompt_templates().get(
            PromptType.SUMMARY.value, 
            "You are a helpful AI assistant that provides accurate and informative responses."
        )
        
        self.app_state.conversation_history = [{
            "role": "system",
            "content": system_prompt
        }]
    
    async def process_query(self, user_query: str) -> Optional[str]:
        """
        Process user query through the complete pipeline.
        
        Args:
            user_query: User's input query
            
        Returns:
            AI response or None if processing failed
        """
        console = self.app_state.console
        
        try:
            # Step 1: Optimize query for search
            optimized_query = await self._optimize_query(user_query)
            console.print(f"[magenta]🔄 Optimized Query: {optimized_query}[/magenta]")
            
            # Step 2: Perform web search
            search_results = await self._perform_web_search(optimized_query)
            if not search_results:
                console.print("[red]⚠️ No search results found[/red]")
                return await self._generate_response_without_search(user_query)
            
            # Step 3: Find most relevant link and extract content
            extracted_content = await self._extract_relevant_content(optimized_query, search_results)
            
            # Step 4: Generate AI response
            return await self._generate_ai_response(user_query, optimized_query, search_results, extracted_content)
            
        except Exception as e:
            console.log(f"[red]Error processing query: {e}[/red]")
            console.print(f"[red]❌ Error processing query: {e}[/red]")
            return None
    
    async def _optimize_query(self, query: str) -> str:
        """Optimize user query for better search results."""
        try:
            optimized_resp = await self.app_state.ollama_client.quick_paraphrase(
                messages=self.app_state.conversation_history,
                query=query
            )
            optimized = json.loads(optimized_resp)

            # Fallback to original if optimization fails
            return optimized.get("improved_query", "").strip() or query
        except Exception as e:
            console.log(f"[yellow]Query optimization failed: {e}[/yellow]")
            return query
    
    async def _perform_web_search(self, query: str) -> List[SearchResult]:
        """Perform web search using the search client."""
        with self.app_state.console.status("[yellow]🔍 Searching the web...[/yellow]", spinner="dots"):
            try:
                return await self.app_state.search_client.search(
                    query, 
                    count=Config.DEFAULT_SEARCH_COUNT
                )
            except WebSearchError as e:
                console.log(f"[red]Web search failed: {e}[/red]")
                return []
    
    async def _extract_relevant_content(self, query: str, results: List[SearchResult]) -> Optional[str]:
        """Extract content from the most relevant search result."""
        if not results:
            return None
        
        try:
            # Use LLM to determine most relevant link
            results_markdown = SearchResultProcessor.convert_results_to_markdown(results)
            prompt_templates = self.app_state.ollama_client.get_prompt_templates()
            top_links_prompt = prompt_templates.get(PromptType.TOP_LINKS.value, "")
            
            if not top_links_prompt:
                # Fallback: use first result
                top_link = results[0].link
            else:
                with self.app_state.console.status("[blue]🎯 Finding most relevant source...[/blue]", spinner="dots"):
                    # Format the prompt with query and results
                    formatted_prompt = top_links_prompt.format(query, results_markdown)
                    
                    # Generate response with JSON format
                    response = await self.app_state.ollama_client.generate(
                        PromptType.SUMMARY.value,
                        user_input=formatted_prompt,
                        format={
                            "type": "object",
                            "properties": {
                                "link": {
                                    "type": "string"
                                }
                            },
                            "required": ["link"]
                        }
                    )
                    
                    try:
                        parsed_response = json.loads(response)
                        top_link = parsed_response.get("link", results[0].link)
                    except (json.JSONDecodeError, KeyError):
                        # Fallback to first result if JSON parsing fails
                        top_link = results[0].link
            
            # Extract content from the top link
            with self.app_state.console.status("[green]📄 Extracting content...[/green]", spinner="dots"):
                return await self.app_state.search_client.extract_content(top_link)
                
        except (ContentExtractionError, Exception) as e:
            console.log(f"[yellow]Content extraction failed: {e}[/yellow]")
            return None
    
    async def _generate_ai_response(
        self, 
        original_query: str, 
        optimized_query: str, 
        search_results: List[SearchResult], 
        extracted_content: Optional[str]
    ) -> str:
        """Generate AI response based on search results and extracted content."""
        
        # Create comprehensive content summary
        content_summary = SearchResultProcessor.create_content_summary(search_results, extracted_content)
        
        # Add user message to conversation
        user_message = {
            "role": "user",
            "content": f"**User Query:** {original_query}\n**Optimized Query:** {optimized_query}\n\n{content_summary}"
        }
        self.app_state.conversation_history.append(user_message)
        
        # Generate response
        with self.app_state.console.status("[bold green]🤔 Generating response...[/bold green]"):
            try:
                response = await self.app_state.ollama_client.chat(
                    messages=self.app_state.conversation_history,
                    options={"num_ctx": Config.NUM_CTX}
                )
                
                # Add assistant response to conversation history
                assistant_message = {"role": "assistant", "content": response}
                self.app_state.conversation_history.append(assistant_message)
                
                return response
                
            except OllamaClientError as e:
                console.log(f"[red]AI response generation failed: {e}[/red]")
                return f"I apologize, but I encountered an error while generating a response: {e}"
    
    async def _generate_response_without_search(self, query: str) -> str:
        """Generate AI response without web search results."""
        user_message = {
            "role": "user", 
            "content": f"Please answer this query without web search: {query}"
        }
        self.app_state.conversation_history.append(user_message)
        
        try:
            response = await self.app_state.ollama_client.chat(
                messages=self.app_state.conversation_history
            )
            
            assistant_message = {"role": "assistant", "content": response}
            self.app_state.conversation_history.append(assistant_message)
            
            return response
            
        except OllamaClientError as e:
            return f"I apologize, but I cannot process your query at the moment: {e}"


def display_welcome_message(console: Console) -> None:
    """Display application welcome message."""
    welcome_text = Text()
    welcome_text.append("🚀 AI-Powered Web Search & Chat\n", style="bold cyan")
    welcome_text.append("Combining local AI with real-time web search\n", style="italic")
    welcome_text.append("\nFeatures:\n", style="bold")
    welcome_text.append("• Query optimization for better search results\n", style="green")
    welcome_text.append("• Multi-provider web search (Google, Brave)\n", style="green")
    welcome_text.append("• Intelligent content extraction\n", style="green")
    welcome_text.append("• Conversational AI responses\n", style="green")
    
    panel = Panel(
        welcome_text,
        title="Welcome",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)


def get_user_input(console: Console) -> Optional[str]:
    """
    Get user input with validation and exit handling.
    
    Args:
        console: Rich console instance
        
    Returns:
        Validated user input or None for exit
    """
    while True:
        console.print("\n[bold yellow]💬 Ask me anything![/bold yellow]")
        console.print("[dim italic]Type 'exit', 'quit', or 'q' to exit[/dim italic]")
        
        try:
            query = input(">> ").strip()
            
            if InputValidator.is_exit_command(query):
                return None
            
            is_valid, error_message = InputValidator.validate_query(query)
            if not is_valid:
                console.print(f"[red]⚠️ {error_message}[/red]")
                continue
            
            return query
            
        except (KeyboardInterrupt, EOFError):
            return None


async def initialize_clients() -> Tuple[OllamaClient, WebSearchClient]:
    """
    Initialize and configure the AI and search clients.
    
    Returns:
        Tuple of (ollama_client, search_client)
        
    Raises:
        Exception: If client initialization fails
    """
    # Initialize Ollama client
    try:
        ollama_client = OllamaClient(timeout=Config.DEFAULT_MODEL_TIMEOUT)
        
        # Test connection
        if not await ollama_client.health_check():
            raise OllamaClientError("Ollama server is not responding")
            
    except Exception as e:
        raise Exception(f"Failed to initialize Ollama client: {e}")
    
    # Initialize search client
    try:
        search_client = WebSearchClient(
            timeout=Config.DEFAULT_SEARCH_TIMEOUT,
            max_retries=2,
            deduplicate=True
        )
        
        # Check configured providers
        providers = search_client.get_configured_providers()
        if not providers:
            console.log("[yellow]No search providers configured - search functionality will be limited[/yellow]")
            
    except Exception as e:
        raise Exception(f"Failed to initialize search client: {e}")
    
    return ollama_client, search_client


@asynccontextmanager
async def application_context():
    """Async context manager for application lifecycle."""
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Display welcome message
        display_welcome_message(console)
        
        # Initialize clients
        console.print("[blue]🔧 Initializing AI and search clients...[/blue]")
        ollama_client, search_client = await initialize_clients()
        
        # Initialize other components
        typer = MarkdownTyper(console)
        
        # Create application state
        app_state = AppState(
            ollama_client=ollama_client,
            search_client=search_client,
            typer=typer,
            console=console
        )
        
        console.print("[green]✅ Initialization complete![/green]")
        
        yield app_state
        
    except Exception as e:
        console.print(f"[red]❌ Initialization failed: {e}[/red]")
        raise
    finally:
        console.print("[blue]🔄 Cleaning up resources...[/blue]")


async def main() -> None:
    """
    Main application entry point.
    
    Orchestrates the entire application flow including:
    - Client initialization
    - Model selection
    - Interactive chat loop
    - Graceful shutdown
    """
    try:
        async with application_context() as app_state:
            console = app_state.console
            
            # Model selection
            console.print("\n[blue]🤖 Setting up AI model...[/blue]")
            models = await ModelManager.get_available_models(app_state.ollama_client)
            
            if not models:
                console.print("[red]❌ No Ollama models available. Please install a model first.[/red]")
                return
            
            selected_model = ModelManager.display_model_selection(console, models)
            app_state.ollama_client.set_model(selected_model)
            app_state.selected_model = selected_model
            
            model_display_name = selected_model.split(':')[0].title()
            console.print(f"[bold green]✅ Selected model: {model_display_name}[/bold green]")
            
            # Initialize conversation manager
            conversation_manager = ConversationManager(app_state)
            
            # Main chat loop
            console.print("\n[bold cyan]🎯 Ready to chat! Ask me anything.[/bold cyan]")
            
            while True:
                # Get user input
                user_query = get_user_input(console)
                if user_query is None:
                    break

                start_time = time.perf_counter()
                
                # Process query and get response
                ai_response = await conversation_manager.process_query(user_query)

                turnaround_time = round(time.perf_counter() - start_time)
                
                if ai_response:
                    # Display response with typewriter effect
                    app_state.typer.type_with_cursor(
                        f"🤖 {model_display_name} Response",
                        ai_response,
                        delay=0.01,
                        border_style="green",
                        turnaround_time=turnaround_time
                    )
                else:
                    console.print("[red]❌ Unable to generate response. Please try again.[/red]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Interrupted by user. Exiting gracefully...[/yellow]")
    except Exception as e:
        console.log(f"[red]Application error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    """Application entry point with proper error handling."""
    # Configure global logger

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!")
    except Exception as e:
        console.log(f"[bold red]Critical application error: {e}[/bold red]")
        console.print(f"💥 Critical error: {e}")
        sys.exit(1)
