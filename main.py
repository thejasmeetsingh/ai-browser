import os
import time
import asyncio
import traceback

from dotenv import load_dotenv
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import IntPrompt
from rich.text import Text

from llm import CustomOllama, SystemPromptType
from web import extract_web_page_content, google_search, brave_search


# Initialize console with rich themes
load_dotenv()
console = Console()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


class CustomIntPrompt(IntPrompt):
    def make_prompt(self, default) -> Text:
        """Make prompt text.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text: Text to display in prompt.
        """

        prompt = self.prompt.copy()
        prompt.end = ""

        if self.show_choices and self.choices:
            _choices = "\n".join(self.choices)
            choices = f"\n{_choices}\n"
            prompt.append(" ")
            prompt.append(choices, "prompt.choices")

        if (
            default != ...
            and self.show_default
            and isinstance(default, (str, self.response_type))
        ):
            prompt.append(" ")
            _default = self.render_default(default)
            prompt.append(_default)

        prompt.append(self.prompt_suffix)

        return prompt

    def check_choice(self, value: str) -> bool:
        """Check value is in the list of valid choices.

        Args:
            value (str): Value entered by user.

        Returns:
            bool: True if choice was valid, otherwise False.
        """

        assert self.choices is not None
        assert value.isdigit()

        choices = set(map(lambda x: int(x.split(".")[0]), self.choices))
        return int(value) in choices


def select_model(models):
    """Select a model with emoji-based choices"""

    choices = [f"{idx}. {model['model'].split(':')[0].title()}" for idx, model in enumerate(models, start=1)]
    choice = CustomIntPrompt.ask(
        "[bold blue]🤖 Available models:[/bold blue]",
        choices=choices
    )
    return models[choice - 1]["model"]


def get_link_choices(web_search_results):
    """Generate link selection options with emoji indicators"""

    choices = [f"{i+1}. {result['title']}" for i, result in enumerate(web_search_results)]
    choices.append("0. Exit")
    return choices


async def web_search(query: str, count: int = 10) -> list:
    web_search_results = []

    if GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID:
        google_search_results = await google_search(GOOGLE_API_KEY, GOOGLE_SEARCH_ENGINE_ID,
                                                    query, count)
        web_search_results.extend(google_search_results)
    
    if BRAVE_API_KEY:
        brave_search_results = await brave_search(BRAVE_API_KEY, query, count)
        web_search_results.extend(brave_search_results)

    return web_search_results


def prompt_ask(message: str) -> str | None:
    exit_msg = "[i][red]enter [b]exit[/b] or [b]quit[/b] to exit[/red][/i]"

    console.print(f"{message}({exit_msg})")
    query = input(">> ")

    if not query:
        console.print(f"[red]⚠️ Error: Query should not be empty[/red]")
        return prompt_ask(message)

    if query.lower() in {"exit", "quit"}:
        return None
    
    return query.strip()

def print_response(markdown_content: str, panel_title: str, start_time: int | None = None) -> None:
    if not start_time:
        start_time = time.perf_counter()

    console.print(Panel(
        Markdown(markdown_content),
        title=f"[purple]{panel_title}[/purple]",
        subtitle=f"[i][light purple]{round(time.perf_counter() - start_time)}s[/light purple][/i]",
        subtitle_align="right"
    ))

async def initiate_disucssion(_ollama: CustomOllama, web_page_contents: str) -> None:
    prompts = _ollama.get_sys_prompts()

    messages = [
        {
            "role": "system",
            "content": prompts[SystemPromptType.SUMMARY.value]
        },
        {
            "role": "user",
            "content": f"# Web Page Contents\n\n{web_page_contents}"
        }
    ]

    while True:
        with console.status("[blue]Thinking...[/blue]"):
            start_time = time.perf_counter()
            response = await _ollama.chat(messages)

            print_response(response, "🤖", start_time)

        question = prompt_ask("[magenta]Ask a follow-up question[/magenta]")
        if question is None:
            return
        
        messages.extend([
            {
                "role": "assistant",
                "content": response
            },
            {
                "role": "user",
                "content": question
            }
        ])


async def extract_url_contents(_ollama: CustomOllama, url: str, title: str | None = None) -> None:
    if not TAVILY_API_KEY:
        console.print("[red]TAVILY_API_KEY is not configured in env. Cannot extract web page content[/red]")
        return

    start_time = time.perf_counter()
    content = None

    if not title:
        title = "Link"

    with console.status("[green]Loading[/green]", spinner="point"):
        content = await extract_web_page_content(TAVILY_API_KEY, url)
        if not content:
            message = f"No content found ⛓️‍💥. Please visit [{title}]({url}) by yourself."
            print_response(message, title, start_time)

    await initiate_disucssion(_ollama, content)


async def main():
    ollama = CustomOllama()
    models = await ollama.get_models()
    
    # Display model selection with visual feedback
    selected_model = select_model(models)
    ollama.set_model(selected_model)
    model_name = selected_model.split(':')[0].title()

    console.print(f"[bold green]✅ Selected model: {model_name}[/bold green]")
    
    while True:
        # Main query prompt with emoji and styling
        prompt = prompt_ask("\n[bold yellow]🔍 What would you like to search for?[/bold yellow]")
        if prompt is None:
            break
        
        # Enhanced search process with visual feedback
        with console.status("[yellow]Searching...[/yellow]", spinner="earth"):
            start_time = time.perf_counter()
            
            # Generate improved query with rich styling
            improved_query = await ollama.generate(
                sys_prompt_type=SystemPromptType.PARAPHRASE.value,
                user_prompt=prompt
            )
            console.print(f"[magenta]🔄 Query: {improved_query}[/magenta]")
            
            # Perform web search with visual feedback
            web_search_results = await web_search(query=prompt)
            
            # Display search results with rich formatting
            if not web_search_results:
                console.print("[red]⚠️ No results found[/red]")
                continue
            
            snippets = ".".join(result['snippet'] for result in web_search_results)
            snippets_summary = await ollama.generate(
                sys_prompt_type=SystemPromptType.SUMMARY.value,
                user_prompt=snippets
            )

            print_response(snippets_summary, "🌐 Search Results", start_time)

        # Link selection interface
        while True:
            link_choices = get_link_choices(web_search_results)
            choice = CustomIntPrompt.ask(
                "[bold red]🔗 Select a link to get more details:[/bold red]",
                choices=link_choices
            )
            
            if choice == 0:
                break

            result = web_search_results[choice - 1]
            link_title = result['title'].title()
            
            await extract_url_contents(ollama, result["link"], link_title)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.log("[red]Ctrl+C detected. Exiting...[/red]")
    except Exception as e:
        console.log(f"[red]{str(e)}[/red]")
