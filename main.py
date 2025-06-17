import time
import asyncio
import traceback

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, IntPrompt
from rich.text import Text

from llm import CustomOllama, SystemPromptType
from tools import google_search, extract_web_page_content


# Initialize console with rich themes
load_dotenv()
console = Console()


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


async def main():
    ollama = CustomOllama()
    models = await ollama.get_models()
    
    # Display model selection with visual feedback
    selected_model = select_model(models)
    ollama.set_model(selected_model)
    console.print(f"[bold green]✅ Selected model: {selected_model.split(':')[0].title()}[/bold green]")
    
    while True:
        # Main query prompt with emoji and styling
        prompt = Prompt.ask("\n[bold yellow]🔍 What would you like to search for?[/bold yellow]")

        if not prompt:
            console.print(f"[red]⚠️ Error: Query should not be empty[/red]")
            continue
        
        if prompt.lower() == "exit":
            break
        
        # Enhanced search process with visual feedback
        with console.status("[yellow]Searching...[/yellow]", spinner="earth"):
            start_time = time.perf_counter()
            
            # Generate improved query with rich styling
            improved_query = await ollama.generate(sys_prompt_type=SystemPromptType.PARAPHRASE.value, user_prompt=prompt)
            console.print(f"[magenta]🔄 Query: {improved_query}[/magenta]")
            
            # Perform web search with visual feedback
            web_search_results = await google_search(query=improved_query)
            
            # Display search results with rich formatting
            if not web_search_results:
                console.print("[red]⚠️ No results found[/red]")
                continue
            
            snippets = ".".join(result['snippet'] for result in web_search_results)
            snippets_summary = await ollama.generate(sys_prompt_type=SystemPromptType.SUMMARY.value, user_prompt=snippets)
            
            console.print(Panel(
                Markdown(snippets_summary),
                title="[purple]🌐 Google Search Results[/purple]",
                subtitle=f"[i][light purple]{round(time.perf_counter() - start_time) }s[/light purple][/i]",
                subtitle_align="right"
            ))

        # Link selection interface
        while True:
            link_choices = get_link_choices(web_search_results)
            choice = CustomIntPrompt.ask(
                "[bold red]🔗 Select a link to get more details:[/bold red]",
                choices=link_choices
            )
            
            if choice == 0:
                break
            
            # Display link selection with visual feedback
            with console.status("[green]Loading[/green]", spinner="point"):
                result = web_search_results[choice - 1]
                link_title = result['title'].title()
                
                # Extract and display content
                start_time = time.perf_counter()
                content = await extract_web_page_content(result["link"])
                content_summary = await ollama.generate(sys_prompt_type=SystemPromptType.SUMMARY.value, user_prompt=content)
                
                console.print(Panel(
                    Markdown(content_summary),
                    title=f"[purple]{link_title}[/purple]",
                    subtitle=f"[i][light purple]{round(time.perf_counter() - start_time) }s[/light purple][/i]",
                    subtitle_align="right"
                ))
                
            # Add a brief pause between selections
            await asyncio.sleep(0.5)
        
        # Add a brief pause between main queries
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.log("[red]Ctrl+C detected. Exiting...[/red]")
    except Exception as e:
        console.log(f"[red]{str(e)}\n{traceback.format_exc()}[/red]")
