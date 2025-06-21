import json
import asyncio

from dotenv import load_dotenv
from rich.console import Console

from llm import CustomOllama, PromptType
from utils import convert_dict_to_markdown
from web import web_search, extract_web_page_content
from custom_rich_mod import MarkdownTyper, CustomIntPrompt


# Initialize console with rich themes
load_dotenv()
console = Console()


def select_model(models):
    choices = [f"{idx}. {model['model'].split(':')[0].title()}" for idx, model in enumerate(models, start=1)]
    choice = CustomIntPrompt.ask(
        "[bold blue]🤖 Available models:[/bold blue]",
        choices=choices
    )
    return models[choice - 1]["model"]


def get_link_choices(web_search_results):
    choices = [f"{i+1}. {result['title']}" for i, result in enumerate(web_search_results)]
    choices.append("0. Exit")
    return choices


def prompt_ask(message: str) -> str | None:
    exit_msg = "[i][red]enter [b]exit[/b] or [b]quit[/b] to exit[/red][/i]"

    console.print(f"{message} ({exit_msg})")
    query = input(">> ")

    if not query:
        console.print(f"[red]⚠️ Error: Query should not be empty[/red]")
        return prompt_ask(message)

    if query.lower() in {"exit", "quit"}:
        return None
    
    return query.strip()


async def main():
    ollama = CustomOllama()
    typer = MarkdownTyper(console)

    models = await ollama.get_models()
    prompts = ollama.get_prompts()
    
    selected_model = select_model(models)
    ollama.set_model(selected_model)
    model_name = selected_model.split(':')[0].title()

    console.print(f"[bold green]✅ Selected model: {model_name}[/bold green]")
    messages = [{
        "role": "system",
        "content": ollama.get_prompts().get(PromptType.SUMMARY.value)
    }]

    while True:
        prompt = prompt_ask(f"\n[bold yellow]Ask me anything![/bold yellow]")
        if prompt is None:
            break

        improved_query = await ollama.chat(messages=messages + [{
            "role": "user",
            "content": prompts.get(PromptType.PARAPHRASE.value).format(prompt)
        }])
        console.print(f"[magenta]🔄 Query: {improved_query}[/magenta]")

        web_search_results = await web_search(query=prompt)
        if not web_search_results:
            console.print("[red]⚠️ No results found[/red]")
            continue
        
        web_search_results_md = convert_dict_to_markdown({"results": web_search_results})
        user_prompt = prompts.get(PromptType.TOP_LINKS.value).format(improved_query, web_search_results_md)
            
        with console.status("[yellow]Searching...[/yellow]", spinner="earth"):    
            top_relevant_link_resp = await ollama.generate(
                sys_prompt_type=PromptType.SUMMARY.value,
                user_prompt=user_prompt,
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
            top_relevant_link = json.loads(top_relevant_link_resp)

            content = web_search_results_md

            if top_relevant_link:
                extracted_content = await extract_web_page_content(top_relevant_link["link"])
                if extracted_content:
                    content += f"\n### More Information From Source: {top_relevant_link['link']}\n\n{extracted_content}"
        
        messages.append({
            "role": "user",
            "content": f"**User Query:** {improved_query}\n---\n### Web Search Results:\n{content}"
        })
        
        with console.status("[bold green]Thinking...[/bold green]"):
            response = await ollama.chat(messages=messages)

        typer.type_with_cursor(f"🤖 {model_name} Response", response, delay=0.01)

        messages.append({
            "role": "assistant",
            "content": response
        })


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.log("[red]Ctrl+C detected. Exiting...[/red]")
    except Exception as e:
        console.log(f"[red]{str(e)}[/red]")
