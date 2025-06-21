import time
import random

from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.console import Console
from rich.prompt import IntPrompt
from rich.markdown import Markdown


class MarkdownTyper:
    def __init__(self, console=None):
        self.console = console or Console()
    
    def type_markdown_line_by_line(self, markdown_text, delay=0.8):
        lines = markdown_text.strip().split('\n')
        accumulated_text = ""
        
        for i, line in enumerate(lines):
            accumulated_text += line + '\n'
            
            self.console.clear()
            md = Markdown(accumulated_text)
            self.console.print(md)
            
            time.sleep(delay)
    
    def type_markdown_char_by_char(self, title, markdown_text, delay=0.05):
        accumulated_text = ""
        
        with Live(console=self.console, auto_refresh=False) as live:
            for char in markdown_text:
                accumulated_text += char
                
                live.update(Panel(Markdown(accumulated_text), title=title, border_style="cyan"))
                live.refresh()
                
                time.sleep(delay + random.uniform(-0.01, 0.01))
            
            time.sleep(0.5)
            live.update(Panel(Markdown(accumulated_text), title=title, border_style="cyan"))
            live.refresh()
    
    def type_with_cursor(self, title, markdown_text, delay=0.03):
        accumulated_text = ""
        cursor_states = ["|", " "]
        cursor_index = 0
        
        with Live(console=self.console, auto_refresh=False) as live:
            for char in markdown_text:
                accumulated_text += char
                display_text = accumulated_text + cursor_states[cursor_index % 2]
                
                live.update(Panel(Markdown(display_text), title=title, border_style="green"))
                live.refresh()
                
                time.sleep(delay)
                cursor_index += 1
            
            for _ in range(6):
                display_text = accumulated_text + cursor_states[cursor_index % 2]
                live.update(Panel(Markdown(display_text), title=title, border_style="green"))
                live.refresh()
                time.sleep(delay)
                cursor_index += 1
            
            live.update(Panel(Markdown(accumulated_text), title=title, border_style="green"))
            live.refresh()


class CustomIntPrompt(IntPrompt):
    def make_prompt(self, default) -> Text:
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
        assert self.choices is not None
        assert value.isdigit()

        choices = set(map(lambda x: int(x.split(".")[0]), self.choices))
        return int(value) in choices
