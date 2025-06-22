import time
import random
from typing import Optional, Union

from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.console import Console
from rich.prompt import IntPrompt
from rich.markdown import Markdown


class MarkdownTyper:
    """
    A class for creating typewriter-style animations with markdown content.
    
    This class provides multiple animation styles for displaying markdown text,
    including line-by-line, character-by-character, and cursor-based animations.
    
    Attributes:
        console (Console): The Rich console instance for output
    """
    
    def __init__(self, console: Optional[Console] = None) -> None:
        """
        Initialize the MarkdownTyper.
        
        Args:
            console: Optional Rich Console instance. If None, creates a new one.
        """
        self.console = console or Console()
    
    def type_markdown_line_by_line(
        self, 
        markdown_text: str, 
        delay: float = 0.8
    ) -> None:
        """
        Display markdown text line by line with animation.
        
        Args:
            markdown_text: The markdown content to display
            delay: Delay between lines in seconds (default: 0.8)
            
        Raises:
            ValueError: If delay is negative
        """
        if delay < 0:
            raise ValueError("Delay must be non-negative")
            
        lines = markdown_text.strip().split('\n')
        accumulated_text = ""
        
        for line in lines:
            accumulated_text += line + '\n'
            
            self.console.clear()
            md = Markdown(accumulated_text)
            self.console.print(md)
            
            time.sleep(delay)
    
    def type_markdown_char_by_char(
        self, 
        title: str, 
        markdown_text: str, 
        delay: float = 0.05,
        border_style: str = "cyan",
        turnaround_time: int = 0
    ) -> None:
        """
        Display markdown text character by character with animation in a panel.
        
        Args:
            title: Panel title
            markdown_text: The markdown content to display
            delay: Base delay between characters in seconds (default: 0.05)
            border_style: Panel border style (default: "cyan")
            turnaround_time: Turnaround time in seconds (default: 0)
            
        Raises:
            ValueError: If delay is negative
        """
        if delay < 0:
            raise ValueError("Delay must be non-negative")
            
        accumulated_text = ""
        
        with Live(console=self.console, auto_refresh=False) as live:
            for char in markdown_text:
                accumulated_text += char
                
                # Add small random variation to delay for more natural typing
                char_delay = delay + random.uniform(-0.01, 0.01)
                char_delay = max(0, char_delay)  # Ensure non-negative
                
                live.update(
                    Panel(
                        Markdown(accumulated_text), 
                        title=title, 
                        border_style=border_style
                    )
                )
                live.refresh()
                
                time.sleep(char_delay)
            
            # Brief pause at the end
            time.sleep(0.5)
            live.update(
                Panel(
                    Markdown(accumulated_text), 
                    title=title,
                    subtitle=f"{turnaround_time}s",
                    subtitle_align="right",
                    border_style=border_style
                )
            )
            live.refresh()
    
    def type_with_cursor(
        self, 
        title: str, 
        markdown_text: str, 
        delay: float = 0.03,
        border_style: str = "green",
        cursor_char: str = "|",
        cursor_blinks: int = 6,
        turnaround_time: int = 0
    ) -> None:
        """
        Display markdown text with a blinking cursor animation.
        
        Args:
            title: Panel title
            markdown_text: The markdown content to display
            delay: Delay between characters/cursor blinks in seconds (default: 0.03)
            border_style: Panel border style (default: "green")
            cursor_char: Character to use for cursor (default: "|")
            cursor_blinks: Number of cursor blinks at the end (default: 6)
            turnaround_time: Turnaround time in seconds (default: 0)
            
        Raises:
            ValueError: If delay is negative or cursor_blinks is negative
        """
        if delay < 0:
            raise ValueError("Delay must be non-negative")
        if cursor_blinks < 0:
            raise ValueError("Cursor blinks must be non-negative")
            
        accumulated_text = ""
        cursor_states = [cursor_char, " "]
        cursor_index = 0
        
        with Live(console=self.console, auto_refresh=False) as live:
            # Type characters with cursor
            for char in markdown_text:
                accumulated_text += char
                display_text = accumulated_text + cursor_states[cursor_index % 2]
                
                live.update(
                    Panel(
                        Markdown(display_text), 
                        title=title, 
                        border_style=border_style
                    )
                )
                live.refresh()
                
                time.sleep(delay)
                cursor_index += 1
            
            # Blink cursor at the end
            for _ in range(cursor_blinks):
                display_text = accumulated_text + cursor_states[cursor_index % 2]
                live.update(
                    Panel(
                        Markdown(display_text), 
                        title=title, 
                        border_style=border_style
                    )
                )
                live.refresh()
                time.sleep(delay)
                cursor_index += 1
            
            # Final display without cursor
            live.update(
                Panel(
                    Markdown(accumulated_text), 
                    title=title,
                    subtitle=f"{turnaround_time}s",
                    subtitle_align="right",
                    border_style=border_style
                )
            )
            live.refresh()


class CustomIntPrompt(IntPrompt):
    """
    Enhanced integer prompt with custom formatting and validation.
    
    Extends Rich's IntPrompt to provide better choice handling and validation
    for numbered menu-style prompts.
    """
    
    def make_prompt(self, default: Union[int, str, None] = None) -> Text:
        """
        Create the prompt text with enhanced formatting.
        
        Args:
            default: Default value to display
            
        Returns:
            Formatted prompt text
        """
        prompt = self.prompt.copy()
        prompt.end = ""

        # Add choices if available
        if self.show_choices and self.choices:
            choices_text = "\n".join(self.choices)
            prompt.append(" ")
            prompt.append(f"\n{choices_text}\n", "prompt.choices")

        # Add default value if provided
        if (
            default is not None
            and default != ...
            and self.show_default
            and isinstance(default, (str, self.response_type))
        ):
            prompt.append(" ")
            default_text = self.render_default(default)
            prompt.append(default_text)

        prompt.append(self.prompt_suffix)
        return prompt

    def check_choice(self, value: str) -> bool:
        """
        Validate if the input value is a valid choice.
        
        Args:
            value: Input value to validate
            
        Returns:
            True if value is a valid choice, False otherwise
            
        Raises:
            AssertionError: If choices is None or value is not a digit
        """
        if self.choices is None:
            raise ValueError("No choices available for validation")
            
        if not value.isdigit():
            return False

        try:
            # Extract numbers from choice strings (e.g., "1. Option" -> 1)
            valid_choices = set()
            for choice in self.choices:
                choice_parts = choice.split(".")
                if choice_parts and choice_parts[0].isdigit():
                    valid_choices.add(int(choice_parts[0]))
            
            return int(value) in valid_choices
            
        except (ValueError, IndexError):
            return False
