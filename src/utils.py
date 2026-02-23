"""
Utility functions for file parsing and shell execution.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

if TYPE_CHECKING:
    pass


@dataclass
class ParsedAction:
    action_type: str
    thought: str = ""
    file_path: str = ""
    content: str = ""
    command: str = ""


class FileParser:
    THOUGHT_PATTERN = re.compile(
        r"\[THOUGHT\](.*?)\[/THOUGHT\]",
        re.DOTALL | re.IGNORECASE,
    )
    FILE_WRITE_PATTERN = re.compile(
        r"\[FILE_WRITE\]\s*\n?\s*([^\n]+)\s*\n?\s*```[^\n]*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    SHELL_CMD_PATTERN = re.compile(
        r"\[SHELL_CMD\]\s*\n?\s*([^\[\]]+?)(?=\[/SHELL_CMD\]|\[|$)",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def parse(self, response: str) -> list[ParsedAction]:
        actions = []

        thoughts = self.THOUGHT_PATTERN.findall(response)
        thought_text = thoughts[0].strip() if thoughts else ""

        for match in self.FILE_WRITE_PATTERN.finditer(response):
            file_path = match.group(1).strip()
            content = match.group(2).strip()
            actions.append(
                ParsedAction(
                    action_type="file_write",
                    thought=thought_text,
                    file_path=file_path,
                    content=content,
                )
            )

        for match in self.SHELL_CMD_PATTERN.finditer(response):
            command = match.group(1).strip()
            if command:
                actions.append(
                    ParsedAction(
                        action_type="shell_cmd",
                        thought=thought_text,
                        command=command,
                    )
                )

        if not actions and thought_text:
            actions.append(
                ParsedAction(
                    action_type="thought",
                    thought=thought_text,
                )
            )

        return actions

    def display_parsed(self, actions: list[ParsedAction]) -> None:
        for action in actions:
            if action.thought:
                thought_panel = Panel(
                    Text(action.thought),
                    title="[bold blue]Thinking[/bold blue]",
                    border_style="blue",
                )
                self.console.print(thought_panel)

            if action.action_type == "file_write":
                self.console.print(
                    f"\n[bold green]File to write:[/bold green] {action.file_path}"
                )
                syntax = Syntax(
                    action.content,
                    self._detect_language(action.file_path),
                    theme="monokai",
                    line_numbers=True,
                )
                self.console.print(syntax)

            elif action.action_type == "shell_cmd":
                self.console.print(
                    f"\n[bold yellow]Shell command:[/bold yellow] {action.command}"
                )

    def _detect_language(self, file_path: str) -> str:
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sh": "bash",
            ".rs": "rust",
            ".go": "go",
            ".java": "java",
            ".kt": "kotlin",
            ".rb": "ruby",
            ".php": "php",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
        }
        ext = Path(file_path).suffix.lower()
        return extension_map.get(ext, "text")


def write_file(
    file_path: Union[str, Path],
    content: str,
    root_path: Optional[Union[str, Path]] = None,
    console: Optional[Console] = None,
) -> Path:
    console = console or Console()
    path = Path(file_path)

    if root_path:
        path = Path(root_path) / path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    console.print(f"[green]✓[/green] Wrote: {path}")
    return path


def execute_shell_command(
    command: str,
    cwd: Optional[Union[Path, str]] = None,
    console: Optional[Console] = None,
) -> tuple[int, str, str]:
    console = console or Console()
    console.print(f"[yellow]Executing:[/yellow] {command}")

    result = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        console.print(f"[red]Error (exit code {result.returncode}):[/red]")
        if result.stderr:
            console.print(result.stderr)
    else:
        console.print("[green]✓ Command executed successfully[/green]")
        if result.stdout:
            console.print(result.stdout)

    return result.returncode, result.stdout, result.stderr


def confirm_action(action: ParsedAction, console: Optional[Console] = None) -> bool:
    console = console or Console()

    if action.action_type == "file_write":
        return console.input(
            f"\n[bold]Write file [cyan]{action.file_path}[/cyan]? [Y/n]:[/bold] "
        ).lower() in ("y", "yes", "")
    elif action.action_type == "shell_cmd":
        return console.input(
            f"\n[bold]Execute command [yellow]{action.command}[/yellow]? [Y/n]:[/bold] "
        ).lower() in ("y", "yes", "")

    return True
