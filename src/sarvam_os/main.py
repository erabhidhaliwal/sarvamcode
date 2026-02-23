"""
Main: Prompt-toolkit chat interface with advanced UX.
High-end terminal UI for Sarvam-OS.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import clear
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.sarvam_os.agent import SarvamAgent, create_agent

console = Console()

STYLE = Style.from_dict(
    {
        "prompt": "bold cyan",
        "user": "bold green",
        "assistant": "bold yellow",
        "system": "bold red",
    }
)

HISTORY_FILE = Path.home() / ".sarvam" / "chat_history"


def create_key_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("c-c")
    def _(event):
        event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

    @kb.add("c-l")
    def _(event):
        clear()
        event.app.renderer.clear()

    @kb.add("c-d")
    def _(event):
        event.app.exit(exception=EOFError)

    return kb


class SarvamCLI:
    def __init__(
        self,
        project_path: Optional[Path] = None,
        stream: bool = True,
    ):
        self.project_path = project_path or Path.cwd()
        self.stream = stream
        self.agent = create_agent(self.project_path)
        self.session: Optional[PromptSession] = None
        self.running = True

        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    def _setup_session(self) -> PromptSession:
        return PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            key_bindings=create_key_bindings(),
            style=STYLE,
            multiline=False,
            mouse_support=True,
        )

    def _print_welcome(self) -> None:
        summary = self.agent.get_memory_summary()
        console.print(
            Panel.fit(
                Text.assemble(
                    ("Sarvam-OS", "bold cyan"),
                    (" - Your AI Co-Developer\n\n", "default"),
                    ("Project: ", "dim"),
                    (str(self.project_path.name), "green"),
                    ("\nMemory: ", "dim"),
                    (str(summary["total_messages"]), "yellow"),
                    (" messages", "dim"),
                ),
                title="[bold]Welcome[/bold]",
                border_style="cyan",
            )
        )
        console.print("\n[dim]Commands: /help, /scan, /clear, /exit[/dim]\n")

    def _print_help(self) -> None:
        help_text = """
[bold]Available Commands:[/bold]

  /help          Show this help message
  /scan          Scan and display project structure
  /clear         Clear conversation memory
  /history       Show conversation history
  /exit          Exit Sarvam-OS

[bold]Tips:[/bold]
  • Use ↑/↓ arrows for command history
  • Ctrl+C to cancel current input
  • Ctrl+L to clear screen
  • The agent remembers context across sessions
"""
        console.print(Panel(help_text, border_style="blue"))

    def _handle_command(self, command: str) -> bool:
        cmd = command.lower().strip()

        if cmd in ("/exit", "/quit", "/q"):
            console.print("[yellow]Goodbye! Your conversation has been saved.[/yellow]")
            return False

        elif cmd == "/help":
            self._print_help()

        elif cmd == "/scan":
            self._scan_project()

        elif cmd == "/clear":
            self.agent.clear_memory()
            console.print("[green]✓ Memory cleared[/green]")

        elif cmd == "/history":
            self._show_history()

        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("[dim]Type /help for available commands[/dim]")

        return True

    def _scan_project(self) -> None:
        from src.agent import ProjectMapper

        mapper = ProjectMapper(self.project_path)
        tree = mapper.to_markdown()
        console.print(Panel(tree, title="Project Structure", border_style="cyan"))

    def _show_history(self) -> None:
        messages = self.agent.memory.get_messages()
        if not messages:
            console.print("[dim]No conversation history[/dim]")
            return

        console.print(f"\n[bold]Conversation History ({len(messages)} messages):[/bold]\n")
        for msg in messages[-20:]:
            role_color = {"user": "green", "assistant": "yellow", "system": "red"}.get(
                msg.role, "white"
            )
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            console.print(f"  [{role_color}]{msg.role}[/{role_color}]: {content}")

    def _process_input(self, user_input: str) -> None:
        console.print("\n[bold cyan]Thinking...[/bold cyan]")

        try:
            response = self.agent.chat(user_input, stream=self.stream)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")

    def run(self) -> None:
        self.session = self._setup_session()
        self._print_welcome()
        self._run_loop()

    def _run_loop(self) -> None:
        while self.running:
            try:
                if self.session is None:
                    break
                try:
                    user_input = self.session.prompt(
                        HTML("<ansigreen><b>You:</b></ansigreen> "),
                    )
                except Exception:
                    user_input = input("You: ")

                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    self.running = self._handle_command(user_input)
                    continue

                self._process_input(user_input)

            except KeyboardInterrupt:
                console.print("\n[yellow]Use /exit to quit[/yellow]")
                continue

            except EOFError:
                console.print("\n[yellow]Goodbye![/yellow]")
                break

            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                continue


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sarvam-OS: AI Co-Developer")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Project path (default: current directory)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output",
    )

    args = parser.parse_args()

    project_path = Path(args.path).resolve()

    if not project_path.exists():
        console.print(f"[red]Error: Path does not exist: {project_path}[/red]")
        sys.exit(1)

    os.environ.setdefault("SARVAM_API_KEY", os.environ.get("SARVAM_API_KEY", ""))
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    cli = SarvamCLI(
        project_path=project_path,
        stream=not args.no_stream,
    )
    cli.run()


if __name__ == "__main__":
    main()
