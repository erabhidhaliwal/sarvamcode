"""
Sarvam-Flow CLI: A local AI coding agent powered by Sarvam-M.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.text import Text

from src.agent import Agent, ProjectMapper
from src.utils import FileParser, confirm_action, execute_shell_command, write_file

app = typer.Typer(
    name="sarvam-flow",
    help="A local AI coding agent CLI powered by Sarvam-M",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    prompt: Annotated[
        str,
        typer.Argument(
            help="The task or question for the AI agent",
        ),
    ],
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Path to the project directory",
            exists=False,
        ),
    ] = Path("."),
    include_files: Annotated[
        bool,
        typer.Option(
            "--include-files",
            "-i",
            help="Include file contents in context",
        ),
    ] = False,
    auto_approve: Annotated[
        bool,
        typer.Option(
            "--auto-approve",
            "-y",
            help="Auto-approve all actions without confirmation",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-d",
            help="Parse and display actions without executing",
        ),
    ] = False,
) -> None:
    _run_agent(
        prompt=prompt,
        project_path=path,
        include_files=include_files,
        auto_approve=auto_approve,
        dry_run=dry_run,
    )


@app.command()
def scan(
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Path to the project directory",
        ),
    ] = Path("."),
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output file for the project structure",
        ),
    ] = None,
) -> None:
    mapper = ProjectMapper(path, console)
    markdown = mapper.to_markdown()

    if output:
        output.write_text(markdown, encoding="utf-8")
        console.print(f"[green]✓[/green] Project structure written to {output}")
    else:
        console.print(Panel(markdown, title="Project Structure", border_style="cyan"))


@app.command()
def clear_history() -> None:
    history_file = Path(".conversation_history.json")
    if history_file.exists():
        history_file.unlink()
        console.print("[green]✓[/green] Conversation history cleared")
    else:
        console.print("[yellow]No conversation history found[/yellow]")


def _run_agent(
    prompt: str,
    project_path: Path,
    include_files: bool,
    auto_approve: bool,
    dry_run: bool,
) -> None:
    project_path = project_path.resolve()
    history_file = project_path / ".conversation_history.json"

    mapper = ProjectMapper(project_path, console)
    agent = Agent(
        project_mapper=mapper,
        history_file=history_file,
    )
    parser = FileParser(console)

    console.print(
        Panel(
            Text(f"Project: {project_path.name}", style="bold cyan"),
            title="Sarvam-Flow",
            border_style="cyan",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Thinking...", total=None)
        response = agent.chat(prompt, include_files=include_files)

    actions = parser.parse(response)

    if not actions:
        console.print(Panel(response, title="Response", border_style="green"))
        return

    parser.display_parsed(actions)

    if dry_run:
        console.print("\n[yellow]Dry run mode - no actions executed[/yellow]")
        return

    for action in actions:
        if action.action_type == "file_write":
            if auto_approve or confirm_action(action, console):
                write_file(
                    action.file_path,
                    action.content,
                    root_path=project_path,
                    console=console,
                )

        elif action.action_type == "shell_cmd":
            if auto_approve or confirm_action(action, console):
                execute_shell_command(
                    action.command,
                    cwd=project_path,
                    console=console,
                )


@app.command()
def chat(
    prompt: Annotated[
        str,
        typer.Argument(
            help="Message to send to the agent",
        ),
    ],
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Path to the project directory",
        ),
    ] = Path("."),
) -> None:
    project_path = path.resolve()
    history_file = project_path / ".conversation_history.json"

    mapper = ProjectMapper(project_path, console)
    agent = Agent(
        project_mapper=mapper,
        history_file=history_file,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Thinking...", total=None)
        response = agent.chat(prompt, include_files=False)

    console.print(Panel(response, title="Response", border_style="green"))


@app.command()
def interactive(
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Path to the project directory",
        ),
    ] = Path("."),
) -> None:
    project_path = path.resolve()
    history_file = project_path / ".conversation_history.json"

    mapper = ProjectMapper(project_path, console)
    agent = Agent(
        project_mapper=mapper,
        history_file=history_file,
    )
    parser = FileParser(console)

    console.print(
        Panel(
            Text("Type your requests. Use 'exit' or 'quit' to stop.", style="cyan"),
            title="Sarvam-Flow Interactive Mode",
            border_style="cyan",
        )
    )

    while True:
        try:
            user_input = console.input("\n[bold green]You:[/bold green] ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() == "clear":
                agent.clear_history()
                console.print("[green]History cleared[/green]")
                continue

            if user_input.lower() == "scan":
                console.print(mapper.to_markdown())
                continue

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Thinking...", total=None)
                response = agent.chat(user_input, include_files=False)

            actions = parser.parse(response)

            if actions:
                parser.display_parsed(actions)

                for action in actions:
                    if action.action_type == "file_write":
                        if confirm_action(action, console):
                            write_file(
                                action.file_path,
                                action.content,
                                root_path=project_path,
                                console=console,
                            )

                    elif action.action_type == "shell_cmd":
                        if confirm_action(action, console):
                            execute_shell_command(
                                action.command,
                                cwd=project_path,
                                console=console,
                            )
            else:
                console.print(Panel(response, title="Response", border_style="green"))

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break


if __name__ == "__main__":
    app()
