"""
ProjectMapper: Scans directory structure and generates Markdown tree.
Agent: Manages conversation history and Sarvam-M API calls.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import pathspec
from rich.syntax import Syntax
from rich.text import Text

if TYPE_CHECKING:
    from rich.console import Console

DEFAULT_IGNORE_PATTERNS = [
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    ".npm/",
    "*.egg-info/",
    "dist/",
    "build/",
    ".tox/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "*.log",
    ".env",
    "*.env",
]


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}


@dataclass
class FileAction:
    action_type: str
    path: Path
    content: str = ""
    command: str = ""


class ProjectMapper:
    def __init__(self, root_path: Union[Path, str], console: Optional[Console] = None):
        self.root_path = Path(root_path).resolve()
        self.console = console
        self._ignore_patterns = DEFAULT_IGNORE_PATTERNS.copy()
        self._gitignore_spec: Optional[pathspec.PathSpec] = None
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        gitignore_path = self.root_path / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, encoding="utf-8") as f:
                gitignore_patterns = f.read().splitlines()
            self._gitignore_spec = pathspec.PathSpec.from_lines(
                "gitwildmatch",
                gitignore_patterns,
            )

    def _should_ignore(self, path: Path) -> bool:
        relative_path = path.relative_to(self.root_path)
        path_str = str(relative_path)
        if path.is_dir():
            path_str += "/"

        for pattern in self._ignore_patterns:
            if pattern.endswith("/") and path.is_dir():
                if relative_path.name == pattern.rstrip("/"):
                    return True
            elif path.match(pattern):
                return True

        if self._gitignore_spec:
            return self._gitignore_spec.match_file(path_str)

        return False

    def scan(self, max_depth: int = 10) -> str:
        return self._generate_tree(self.root_path, max_depth=max_depth)

    def _generate_tree(self, current_path: Path, prefix: str = "", depth: int = 0, max_depth: int = 10) -> str:
        if depth > max_depth:
            return ""

        entries = sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        entries = [e for e in entries if not self._should_ignore(e)]

        tree_lines = []
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            tree_lines.append(f"{prefix}{connector}{entry.name}")

            if entry.is_dir():
                extension = "    " if is_last else "│   "
                subtree = self._generate_tree(
                    entry, prefix + extension, depth + 1, max_depth
                )
                if subtree:
                    tree_lines.append(subtree)

        return "\n".join(tree_lines)

    def to_markdown(self, max_depth: int = 10) -> str:
        tree = self.scan(max_depth)
        return f"# Project Structure: {self.root_path.name}\n\n```\n{self.root_path.name}/\n{tree}\n```\n"

    def get_file_contents(self, file_patterns: list[str] | None = None) -> dict[str, str]:
        contents = {}
        patterns = file_patterns or ["*.py", "*.md", "*.toml", "*.yaml", "*.yml", "*.json"]

        for pattern in patterns:
            for file_path in self.root_path.rglob(pattern):
                if self._should_ignore(file_path):
                    continue
                try:
                    relative_path = file_path.relative_to(self.root_path)
                    contents[str(relative_path)] = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, PermissionError):
                    continue

        return contents


class Agent:
    SYSTEM_PROMPT = """You are Sarvam-Flow, an expert coding assistant. You help users edit their local project files.

You MUST respond using the following tag format:

[THOUGHT]
Your internal reasoning and analysis of the task.
[/THOUGHT]

[FILE_WRITE]
path/to/file.py
```python
# file content here
```
[/FILE_WRITE]

[SHELL_CMD]
command to execute
[/SHELL_CMD]

Rules:
1. Always start with [THOUGHT] to explain your reasoning
2. Use [FILE_WRITE] for creating or modifying files - include the relative path followed by a code block
3. Use [SHELL_CMD] for shell commands like pip install, pytest, etc.
4. Multiple [FILE_WRITE] and [SHELL_CMD] blocks are allowed
5. Be precise with file paths relative to the project root
6. Write complete, production-ready code
7. Follow existing project conventions and style"""

    def __init__(
        self,
        project_mapper: ProjectMapper,
        conversation_history: Optional[list[Message]] = None,
        history_file: Optional[Union[Path, str]] = None,
    ):
        self.project_mapper = project_mapper
        self.conversation_history = conversation_history or []
        self.history_file = Path(history_file) if history_file else None
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI

                base_url = os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai/v1")
                api_key = os.environ.get("SARVAM_API_KEY")
                self._client = OpenAI(base_url=base_url, api_key=api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai"
                )
        return self._client

    def _build_context(self, user_request: str, include_files: bool = False) -> str:
        context_parts = [
            self.project_mapper.to_markdown(),
            "\n## User Request\n",
            user_request,
        ]

        if include_files:
            relevant_files = self.project_mapper.get_file_contents()
            if relevant_files:
                context_parts.append("\n## Relevant File Contents\n")
                for path, content in relevant_files.items():
                    context_parts.append(f"\n### {path}\n```\n{content}\n```\n")

        return "\n".join(context_parts)

    def add_message(self, role: str, content: str) -> Message:
        message = Message(role=role, content=content)
        self.conversation_history.append(message)
        self._save_history()
        return message

    def _save_history(self) -> None:
        if self.history_file:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(
                    [m.to_dict() for m in self.conversation_history],
                    f,
                    indent=2,
                )

    def _load_history(self) -> None:
        if self.history_file and self.history_file.exists():
            with open(self.history_file, encoding="utf-8") as f:
                data = json.load(f)
                self.conversation_history = [
                    Message(**item) for item in data
                ]

    def chat(self, user_input: str, include_files: bool = False) -> str:
        context = self._build_context(user_input, include_files)

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        for msg in self.conversation_history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": context})

        client = self._get_client()

        response = client.chat.completions.create(
            model=os.environ.get("SARVAM_MODEL", "sarvam-m"),
            messages=messages,
        )

        assistant_message = response.choices[0].message.content or ""

        self.add_message("user", user_input)
        self.add_message("assistant", assistant_message)

        return assistant_message

    def clear_history(self) -> None:
        self.conversation_history = []
        if self.history_file and self.history_file.exists():
            self.history_file.unlink()
