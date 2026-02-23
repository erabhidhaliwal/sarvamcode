"""
Agentic Tools: Tools for Sarvam-OS agent.
These are the "Hands" of the agent - read, write, execute.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

from src.agent import ProjectMapper


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDeps:
    project_path: Path
    max_retries: int = 3
    auto_git: bool = False
    safe_mode: bool = True


def read_codebase(
    deps: AgentDeps,
    max_depth: int = 10,
    include_patterns: Optional[list[str]] = None,
    include_contents: bool = False,
) -> ToolResult:
    try:
        mapper = ProjectMapper(deps.project_path)
        tree = mapper.to_markdown(max_depth=max_depth)

        files = {}
        if include_contents:
            patterns = include_patterns or ["*.py", "*.md", "*.toml", "*.yaml", "*.yml", "*.json"]
            files = mapper.get_file_contents(patterns)

        result_data = {
            "tree": tree,
            "files": files,
            "summary": {
                "project_name": deps.project_path.name,
                "total_files": len(files),
            },
        }

        return ToolResult(
            success=True,
            output=f"Scanned project: {deps.project_path.name}\n{tree}",
            data=result_data,
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def read_file(
    deps: AgentDeps,
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> ToolResult:
    full_path = deps.project_path / file_path

    if not full_path.exists():
        return ToolResult(
            success=False,
            output="",
            error=f"File not found: {file_path}",
        )

    try:
        content = full_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        if start_line is not None or end_line is not None:
            start = start_line or 0
            end = end_line or len(lines)
            content = "\n".join(lines[start:end])

        return ToolResult(
            success=True,
            output=content,
            data={"lines": len(lines), "file_path": file_path},
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def list_files(
    deps: AgentDeps,
    pattern: str = "**/*",
    exclude_dirs: Optional[list[str]] = None,
) -> ToolResult:
    exclude = exclude_dirs or ["__pycache__", ".git", "node_modules", ".venv", ".sarvam"]
    files = []

    for f in deps.project_path.glob(pattern):
        if f.is_file():
            rel_path = str(f.relative_to(deps.project_path))
            skip = any(exc in rel_path for exc in exclude)
            if not skip:
                files.append(rel_path)

    return ToolResult(
        success=True,
        output=f"Found {len(files)} files:\n" + "\n".join(sorted(files)[:50]),
        data={"files": sorted(files), "count": len(files)},
    )


def edit_file(
    deps: AgentDeps,
    file_path: str,
    content: str,
    mode: str = "overwrite",
    search: Optional[str] = None,
    replace: Optional[str] = None,
    create_dirs: bool = True,
) -> ToolResult:
    full_path = deps.project_path / file_path

    if mode == "overwrite":
        if create_dirs:
            full_path.parent.mkdir(parents=True, exist_ok=True)

        full_path.write_text(content, encoding="utf-8")
        return ToolResult(
            success=True,
            output=f"File written: {file_path}",
            data={"file_path": file_path, "lines": content.count("\n") + 1},
        )

    elif mode == "append":
        if not full_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File does not exist: {file_path}",
            )

        with open(full_path, "a", encoding="utf-8") as f:
            f.write("\n" + content)

        return ToolResult(
            success=True,
            output=f"Content appended to: {file_path}",
            data={"file_path": file_path},
        )

    elif mode == "search_replace":
        if not full_path.exists():
            return ToolResult(
                success=False,
                output="",
                error=f"File does not exist: {file_path}",
            )

        if not search:
            return ToolResult(
                success=False,
                output="",
                error="search_replace mode requires 'search' parameter",
            )

        original = full_path.read_text(encoding="utf-8")

        if search not in original:
            return ToolResult(
                success=False,
                output="",
                error=f"Search text not found in: {file_path}",
            )

        new_content = original.replace(search, replace or content)
        full_path.write_text(new_content, encoding="utf-8")

        return ToolResult(
            success=True,
            output=f"Text replaced in: {file_path}",
            data={"file_path": file_path},
        )

    return ToolResult(
        success=False,
        output="",
        error=f"Unknown mode: {mode}",
    )


def execute_command(
    deps: AgentDeps,
    command: str,
    timeout: int = 60,
) -> ToolResult:
    dangerous_commands = ["rm -rf /", "sudo rm", "chmod 777 /", "> /dev/sda"]
    if deps.safe_mode:
        for dangerous in dangerous_commands:
            if dangerous in command:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Blocked dangerous command: {dangerous}",
                )

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=deps.project_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout or ""
        error = result.stderr or ""

        return ToolResult(
            success=result.returncode == 0,
            output=output,
            error=error if result.returncode != 0 else "",
            data={
                "return_code": result.returncode,
                "command": command,
            },
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            success=False,
            output="",
            error=f"Command timed out after {timeout} seconds",
        )
    except Exception as e:
        return ToolResult(success=False, output="", error=str(e))


def git_commit(
    deps: AgentDeps,
    message: str,
    add_all: bool = True,
) -> ToolResult:
    try:
        if add_all:
            subprocess.run(
                ["git", "add", "-A"],
                cwd=deps.project_path,
                capture_output=True,
                check=True,
            )

        result = subprocess.run(
            ["git", "commit", "-m", message],
            cwd=deps.project_path,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            if "nothing to commit" in result.stdout:
                return ToolResult(
                    success=True,
                    output="Nothing to commit",
                )
            return ToolResult(
                success=False,
                output="",
                error=result.stderr or result.stdout,
            )

        push_result = subprocess.run(
            ["git", "push"],
            cwd=deps.project_path,
            capture_output=True,
            text=True,
        )

        return ToolResult(
            success=True,
            output="Committed and pushed successfully",
            data={"commit_message": message},
        )
    except subprocess.CalledProcessError as e:
        return ToolResult(success=False, output="", error=str(e))
    except FileNotFoundError:
        return ToolResult(success=False, output="", error="Git not found")


TOOL_DEFINITIONS = {
    "read_codebase": {
        "function": read_codebase,
        "description": "Scan project directory structure and optionally read file contents",
        "parameters": {
            "max_depth": {"type": "integer", "default": 10, "description": "Maximum depth to scan"},
            "include_contents": {"type": "boolean", "default": False, "description": "Include file contents"},
        },
    },
    "read_file": {
        "function": read_file,
        "description": "Read a specific file's contents",
        "parameters": {
            "file_path": {"type": "string", "required": True, "description": "Path to file"},
            "start_line": {"type": "integer", "description": "Start line (0-indexed)"},
            "end_line": {"type": "integer", "description": "End line"},
        },
    },
    "list_files": {
        "function": list_files,
        "description": "List files matching a pattern",
        "parameters": {
            "pattern": {"type": "string", "default": "**/*", "description": "Glob pattern"},
        },
    },
    "edit_file": {
        "function": edit_file,
        "description": "Create, overwrite, append, or search-replace content in a file",
        "parameters": {
            "file_path": {"type": "string", "required": True, "description": "Path to file"},
            "content": {"type": "string", "required": True, "description": "Content to write"},
            "mode": {"type": "string", "default": "overwrite", "description": "overwrite, append, or search_replace"},
            "search": {"type": "string", "description": "Text to search (search_replace mode)"},
            "replace": {"type": "string", "description": "Text to replace with"},
        },
    },
    "execute_command": {
        "function": execute_command,
        "description": "Execute a shell command safely",
        "parameters": {
            "command": {"type": "string", "required": True, "description": "Command to execute"},
            "timeout": {"type": "integer", "default": 60, "description": "Timeout in seconds"},
        },
    },
    "git_commit": {
        "function": git_commit,
        "description": "Commit and push changes to git",
        "parameters": {
            "message": {"type": "string", "required": True, "description": "Commit message"},
            "add_all": {"type": "boolean", "default": True, "description": "Stage all changes"},
        },
    },
}


def get_tool_function(name: str) -> Optional[Callable]:
    if name in TOOL_DEFINITIONS:
        return TOOL_DEFINITIONS[name]["function"]
    return None


def get_tools_description() -> str:
    descriptions = []
    for name, tool in TOOL_DEFINITIONS.items():
        params = ", ".join(
            f"{p}: {d.get('type', 'any')}"
            for p, d in tool["parameters"].items()
        )
        descriptions.append(f"- {name}({params}): {tool['description']}")
    return "\n".join(descriptions)
