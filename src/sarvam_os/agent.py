"""
Agent: Core agent with reasoning loop and system prompt.
The "Brain" of Sarvam-OS.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional, Union

from openai import OpenAI
from rich.console import Console

from src.sarvam_os.memory import MemoryStore, Message
from src.sarvam_os.tools import (
    AgentDeps,
    TOOL_DEFINITIONS,
    ToolResult,
    execute_command,
    edit_file,
    get_tool_function,
    get_tools_description,
    git_commit,
    list_files,
    read_codebase,
    read_file,
)

console = Console()

SYSTEM_PROMPT = """You are Sarvam-OS, an autonomous AI coding agent that operates as a co-developer on the user's local machine.

## Your Identity
You are a persistent agent with memory across sessions. You can read, write, and execute commands on the user's behalf.

## Response Format
You MUST follow this Thought-Action-Observation loop:

[PLAN]
1. Analyze the task
2. Break it into steps
3. Identify required tools
[/PLAN]

[THOUGHT]
Your internal reasoning about the current step...
[/THOUGHT]

[ACTION]
tool_name({"param": "value"})
[/ACTION]

After each action, you will receive an [OBSERVATION] with the result. Use this to inform your next step.

## Available Tools
{tools}

## Autonomous Error Correction
When a command fails:
1. Analyze the error in [THOUGHT]
2. Attempt a fix automatically
3. Retry up to 3 times before asking the user
4. Always explain what went wrong and how you fixed it

## Best Practices
1. Always scan the codebase before making changes
2. Use search_replace mode for targeted edits
3. Run tests after code changes
4. Commit meaningful changes with clear messages
5. Preserve existing code style and conventions
6. Never delete files without explicit permission

Be helpful, proactive, and thorough. You are not just a chatbot - you are a capable developer assistant."""


class SarvamAgent:
    def __init__(
        self,
        project_path: Union[str, os.PathLike],
        max_retries: int = 3,
        auto_git: bool = False,
    ):
        self.project_path = Path(project_path).resolve()
        self.memory = MemoryStore(project_path=self.project_path)
        self.max_retries = max_retries
        self.auto_git = auto_git

        self.deps = AgentDeps(
            project_path=self.project_path,
            max_retries=max_retries,
            auto_git=auto_git,
        )

        self._client = self._get_client()
        self._retry_count = 0

    def _get_client(self) -> OpenAI:
        base_url = os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai/v1")
        api_key = os.environ.get("SARVAM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return OpenAI(base_url=base_url, api_key=api_key)

    def _get_model(self) -> str:
        return os.environ.get("SARVAM_MODEL", "sarvam-m")

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(tools=get_tools_description()),
            }
        ]

        for msg in self.memory.get_context_window(max_tokens=6000):
            messages.append(msg)

        messages.append({"role": "user", "content": user_input})

        return messages

    def _parse_tool_call(self, text: str) -> Optional[tuple[str, dict[str, Any]]]:
        action_match = re.search(r"\[ACTION\]\s*(\w+)\s*\((.*?)\)\s*\[/ACTION\]", text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            params_str = action_match.group(2).strip()

            try:
                if params_str:
                    params = json.loads(params_str)
                else:
                    params = {}
                return tool_name, params
            except json.JSONDecodeError:
                params = {}
                if params_str:
                    for part in params_str.split(","):
                        if ":" in part:
                            key, value = part.split(":", 1)
                            params[key.strip().strip('"')] = value.strip().strip('"')
                return tool_name, params

        return None

    def _execute_tool(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        tool_func = get_tool_function(tool_name)
        if not tool_func:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}",
            )

        return tool_func(self.deps, **params)

    def chat(self, user_input: str, stream: bool = True) -> str:
        self.memory.add("user", user_input)

        messages = self._build_messages(user_input)
        model = self._get_model()

        try:
            if stream:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                response_text = ""
                console.print("\n[bold cyan]Sarvam:[/bold cyan] ", end="")

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        console.print(content, end="")
                        response_text += content

                console.print()
            else:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                response_text = response.choices[0].message.content or ""

            self.memory.add("assistant", response_text)

            tool_call = self._parse_tool_call(response_text)
            if tool_call:
                tool_name, params = tool_call
                result = self._execute_tool(tool_name, params)

                observation = f"\n[OBSERVATION]\nSuccess: {result.success}\n"
                if result.output:
                    observation += f"Output: {result.output[:1000]}\n"
                if result.error:
                    observation += f"Error: {result.error}\n"
                observation += "[/OBSERVATION]"

                if not result.success and self._retry_count < self.max_retries:
                    self._retry_count += 1
                    observation += f"\n[RETRY {self._retry_count}/{self.max_retries}]"
                else:
                    self._retry_count = 0

                self.memory.add("system", observation, metadata={"observation": True})

                if not result.success and self._retry_count > 0:
                    return self.chat("Continue with the fix", stream=stream)

            return response_text

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.memory.add("system", error_msg, metadata={"error": True})
            return error_msg

    def clear_memory(self) -> None:
        self.memory.clear()

    def get_memory_summary(self) -> dict[str, Any]:
        return self.memory.get_summary()

    def add_system_message(self, content: str) -> None:
        self.memory.add("system", content)


def create_agent(
    project_path: Optional[Union[str, os.PathLike]] = None,
    **kwargs,
) -> SarvamAgent:
    path = project_path or os.getcwd()
    return SarvamAgent(project_path=path, **kwargs)
