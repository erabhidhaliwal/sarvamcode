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
tool_name(file_path="path/to/file")
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

        history = self.memory.get_context_window(max_tokens=6000)
        
        # Filter and ensure proper alternation
        last_role = "system"
        for msg in history:
            role = msg["role"]
            # Skip system messages in history (they're for observations only)
            if role == "system":
                continue
            # Ensure alternation
            if role == last_role:
                continue
            messages.append(msg)
            last_role = role

        # Always add user message at the end
        messages.append({"role": "user", "content": user_input})

        return messages

    def _parse_tool_call(self, text: str) -> Optional[tuple[str, dict[str, Any]]]:
        action_match = re.search(r"\[ACTION\]\s*(\w+)\s*\((.*?)\)\s*\[/ACTION\]", text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            params_str = action_match.group(2).strip()

            if not params_str:
                return tool_name, {}

            try:
                params = json.loads(params_str)
                return tool_name, params
            except json.JSONDecodeError:
                params = {}
                in_string = False
                current_key = ""
                current_value = ""
                buffer = ""
                
                for char in params_str:
                    if char == '"' and (not buffer or buffer[-1] != '\\'):
                        in_string = not in_string
                        buffer += char
                    elif char == ':' and not in_string:
                        current_key = buffer.strip().strip('"')
                        buffer = ""
                    elif char == ',' and not in_string:
                        current_value = buffer.strip().strip('"')
                        if current_key:
                            params[current_key] = current_value
                        current_key = ""
                        current_value = ""
                        buffer = ""
                    else:
                        buffer += char
                
                if current_key and buffer:
                    current_value = buffer.strip().strip('"')
                    params[current_key] = current_value

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
        messages = self._build_messages(user_input)
        model = self._get_model()

        # Add user message to memory AFTER building messages
        self.memory.add("user", user_input)

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
                    # Add as assistant message to maintain alternation
                    self.memory.add("assistant", observation, metadata={"observation": True})
                    return self.chat("Continue with the fix", stream=stream)
                else:
                    self._retry_count = 0
                    self.memory.add("assistant", observation, metadata={"observation": True})

            return response_text

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            # Don't add to memory - just return the error
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
