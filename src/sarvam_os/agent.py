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

SYSTEM_PROMPT = """You are Sarvam-OS, an AI coding agent. You are NOT a human. You are a tool-executing agent.

## CRITICAL RULES - NEVER VIOLATE
1. NEVER claim to have done something unless you have [OBSERVATION] output confirming success
2. NEVER make up file contents, command outputs, or results
3. NEVER say "I have created", "I have modified", "I have executed" without actual tool confirmation
4. If a tool fails, report the EXACT error from the [OBSERVATION]
5. Always VERIFY your actions by reading files back or checking outputs
6. Current date/time: {current_datetime}

## Response Format
Follow this EXACT format:

[THOUGHT]
What I need to do and which tool to use...
[/THOUGHT]

[ACTION]
tool_name(param="value")
[/ACTION]

STOP. Wait for [OBSERVATION] before continuing.

## Multi-File Tasks - IMPORTANT
When asked to create MULTIPLE files, you must:
1. Create ONE file at a time
2. Wait for [OBSERVATION] confirming success
3. Then create the NEXT file
4. Repeat until ALL files are created
5. ONLY report completion after ALL files are verified

Example for creating 3 files:
- Create file 1 -> Wait for observation -> Create file 2 -> Wait for observation -> Create file 3 -> Wait for observation -> Then report completion

## Available Tools
{tools}

## Tool Usage Rules
1. edit_file: Use mode="overwrite" for new files, mode="search_replace" for modifications
2. execute_command: Returns stdout/stderr - READ THE OUTPUT before claiming success
3. read_file: Always use to verify file contents after creating/modifying
4. After ANY file edit, run read_file to confirm the change was applied

## Example Correct Flow
[THOUGHT] I need to create a file test.py
[/THOUGHT]

[ACTION]
edit_file(file_path="test.py", content="print('hello')", mode="overwrite")
[/ACTION]

[OBSERVATION]
Success: True
Output: File written: test.py
[/OBSERVATION]

[THOUGHT] The file was created. Let me verify by reading it.
[/THOUGHT]

[ACTION]
read_file(file_path="test.py")
[/ACTION]

[OBSERVATION]
Success: True
Output: print('hello')
[/OBSERVATION]

The file test.py has been created and verified. It contains: print('hello')

## Example Error Handling
[OBSERVATION]
Success: False
Error: File not found
[/OBSERVATION]

Report: "The operation failed with error: File not found. I need to create the file first."

NEVER fabricate success. ONLY report what observations show."""


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

    def _get_datetime(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")

    def _build_messages(self, user_input: str) -> list[dict[str, str]]:
        current_dt = self._get_datetime()
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tools=get_tools_description(),
                    current_datetime=current_dt
                ),
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

    def _build_messages_from_history(self) -> list[dict[str, str]]:
        """Build messages from memory history without adding new user input."""
        current_dt = self._get_datetime()
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tools=get_tools_description(),
                    current_datetime=current_dt
                ),
            }
        ]

        history = self.memory.get_context_window(max_tokens=6000)
        
        # Filter and ensure proper alternation
        last_role = "system"
        for msg in history:
            role = msg["role"]
            if role == "system":
                continue
            if role == last_role:
                continue
            messages.append(msg)
            last_role = role

        return messages

    def _parse_tool_call(self, text: str) -> Optional[tuple[str, dict[str, Any]]]:
        action_match = re.search(r"\[ACTION\]\s*(\w+)\s*\((.*?)\)\s*\[/ACTION\]", text, re.DOTALL)
        if action_match:
            tool_name = action_match.group(1)
            params_str = action_match.group(2).strip()

            if not params_str:
                return tool_name, {}

            params = {}
            
            # More robust parsing that handles multi-line content
            # Match key= followed by either "value" or 'value' or unquoted value
            # Use a state machine approach for better quote handling
            i = 0
            while i < len(params_str):
                # Skip whitespace
                while i < len(params_str) and params_str[i] in ' \t\n':
                    i += 1
                if i >= len(params_str):
                    break
                    
                # Find key
                key_start = i
                while i < len(params_str) and params_str[i] not in '=, \t\n':
                    i += 1
                key = params_str[key_start:i].strip()
                
                if not key:
                    i += 1
                    continue
                    
                # Skip to =
                while i < len(params_str) and params_str[i] in ' \t\n':
                    i += 1
                if i >= len(params_str) or params_str[i] != '=':
                    i += 1
                    continue
                i += 1  # Skip =
                
                # Skip whitespace after =
                while i < len(params_str) and params_str[i] in ' \t\n':
                    i += 1
                    
                if i >= len(params_str):
                    break
                    
                # Parse value
                if params_str[i] == '"':
                    # Double quoted string
                    i += 1
                    value_start = i
                    value_chars = []
                    while i < len(params_str):
                        if params_str[i] == '\\' and i + 1 < len(params_str):
                            # Handle escape sequences
                            next_char = params_str[i + 1]
                            if next_char == 'n':
                                value_chars.append('\n')
                            elif next_char == 't':
                                value_chars.append('\t')
                            elif next_char == '"':
                                value_chars.append('"')
                            elif next_char == '\\':
                                value_chars.append('\\')
                            else:
                                value_chars.append(next_char)
                            i += 2
                        elif params_str[i] == '"':
                            break
                        else:
                            value_chars.append(params_str[i])
                            i += 1
                    params[key] = ''.join(value_chars)
                    i += 1  # Skip closing quote
                elif params_str[i] == "'":
                    # Single quoted string
                    i += 1
                    value_start = i
                    while i < len(params_str) and params_str[i] != "'":
                        i += 1
                    params[key] = params_str[value_start:i]
                    i += 1
                else:
                    # Unquoted value (until comma or end)
                    value_start = i
                    while i < len(params_str) and params_str[i] not in ',\n':
                        i += 1
                    params[key] = params_str[value_start:i].strip()
                
                # Skip comma if present
                while i < len(params_str) and params_str[i] in ' \t\n':
                    i += 1
                if i < len(params_str) and params_str[i] == ',':
                    i += 1

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

                observation = f"[OBSERVATION]\n"
                observation += f"Tool: {tool_name}\n"
                observation += f"Parameters: {params}\n"
                observation += f"Success: {result.success}\n"
                if result.output:
                    observation += f"--- OUTPUT START ---\n{result.output}\n--- OUTPUT END ---\n"
                if result.error:
                    observation += f"--- ERROR START ---\n{result.error}\n--- ERROR END ---\n"
                observation += "[/OBSERVATION]\n\n"
                
                if result.success:
                    observation += "IMPORTANT: Verify this action succeeded by using read_file or list_files before telling the user it is done."
                else:
                    observation += "The action FAILED. Report the exact error above. Do NOT claim success."

                # Add observation as user message for the next turn
                self.memory.add("user", observation)
                
                # Build messages for continuation
                messages = self._build_messages_from_history()
                
                # Continue the conversation (recursive to handle multiple tool calls)
                return self._continue_with_observation(model, messages)

            return response_text

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return error_msg

    def _continue_with_observation(self, model: str, messages: list[dict[str, str]], max_loops: int = 15) -> str:
        """Continue conversation after tool execution, handling multiple tool calls."""
        if max_loops <= 0:
            return "Maximum tool calls reached. Task may not be complete."

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

            observation = f"[OBSERVATION]\n"
            observation += f"Tool: {tool_name}\n"
            observation += f"Parameters: {params}\n"
            observation += f"Success: {result.success}\n"
            if result.output:
                observation += f"--- OUTPUT START ---\n{result.output}\n--- OUTPUT END ---\n"
            if result.error:
                observation += f"--- ERROR START ---\n{result.error}\n--- ERROR END ---\n"
            observation += "[/OBSERVATION]\n\n"
            
            if result.success:
                observation += "IMPORTANT: Verify this action succeeded by using read_file or list_files before telling the user it is done."
            else:
                observation += "The action FAILED. Report the exact error above. Do NOT claim success."

            self.memory.add("user", observation)
            messages = self._build_messages_from_history()
            return self._continue_with_observation(model, messages, max_loops - 1)

        return response_text

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
