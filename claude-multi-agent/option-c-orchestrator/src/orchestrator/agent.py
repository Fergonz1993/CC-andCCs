"""
Claude Code Agent - Manages a single Claude Code subprocess.

This class handles:
- Spawning Claude Code processes
- Sending prompts via stdin
- Receiving responses via stdout
- Managing process lifecycle
- Handling timeouts and errors

Note: This uses asyncio.create_subprocess_exec which is the safe way to spawn
processes without shell injection risks (equivalent to execFile in Node.js).
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Optional, Callable, AsyncIterator
from pathlib import Path

from .models import Agent, AgentRole, Task, TaskResult


class ClaudeCodeAgent:
    """
    Manages a single Claude Code subprocess.

    The agent communicates with Claude Code using the --print flag for
    non-interactive mode and --output-format json for structured output.
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole = AgentRole.WORKER,
        working_directory: str = ".",
        model: str = "claude-sonnet-4-20250514",
        on_output: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.working_directory = Path(working_directory).resolve()
        self.model = model
        self.on_output = on_output
        self.on_error = on_error

        # Process state
        self._process: Optional[asyncio.subprocess.Process] = None
        self._is_running = False
        self._current_task: Optional[str] = None
        self._stderr_task: Optional[asyncio.Task] = None

        # Metrics
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._start_time: Optional[datetime] = None

        # Output buffer for streaming
        self._output_buffer: list[str] = []

    @property
    def is_running(self) -> bool:
        return self._is_running and self._process is not None

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None

    def to_model(self) -> Agent:
        """Convert to Agent model for state tracking."""
        return Agent(
            id=self.agent_id,
            role=self.role,
            is_active=self.is_running,
            current_task=self._current_task,
            pid=self.pid,
            last_heartbeat=datetime.now(),
            working_directory=str(self.working_directory),
            model=self.model,
        )

    async def start(self) -> bool:
        """
        Start the Claude Code subprocess in conversation mode.

        Uses create_subprocess_exec (not shell) for security - arguments
        are passed directly without shell interpretation.

        Returns True if started successfully.
        """
        if self.is_running:
            return True

        try:
            # Build command arguments (no shell interpretation)
            cmd = ["claude"]
            args = [
                "--print",  # Non-interactive, print output
                "--output-format", "stream-json",  # JSON output for parsing
                "--model", self.model,
                "--verbose",  # More detailed output
            ]

            # Start process using exec (not shell) for safety
            self._process = await asyncio.create_subprocess_exec(
                cmd[0],
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_directory,
                env={**os.environ, "CLAUDE_CODE_AGENT_ID": self.agent_id},
            )

            self._is_running = True
            self._start_time = datetime.now()

            # Start stderr reader and store reference to prevent GC
            self._stderr_task = asyncio.create_task(self._read_stderr())

            return True

        except FileNotFoundError:
            if self.on_error:
                self.on_error(
                    "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
                )
            return False
        except Exception as e:
            if self.on_error:
                self.on_error(f"Failed to start Claude Code: {e}")
            return False

    async def stop(self) -> None:
        """Stop the Claude Code subprocess gracefully."""
        if not self._process:
            return

        self._is_running = False

        # Cancel stderr reader task
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None

        try:
            # Try graceful shutdown first
            if self._process.stdin:
                self._process.stdin.close()
                await self._process.stdin.wait_closed()

            # Wait briefly for natural exit
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                # Force kill if it doesn't exit gracefully
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()

        except Exception as e:
            if self.on_error:
                self.on_error(f"Error stopping agent {self.agent_id}: {e}")

        self._process = None

    async def send_prompt(
        self,
        prompt: str,
        timeout: float = 600.0,  # 10 minutes default
    ) -> str:
        """
        Send a prompt to Claude Code and wait for the complete response.

        Args:
            prompt: The prompt to send
            timeout: Maximum time to wait for response

        Returns:
            The complete response text
        """
        if not self.is_running:
            raise RuntimeError(f"Agent {self.agent_id} is not running")

        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError(f"Agent {self.agent_id} has no valid streams")

        self._output_buffer.clear()

        try:
            # Send prompt with newline
            prompt_bytes = (prompt + "\n").encode("utf-8")
            self._process.stdin.write(prompt_bytes)
            await self._process.stdin.drain()

            # Read response with timeout
            response = await asyncio.wait_for(
                self._read_response(),
                timeout=timeout
            )

            return response

        except asyncio.TimeoutError:
            if self.on_error:
                self.on_error(f"Agent {self.agent_id} timed out after {timeout}s")
            raise

    async def send_prompt_streaming(
        self,
        prompt: str,
        timeout: float = 600.0,
    ) -> AsyncIterator[str]:
        """
        Send a prompt and yield response chunks as they arrive.
        """
        if not self.is_running:
            raise RuntimeError(f"Agent {self.agent_id} is not running")

        if not self._process or not self._process.stdin or not self._process.stdout:
            raise RuntimeError(f"Agent {self.agent_id} has no valid streams")

        try:
            # Send prompt
            prompt_bytes = (prompt + "\n").encode("utf-8")
            self._process.stdin.write(prompt_bytes)
            await self._process.stdin.drain()

            # Stream response
            async for chunk in self._stream_response(timeout):
                yield chunk

        except asyncio.TimeoutError:
            if self.on_error:
                self.on_error(f"Agent {self.agent_id} timed out")
            raise

    async def execute_task(self, task: Task) -> TaskResult:
        """
        Execute a task and return the result.

        This method constructs an appropriate prompt from the task
        and handles the response parsing.
        """
        self._current_task = task.id

        # Build task prompt
        prompt = self._build_task_prompt(task)

        try:
            # Send prompt and get response
            response = await self.send_prompt(prompt)

            # Parse response into result
            result = self._parse_task_response(response, task)

            self._tasks_completed += 1
            return result

        except asyncio.TimeoutError:
            self._tasks_failed += 1
            return TaskResult(
                output="",
                error=f"Task timed out after {task.context.metadata.get('timeout', 600)}s"
            )
        except Exception as e:
            self._tasks_failed += 1
            return TaskResult(output="", error=str(e))
        finally:
            self._current_task = None

    def _build_task_prompt(self, task: Task) -> str:
        """Build a prompt from a task."""
        parts = [
            f"## Task: {task.id}",
            f"**Description**: {task.description}",
        ]

        if task.context.files:
            parts.append(f"**Relevant files**: {', '.join(task.context.files)}")

        if task.context.hints:
            parts.append(f"**Hints**: {task.context.hints}")

        parts.extend([
            "",
            "Please complete this task. When done, summarize what you did.",
            "List any files you modified or created.",
            "Note any important discoveries that other team members should know about.",
        ])

        return "\n".join(parts)

    def _parse_task_response(self, response: str, task: Task) -> TaskResult:
        """Parse Claude's response into a TaskResult."""
        # Try to extract structured information from the response
        files_modified: list[str] = []
        files_created: list[str] = []
        discoveries: list[str] = []

        # Simple heuristic parsing (could be improved with more structure)
        lines = response.split("\n")
        current_section: Optional[str] = None

        for line in lines:
            line_lower = line.lower().strip()

            if "modified" in line_lower or "changed" in line_lower:
                current_section = "modified"
            elif "created" in line_lower or "new file" in line_lower:
                current_section = "created"
            elif "discover" in line_lower or "found" in line_lower or "note:" in line_lower:
                current_section = "discoveries"
            elif line.strip().startswith("- ") or line.strip().startswith("* "):
                item = line.strip()[2:].strip()
                if current_section == "modified":
                    files_modified.append(item)
                elif current_section == "created":
                    files_created.append(item)
                elif current_section == "discoveries":
                    discoveries.append(item)

        return TaskResult(
            output=response,
            files_modified=files_modified,
            files_created=files_created,
            discoveries=discoveries,
        )

    async def _read_response(self) -> str:
        """Read the complete response from stdout."""
        if not self._process or not self._process.stdout:
            return ""

        response_parts: list[str] = []

        while True:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    break

                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue

                # Try to parse as JSON (stream-json format)
                try:
                    data = json.loads(decoded)
                    if data.get("type") == "assistant":
                        # Extract text content
                        message = data.get("message", {})
                        content = message.get("content", [])
                        for block in content:
                            if block.get("type") == "text":
                                response_parts.append(block.get("text", ""))
                    elif data.get("type") == "result":
                        # End of response
                        break
                except json.JSONDecodeError:
                    # Plain text output
                    response_parts.append(decoded)

                if self.on_output:
                    self.on_output(decoded)

            except Exception:
                break

        return "\n".join(response_parts)

    async def _stream_response(self, timeout: float) -> AsyncIterator[str]:
        """Stream response chunks."""
        if not self._process or not self._process.stdout:
            return

        loop = asyncio.get_running_loop()
        start_time = loop.time()

        while True:
            elapsed = loop.time() - start_time
            remaining = timeout - elapsed

            if remaining <= 0:
                raise asyncio.TimeoutError()

            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=min(30.0, remaining)
                )

                if not line:
                    break

                decoded = line.decode("utf-8").strip()
                if not decoded:
                    continue

                try:
                    data = json.loads(decoded)
                    if data.get("type") == "assistant":
                        for block in data.get("message", {}).get("content", []):
                            if block.get("type") == "text":
                                yield block.get("text", "")
                    elif data.get("type") == "result":
                        break
                except json.JSONDecodeError:
                    yield decoded

            except asyncio.TimeoutError:
                # Check if overall timeout has expired
                if loop.time() - start_time >= timeout:
                    raise
                # Otherwise continue waiting for more data
                continue

    async def _read_stderr(self) -> None:
        """Background task to read stderr."""
        if not self._process or not self._process.stderr:
            return

        while self._is_running:
            try:
                line = await self._process.stderr.readline()
                if not line:
                    break

                decoded = line.decode("utf-8").strip()
                if decoded and self.on_error:
                    self.on_error(f"[{self.agent_id}] {decoded}")

            except Exception:
                break


class AgentPool:
    """
    Manages a pool of Claude Code agents.

    Handles agent lifecycle, task distribution, and load balancing.
    """

    def __init__(
        self,
        working_directory: str = ".",
        max_workers: int = 3,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.working_directory = working_directory
        self.max_workers = max_workers
        self.model = model

        self._agents: dict[str, ClaudeCodeAgent] = {}
        self._leader: Optional[ClaudeCodeAgent] = None

    async def start_leader(self) -> ClaudeCodeAgent:
        """Start the leader agent."""
        if self._leader and self._leader.is_running:
            return self._leader

        self._leader = ClaudeCodeAgent(
            agent_id="leader",
            role=AgentRole.LEADER,
            working_directory=self.working_directory,
            model=self.model,
        )

        await self._leader.start()
        return self._leader

    async def start_worker(self, worker_id: Optional[str] = None) -> ClaudeCodeAgent:
        """Start a new worker agent."""
        if len(self._agents) >= self.max_workers:
            raise RuntimeError(f"Maximum workers ({self.max_workers}) reached")

        if worker_id is None:
            worker_id = f"worker-{len(self._agents) + 1}"

        agent = ClaudeCodeAgent(
            agent_id=worker_id,
            role=AgentRole.WORKER,
            working_directory=self.working_directory,
            model=self.model,
        )

        await agent.start()
        self._agents[worker_id] = agent

        return agent

    async def stop_all(self) -> None:
        """Stop all agents."""
        tasks = []

        if self._leader:
            tasks.append(self._leader.stop())

        for agent in self._agents.values():
            tasks.append(agent.stop())

        await asyncio.gather(*tasks, return_exceptions=True)

        self._leader = None
        self._agents.clear()

    def get_idle_worker(self) -> Optional[ClaudeCodeAgent]:
        """Get an idle worker agent."""
        for agent in self._agents.values():
            if agent.is_running and agent._current_task is None:
                return agent
        return None

    def get_all_agents(self) -> list[ClaudeCodeAgent]:
        """Get all agents including leader."""
        agents = list(self._agents.values())
        if self._leader:
            agents.insert(0, self._leader)
        return agents
