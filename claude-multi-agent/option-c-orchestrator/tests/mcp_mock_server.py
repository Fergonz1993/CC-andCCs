"""
Mock server for MCP (Model Context Protocol) testing.

Feature: adv-test-016 - Mock server for MCP testing

This module provides a mock MCP server for testing the orchestrator's
interaction with MCP-based coordination systems (Option B).

The mock server simulates all MCP tool calls without requiring
actual Claude Code instances.

Usage:
    from tests.mcp_mock_server import MockMCPServer, MCPClient

    # Start mock server
    server = MockMCPServer()
    server.start()

    # Use mock client
    async with MCPClient(server.url) as client:
        result = await client.call_tool("create_task", {...})

    server.stop()
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Awaitable
from enum import Enum
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import queue


class MCPToolType(str, Enum):
    """Available MCP tools in the coordination system."""
    CREATE_TASK = "create_task"
    CLAIM_TASK = "claim_task"
    COMPLETE_TASK = "complete_task"
    FAIL_TASK = "fail_task"
    GET_TASKS = "get_tasks"
    GET_DISCOVERIES = "get_discoveries"
    ADD_DISCOVERY = "add_discovery"
    REGISTER_AGENT = "register_agent"
    GET_STATUS = "get_status"


@dataclass
class MCPToolCall:
    """Record of an MCP tool call."""
    tool: str
    params: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class MockTask:
    """Simple task representation for the mock server."""
    id: str
    description: str
    status: str = "available"
    priority: int = 5
    claimed_by: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MockDiscovery:
    """Simple discovery representation for the mock server."""
    id: str
    agent_id: str
    content: str
    tags: List[str] = field(default_factory=list)
    related_task: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MockAgent:
    """Simple agent representation for the mock server."""
    id: str
    role: str = "worker"
    is_active: bool = True
    current_task: Optional[str] = None
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())


class MockMCPState:
    """
    In-memory state for the mock MCP server.

    Tracks tasks, discoveries, and agents.
    """

    def __init__(self):
        self.tasks: Dict[str, MockTask] = {}
        self.discoveries: Dict[str, MockDiscovery] = {}
        self.agents: Dict[str, MockAgent] = {}
        self.goal: str = ""
        self.master_plan: str = ""
        self._lock = threading.Lock()

    def reset(self):
        """Reset all state."""
        with self._lock:
            self.tasks.clear()
            self.discoveries.clear()
            self.agents.clear()
            self.goal = ""
            self.master_plan = ""

    def add_task(self, task: MockTask) -> None:
        with self._lock:
            self.tasks[task.id] = task

    def get_task(self, task_id: str) -> Optional[MockTask]:
        with self._lock:
            return self.tasks.get(task_id)

    def get_available_tasks(self) -> List[MockTask]:
        with self._lock:
            done_ids = {t.id for t in self.tasks.values() if t.status == "done"}
            return [
                t for t in self.tasks.values()
                if t.status == "available"
                and all(dep in done_ids for dep in t.dependencies)
            ]


class MockMCPToolHandler:
    """
    Handles MCP tool calls for the mock server.

    Each method corresponds to an MCP tool and returns
    a result or raises an exception.
    """

    def __init__(self, state: MockMCPState):
        self.state = state
        self.call_history: List[MCPToolCall] = []

    def record_call(self, tool: str, params: Dict[str, Any]) -> MCPToolCall:
        """Record a tool call in history."""
        call = MCPToolCall(tool=tool, params=params)
        self.call_history.append(call)
        return call

    def handle_create_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_task tool call."""
        call = self.record_call("create_task", params)

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        task = MockTask(
            id=task_id,
            description=params.get("description", ""),
            priority=params.get("priority", 5),
            dependencies=params.get("dependencies", []),
        )

        self.state.add_task(task)

        result = {"task_id": task_id, "status": "created"}
        call.result = result
        return result

    def handle_claim_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle claim_task tool call."""
        call = self.record_call("claim_task", params)

        agent_id = params.get("agent_id")
        task_id = params.get("task_id")

        if task_id:
            task = self.state.get_task(task_id)
        else:
            # Auto-claim highest priority available task
            available = self.state.get_available_tasks()
            if not available:
                call.error = "No available tasks"
                return {"error": "No available tasks"}
            available.sort(key=lambda t: t.priority)
            task = available[0]

        if not task:
            call.error = "Task not found"
            return {"error": "Task not found"}

        if task.status != "available":
            call.error = f"Task is {task.status}, not available"
            return {"error": f"Task is {task.status}"}

        task.status = "claimed"
        task.claimed_by = agent_id

        result = {
            "task_id": task.id,
            "description": task.description,
            "status": "claimed",
        }
        call.result = result
        return result

    def handle_complete_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle complete_task tool call."""
        call = self.record_call("complete_task", params)

        task_id = params.get("task_id")
        task = self.state.get_task(task_id)

        if not task:
            call.error = "Task not found"
            return {"error": "Task not found"}

        task.status = "done"
        task.result = {
            "output": params.get("output", ""),
            "files_modified": params.get("files_modified", []),
            "discoveries": params.get("discoveries", []),
        }

        # Create discoveries
        for disc_content in task.result.get("discoveries", []):
            disc = MockDiscovery(
                id=f"disc-{uuid.uuid4().hex[:8]}",
                agent_id=task.claimed_by or "unknown",
                content=disc_content,
                related_task=task_id,
            )
            self.state.discoveries[disc.id] = disc

        result = {"task_id": task_id, "status": "done"}
        call.result = result
        return result

    def handle_fail_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle fail_task tool call."""
        call = self.record_call("fail_task", params)

        task_id = params.get("task_id")
        error = params.get("error", "Unknown error")
        task = self.state.get_task(task_id)

        if not task:
            call.error = "Task not found"
            return {"error": "Task not found"}

        task.status = "failed"
        task.result = {"error": error}

        result = {"task_id": task_id, "status": "failed", "error": error}
        call.result = result
        return result

    def handle_get_tasks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_tasks tool call."""
        call = self.record_call("get_tasks", params)

        status_filter = params.get("status")
        tasks_list = []

        for task in self.state.tasks.values():
            if status_filter and task.status != status_filter:
                continue
            tasks_list.append({
                "id": task.id,
                "description": task.description,
                "status": task.status,
                "priority": task.priority,
                "claimed_by": task.claimed_by,
            })

        result = {"tasks": tasks_list}
        call.result = result
        return result

    def handle_get_discoveries(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_discoveries tool call."""
        call = self.record_call("get_discoveries", params)

        discoveries_list = [
            {
                "id": d.id,
                "agent_id": d.agent_id,
                "content": d.content,
                "tags": d.tags,
                "related_task": d.related_task,
            }
            for d in self.state.discoveries.values()
        ]

        result = {"discoveries": discoveries_list}
        call.result = result
        return result

    def handle_add_discovery(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle add_discovery tool call."""
        call = self.record_call("add_discovery", params)

        disc_id = f"disc-{uuid.uuid4().hex[:8]}"
        disc = MockDiscovery(
            id=disc_id,
            agent_id=params.get("agent_id", "unknown"),
            content=params.get("content", ""),
            tags=params.get("tags", []),
            related_task=params.get("related_task"),
        )

        self.state.discoveries[disc_id] = disc

        result = {"discovery_id": disc_id}
        call.result = result
        return result

    def handle_register_agent(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle register_agent tool call."""
        call = self.record_call("register_agent", params)

        agent_id = params.get("agent_id")
        role = params.get("role", "worker")

        agent = MockAgent(id=agent_id, role=role)
        self.state.agents[agent_id] = agent

        result = {"agent_id": agent_id, "status": "registered"}
        call.result = result
        return result

    def handle_get_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_status tool call."""
        call = self.record_call("get_status", params)

        total = len(self.state.tasks)
        by_status = {}
        for task in self.state.tasks.values():
            by_status[task.status] = by_status.get(task.status, 0) + 1

        done = by_status.get("done", 0)

        result = {
            "goal": self.state.goal,
            "total_tasks": total,
            "by_status": by_status,
            "percent_complete": round(done / total * 100, 1) if total > 0 else 0,
            "discoveries_count": len(self.state.discoveries),
            "agents_count": len(self.state.agents),
        }
        call.result = result
        return result

    def dispatch(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call to the appropriate handler."""
        handlers = {
            "create_task": self.handle_create_task,
            "claim_task": self.handle_claim_task,
            "complete_task": self.handle_complete_task,
            "fail_task": self.handle_fail_task,
            "get_tasks": self.handle_get_tasks,
            "get_discoveries": self.handle_get_discoveries,
            "add_discovery": self.handle_add_discovery,
            "register_agent": self.handle_register_agent,
            "get_status": self.handle_get_status,
        }

        handler = handlers.get(tool)
        if not handler:
            return {"error": f"Unknown tool: {tool}"}

        return handler(params)


class MCPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the mock MCP server."""

    def __init__(self, *args, handler: MockMCPToolHandler, **kwargs):
        self.tool_handler = handler
        super().__init__(*args, **kwargs)

    def do_POST(self):
        """Handle POST requests (tool calls)."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            request = json.loads(body)
            tool = request.get("tool", "")
            params = request.get("params", {})

            result = self.tool_handler.dispatch(tool, params)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())

    def do_GET(self):
        """Handle GET requests (status check)."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def log_message(self, format, *args):
        """Suppress logging."""
        pass


class MockMCPServer:
    """
    Mock MCP server for testing.

    Provides a full simulation of the MCP coordination server
    for integration testing.
    """

    def __init__(self, host: str = "localhost", port: int = 0):
        self.host = host
        self.port = port
        self.state = MockMCPState()
        self.handler = MockMCPToolHandler(self.state)
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def url(self) -> str:
        """Get the server URL."""
        if self._server:
            return f"http://{self.host}:{self._server.server_address[1]}"
        return ""

    def start(self) -> None:
        """Start the mock server in a background thread."""
        def create_handler(*args, **kwargs):
            return MCPRequestHandler(*args, handler=self.handler, **kwargs)

        self._server = HTTPServer((self.host, self.port), create_handler)
        self.port = self._server.server_address[1]
        self._running = True

        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def _run(self) -> None:
        """Run the server."""
        while self._running:
            self._server.handle_request()

    def stop(self) -> None:
        """Stop the mock server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def reset(self) -> None:
        """Reset server state."""
        self.state.reset()
        self.handler.call_history.clear()

    def get_call_history(self) -> List[MCPToolCall]:
        """Get history of all tool calls."""
        return self.handler.call_history

    def assert_tool_called(self, tool: str, times: int = 1) -> None:
        """Assert a tool was called a specific number of times."""
        calls = [c for c in self.handler.call_history if c.tool == tool]
        assert len(calls) == times, f"Expected {times} calls to {tool}, got {len(calls)}"

    def assert_tool_called_with(
        self,
        tool: str,
        params: Dict[str, Any],
    ) -> None:
        """Assert a tool was called with specific parameters."""
        for call in self.handler.call_history:
            if call.tool == tool:
                for key, value in params.items():
                    if call.params.get(key) == value:
                        return
        raise AssertionError(f"No call to {tool} with params {params}")


# =============================================================================
# Async MCP Client for Testing
# =============================================================================

class MCPClient:
    """
    Async client for interacting with the mock MCP server.

    Usage:
        async with MCPClient(server.url) as client:
            result = await client.call_tool("create_task", {...})
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._session = None

    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, *args):
        """Exit async context."""
        pass

    async def call_tool(
        self,
        tool: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call an MCP tool on the server.

        Args:
            tool: Tool name
            params: Tool parameters

        Returns:
            Tool result
        """
        import urllib.request

        request_data = json.dumps({
            "tool": tool,
            "params": params or {}
        }).encode()

        req = urllib.request.Request(
            self.base_url,
            data=request_data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req)
        )

        return json.loads(response.read().decode())

    async def create_task(
        self,
        description: str,
        priority: int = 5,
        dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a task."""
        return await self.call_tool("create_task", {
            "description": description,
            "priority": priority,
            "dependencies": dependencies or [],
        })

    async def claim_task(
        self,
        agent_id: str,
        task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Claim a task."""
        params = {"agent_id": agent_id}
        if task_id:
            params["task_id"] = task_id
        return await self.call_tool("claim_task", params)

    async def complete_task(
        self,
        task_id: str,
        output: str,
        files_modified: Optional[List[str]] = None,
        discoveries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Complete a task."""
        return await self.call_tool("complete_task", {
            "task_id": task_id,
            "output": output,
            "files_modified": files_modified or [],
            "discoveries": discoveries or [],
        })

    async def get_status(self) -> Dict[str, Any]:
        """Get coordination status."""
        return await self.call_tool("get_status", {})


# =============================================================================
# Pytest Fixtures
# =============================================================================

import pytest


@pytest.fixture
def mock_mcp_server():
    """Provide a mock MCP server for tests."""
    server = MockMCPServer()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def mcp_client(mock_mcp_server):
    """Provide an MCP client connected to the mock server."""
    return MCPClient(mock_mcp_server.url)


# =============================================================================
# Tests for the Mock Server
# =============================================================================

class TestMockMCPServer:
    """Tests for the mock MCP server itself."""

    def test_server_starts_and_stops(self):
        """Mock server can start and stop."""
        server = MockMCPServer()
        server.start()
        assert server.url.startswith("http://")
        server.stop()

    @pytest.mark.asyncio
    async def test_create_task(self, mock_mcp_server, mcp_client):
        """Mock server handles create_task."""
        result = await mcp_client.create_task("Test task")

        assert "task_id" in result
        assert result["status"] == "created"
        mock_mcp_server.assert_tool_called("create_task")

    @pytest.mark.asyncio
    async def test_claim_task(self, mock_mcp_server, mcp_client):
        """Mock server handles claim_task."""
        # Create task first
        create_result = await mcp_client.create_task("Task to claim")
        task_id = create_result["task_id"]

        # Claim it
        claim_result = await mcp_client.claim_task("worker-1", task_id)

        assert claim_result["task_id"] == task_id
        assert claim_result["status"] == "claimed"

    @pytest.mark.asyncio
    async def test_complete_task(self, mock_mcp_server, mcp_client):
        """Mock server handles complete_task."""
        # Create and claim
        create_result = await mcp_client.create_task("Task to complete")
        task_id = create_result["task_id"]
        await mcp_client.claim_task("worker-1", task_id)

        # Complete
        complete_result = await mcp_client.complete_task(
            task_id,
            "Successfully completed",
            discoveries=["Found something"]
        )

        assert complete_result["status"] == "done"

        # Verify discovery was created
        status = await mcp_client.get_status()
        assert status["discoveries_count"] == 1

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_mcp_server, mcp_client):
        """Mock server handles a full workflow."""
        # Create multiple tasks
        task1 = await mcp_client.create_task("Task 1", priority=1)
        task2 = await mcp_client.create_task("Task 2", priority=2)

        # Claim and complete task 1
        await mcp_client.claim_task("worker-1", task1["task_id"])
        await mcp_client.complete_task(task1["task_id"], "Done 1")

        # Claim and complete task 2
        await mcp_client.claim_task("worker-1", task2["task_id"])
        await mcp_client.complete_task(task2["task_id"], "Done 2")

        # Check status
        status = await mcp_client.get_status()
        assert status["total_tasks"] == 2
        assert status["percent_complete"] == 100.0

    def test_call_history_tracking(self, mock_mcp_server):
        """Mock server tracks call history."""
        # Make some calls through the handler directly
        mock_mcp_server.handler.handle_create_task({"description": "Test"})
        mock_mcp_server.handler.handle_get_status({})

        history = mock_mcp_server.get_call_history()
        assert len(history) == 2
        assert history[0].tool == "create_task"
        assert history[1].tool == "get_status"

    def test_state_reset(self, mock_mcp_server):
        """Mock server state can be reset."""
        mock_mcp_server.handler.handle_create_task({"description": "Test"})
        assert len(mock_mcp_server.state.tasks) == 1

        mock_mcp_server.reset()
        assert len(mock_mcp_server.state.tasks) == 0
        assert len(mock_mcp_server.get_call_history()) == 0
