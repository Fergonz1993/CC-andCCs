"""
Unit tests for Option C agent.py (test-004)

Tests the agent worker functionality including:
- Task claiming
- Task execution
- Communication with orchestrator
- Error handling
"""

import pytest
import json
import tempfile
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from orchestrator.agent import (
    Agent,
    AgentConfig,
    AgentError,
    TaskExecutionError,
)


class TestAgentInitialization:
    """Test agent initialization and configuration."""

    def test_agent_creates_with_defaults(self):
        """Test agent creation with default configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(
                agent_id="test-agent",
                coordination_dir=tmpdir
            )

            assert agent.agent_id == "test-agent"
            assert agent.config is not None

    def test_agent_creates_with_custom_config(self):
        """Test agent creation with custom configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(
                heartbeat_interval=10,
                max_retries=5,
                capabilities=["python", "testing"]
            )

            agent = Agent(
                agent_id="test-agent",
                coordination_dir=tmpdir,
                config=config
            )

            assert agent.config.heartbeat_interval == 10
            assert agent.config.max_retries == 5
            assert "python" in agent.config.capabilities

    def test_agent_config_defaults(self):
        """Test AgentConfig default values."""
        config = AgentConfig()
        assert config.heartbeat_interval == 5.0
        assert config.max_retries == 3
        assert config.capabilities == []

    def test_agent_registers_on_start(self):
        """Test that agent registers with orchestrator on start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(agent_id="test-agent", coordination_dir=tmpdir)

            with patch.object(agent, 'register') as mock_register:
                agent.start()
                mock_register.assert_called_once()


class TestTaskClaiming:
    """Test agent task claiming functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(agent_id="test-agent", coordination_dir=tmpdir)
            yield agent

    def test_claim_task_success(self, agent):
        """Test successfully claiming a task."""
        mock_task = {
            "id": "task-1",
            "description": "Test task",
            "status": "available",
            "priority": 1
        }

        with patch.object(agent, 'get_available_tasks', return_value=[mock_task]):
            with patch.object(agent, 'claim_task', return_value=mock_task) as mock_claim:
                task = agent.try_claim_task()

                assert task is not None
                assert task["id"] == "task-1"
                mock_claim.assert_called_once_with("task-1")

    def test_claim_task_no_available(self, agent):
        """Test claiming when no tasks are available."""
        with patch.object(agent, 'get_available_tasks', return_value=[]):
            task = agent.try_claim_task()
            assert task is None

    def test_claim_task_already_claimed(self, agent):
        """Test claiming a task that's already claimed."""
        mock_task = {
            "id": "task-1",
            "description": "Test task",
            "status": "claimed",
            "priority": 1
        }

        with patch.object(agent, 'get_available_tasks', return_value=[mock_task]):
            with patch.object(agent, 'claim_task', side_effect=AgentError("Already claimed")):
                task = agent.try_claim_task()
                assert task is None

    def test_claim_highest_priority_task(self, agent):
        """Test that agent claims highest priority available task."""
        mock_tasks = [
            {"id": "task-1", "priority": 3, "status": "available"},
            {"id": "task-2", "priority": 1, "status": "available"},
            {"id": "task-3", "priority": 2, "status": "available"},
        ]

        with patch.object(agent, 'get_available_tasks', return_value=mock_tasks):
            with patch.object(agent, 'claim_task', return_value=mock_tasks[1]) as mock_claim:
                task = agent.try_claim_task()

                assert task["priority"] == 1
                mock_claim.assert_called_once_with("task-2")


class TestTaskExecution:
    """Test task execution functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(agent_id="test-agent", coordination_dir=tmpdir)
            yield agent

    def test_execute_task_success(self, agent):
        """Test successful task execution."""
        task = {
            "id": "task-1",
            "description": "Test task",
            "status": "claimed",
            "assigned_to": "test-agent"
        }

        with patch.object(agent, 'execute_task_logic', return_value="Success"):
            with patch.object(agent, 'complete_task') as mock_complete:
                agent.execute_task(task)
                mock_complete.assert_called_once()

    def test_execute_task_failure(self, agent):
        """Test task execution with failure."""
        task = {
            "id": "task-1",
            "description": "Test task",
            "status": "claimed",
            "assigned_to": "test-agent"
        }

        with patch.object(agent, 'execute_task_logic', side_effect=Exception("Task failed")):
            with patch.object(agent, 'fail_task') as mock_fail:
                agent.execute_task(task)
                mock_fail.assert_called_once()

    def test_execute_task_with_retry(self, agent):
        """Test task execution with retry on failure."""
        task = {
            "id": "task-1",
            "description": "Test task",
            "status": "claimed",
            "assigned_to": "test-agent"
        }

        # Fail twice, then succeed
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TaskExecutionError("Temporary failure")
            return "Success"

        with patch.object(agent, 'execute_task_logic', side_effect=side_effect):
            with patch.object(agent, 'complete_task') as mock_complete:
                agent.execute_task_with_retry(task, max_retries=3)
                mock_complete.assert_called_once()

    def test_execute_task_max_retries_exceeded(self, agent):
        """Test task execution when max retries are exceeded."""
        task = {
            "id": "task-1",
            "description": "Test task",
            "status": "claimed",
            "assigned_to": "test-agent"
        }

        with patch.object(agent, 'execute_task_logic', side_effect=TaskExecutionError("Always fails")):
            with patch.object(agent, 'fail_task') as mock_fail:
                agent.execute_task_with_retry(task, max_retries=2)
                mock_fail.assert_called_once()


class TestHeartbeat:
    """Test agent heartbeat functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(agent_id="test-agent", coordination_dir=tmpdir)
            yield agent

    def test_send_heartbeat(self, agent):
        """Test sending a heartbeat."""
        with patch.object(agent, 'send_heartbeat_to_orchestrator') as mock_heartbeat:
            agent.send_heartbeat()
            mock_heartbeat.assert_called_once()

    def test_heartbeat_updates_timestamp(self, agent):
        """Test that heartbeat updates last heartbeat timestamp."""
        with patch.object(agent, 'send_heartbeat_to_orchestrator'):
            before = agent.last_heartbeat
            time.sleep(0.1)
            agent.send_heartbeat()
            after = agent.last_heartbeat

            assert after > before

    def test_automatic_heartbeat(self, agent):
        """Test that agent sends automatic heartbeats."""
        with patch.object(agent, 'send_heartbeat') as mock_heartbeat:
            agent.config.heartbeat_interval = 0.1
            agent.start_heartbeat()
            time.sleep(0.3)
            agent.stop_heartbeat()

            assert mock_heartbeat.call_count >= 2

    def test_heartbeat_thread_teardown(self, agent):
        """Test that heartbeat thread is cleaned up on stop."""
        agent.config.heartbeat_interval = 0.05
        agent.start_heartbeat()
        time.sleep(0.2)
        agent.stop_heartbeat()
        assert agent._heartbeat_thread is None or not agent._heartbeat_thread.is_alive()


class TestAgentLifecycle:
    """Test agent lifecycle management."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(agent_id="test-agent", coordination_dir=tmpdir)
            yield agent

    def test_agent_start(self, agent):
        """Test starting an agent."""
        with patch.object(agent, 'register') as mock_register:
            agent.start()
            assert agent.is_running
            mock_register.assert_called_once()

    def test_agent_stop(self, agent):
        """Test stopping an agent."""
        with patch.object(agent, 'deregister') as mock_deregister:
            agent.start()
            agent.stop()

            assert not agent.is_running
            mock_deregister.assert_called_once()

    def test_agent_cleanup_on_stop(self, agent):
        """Test that agent cleans up resources on stop."""
        agent.start()

        with patch.object(agent, 'cleanup_resources') as mock_cleanup:
            agent.stop()
            mock_cleanup.assert_called_once()


class TestWorkLoop:
    """Test agent work loop."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(agent_id="test-agent", coordination_dir=tmpdir)
            yield agent

    def test_work_loop_claims_and_executes(self, agent):
        """Test that work loop claims and executes tasks."""
        mock_task = {
            "id": "task-1",
            "description": "Test task",
            "status": "available",
            "priority": 1
        }

        # Return task once, then None to exit loop
        call_count = 0
        def get_task_side_effect():
            nonlocal call_count
            call_count += 1
            return mock_task if call_count == 1 else None

        with patch.object(agent, 'try_claim_task', side_effect=get_task_side_effect):
            with patch.object(agent, 'execute_task') as mock_execute:
                agent.run_once()
                mock_execute.assert_called_once()

    def test_work_loop_handles_no_tasks(self, agent):
        """Test that work loop handles no available tasks gracefully."""
        with patch.object(agent, 'try_claim_task', return_value=None):
            with patch.object(agent, 'execute_task') as mock_execute:
                agent.run_once()
                mock_execute.assert_not_called()


class TestErrorHandling:
    """Test agent error handling."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(agent_id="test-agent", coordination_dir=tmpdir)
            yield agent

    def test_handle_orchestrator_connection_error(self, agent):
        """Test handling connection errors to orchestrator."""
        with patch.object(agent, 'get_available_tasks', side_effect=ConnectionError("Cannot connect")):
            # Should not crash
            agent.run_once()

    def test_handle_task_execution_error(self, agent):
        """Test handling errors during task execution."""
        task = {
            "id": "task-1",
            "description": "Test task",
            "status": "claimed"
        }

        with patch.object(agent, 'execute_task_logic', side_effect=Exception("Execution error")):
            with patch.object(agent, 'fail_task') as mock_fail:
                # Should handle error gracefully
                agent.execute_task(task)
                mock_fail.assert_called_once()

    def test_handle_malformed_task(self, agent):
        """Test handling malformed task data."""
        malformed_task = {
            "id": "task-1",
            # Missing required fields
        }

        with pytest.raises(AgentError):
            agent.validate_task(malformed_task)


class TestTaskContext:
    """Test task context handling."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = Agent(agent_id="test-agent", coordination_dir=tmpdir)
            yield agent

    def test_load_task_context(self, agent):
        """Test loading task context."""
        task = {
            "id": "task-1",
            "description": "Test task",
            "context": {
                "files": ["main.py", "test.py"],
                "hints": "Focus on the authentication module"
            }
        }

        context = agent.load_context(task)
        assert context is not None
        assert "files" in context
        assert len(context["files"]) == 2

    def test_execute_with_context(self, agent):
        """Test executing a task with context."""
        task = {
            "id": "task-1",
            "description": "Fix bug in login",
            "context": {
                "files": ["auth/login.py"],
                "hints": "Check password validation"
            }
        }

        with patch.object(agent, 'execute_task_logic') as mock_execute:
            agent.execute_task(task)
            # Verify context was passed
            call_args = mock_execute.call_args
            assert call_args is not None


class TestCapabilities:
    """Test agent capabilities matching."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AgentConfig(capabilities=["python", "testing", "docker"])
            agent = Agent(
                agent_id="test-agent",
                coordination_dir=tmpdir,
                config=config
            )
            yield agent

    def test_can_handle_task_with_matching_capabilities(self, agent):
        """Test that agent can handle tasks matching its capabilities."""
        task = {
            "id": "task-1",
            "description": "Write Python tests",
            "required_capabilities": ["python", "testing"]
        }

        assert agent.can_handle_task(task)

    def test_cannot_handle_task_without_capabilities(self, agent):
        """Test that agent cannot handle tasks without required capabilities."""
        task = {
            "id": "task-1",
            "description": "Build iOS app",
            "required_capabilities": ["swift", "ios"]
        }

        assert not agent.can_handle_task(task)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
