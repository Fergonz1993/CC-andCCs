#!/usr/bin/env python3

"""Ralph Loop helper.

Usage:
  ./scripts/ralph_loop.py init
  ./scripts/ralph_loop.py run-tests
  ./scripts/ralph_loop.py prompt
  ./scripts/ralph_loop.py loop [--auto] [--agent-command CMD] [--max-iterations N] [--sleep SECONDS]
                             [--verify-passes] [--features-first]
  ./scripts/ralph_loop.py step [--auto] [--agent-command CMD] [--verify-passes] [--features-first]
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

_STOP_REQUESTED = False


def _handle_stop_signal(signum: int, frame: object) -> None:
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\nLoop stop requested. Finishing current step...", file=sys.stderr)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_log_dir(repo_root: Path) -> Path:
    log_dir = repo_root / ".coordination" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def load_config(path: Path) -> dict:
    if not path.exists():
        print(f"ralph_config.json not found at {path}", file=sys.stderr)
        sys.exit(2)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_test_results_json(results: list[dict], overall_rc: int, output_path: Path) -> None:
    """Write test results to JSON file (ATOM-112)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    total_duration = sum(r["duration_seconds"] for r in results)

    report = {
        "timestamp": _timestamp(),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "overall_status": "pass" if overall_rc == 0 else "fail",
            "overall_return_code": overall_rc,
            "total_duration_seconds": round(total_duration, 3),
        },
        "commands": results,
    }

    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def should_run_optional(command: str) -> bool:
    """Run optional commands only when their target script exists.

    Heuristic: if the first token is a path and exists, run it. Otherwise skip.
    """
    try:
        first = shlex.split(command)[0]
    except ValueError:
        return False
    if first.startswith("./") or first.startswith("/") or first.endswith(".sh"):
        return Path(first).exists()
    return False


def run_commands(
    commands: list[str],
    label: str,
    log_path: Optional[Path] = None,
    results_collector: Optional[list[dict]] = None,
) -> int:
    for cmd in commands:
        print(f"\n==> {label}: {cmd}")
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
        elapsed = time.time() - start_time

        # Collect result if collector provided (ATOM-112)
        if results_collector is not None:
            results_collector.append({
                "command": cmd,
                "label": label,
                "return_code": result.returncode,
                "duration_seconds": round(elapsed, 3),
                "status": "pass" if result.returncode == 0 else "fail",
                "stdout": result.stdout[:1000] if result.stdout else "",  # Truncate for JSON
                "stderr": result.stderr[:1000] if result.stderr else "",
                "timestamp": _timestamp(),
            })

        # Print output to console
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)

        if result.returncode != 0:
            if log_path:
                _append_log(
                    log_path,
                    f"[{_timestamp()}] {label} status=fail rc={result.returncode} cmd={cmd}",
                )
            return result.returncode
        if log_path:
            _append_log(
                log_path,
                f"[{_timestamp()}] {label} status=ok rc=0 cmd={cmd}",
            )
    return 0


def _find_next_backlog_item(todo_path: Path) -> Optional[str]:
    if not todo_path.exists():
        return None
    for line in todo_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("- [ ]"):
            return line.strip()
    return None


def _ensure_file(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def init_state(repo_root: Path, cfg: dict) -> None:
    todo_path = repo_root / "TODO_RALPH.md"
    agents_path = repo_root / "AGENTS.md"

    _ensure_file(
        todo_path,
        "# TODO_RALPH.md\n\n## P0 - Correctness / Security / Critical Path\n- [ ] ATOM-001: Describe the first atomic feature\n",
    )

    _ensure_file(
        agents_path,
        "# AGENTS.md\n\n## RALPH_STATE\n- Goal: Define the next backlog goal\n- Backlog: TODO_RALPH.md\n- Last updated: \n- Status: backlog ready (no items started)\n",
    )

    next_item = _find_next_backlog_item(todo_path) or "none"
    completion_token = cfg.get("completion_token", "<<<RALPH_DONE>>>")

    print("RALPH init complete")
    print(f"Next backlog item: {next_item}")
    print(f"Completion token: {completion_token}")


def write_prompt(repo_root: Path, cfg: dict) -> None:
    agents_path = repo_root / "AGENTS.md"
    todo_path = repo_root / "TODO_RALPH.md"
    resume_path = repo_root / "RALPH_RESUME.md"
    prompt_path = repo_root / "RALPH_PROMPT.md"

    next_item = _find_next_backlog_item(todo_path) or "none"
    completion_token = cfg.get("completion_token", "<<<RALPH_DONE>>>")

    resume_body = "\n".join(
        [
            "# RALPH_RESUME.md",
            "",
            "State:",
            agents_path.read_text(encoding="utf-8") if agents_path.exists() else "(missing AGENTS.md)",
            "",
            "Next backlog item:",
            next_item,
            "",
            f"Completion token: {completion_token}",
        ]
    )
    resume_path.write_text(resume_body, encoding="utf-8")

    prompt_body = "\n".join(
        [
            "# RALPH_PROMPT.md",
            "",
            "Run the Codex feature loop:",
            "1) Read AGENTS.md (RALPH_STATE) and TODO_RALPH.md.",
            "2) Pick the next unchecked item (if TODO is empty, check feature_list.json for passes=false).",
            "3) Implement minimal change and run: ./scripts/ralph_loop.py run-tests",
            "4) Update AGENTS.md with results and TODO_RALPH.md with progress (or feature_list.json if used).",
            f"5) Repeat until backlog complete, then emit {completion_token}.",
        ]
    )
    prompt_path.write_text(prompt_body, encoding="utf-8")

    print(f"Wrote {resume_path}")
    print(f"Wrote {prompt_path}")


def write_loop_prompt(
    repo_root: Path,
    cfg: dict,
    next_item: str,
    *,
    is_feature_item: bool,
    verify_passes: bool,
) -> Path:
    prompt_path = repo_root / "RALPH_LOOP_PROMPT.md"
    completion_token = cfg.get("completion_token", "<<<RALPH_DONE>>>")
    lines = [
        "# RALPH_LOOP_PROMPT.md",
        "",
        "Run ONE iteration only:",
        "1) Read AGENTS.md (RALPH_STATE), TODO_RALPH.md, README.md/ARCHITECTURE.md.",
        "2) If TODO has unchecked items, pick the next unchecked.",
        "   Else, open feature_list.json and pick the first item where passes != true",
        "   (or, in verification mode, where verified != true).",
        f"3) Current target: {next_item}",
        "4) Implement the minimal safe change for that single item.",
        "5) Update TODO_RALPH.md (mark done) or feature_list.json accordingly.",
        "6) Update AGENTS.md with results, pre-mortem, adversarial check, regression net.",
        "7) Stop. Do NOT run tests; the loop will run ./scripts/ralph_loop.py run-tests.",
        f"8) If no pending items exist in either list, emit {completion_token} and stop.",
    ]

    if is_feature_item and verify_passes:
        lines.extend(
            [
                "",
                "Verification mode (feature_list.json):",
                "- Re-test or reproduce the feature claim.",
                "- If it holds, add \"verified\": true plus \"verification_date\" (YYYY-MM-DD)",
                "  and short \"verification_notes\".",
                "- If it fails, set \"passes\": false and add \"failure_notes\".",
            ]
        )

    prompt_body = "\n".join(lines)
    prompt_path.write_text(prompt_body, encoding="utf-8")
    return prompt_path


def _load_feature_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        features = data.get("features", [])
        return features if isinstance(features, list) else []
    if isinstance(data, list):
        return data
    return []


def _find_next_feature_item(path: Path, verify_passes: bool) -> Optional[dict]:
    for item in _load_feature_list(path):
        if verify_passes:
            if item.get("verified") is not True:
                return item
        else:
            if item.get("passes") is not True:
                return item
    return None


def _format_feature_item(item: dict) -> str:
    feature_id = item.get("id", "feature")
    description = item.get("description", "(no description)")
    return f"feature_list:{feature_id} - {description}"


def _resolve_agent_command(repo_root: Path, cfg: dict, agent_command: Optional[str]) -> str:
    if agent_command:
        return agent_command
    if cfg.get("agent_command"):
        return str(cfg["agent_command"])
    if shutil.which("codex"):
        return f"codex exec --full-auto --sandbox workspace-write -C {repo_root}"
    return ""


def _run_agent_command(command: str, prompt_path: Path, log_path: Optional[Path]) -> int:
    if "{prompt}" in command:
        final_command = command.format(prompt=prompt_path)
        input_text = None
    else:
        final_command = command
        input_text = prompt_path.read_text(encoding="utf-8")

    if log_path:
        _append_log(log_path, f"[{_timestamp()}] agent cmd={final_command}")

    result = subprocess.run(
        final_command,
        shell=True,
        executable="/bin/bash",
        input=input_text,
        text=True,
    )
    if log_path:
        _append_log(log_path, f"[{_timestamp()}] agent rc={result.returncode}")
    return result.returncode


def run_tests(repo_root: Path, cfg: dict, *, json_output: bool = False, json_path: Optional[Path] = None) -> None:
    """Run the test gate commands (ATOM-112: supports --json flag)."""
    test_commands = cfg.get("test_commands", [])
    optional_commands = cfg.get("optional_test_commands", [])

    if not isinstance(test_commands, list) or not isinstance(optional_commands, list):
        print("Invalid ralph_config.json: test_commands/optional_test_commands must be lists", file=sys.stderr)
        sys.exit(2)

    if not test_commands:
        print("No test_commands defined in ralph_config.json", file=sys.stderr)
        sys.exit(2)

    log_dir = _ensure_log_dir(repo_root)
    log_path = log_dir / "ralph_test_gate.log"
    _append_log(log_path, f"[{_timestamp()}] test-gate status=start")

    # Collect results for JSON output (ATOM-112)
    results_collector: list[dict] = [] if json_output else None

    rc = run_commands(test_commands, "test", log_path=log_path, results_collector=results_collector)
    if rc != 0:
        _append_log(log_path, f"[{_timestamp()}] test-gate status=fail rc={rc}")
        if json_output:
            _write_test_results_json(results_collector, rc, json_path or (log_dir / "test_results.json"))
        print(f"Command failed ({rc}). See {log_path}", file=sys.stderr)
        sys.exit(rc)

    optional_to_run = [cmd for cmd in optional_commands if should_run_optional(cmd)]
    if optional_commands and not optional_to_run:
        print("No optional tests matched existing scripts; skipping optional tests.")
        _append_log(log_path, f"[{_timestamp()}] optional-tests status=skipped")
    if optional_to_run:
        rc = run_commands(optional_to_run, "optional", log_path=log_path, results_collector=results_collector)
        if rc != 0:
            _append_log(log_path, f"[{_timestamp()}] test-gate status=fail rc={rc}")
            if json_output:
                _write_test_results_json(results_collector, rc, json_path or (log_dir / "test_results.json"))
            print(f"Command failed ({rc}). See {log_path}", file=sys.stderr)
            sys.exit(rc)

    _append_log(log_path, f"[{_timestamp()}] test-gate status=ok")

    # Write JSON results if requested (ATOM-112)
    if json_output:
        output_path = json_path or (log_dir / "test_results.json")
        _write_test_results_json(results_collector, 0, output_path)
        print(f"\nTest results exported to: {output_path}")

    print("\nAll test commands passed.")


def run_loop(
    repo_root: Path,
    cfg: dict,
    *,
    max_iterations: Optional[int],
    sleep_seconds: float,
    auto: bool,
    agent_command: Optional[str],
    sync_features: bool,
    run_test_gate: bool,
    verify_passes: bool,
    features_first: bool,
) -> None:
    todo_path = repo_root / "TODO_RALPH.md"
    feature_list_path = repo_root / "feature_list.json"
    agents_path = repo_root / "AGENTS.md"

    if not todo_path.exists() or not agents_path.exists():
        init_state(repo_root, cfg)

    log_dir = _ensure_log_dir(repo_root)
    loop_log = log_dir / "ralph_loop.log"

    iteration = 0
    while True:
        if _STOP_REQUESTED:
            _append_log(loop_log, f"[{_timestamp()}] loop status=stopped")
            print("Loop stopped.")
            return

        iteration += 1
        next_item = None
        is_feature_item = False
        feature_item: Optional[dict] = None

        if features_first and sync_features:
            feature_item = _find_next_feature_item(feature_list_path, verify_passes)
            if feature_item:
                next_item = _format_feature_item(feature_item)
                is_feature_item = True
            else:
                next_item = _find_next_backlog_item(todo_path)
        else:
            next_item = _find_next_backlog_item(todo_path)
            if not next_item and sync_features:
                feature_item = _find_next_feature_item(feature_list_path, verify_passes)
                if feature_item:
                    next_item = _format_feature_item(feature_item)
                    is_feature_item = True

        if not next_item:
            _append_log(loop_log, f"[{_timestamp()}] loop status=empty")
            write_prompt(repo_root, cfg)
            print("Backlog empty. Nothing to do.")
            return

        write_prompt(repo_root, cfg)
        loop_prompt_path = write_loop_prompt(
            repo_root,
            cfg,
            next_item,
            is_feature_item=is_feature_item,
            verify_passes=verify_passes,
        )
        _append_log(loop_log, f"[{_timestamp()}] iteration={iteration} target={next_item}")

        if not auto:
            print("Loop prepared next item. Re-run with --auto to execute automatically.")
            return

        command = _resolve_agent_command(repo_root, cfg, agent_command)
        if not command:
            print("No agent command configured. Pass --agent-command or set agent_command in ralph_config.json.", file=sys.stderr)
            sys.exit(2)

        rc = _run_agent_command(command, loop_prompt_path, loop_log)
        if rc != 0:
            print(f"Agent command failed ({rc}). See {loop_log}", file=sys.stderr)
            sys.exit(rc)

        if run_test_gate:
            run_tests(repo_root, cfg)

        _append_log(loop_log, f"[{_timestamp()}] iteration={iteration} status=complete")

        if max_iterations is not None and iteration >= max_iterations:
            print(f"Reached max iterations: {max_iterations}")
            return

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ralph loop helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")

    run_tests_parser = subparsers.add_parser("run-tests")
    run_tests_parser.add_argument(
        "--json",
        action="store_true",
        help="Export test results as JSON (ATOM-112)",
    )
    run_tests_parser.add_argument(
        "--json-output",
        type=str,
        help="Path for JSON output file (default: .coordination/logs/test_results.json)",
    )

    subparsers.add_parser("prompt")

    loop_parser = subparsers.add_parser("loop")
    loop_parser.add_argument("--auto", action="store_true", help="Run agent command each iteration")
    loop_parser.add_argument("--agent-command", help="Override agent command (supports {prompt})")
    loop_parser.add_argument("--max-iterations", type=int, help="Stop after N iterations")
    loop_parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between iterations")
    loop_parser.add_argument("--no-sync-features", action="store_true", help="Skip feature_list.json fallback")
    loop_parser.add_argument("--no-tests", action="store_true", help="Skip running test gate")
    loop_parser.add_argument(
        "--verify-passes",
        action="store_true",
        help="Treat feature_list items without verified=true as pending",
    )
    loop_parser.add_argument(
        "--no-verify-passes",
        action="store_true",
        help="Disable verification mode even if config enables it",
    )
    loop_parser.add_argument(
        "--features-first",
        action="store_true",
        help="Prefer feature_list.json items before TODO items",
    )

    step_parser = subparsers.add_parser("step")
    step_parser.add_argument("--auto", action="store_true", help="Run agent command once")
    step_parser.add_argument("--agent-command", help="Override agent command (supports {prompt})")
    step_parser.add_argument("--no-sync-features", action="store_true", help="Skip feature_list.json fallback")
    step_parser.add_argument("--no-tests", action="store_true", help="Skip running test gate")
    step_parser.add_argument(
        "--verify-passes",
        action="store_true",
        help="Treat feature_list items without verified=true as pending",
    )
    step_parser.add_argument(
        "--no-verify-passes",
        action="store_true",
        help="Disable verification mode even if config enables it",
    )
    step_parser.add_argument(
        "--features-first",
        action="store_true",
        help="Prefer feature_list.json items before TODO items",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_config(repo_root / "ralph_config.json")

    if args.command == "init":
        init_state(repo_root, cfg)
        return

    if args.command == "prompt":
        write_prompt(repo_root, cfg)
        return

    if args.command == "run-tests":
        json_path = Path(args.json_output) if args.json_output else None
        run_tests(repo_root, cfg, json_output=args.json, json_path=json_path)
        return

    if args.command == "loop":
        verify_default = bool(cfg.get("verify_feature_passes", False))
        verify_passes = (args.verify_passes or verify_default) and not args.no_verify_passes
        features_first = bool(cfg.get("features_first", False)) or args.features_first
        signal.signal(signal.SIGINT, _handle_stop_signal)
        signal.signal(signal.SIGTERM, _handle_stop_signal)
        run_loop(
            repo_root,
            cfg,
            max_iterations=args.max_iterations,
            sleep_seconds=args.sleep,
            auto=args.auto,
            agent_command=args.agent_command,
            sync_features=not args.no_sync_features,
            run_test_gate=not args.no_tests,
            verify_passes=verify_passes,
            features_first=features_first,
        )
        return

    if args.command == "step":
        verify_default = bool(cfg.get("verify_feature_passes", False))
        verify_passes = (args.verify_passes or verify_default) and not args.no_verify_passes
        features_first = bool(cfg.get("features_first", False)) or args.features_first
        run_loop(
            repo_root,
            cfg,
            max_iterations=1,
            sleep_seconds=0.0,
            auto=args.auto,
            agent_command=args.agent_command,
            sync_features=not args.no_sync_features,
            run_test_gate=not args.no_tests,
            verify_passes=verify_passes,
            features_first=features_first,
        )
        return

    print("Unknown command", file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
