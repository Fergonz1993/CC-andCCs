#!/usr/bin/env python3
"""
Clean coordination artifacts from the filesystem (ATOM-209).

Usage:
    ./scripts/clean_coordination.py [--dry-run] [--all] [--dir PATH]

Removes:
    - .coordination/tasks.json
    - .coordination/agents.json
    - .coordination/discoveries.json
    - .coordination/logs/*
    - .coordination/results/*
    - .coordination/metrics/*

Does NOT remove:
    - .coordination/master-plan.md (unless --all)
    - .coordination/.lock
"""

import argparse
import shutil
from pathlib import Path


def find_coordination_dirs(root: Path) -> list[Path]:
    """Find all .coordination directories."""
    dirs = []

    # Root level
    root_coord = root / ".coordination"
    if root_coord.exists():
        dirs.append(root_coord)

    # Under claude-multi-agent
    for option_dir in root.glob("claude-multi-agent/option-*"):
        coord = option_dir / ".coordination"
        if coord.exists():
            dirs.append(coord)

    # Under test dirs
    for test_coord in root.glob("**/tests/.coordination"):
        if "node_modules" not in str(test_coord):
            dirs.append(test_coord)

    return dirs


def clean_artifacts(
    coord_dir: Path,
    *,
    dry_run: bool = False,
    remove_all: bool = False
) -> tuple[int, int]:
    """
    Clean coordination artifacts from a directory.

    Returns:
        Tuple of (files_removed, bytes_freed)
    """
    files_removed = 0
    bytes_freed = 0

    # Files to always clean
    files_to_remove = [
        "tasks.json",
        "agents.json",
        "discoveries.json",
    ]

    # Directories to clean
    dirs_to_clean = [
        "logs",
        "results",
        "metrics",
    ]

    if remove_all:
        files_to_remove.extend([
            "master-plan.md",
            ".lock",
        ])
        dirs_to_clean.extend([
            "context",
        ])

    # Remove individual files
    for filename in files_to_remove:
        file_path = coord_dir / filename
        if file_path.exists():
            size = file_path.stat().st_size
            if dry_run:
                print(f"  Would remove: {file_path} ({size} bytes)")
            else:
                file_path.unlink()
                print(f"  Removed: {file_path}")
            files_removed += 1
            bytes_freed += size

    # Clean directories
    for dirname in dirs_to_clean:
        dir_path = coord_dir / dirname
        if dir_path.exists() and dir_path.is_dir():
            for file_path in dir_path.glob("**/*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    if dry_run:
                        print(f"  Would remove: {file_path} ({size} bytes)")
                    else:
                        file_path.unlink()
                        print(f"  Removed: {file_path}")
                    files_removed += 1
                    bytes_freed += size

    return files_removed, bytes_freed


def format_bytes(num_bytes: int) -> str:
    """Format bytes in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean coordination artifacts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without removing"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also remove master-plan.md and context/"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Specific .coordination directory to clean"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    if args.dir:
        coord_dirs = [args.dir] if args.dir.exists() else []
    else:
        coord_dirs = find_coordination_dirs(repo_root)

    if not coord_dirs:
        print("No .coordination directories found")
        return

    print(f"Found {len(coord_dirs)} coordination director{'ies' if len(coord_dirs) != 1 else 'y'}:")
    for d in coord_dirs:
        print(f"  - {d}")
    print()

    if not args.dry_run and not args.yes:
        response = input("Proceed with cleanup? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            return

    total_files = 0
    total_bytes = 0

    for coord_dir in coord_dirs:
        print(f"\nCleaning: {coord_dir}")
        files, bytes_freed = clean_artifacts(
            coord_dir,
            dry_run=args.dry_run,
            remove_all=args.all
        )
        total_files += files
        total_bytes += bytes_freed

    print()
    if args.dry_run:
        print(f"Dry run: would remove {total_files} files ({format_bytes(total_bytes)})")
    else:
        print(f"Cleaned {total_files} files ({format_bytes(total_bytes)})")


if __name__ == "__main__":
    main()
