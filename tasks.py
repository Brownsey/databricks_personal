"""
Task runner for databricks-personal project.
Replaces Makefile with cross-platform UV-native commands.

Usage: uv run task <target>
"""

import subprocess
import sys

SRC = ["src", "register_model.py", "tasks.py"]


def _run(*cmd: str) -> int:
    result = subprocess.run(cmd)
    return result.returncode


# ── Code Quality ───────────────────────────────────────


def task_ruff() -> int:
    print("Running ruff linter...")
    return _run("uv", "run", "ruff", "check", *SRC)


def task_ruff_fix() -> int:
    print("Running ruff with auto-fix...")
    return _run("uv", "run", "ruff", "check", "--fix", *SRC)


def task_format() -> int:
    print("Formatting code with ruff...")
    return _run("uv", "run", "ruff", "format", *SRC)


def task_format_check() -> int:
    print("Checking code formatting...")
    return _run("uv", "run", "ruff", "format", "--check", *SRC)


def task_ty() -> int:
    print("Running ty type checker...")
    return _run("uv", "tool", "run", "ty", "check", *SRC)


def task_lint() -> int:
    rc = task_ruff()
    if rc != 0:
        return rc
    rc = task_format_check()
    if rc != 0:
        return rc
    print("[OK] All linting checks passed")
    return 0


# ── Testing ────────────────────────────────────────────


def task_test() -> int:
    print("Running tests...")
    return _run("uv", "run", "pytest", "tests/", "-v")


def task_coverage() -> int:
    print("Running tests with coverage...")
    return _run(
        "uv",
        "run",
        "pytest",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html",
        "tests/",
    )


# ── Setup ──────────────────────────────────────────────


def task_install() -> int:
    print("Installing dependencies with UV...")
    rc = _run("uv", "sync", "--all-groups")
    if rc == 0:
        print("[OK] Installation complete")
    return rc


# ── Full Pipeline ──────────────────────────────────────


def task_all() -> int:
    rc = task_install()
    if rc != 0:
        return rc
    rc = task_lint()
    if rc != 0:
        return rc
    rc = task_test()
    if rc != 0:
        return rc
    print("\n[OK] Full pipeline complete!")
    return 0


# ── CLI ────────────────────────────────────────────────

TASKS: dict[str, tuple[str, object]] = {
    "install": ("Install dependencies with UV", task_install),
    "ruff": ("Run ruff linter", task_ruff),
    "ruff-fix": ("Run ruff with auto-fix", task_ruff_fix),
    "format": ("Format code with ruff", task_format),
    "format-check": ("Check code formatting", task_format_check),
    "ty": ("Run ty type checker (Astral)", task_ty),
    "lint": ("Run all linters (ruff + format check)", task_lint),
    "test": ("Run tests with pytest", task_test),
    "coverage": ("Run tests with coverage report", task_coverage),
    "all": ("Full pipeline: install, lint, test", task_all),
}

SECTIONS = {
    "Setup": ["install"],
    "Code Quality": ["ruff", "ruff-fix", "format", "format-check", "ty", "lint"],
    "Testing": ["test", "coverage"],
    "Pipeline": ["all"],
}


def show_help() -> None:
    print("Databricks Personal - Available Tasks:\n")
    for section, keys in SECTIONS.items():
        print(f"  {section}:")
        for key in keys:
            desc = TASKS[key][0]
            print(f"    uv run task {key:<20} {desc}")
        print()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Databricks Personal task runner",
        usage="uv run task <target>",
    )
    parser.add_argument("target", nargs="?", default="help", help="Task to run")
    args = parser.parse_args()

    if args.target == "help":
        show_help()
        return 0

    if args.target not in TASKS:
        print(f"Unknown task: {args.target}")
        print("Run 'uv run task help' for available tasks")
        return 1

    _, task_fn = TASKS[args.target]
    return task_fn()


def cli() -> None:
    sys.exit(main())


if __name__ == "__main__":
    cli()
