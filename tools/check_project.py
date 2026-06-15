"""Check project directory structure."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_DIRS = [
    "configs",
    "data",
    "models",
    "utils",
    "scripts",
    "tools",
    "docs",
    "experiments",
]

REQUIRED_ENTRIES = [
    "train.py",
    "eval.py",
    "infer.py",
]

FORBIDDEN_DIRS = [
    "src",
    "model",
    "losses",
    "metrics",
    "visualization",
    "engine",
    "core",
    "tasks",
]


def main() -> None:
    """Verify that the project has the expected lightweight research layout."""
    print(f"Project root: {PROJECT_ROOT}")

    for dirname in REQUIRED_DIRS:
        path = PROJECT_ROOT / dirname
        if path.is_dir():
            print(f"[OK]   {dirname}/")
        else:
            print(f"[MISS] {dirname}/  (expected directory)")

    for entry in REQUIRED_ENTRIES:
        path = PROJECT_ROOT / entry
        if path.is_file():
            print(f"[OK]   {entry}")
        else:
            print(f"[MISS] {entry}  (expected file)")

    for dirname in FORBIDDEN_DIRS:
        path = PROJECT_ROOT / dirname
        if path.exists():
            print(f"[WARN] {dirname}/ should not exist in this template")

    print("Project structure check finished.")


if __name__ == "__main__":
    main()
