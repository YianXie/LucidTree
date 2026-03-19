from pathlib import Path


def get_project_root() -> Path:
    """
    Find the project root by searching for a .git directory or a pyproject.toml file

    Raises:
        FileNotFoundError: if root could not be found

    Returns:
        Path: the path to start with
    """
    current_file_path = Path(__file__).resolve()
    for parent in current_file_path.parents:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback or error handling if the root isn't found
    raise FileNotFoundError(
        "Project root could not be found based on standard markers."
    )
