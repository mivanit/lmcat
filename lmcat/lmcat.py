from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Handle Python 3.11+ vs older Python for TOML parsing
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


@dataclass
class LMCatConfig:
    """Configuration dataclass for lmcat

    # Parameters:
     - `tree_divider: str`
     - `indent: str`
     - `file_divider: str`
     - `content_divider: str`
    """

    tree_divider: str = "│   "
    indent: str = "    "
    file_divider: str = "├── "
    content_divider: str = "``````"

    @classmethod
    def load(cls, cfg_data: dict[str, Any]) -> LMCatConfig:
        """Load an LMCatConfig from a dictionary of config values

        # Parameters:
         - `cfg_data: dict[str, Any]`

        # Returns:
         - `LMCatConfig`
        """
        config = cls()
        for key, val in cfg_data.items():
            if key in config.__dataclass_fields__:
                setattr(config, key, val)
        return config

    @classmethod
    def read(cls, root_dir: Path) -> LMCatConfig:
        """Attempt to read config from pyproject.toml, lmcat.toml, or lmcat.json.
        Fallback to defaults if none found.

        # Parameters:
         - `root_dir: Path`

        # Returns:
         - `LMCatConfig`
        """
        pyproject_path = root_dir / "pyproject.toml"
        lmcat_toml_path = root_dir / "lmcat.toml"
        lmcat_json_path = root_dir / "lmcat.json"

        # If tomllib is available (or tomli fallback), try pyproject
        if tomllib is not None and pyproject_path.is_file():
            with pyproject_path.open("rb") as f:
                pyproject_data = tomllib.load(f)
            if "tool" in pyproject_data and "lmcat" in pyproject_data["tool"]:
                return cls.load(pyproject_data["tool"]["lmcat"])

        # If tomllib is available (or tomli fallback), try lmcat.toml
        if tomllib is not None and lmcat_toml_path.is_file():
            with lmcat_toml_path.open("rb") as f:
                toml_data = tomllib.load(f)
            return cls.load(toml_data)

        # Try lmcat.json
        if lmcat_json_path.is_file():
            with lmcat_json_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f)
            return cls.load(json_data)

        # Fallback to defaults
        return cls()


def load_ignore_patterns(root_dir: Path) -> dict[Path, list[str]]:
    """Traverse `root_dir`, collecting `.lmignore` patterns

    # Parameters:
     - `root_dir: Path`

    # Returns:
     - `dict[Path, list[str]]`
       A mapping of directory paths to lists of ignore patterns.
    """
    ignore_dict: dict[Path, list[str]] = {}
    for current_path, dirs, files in os.walk(root_dir):
        current_dir = Path(current_path)
        if ".lmignore" in files:
            lmignore_path = current_dir / ".lmignore"
            patterns: list[str] = []
            with lmignore_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line_stripped = line.strip()
                    if not line_stripped or line_stripped.startswith("#"):
                        continue
                    patterns.append(line_stripped)
            ignore_dict[current_dir] = patterns
    return ignore_dict


def should_ignore_path(rel_path: str, pattern: str) -> bool:
    """
    Decide if rel_path matches a given ignore pattern.
    If pattern ends with '/', it means ignore a directory (and all contents).
    Example: 'subdir2/' => ignore subdir2 and anything under it.
    """
    # If pattern ends with '/', treat that as "ignore subdir" 
    # plus everything under it. So if rel_path == 'subdir2'
    # or rel_path starts with 'subdir2/', we match.
    if pattern.endswith("/"):
        dir_pat = pattern.rstrip("/")
        if rel_path == dir_pat or rel_path.startswith(dir_pat + os.sep):
            return True
        return False
    else:
        # Regular fnmatch
        return fnmatch.fnmatch(rel_path, pattern)


def is_ignored(path: Path, root_dir: Path, ignore_dict: dict[Path, list[str]]) -> bool:
    """Check if a file/directory `path` should be ignored,
    by searching for matching patterns from `.lmignore` files in its ancestry.

    # Parameters:
     - `path: Path`
     - `root_dir: Path`
     - `ignore_dict: dict[Path, list[str]]`

    # Returns:
     - `bool`
       `True` if ignored, `False` otherwise.
    """
    path_abs = path.resolve()
    root_abs = root_dir.resolve()

    # We'll gather ignore rules from path's ancestry
    current_dir = path_abs.parent
    while True:
        if current_dir in ignore_dict:
            rel_path = str(path_abs.relative_to(current_dir))
            for pattern in ignore_dict[current_dir]:
                if should_ignore_path(rel_path, pattern):
                    return True
        if current_dir == root_abs:
            break
        parent = current_dir.parent
        if parent == current_dir:
            break
        current_dir = parent

    return False


def sorted_entries(directory: Path) -> list[Path]:
    """
    Return the contents of `directory`, with directories first (alphabetically),
    then files (alphabetically).
    """
    subdirs = sorted([p for p in directory.iterdir() if p.is_dir()], key=lambda x: x.name)
    files = sorted([p for p in directory.iterdir() if p.is_file()], key=lambda x: x.name)
    return subdirs + files


def walk_dir(
    directory: Path,
    root_dir: Path,
    ignore_dict: dict[Path, list[str]],
    config: LMCatConfig,
    prefix: str = "",
) -> tuple[list[str], list[Path]]:
    """Recursively walk a directory, building tree lines and collecting file paths

    # Parameters:
     - `directory: Path`
     - `root_dir: Path`
     - `ignore_dict: dict[Path, list[str]]`
     - `config: LMCatConfig`
     - `prefix: str` (used for tree indentation)

    # Returns:
     - `tuple[list[str], list[Path]]`
       The text lines for the directory structure and the included file paths
    """
    tree_output: list[str] = []
    collected_files: list[Path] = []

    entries: list[Path] = sorted_entries(directory)

    for i, entry in enumerate(entries):
        # Always skip listing the .lmignore file itself
        if entry.name == ".lmignore":
            continue

        if is_ignored(entry, root_dir, ignore_dict):
            continue

        is_last = (i == len(entries) - 1)
        connector = config.file_divider if not is_last else config.file_divider.replace("├", "└")

        if entry.is_dir():
            tree_output.append(f"{prefix}{connector}{entry.name}")
            extension = config.tree_divider if not is_last else config.indent
            sub_output, sub_files = walk_dir(entry, root_dir, ignore_dict, config, prefix + extension)
            tree_output.extend(sub_output)
            collected_files.extend(sub_files)
        else:
            tree_output.append(f"{prefix}{connector}{entry.name}")
            collected_files.append(entry)

    return tree_output, collected_files


def walk_and_collect(root_dir: Path, config: LMCatConfig) -> tuple[list[str], list[Path]]:
    """Walk the filesystem from `root_dir`, using `config` for formatting,
    and gather a tree listing plus file paths to stitch

    # Parameters:
     - `root_dir: Path`
     - `config: LMCatConfig`

    # Returns:
     - `tuple[list[str], list[Path]]`
       (tree_output, collected_files)
    """
    ignore_dict = load_ignore_patterns(root_dir)
    base_name = root_dir.resolve().name

    # Start with the top-level name
    tree_output: list[str] = [base_name]

    sub_output, sub_files = walk_dir(root_dir, root_dir, ignore_dict, config)
    tree_output.extend(sub_output)
    return tree_output, sub_files


def main() -> None:
    """Main entry point for the script

    # Parameters:
     - None

    # Returns:
     - `None`
    """
    root_dir = Path(".").resolve()
    config = LMCatConfig.read(root_dir)

    tree_output, collected_files = walk_and_collect(root_dir, config)

    # Print the directory tree
    for line in tree_output:
        print(line)

    # Print the contents
    for fpath in collected_files:
        print()
        print(config.content_divider)
        with fpath.open("r", encoding="utf-8", errors="ignore") as fobj:
            print(fobj.read(), end="")
        print(config.content_divider)
