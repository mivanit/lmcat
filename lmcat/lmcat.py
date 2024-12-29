from __future__ import annotations

import argparse
import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
	 - `include_gitignore: bool`     (default True)
	 - `suppress_contents: bool`     (default False)
	"""

	tree_divider: str = "│   "
	indent: str = "    "
	file_divider: str = "├── "
	content_divider: str = "``````"
	include_gitignore: bool = True
	suppress_contents: bool = False

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
				# Convert booleans if needed
				if isinstance(getattr(config, key), bool) and isinstance(val, str):
					lower_val = val.strip().lower()
					if lower_val in ("true", "1", "yes"):
						val = True
					elif lower_val in ("false", "0", "no"):
						val = False
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


def load_dir_ignore_patterns(dir_path: Path, config: LMCatConfig) -> list[str]:
	"""
	Return the combined ignore patterns for `dir_path`:
	  1) .gitignore (if present and config.include_gitignore == True)
	  2) .lmignore (if present)

	.lmignore lines come last, so they override .gitignore if there's a conflict or negation.
	"""
	patterns: list[str] = []

	if config.include_gitignore:
		gitignore = dir_path / ".gitignore"
		if gitignore.is_file():
			with gitignore.open("r", encoding="utf-8") as f:
				for line in f:
					line_stripped = line.strip()
					if not line_stripped or line_stripped.startswith("#"):
						continue
					patterns.append(line_stripped)

	lmignore = dir_path / ".lmignore"
	if lmignore.is_file():
		with lmignore.open("r", encoding="utf-8") as f:
			for line in f:
				line_stripped = line.strip()
				if not line_stripped or line_stripped.startswith("#"):
					continue
				patterns.append(line_stripped)

	return patterns


def load_ignore_patterns(
	root_dir: Path, config: Optional[LMCatConfig] = None
) -> dict[Path, list[str]]:
	"""Traverse `root_dir`, collecting `.gitignore` + `.lmignore` patterns for each directory.

	# Parameters:
	 - `root_dir: Path`
	 - `config: LMCatConfig|None`

	# Returns:
	 - `dict[Path, list[str]]`
	   A mapping of directory paths to combined patterns from .gitignore + .lmignore
	"""
	if config is None:
		config = LMCatConfig()  # default if none provided

	ignore_dict: dict[Path, list[str]] = {}
	for current_path, dirs, files in os.walk(root_dir):
		current_dir = Path(current_path)
		dir_patterns = load_dir_ignore_patterns(current_dir, config)
		if dir_patterns:
			ignore_dict[current_dir] = dir_patterns
	return ignore_dict


def git_like_match(rel_path: str, pattern: str) -> bool:
	"""
	Return True if rel_path matches 'pattern' in a Git-like manner:
	- If pattern ends with '/', it means directory match => subdir plus all inside it.
	- If pattern contains a '/', we match the entire (sub)path (with Unix slashes).
	- If pattern has no '/', we match only the basename (like Git).
	"""
	# If pattern ends with '/', strip it for matching but remember directory intent
	dir_rule = pattern.endswith("/")
	trimmed = pattern.rstrip("/")
	# unify path to slash style
	rel_unix = rel_path.replace("\\", "/")
	pat_unix = trimmed.replace("\\", "/")

	if dir_rule:
		# e.g. "subdir/" => ignore "subdir" or anything under "subdir/"
		# That means either rel_unix == 'subdir' or startswith 'subdir/'
		if rel_unix == pat_unix or rel_unix.startswith(pat_unix + "/"):
			return True
		return False

	# If pattern has no slash, match only the basename
	if "/" not in pat_unix:
		base = os.path.basename(rel_unix)
		return fnmatch.fnmatch(base, pat_unix)
	else:
		# If pattern has slash, match entire path
		return fnmatch.fnmatch(rel_unix, pat_unix)


def is_ignored(path: Path, root_dir: Path, ignore_dict: dict[Path, list[str]]) -> bool:
	"""Check if `path` is ignored or not, combining .gitignore + .lmignore lines
	in the order they appear. The last matching pattern wins.

	- If the pattern starts with '!' and matches, we "un-ignore" the path.
	- Otherwise, if it matches, we "ignore" the path.

	# Parameters:
	 - `path: Path`
	 - `root_dir: Path`
	 - `ignore_dict: dict[Path, list[str]]`

	# Returns:
	 - `bool` => True if path is currently ignored
	"""
	path_abs = path.resolve()
	root_abs = root_dir.resolve()
	is_ignored_flag = False

	current_dir = path_abs.parent
	while True:
		if current_dir in ignore_dict:
			rel_path = str(path_abs.relative_to(current_dir))
			# We check patterns in order
			for pattern in ignore_dict[current_dir]:
				negation = pattern.startswith("!")
				raw_pat = pattern[1:].lstrip() if negation else pattern

				if git_like_match(rel_path, raw_pat):
					if negation:
						# un-ignore
						is_ignored_flag = False
					else:
						# ignore
						is_ignored_flag = True

		if current_dir == root_abs:
			break
		parent = current_dir.parent
		if parent == current_dir:
			break
		current_dir = parent

	return is_ignored_flag


def sorted_entries(directory: Path) -> list[Path]:
	"""
	Return the contents of `directory`, with directories first (alphabetically),
	then files (alphabetically).
	"""
	subdirs = sorted(
		[p for p in directory.iterdir() if p.is_dir()], key=lambda x: x.name
	)
	files = sorted(
		[p for p in directory.iterdir() if p.is_file()], key=lambda x: x.name
	)
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
		# Never list .gitignore or .lmignore themselves
		if entry.name in (".gitignore", ".lmignore"):
			continue

		if is_ignored(entry, root_dir, ignore_dict):
			continue

		is_last = i == len(entries) - 1
		connector = (
			config.file_divider
			if not is_last
			else config.file_divider.replace("├", "└")
		)

		if entry.is_dir():
			tree_output.append(f"{prefix}{connector}{entry.name}")
			extension = config.tree_divider if not is_last else config.indent
			sub_output, sub_files = walk_dir(
				entry, root_dir, ignore_dict, config, prefix + extension
			)
			tree_output.extend(sub_output)
			collected_files.extend(sub_files)
		else:
			tree_output.append(f"{prefix}{connector}{entry.name}")
			collected_files.append(entry)

	return tree_output, collected_files


def walk_and_collect(
	root_dir: Path, config: Optional[LMCatConfig] = None
) -> tuple[list[str], list[Path]]:
	"""Walk the filesystem from `root_dir`, using `config` for formatting,
	and gather a tree listing plus file paths to stitch

	# Parameters:
	 - `root_dir: Path`
	 - `config: LMCatConfig|None`

	# Returns:
	 - `tuple[list[str], list[Path]]`
	   (tree_output, collected_files)
	"""
	if config is None:
		config = LMCatConfig()
	ignore_dict = load_ignore_patterns(root_dir, config)
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
	parser = argparse.ArgumentParser(
		description="lmcat - list tree and content, combining .gitignore + .lmignore",
		add_help=False,  # We'll parse known args to avoid confusion with Pytest's arguments
	)
	parser.add_argument(
		"--no-include-gitignore",
		action="store_false",
		dest="include_gitignore",
		default=True,
		help="Do not parse .gitignore files (default: parse them).",
	)
	parser.add_argument(
		"--suppress-contents",
		action="store_true",
		default=False,
		help="Only print the tree, not the file contents.",
	)

	# parse_known_args to avoid crashing on e.g. --cov=. tests/
	args, unknown = parser.parse_known_args()

	root_dir = Path(".").resolve()
	config = LMCatConfig.read(root_dir)

	# CLI overrides
	config.include_gitignore = args.include_gitignore
	config.suppress_contents = args.suppress_contents

	tree_output, collected_files = walk_and_collect(root_dir, config)

	# Print the directory tree
	for line in tree_output:
		print(line)

	# If not suppressing contents, print them
	if not config.suppress_contents:
		for fpath in collected_files:
			print()
			print(config.content_divider)
			with fpath.open("r", encoding="utf-8", errors="ignore") as fobj:
				print(fobj.read(), end="")
			print(config.content_divider)


if __name__ == "__main__":
	main()
