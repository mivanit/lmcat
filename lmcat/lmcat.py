from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional

# Handle Python 3.11+ vs older Python for TOML parsing
try:
	import tomllib
except ImportError:
	try:
		import tomli as tomllib
	except ImportError:
		tomllib = None

import igittigitt


@dataclass
class LMCatConfig:
	"""Configuration dataclass for lmcat

	# Parameters:
	 - `tree_divider: str`
	 - `indent: str`
	 - `file_divider: str`
	 - `content_divider: str`
	 - `include_gitignore: bool`  (default True)
	 - `tree_only: bool`  (default False)
	"""

	tree_divider: str = "│   "
	indent: str = " "
	file_divider: str = "├── "
	content_divider: str = "``````"
	include_gitignore: bool = True
	tree_only: bool = False

	@classmethod
	def load(cls, cfg_data: dict[str, Any]) -> LMCatConfig:
		"""Load an LMCatConfig from a dictionary of config values"""
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
		"""Attempt to read config from pyproject.toml, lmcat.toml, or lmcat.json."""
		pyproject_path = root_dir / "pyproject.toml"
		lmcat_toml_path = root_dir / "lmcat.toml"
		lmcat_json_path = root_dir / "lmcat.json"

		# Try pyproject.toml first
		if tomllib is not None and pyproject_path.is_file():
			with pyproject_path.open("rb") as f:
				pyproject_data = tomllib.load(f)
			if "tool" in pyproject_data and "lmcat" in pyproject_data["tool"]:
				return cls.load(pyproject_data["tool"]["lmcat"])

		# Then try lmcat.toml
		if tomllib is not None and lmcat_toml_path.is_file():
			with lmcat_toml_path.open("rb") as f:
				toml_data = tomllib.load(f)
			return cls.load(toml_data)

		# Finally try lmcat.json
		if lmcat_json_path.is_file():
			with lmcat_json_path.open("r", encoding="utf-8") as f:
				json_data = json.load(f)
			return cls.load(json_data)

		# Fallback to defaults
		return cls()


class IgnoreHandler:
	"""Handles all ignore pattern matching using igittigitt"""
	
	def __init__(self, root_dir: Path, config: LMCatConfig):
		self.parser: igittigitt.IgnoreParser = igittigitt.IgnoreParser()
		self.root_dir: Path = root_dir
		self.config: LMCatConfig = config
		self._init_parser()
		
	def _init_parser(self) -> None:
		"""Initialize the parser with all relevant ignore files"""
		# If we're including gitignore, let igittigitt handle it natively
		if self.config.include_gitignore:
			self.parser.parse_rule_files(self.root_dir, filename=".gitignore")
			
		# Add all .lmignore files
		for current_dir, _, files in os.walk(self.root_dir):
			current_path: Path = Path(current_dir)
			lmignore: Path = current_path / ".lmignore"
			if lmignore.is_file():
				self.parser.parse_rule_files(current_path, filename=".lmignore")
				
	def is_ignored(self, path: Path) -> bool:
		"""Check if a path should be ignored"""
		# Never ignore the gitignore/lmignore files themselves
		if path.name in {".gitignore", ".lmignore"}:
			return True
			
		# Use igittigitt's matching
		return self.parser.match(path)


def sorted_entries(directory: Path) -> list[Path]:
	"""Return directory contents sorted: directories first, then files"""
	subdirs: list[Path] = sorted([p for p in directory.iterdir() if p.is_dir()], key=lambda x: x.name)
	files: list[Path] = sorted([p for p in directory.iterdir() if p.is_file()], key=lambda x: x.name)
	return subdirs + files


def walk_dir(
	directory: Path,
	ignore_handler: IgnoreHandler,
	config: LMCatConfig,
	prefix: str = "",
) -> tuple[list[str], list[Path]]:
	"""Recursively walk a directory, building tree lines and collecting file paths"""
	tree_output: list[str] = []
	collected_files: list[Path] = []

	entries: list[Path] = sorted_entries(directory)
	for i, entry in enumerate(entries):
		if ignore_handler.is_ignored(entry):
			continue

		is_last: bool = (i == len(entries) - 1)
		connector: str = (
			config.file_divider if not is_last 
			else config.file_divider.replace("├", "└")
		)

		if entry.is_dir():
			tree_output.append(f"{prefix}{connector}{entry.name}")
			extension: str = config.tree_divider if not is_last else config.indent
			sub_output: list[str]; sub_files: list[Path]
			sub_output, sub_files = walk_dir(
				entry, ignore_handler, config, prefix + extension
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
	"""Walk filesystem from root_dir and gather tree listing plus file paths"""
	if config is None:
		config = LMCatConfig()
		
	ignore_handler: IgnoreHandler = IgnoreHandler(root_dir, config)
	base_name: str = root_dir.resolve().name

	# Start with root directory name
	tree_output: list[str] = [base_name]

	# Walk the directory tree
	sub_output: list[str]; sub_files: list[Path]
	sub_output, sub_files = walk_dir(root_dir, ignore_handler, config)
	tree_output.extend(sub_output)
	
	return tree_output, sub_files


def main() -> None:
	"""Main entry point for the script"""
	parser = argparse.ArgumentParser(
		description="lmcat - list tree and content, combining .gitignore + .lmignore",
		add_help=False,
	)
	parser.add_argument(
		"-g",
		"--no-include-gitignore",
		action="store_false",
		dest="include_gitignore",
		default=True,
		help="Do not parse .gitignore files, only .lmignore (default: parse them).",
	)
	parser.add_argument(
		"-t",
		"--tree-only",
		action="store_true",
		default=False,
		help="Only print the tree, not the file contents.",
	)
	parser.add_argument(
		"-o",
		"--output",
		action="store",
		default=None,
		help="Output file to write the tree and contents to.",
	)
	parser.add_argument(
		"-h", "--help", action="help", help="Show this help message and exit."
	)

	args, unknown = parser.parse_known_args()

	root_dir = Path(".").resolve()
	config = LMCatConfig.read(root_dir)

	# CLI overrides
	config.include_gitignore = args.include_gitignore
	config.tree_only = args.tree_only

	tree_output, collected_files = walk_and_collect(root_dir, config)

	output: list[str] = []
	output.append("# File Tree")
	output.append("\n```")
	output.extend(tree_output)
	output.append("```\n")

	cwd = Path.cwd()

	# Add file contents if not suppressed
	if not config.tree_only:
		output.append("# File Contents")

		for fpath in collected_files:
			relpath_posix = fpath.relative_to(cwd).as_posix()
			pathspec_start = f'{{ path: "{relpath_posix}" }}'
			pathspec_end = f'{{ end_of_file: "{relpath_posix}" }}'
			output.append("")
			output.append(config.content_divider + pathspec_start)
			with fpath.open("r", encoding="utf-8", errors="ignore") as fobj:
				output.append(fobj.read())
			output.append(config.content_divider + pathspec_end)

	# Write output
	if args.output:
		with open(args.output, "w", encoding="utf-8") as f:
			f.write("\n".join(output))
	else:
		if sys.platform == "win32":
			sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
			sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
	
		print("\n".join(output))


if __name__ == "__main__":
	main()