import argparse
import io
import json
import os
# from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any


# Handle Python 3.11+ vs older Python for TOML parsing
try:
	import tomllib
except ImportError:
	try:
		import tomli as tomllib  # type: ignore
	except ImportError:
		tomllib = None  # type: ignore[assignment]

import igittigitt  # noqa: E402

from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field
from muutils.misc import shorten_numerical_to_str  # noqa: E402


from lmcat.file_stats import FileStats, TokenizerWrapper, TreeEntry, TOKENIZERS_PRESENT


@serializable_dataclass(kw_only=True)
class LMCatConfig(SerializableDataclass):
	"""Configuration dataclass for lmcat

	# Parameters:
	 - `tree_divider: str`
	 - `tree_indent: str`
	 - `tree_file_divider: str`
	 - `content_divider: str`
	 - `include_gitignore: bool`  (default True)
	 - `tree_only: bool`  (default False)
	"""


	content_divider: str = serializable_field(default="``````")
	tree_only: bool = serializable_field(default=False)

	# ignoring
	ignore_patterns: list[str] = serializable_field(default_factory=list)
	ignore_patterns_files: list[Path] = serializable_field(
		default_factory=lambda: [Path(".gitignore"), Path(".lmignore")],
		serialization_fn=lambda x: [p.as_posix() for p in x],
		deserialize_fn=lambda x: [Path(p) for p in x],
	)

	# this file will be imported, and if the functions in it are decorated
	# with one of the `register_*` decorators, they will be added to the functions
	# which can be used in the processing pipeline
	plugins_file: Path = serializable_field(
		default=Path("lmcat_plugins.py"),
		serialization_fn=lambda x: x.as_posix(),
		deserialize_fn=lambda x: Path(x),
	)

	# processing pipeline
	glob_process: dict[str, str] = serializable_field(default_factory=dict)
	decider_process: dict[str, str] = serializable_field(default_factory=dict)

	# tokenization
	tokenizer: str = serializable_field(default="gpt2" if TOKENIZERS_PRESENT else "whitespace-split")
	"Tokenizer to use for tokenizing the output. `gpt2` by default. passed to `tokenizers.Tokenizer.from_pretrained()`. If specified and `tokenizers` not installed, will throw exception. fallback `whitespace-split` used to avoid exception when `tokenizers` not installed."

	# tree formatting
	tree_divider: str = serializable_field(default="│   ")
	tree_file_divider: str = serializable_field(default="├── ")
	tree_indent: str = serializable_field(default=" ")

	@property
	def get_tokenizer_obj(self) -> TokenizerWrapper:
		"""Get the tokenizer object"""
		return TokenizerWrapper(self.tokenizer)

	@classmethod
	def read(cls, root_dir: Path) -> "LMCatConfig":
		"""Attempt to read config from pyproject.toml, lmcat.toml, or lmcat.json."""
		pyproject_path: Path = root_dir / "pyproject.toml"
		lmcat_toml_path: Path = root_dir / "lmcat.toml"
		lmcat_json_path: Path = root_dir / "lmcat.json"


		if sum(int(p.is_file()) for p in (pyproject_path, lmcat_toml_path, lmcat_json_path)) > 1:
			raise ValueError(
				"Multiple configuration files found. Please only use one of pyproject.toml, lmcat.toml, or lmcat.json."
			)

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
		self.root_dir: Path = root_dir
		self.config: LMCatConfig = config
		
		# set up parser
		self.parser: igittigitt.IgnoreParser = igittigitt.IgnoreParser()

		# first from the files
		for ignore_file in self.config.ignore_patterns_files:
			self.parser.parse_rule_files(self.root_dir, filename=ignore_file.name)

		# then from the config itself
		for pattern in self.config.ignore_patterns:
			self.parser.add_rule(pattern)

	def is_ignored(self, path: Path) -> bool:
		"""Check if a path should be ignored"""
		# Never ignore the gitignore/lmignore files themselves
		if path.name in {".gitignore", ".lmignore"}:
			return True
		
		# Use igittigitt's matching
		return self.parser.match(path)


def sorted_entries(directory: Path) -> list[Path]:
	"""Return directory contents sorted: directories first, then files"""
	subdirs: list[Path] = sorted(
		[p for p in directory.iterdir() if p.is_dir()], key=lambda x: x.name
	)
	files: list[Path] = sorted(
		[p for p in directory.iterdir() if p.is_file()], key=lambda x: x.name
	)
	return subdirs + files


def walk_dir(
	directory: Path,
	ignore_handler: IgnoreHandler,
	config: LMCatConfig,
	tokenizer: TokenizerWrapper,
	prefix: str = "",
) -> tuple[list[TreeEntry], list[Path]]:
	"""Recursively walk a directory, building tree lines and collecting file paths"""
	tree_output: list[TreeEntry] = []
	collected_files: list[Path] = []

	entries: list[Path] = sorted_entries(directory)
	for i, entry in enumerate(entries):
		if ignore_handler.is_ignored(entry):
			continue

		is_last: bool = i == len(entries) - 1
		connector: str = (
			config.tree_file_divider
			if not is_last
			else config.tree_file_divider.replace("├", "└")
		)

		if entry.is_dir():
			tree_output.append(TreeEntry(f"{prefix}{connector}{entry.name}", None))
			extension: str = config.tree_divider if not is_last else config.tree_indent
			sub_output: list[TreeEntry]
			sub_files: list[Path]
			sub_output, sub_files = walk_dir(
				directory=entry,
				ignore_handler=ignore_handler,
				config=config,
				tokenizer=tokenizer,
				prefix=prefix + extension,
			)
			tree_output.extend(sub_output)
			collected_files.extend(sub_files)
		else:
			stats: FileStats = FileStats.from_file(entry, tokenizer)
			tree_output.append(TreeEntry(f"{prefix}{connector}{entry.name}", stats))
			collected_files.append(entry)

	return tree_output, collected_files


def format_tree_with_stats(
	entries: list[TreeEntry], show_tokens: bool = False
) -> list[str]:
	"""Format tree entries with aligned statistics

	# Parameters:
	 - `entries : list[TreeEntry]`
		List of tree entries with optional stats
	 - `show_tokens : bool`
		Whether to show token counts

	# Returns:
	 - `list[str]`
		Formatted tree lines with aligned stats
	"""
	# Find max widths for alignment
	max_line_len: int = max(len(entry.line) for entry in entries)
	max_lines: int = max(
		(len(f"{entry.stats.lines:,}") if entry.stats else 0) for entry in entries
	)
	max_chars: int = max(
		(len(f"{entry.stats.chars:,}") if entry.stats else 0) for entry in entries
	)
	max_tokens: int = (
		max(
			(
				len(f"{entry.stats.tokens:,}")
				if entry.stats and entry.stats.tokens
				else 0
			)
			for entry in entries
		)
		if show_tokens
		else 0
	)

	formatted: list[str] = []
	for entry in entries:
		line: str = entry.line.ljust(max_line_len + 2)
		if entry.stats:
			lines_str: str = f"{entry.stats.lines:,}L".rjust(max_lines + 1)
			chars_str: str = f"{entry.stats.chars:,}C".rjust(max_chars + 1)
			stats_str: str = f"[{lines_str} {chars_str}"
			if show_tokens and entry.stats.tokens is not None:
				tokens_str: str = f"{entry.stats.tokens:,}T".rjust(max_tokens + 1)
				stats_str += f" {tokens_str}"
			stats_str += "]"
			formatted.append(f"{line}{stats_str}")
		else:
			formatted.append(line)

	return formatted


def walk_and_collect(
	root_dir: Path,
	config: LMCatConfig,
) -> tuple[list[str], list[Path]]:
	"""Walk filesystem from root_dir and gather tree listing plus file paths"""
	if config is None:
		config = LMCatConfig()


	tokenizer: TokenizerWrapper = config.get_tokenizer_obj()

	ignore_handler = IgnoreHandler(root_dir, config)
	base_name = root_dir.resolve().name

	# Start with root directory name
	tree_output = [TreeEntry(base_name)]

	# Walk the directory tree
	sub_output, sub_files = walk_dir(
		directory=root_dir,
		ignore_handler=ignore_handler,
		config=config,
		tokenizer=tokenizer,
		prefix="",
	)
	tree_output.extend(sub_output)

	# Format tree with stats
	formatted_tree = format_tree_with_stats(
		tree_output, show_tokens=tokenizer is not None
	)

	return formatted_tree, sub_files



def assemble_summary(
	root_dir: Path,
	config: LMCatConfig,
) -> str:
	"""Assemble the summary output and return"""

	tree_output: list[str]
	collected_files: list[Path]
	tree_output, collected_files = walk_and_collect(
		root_dir=root_dir,
		config=config,
	)

	output: list[str] = []
	output.append("# File Tree")
	output.append("\n```")
	output.extend(tree_output)
	output.append("```\n")

	cwd: Path = Path.cwd()

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

	output_joined: str = "\n".join(output)

	stats_dict_ints: dict[str, int] = {
		"files": len(collected_files),
		"lines": len(output),
		"chars": len(output_joined),
	}

	tokenizer: TokenizerWrapper = config.get_tokenizer_obj()

	n_tokens: int = tokenizer.n_tokens(output_joined)
	stats_dict_ints[f"`{tokenizer.name}` tokens"] = n_tokens

	stats_header: list[str] = ["# Stats"]
	for key, val in stats_dict_ints.items():
		val_str: str = str(val)
		val_short: str = shorten_numerical_to_str(val)
		if val_str != val_short:
			stats_header.append(f"- {val} ({val_short}) {key}")
		else:
			stats_header.append(f"- {val} {key}")

	output_complete: str = "\n".join(stats_header) + "\n\n" + output_joined

	return output_complete


def main() -> None:
	"""Main entry point for the script"""
	arg_parser = argparse.ArgumentParser(
		description="lmcat - list tree and content, combining .gitignore + .lmignore",
		add_help=False,
	)
	arg_parser.add_argument(
		"-t",
		"--tree-only",
		action="store_true",
		default=False,
		help="Only print the tree, not the file contents.",
	)
	arg_parser.add_argument(
		"-o",
		"--output",
		action="store",
		default=None,
		help="Output file to write the tree and contents to.",
	)
	arg_parser.add_argument(
		"-h", "--help", action="help", help="Show this help message and exit."
	)
	arg_parser.add_argument(
		"--print-cfg",
		action="store_true",
		default=False,
		help="Print the configuration as json and exit.",
	)

	args: argparse.Namespace = arg_parser.parse_known_args()[0]
	root_dir: Path = Path(".").resolve()
	config: LMCatConfig = LMCatConfig.read(root_dir)

	# CLI overrides
	config.tree_only = args.tree_only
	

	# print cfg and exit if requested
	if args.print_cfg:
		print(json.dumps(config.serialize(), indent="\t"))
		return

	# assemble summary
	summary: str = assemble_summary(root_dir=root_dir, config=config)

	# Write output
	if args.output:
		output_path: Path = Path(args.output)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(summary, encoding="utf-8")
	else:
		if sys.platform == "win32":
			sys.stdout = io.TextIOWrapper(
				sys.stdout.buffer, encoding="utf-8", errors="replace"
			)
			sys.stderr = io.TextIOWrapper(
				sys.stderr.buffer, encoding="utf-8", errors="replace"
			)

		print(summary)



if __name__ == "__main__":
	main()
