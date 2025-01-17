# Stats
- 9 files
- 52 lines
- 205182 (205K) chars
- 77358 (77K) tokens

# File Tree

```
lmcat
├── lmcat
│   ├── __init__.py
│   ├── __main__.py
│   ├── index.html
│   └── lmcat.py
├── tests
│   └── test_lmcat.py
├── README.md
├── example_output.md
├── makefile
├── pyproject.toml
```

# File Contents

``````{ path: "lmcat/__init__.py" }
"""
.. include:: ../README.md
"""

from lmcat.lmcat import main

__all__ = ["main"]

``````{ end_of_file: "lmcat/__init__.py" }

``````{ path: "lmcat/__main__.py" }
from lmcat import main

if __name__ == "__main__":
	main()

``````{ end_of_file: "lmcat/__main__.py" }

``````{ path: "lmcat/index.html" }
<!DOCTYPE html>
<html>
<head>
    <title>Minimal Git Browser</title>
    <script src="https://unpkg.com/@isomorphic-git/lightning-fs@4.6.0/dist/lightning-fs.min.js"></script>
    <script src="https://unpkg.com/isomorphic-git@1.24.0/index.umd.min.js"></script>
    <script src="https://unpkg.com/isomorphic-git@1.24.0/http/web/index.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
</head>
<body>
    <div style="margin: 20px;">
        <label for="url">Repository URL:</label>
        <input id="url" type="text" value="https://github.com/mivanit/lmcat" style="width: 300px; margin-right: 10px;">
        <button onclick="process()">Process</button>
        <div id="status" style="margin-top: 10px; color: gray;"></div>
    </div>
    <pre id="output" style="margin: 20px; padding: 10px; background: #f5f5f5;"></pre>

    <script>
        let fs, pfs, pyodide;

        // Debug function to check available objects
        function debugGlobals() {
            console.log('Available globals:');
            console.log('git:', typeof window.git);
            console.log('http:', typeof window.http);
            console.log('GitHttp:', typeof window.GitHttp);
            console.log('GitHttpClient:', typeof window.GitHttpClient);
        }

        async function init() {
            try {
                fs = new LightningFS('fs');
                pfs = fs.promises;
                
                // Initialize Pyodide
                pyodide = await loadPyodide();
                await pyodide.runPythonAsync(`
                    import os
                    def list_files(path):
                        try:
                            return str(list(os.listdir(path)))
                        except Exception as e:
                            return str(e)
                    def some_string():
                        return 'Hello from Python!'
                `);

                // Debug available objects
                debugGlobals();
                
                document.getElementById('status').textContent = 'Initialized successfully';
            } catch (err) {
                console.error('Init error:', err);
                document.getElementById('status').textContent = 'Initialization failed: ' + err.message;
            }
        }

        async function process() {
            const output = document.getElementById('output');
            const status = document.getElementById('status');
            status.textContent = 'Processing...';
            
            try {
                const dir = '/repo';
                await pfs.rmdir(dir, { recursive: true }).catch(() => {});
                await pfs.mkdir(dir).catch(() => {});
                
                                // Use the GitHttp object that's available globally
                if (!window.GitHttp) {
                    throw new Error('GitHttp is not available');
                }

                status.textContent = 'Cloning repository...';
                
                await git.clone({
                    fs,
                    http: GitHttp,
                    dir,
                    url: document.getElementById('url').value,
                    depth: 1,
                    singleBranch: true,
                    corsProxy: 'https://cors.isomorphic-git.org'
                });

                status.textContent = 'Listing files...';
                console.log('Listing files...');
                const result = await pyodide.runPythonAsync(`list_files('.')`);
                // const result = await pyodide.runPythonAsync(`some_string()`);
                console.log('result:', result);
                output.textContent = JSON.stringify(result, null, 2);
                status.textContent = 'Done!';
            } catch (err) {
                console.error('Process error:', err);
                status.textContent = 'Error: ' + err.message;
                output.textContent = err.stack || err.message;
            }
        }

        // Initialize on page load
        init();
    </script>
</body>
</html>
``````{ end_of_file: "lmcat/index.html" }

``````{ path: "lmcat/lmcat.py" }
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
		import tomli as tomllib # type: ignore
	except ImportError:
		tomllib = None # type: ignore[assignment]


# tokenizers (optional dep)
TOKENIZERS_PRESENT: bool = False
try:
    import tokenizers
    TOKENIZERS_PRESENT = True
except ImportError:
    pass

import igittigitt
from muutils.misc import shorten_numerical_to_str


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
	prefix: str = "",
) -> tuple[list[str], list[Path]]:
	"""Recursively walk a directory, building tree lines and collecting file paths"""
	tree_output: list[str] = []
	collected_files: list[Path] = []

	entries: list[Path] = sorted_entries(directory)
	for i, entry in enumerate(entries):
		if ignore_handler.is_ignored(entry):
			continue

		is_last: bool = i == len(entries) - 1
		connector: str = (
			config.file_divider
			if not is_last
			else config.file_divider.replace("├", "└")
		)

		if entry.is_dir():
			tree_output.append(f"{prefix}{connector}{entry.name}")
			extension: str = config.tree_divider if not is_last else config.indent
			sub_output: list[str]
			sub_files: list[Path]
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
	sub_output: list[str]
	sub_files: list[Path]
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

	output_joined: str = "\n".join(output)

	stats_dict_ints: dict[str, int] = {
		"files": len(collected_files),
		"lines": len(output),
		"chars": len(output_joined),
	}
	if TOKENIZERS_PRESENT:
		tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_pretrained("gpt2")
		n_tokens: int = len(tokenizer.encode(output_joined).tokens)
		stats_dict_ints["tokens"] = n_tokens


	stats_header: list[str] = ["# Stats"]
	for key, val in stats_dict_ints.items():
		val_str: str = str(val)
		val_short: str = shorten_numerical_to_str(val)
		if val_str != val_short:
			stats_header.append(f"- {val} ({val_short}) {key}")
		else:
			stats_header.append(f"- {val} {key}")

	output_complete: str = "\n".join(stats_header) + "\n\n" + output_joined

	# Write output
	if args.output:
		Path(args.output).parent.mkdir(parents=True, exist_ok=True)
		with open(args.output, "w", encoding="utf-8") as f:
			f.write(output_complete)
	else:
		if sys.platform == "win32":
			sys.stdout = io.TextIOWrapper(
				sys.stdout.buffer, encoding="utf-8", errors="replace"
			)
			sys.stderr = io.TextIOWrapper(
				sys.stderr.buffer, encoding="utf-8", errors="replace"
			)

		print(output_complete)


if __name__ == "__main__":
	main()

``````{ end_of_file: "lmcat/lmcat.py" }

``````{ path: "tests/test_lmcat.py" }
import sys
import os
import shutil
import subprocess
from pathlib import Path

from lmcat.lmcat import (
	LMCatConfig,
	IgnoreHandler,
	walk_dir,
	walk_and_collect,
)

# We will store all test directories under this path:
TEMP_PATH: Path = Path("tests/_temp")


def ensure_clean_dir(dirpath: Path) -> None:
	"""Remove `dirpath` if it exists, then re-create it."""
	if dirpath.is_dir():
		shutil.rmtree(dirpath)
	dirpath.mkdir(parents=True, exist_ok=True)


# Test LMCatConfig - these tests remain largely unchanged
def test_lmcat_config_defaults():
	config = LMCatConfig()
	assert config.tree_divider == "│   "
	assert config.indent == " "
	assert config.file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_partial():
	data = {"tree_divider": "|---"}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "|---"
	assert config.indent == " "
	assert config.file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_all():
	data = {
		"tree_divider": "XX",
		"indent": "YY",
		"file_divider": "ZZ",
		"content_divider": "@@@",
	}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "XX"
	assert config.indent == "YY"
	assert config.file_divider == "ZZ"
	assert config.content_divider == "@@@"


# Test IgnoreHandler class
def test_ignore_handler_init():
	"""Test basic initialization of IgnoreHandler"""
	test_dir = TEMP_PATH / "test_ignore_handler_init"
	ensure_clean_dir(test_dir)
	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)
	assert handler.root_dir == test_dir
	assert handler.config == config


def test_ignore_handler_basic_ignore():
	"""Test basic ignore patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_basic_ignore"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / "file2.log").write_text("content2")
	(test_dir / ".lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches
	assert not handler.is_ignored(test_dir / "file1.txt")
	assert handler.is_ignored(test_dir / "file2.log")


def test_ignore_handler_directory_patterns():
	"""Test directory ignore patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_directory"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.txt").write_text("content2")
	(test_dir / ".lmignore").write_text("subdir2/\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches
	assert not handler.is_ignored(test_dir / "subdir1")
	assert handler.is_ignored(test_dir / "subdir2")
	assert not handler.is_ignored(test_dir / "subdir1/file1.txt")
	assert handler.is_ignored(test_dir / "subdir2/file2.txt")


def test_ignore_handler_negation():
	"""Test negation patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_negation"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / "file2.txt").write_text("content2")
	(test_dir / ".gitignore").write_text("*.txt\n")
	(test_dir / ".lmignore").write_text("!file2.txt\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches - file2.txt should be unignored by the negation
	assert handler.is_ignored(test_dir / "file1.txt")
	assert not handler.is_ignored(test_dir / "file2.txt")


def test_ignore_handler_nested_ignore_files():
	"""Test nested ignore files with different patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_nested"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir").mkdir()
	(test_dir / "subdir/file1.txt").write_text("content1")
	(test_dir / "subdir/file2.log").write_text("content2")

	# Root ignores .txt, subdir ignores .log
	(test_dir / ".lmignore").write_text("*.txt\n")
	(test_dir / "subdir/.lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test both patterns are active
	assert handler.is_ignored(test_dir / "subdir/file1.txt")
	assert handler.is_ignored(test_dir / "subdir/file2.log")


def test_ignore_handler_gitignore_disabled():
	"""Test that gitignore patterns are ignored when disabled"""
	test_dir = TEMP_PATH / "test_ignore_handler_gitignore_disabled"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / ".gitignore").write_text("*.txt\n")

	config = LMCatConfig(include_gitignore=False)
	handler = IgnoreHandler(test_dir, config)

	# File should not be ignored since gitignore is disabled
	assert not handler.is_ignored(test_dir / "file1.txt")


# Test walking functions with new IgnoreHandler
def test_walk_dir_basic():
	"""Test basic directory walking with no ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_dir_basic"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.txt").write_text("content2")
	(test_dir / "file3.txt").write_text("content3")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	tree_output, files = walk_dir(test_dir, handler, config)
	joined_output = "\n".join(tree_output)

	# Check output contains all entries
	assert "subdir1" in joined_output
	assert "subdir2" in joined_output
	assert "file1.txt" in joined_output
	assert "file2.txt" in joined_output
	assert "file3.txt" in joined_output

	# Check collected files
	assert len(files) == 3
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file2.txt", "file3.txt"}


def test_walk_dir_with_ignore():
	"""Test directory walking with ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_dir_with_ignore"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.log").write_text("content2")
	(test_dir / "file3.txt").write_text("content3")

	# Ignore .log files
	(test_dir / ".lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	tree_output, files = walk_dir(test_dir, handler, config)
	joined_output = "\n".join(tree_output)

	# Check output excludes .log file
	assert "file2.log" not in joined_output
	assert "file1.txt" in joined_output
	assert "file3.txt" in joined_output

	# Check collected files
	assert len(files) == 2
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file3.txt"}


def test_walk_and_collect_complex():
	"""Test full directory walking with multiple ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_and_collect_complex"
	ensure_clean_dir(test_dir)

	# Create complex directory structure
	(test_dir / "subdir1/nested").mkdir(parents=True)
	(test_dir / "subdir2/nested").mkdir(parents=True)
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir1/nested/file2.log").write_text("content2")
	(test_dir / "subdir2/file3.txt").write_text("content3")
	(test_dir / "subdir2/nested/file4.log").write_text("content4")

	# Root ignores .log files
	(test_dir / ".lmignore").write_text("*.log\n")
	# subdir2 ignores nested dir
	(test_dir / "subdir2/.lmignore").write_text("nested/\n")

	config = LMCatConfig()
	tree_output, files = walk_and_collect(test_dir, config)
	joined_output = "\n".join(tree_output)

	# Check correct files are excluded
	assert "file1.txt" in joined_output
	assert "file2.log" not in joined_output
	assert "file3.txt" in joined_output
	assert "file4.log" not in joined_output
	assert "nested" not in joined_output.split("\n")[-5:]  # Check last few lines

	# Check collected files
	assert len(files) == 2
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file3.txt"}


# Test CLI functionality
def test_cli_output_file():
	"""Test writing output to a file"""
	test_dir = TEMP_PATH / "test_cli_output_file"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	output_file = test_dir / "output.md"

	original_cwd = os.getcwd()
	try:
		os.chdir(test_dir)
		subprocess.run(
			["uv", "run", "python", "-m", "lmcat", "--output", str(output_file)],
			check=True,
		)

		# Check output file exists and contains expected content
		assert output_file.is_file()
		content = output_file.read_text()
		assert "# File Tree" in content
		assert "file1.txt" in content
		assert "content1" in content
	except subprocess.CalledProcessError as e:
		print(f"{e = }", file=sys.stderr)
		print(e.stdout, file=sys.stderr)
		print(e.stderr, file=sys.stderr)
		raise e
	finally:
		os.chdir(original_cwd)


def test_cli_tree_only():
	"""Test --tree-only option"""
	test_dir = TEMP_PATH / "test_cli_tree_only"
	ensure_clean_dir(test_dir)

	# Create test file
	(test_dir / "file1.txt").write_text("content1")

	original_cwd = os.getcwd()
	try:
		os.chdir(test_dir)
		result = subprocess.run(
			["uv", "run", "python", "-m", "lmcat", "--tree-only"],
			capture_output=True,
			text=True,
			check=True,
		)

		# Check output has tree but not content
		assert "# File Tree" in result.stdout
		assert "file1.txt" in result.stdout
		assert "# File Contents" not in result.stdout
		assert "content1" not in result.stdout
	except subprocess.CalledProcessError as e:
		print(f"{e = }", file=sys.stderr)
		print(e.stdout, file=sys.stderr)
		print(e.stderr, file=sys.stderr)
		raise e
	finally:
		os.chdir(original_cwd)

``````{ end_of_file: "tests/test_lmcat.py" }

``````{ path: "README.md" }
# lmcat

A Python tool for concatenating files and directory structures into a single document, perfect for sharing code with language models. It respects `.gitignore` and `.lmignore` patterns and provides configurable output formatting.

## Features

- Creates a tree view of your directory structure
- Includes file contents with clear delimiters
- Respects `.gitignore` patterns (can be disabled)
- Supports custom ignore patterns via `.lmignore`
- Configurable via `pyproject.toml`, `lmcat.toml`, or `lmcat.json`
- Python 3.11+ native, with fallback support for older versions

## Installation

Install from PyPI:

```bash
pip install lmcat
```

## Usage

Basic usage - concatenate current directory:

```bash
python -m lmcat
```

The output will include a directory tree and the contents of each non-ignored file.

### Command Line Options

- `-g`, `--no-include-gitignore`: Ignore `.gitignore` files (they are included by default)
- `-t`, `--tree-only`: Only print the directory tree, not file contents
- `-o`, `--output`: Specify an output file (defaults to stdout)
- `-h`, `--help`: Show help message

### Configuration

lmcat can be configured using any of these files (in order of precedence):

1. `pyproject.toml` (under `[tool.lmcat]`)
2. `lmcat.toml`
3. `lmcat.json`

Configuration options:

```toml
[tool.lmcat]
tree_divider = "│   "    # Used for vertical lines in the tree
indent = "    "          # Used for indentation
file_divider = "├── "    # Used for file/directory entries
content_divider = "``````" # Used to delimit file contents
include_gitignore = true # Whether to respect .gitignore files
tree_only = false       # Whether to only show the tree
```

### Ignore Patterns

lmcat supports two types of ignore files:

1. `.gitignore` - Standard Git ignore patterns (used by default)
2. `.lmignore` - Custom ignore patterns specific to lmcat

`.lmignore` follows the same pattern syntax as `.gitignore`. Patterns in `.lmignore` take precedence over `.gitignore`.

Example `.lmignore`:
```gitignore
# Ignore all .log files
*.log

# Ignore the build directory and its contents
build/

# Un-ignore a specific file (overrides previous patterns)
!important.log
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mivanit/lmcat
cd lmcat
```

2. Set up the development environment:
```bash
make setup
```

This will:
- Create a virtual environment
- Install development dependencies
- Set up pre-commit hooks

### Development Commands

The project uses `make` for common development tasks:

- `make dep`: Install/update dependencies
- `make format`: Format code using ruff and pycln
- `make test`: Run tests
- `make typing`: Run type checks
- `make check`: Run all checks (format, test, typing)
- `make clean`: Clean temporary files
- `make docs`: Generate documentation
- `make build`: Build the package
- `make publish`: Publish to PyPI (maintainers only)

Run `make help` to see all available commands.

### Running Tests

```bash
make test
```

For verbose output:
```bash
VERBOSE=1 make test
```

For test coverage:
```bash
make cov
```


### Roadmap

- better tests, I feel like gitignore/lmignore interaction is broken
- llm summarization and caching of those summaries in `.lmsummary/`
- reasonable defaults for file extensions to ignore
- web interface
``````{ end_of_file: "README.md" }

``````{ path: "example_output.md" }
# File Tree

```
lmcat
├── lmcat
│   ├── __init__.py
│   ├── __main__.py
│   ├── index.html
│   └── lmcat.py
├── tests
│   └── test_lmcat.py
├── README.md
├── example_output.md
├── makefile
├── pyproject.toml
```

# File Contents

``````{ path: "lmcat/__init__.py" }
"""
.. include:: ../README.md
"""

from lmcat.lmcat import main

__all__ = ["main"]

``````{ end_of_file: "lmcat/__init__.py" }

``````{ path: "lmcat/__main__.py" }
from lmcat import main

if __name__ == "__main__":
	main()

``````{ end_of_file: "lmcat/__main__.py" }

``````{ path: "lmcat/index.html" }
<!DOCTYPE html>
<html>
<head>
    <title>Minimal Git Browser</title>
    <script src="https://unpkg.com/@isomorphic-git/lightning-fs@4.6.0/dist/lightning-fs.min.js"></script>
    <script src="https://unpkg.com/isomorphic-git@1.24.0/index.umd.min.js"></script>
    <script src="https://unpkg.com/isomorphic-git@1.24.0/http/web/index.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
</head>
<body>
    <div style="margin: 20px;">
        <label for="url">Repository URL:</label>
        <input id="url" type="text" value="https://github.com/mivanit/lmcat" style="width: 300px; margin-right: 10px;">
        <button onclick="process()">Process</button>
        <div id="status" style="margin-top: 10px; color: gray;"></div>
    </div>
    <pre id="output" style="margin: 20px; padding: 10px; background: #f5f5f5;"></pre>

    <script>
        let fs, pfs, pyodide;

        // Debug function to check available objects
        function debugGlobals() {
            console.log('Available globals:');
            console.log('git:', typeof window.git);
            console.log('http:', typeof window.http);
            console.log('GitHttp:', typeof window.GitHttp);
            console.log('GitHttpClient:', typeof window.GitHttpClient);
        }

        async function init() {
            try {
                fs = new LightningFS('fs');
                pfs = fs.promises;
                
                // Initialize Pyodide
                pyodide = await loadPyodide();
                await pyodide.runPythonAsync(`
                    import os
                    def list_files(path):
                        try:
                            return str(list(os.listdir(path)))
                        except Exception as e:
                            return str(e)
                    def some_string():
                        return 'Hello from Python!'
                `);

                // Debug available objects
                debugGlobals();
                
                document.getElementById('status').textContent = 'Initialized successfully';
            } catch (err) {
                console.error('Init error:', err);
                document.getElementById('status').textContent = 'Initialization failed: ' + err.message;
            }
        }

        async function process() {
            const output = document.getElementById('output');
            const status = document.getElementById('status');
            status.textContent = 'Processing...';
            
            try {
                const dir = '/repo';
                await pfs.rmdir(dir, { recursive: true }).catch(() => {});
                await pfs.mkdir(dir).catch(() => {});
                
                                // Use the GitHttp object that's available globally
                if (!window.GitHttp) {
                    throw new Error('GitHttp is not available');
                }

                status.textContent = 'Cloning repository...';
                
                await git.clone({
                    fs,
                    http: GitHttp,
                    dir,
                    url: document.getElementById('url').value,
                    depth: 1,
                    singleBranch: true,
                    corsProxy: 'https://cors.isomorphic-git.org'
                });

                status.textContent = 'Listing files...';
                console.log('Listing files...');
                const result = await pyodide.runPythonAsync(`list_files('.')`);
                // const result = await pyodide.runPythonAsync(`some_string()`);
                console.log('result:', result);
                output.textContent = JSON.stringify(result, null, 2);
                status.textContent = 'Done!';
            } catch (err) {
                console.error('Process error:', err);
                status.textContent = 'Error: ' + err.message;
                output.textContent = err.stack || err.message;
            }
        }

        // Initialize on page load
        init();
    </script>
</body>
</html>
``````{ end_of_file: "lmcat/index.html" }

``````{ path: "lmcat/lmcat.py" }
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
		import tomli as tomllib # type: ignore
	except ImportError:
		tomllib = None # type: ignore[assignment]

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
	prefix: str = "",
) -> tuple[list[str], list[Path]]:
	"""Recursively walk a directory, building tree lines and collecting file paths"""
	tree_output: list[str] = []
	collected_files: list[Path] = []

	entries: list[Path] = sorted_entries(directory)
	for i, entry in enumerate(entries):
		if ignore_handler.is_ignored(entry):
			continue

		is_last: bool = i == len(entries) - 1
		connector: str = (
			config.file_divider
			if not is_last
			else config.file_divider.replace("├", "└")
		)

		if entry.is_dir():
			tree_output.append(f"{prefix}{connector}{entry.name}")
			extension: str = config.tree_divider if not is_last else config.indent
			sub_output: list[str]
			sub_files: list[Path]
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
	sub_output: list[str]
	sub_files: list[Path]
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
		Path(args.output).parent.mkdir(parents=True, exist_ok=True)
		with open(args.output, "w", encoding="utf-8") as f:
			f.write("\n".join(output))
	else:
		if sys.platform == "win32":
			sys.stdout = io.TextIOWrapper(
				sys.stdout.buffer, encoding="utf-8", errors="replace"
			)
			sys.stderr = io.TextIOWrapper(
				sys.stderr.buffer, encoding="utf-8", errors="replace"
			)

		print("\n".join(output))


if __name__ == "__main__":
	main()

``````{ end_of_file: "lmcat/lmcat.py" }

``````{ path: "tests/test_lmcat.py" }
import sys
import os
import shutil
import subprocess
from pathlib import Path

from lmcat.lmcat import (
	LMCatConfig,
	IgnoreHandler,
	walk_dir,
	walk_and_collect,
)

# We will store all test directories under this path:
TEMP_PATH: Path = Path("tests/_temp")


def ensure_clean_dir(dirpath: Path) -> None:
	"""Remove `dirpath` if it exists, then re-create it."""
	if dirpath.is_dir():
		shutil.rmtree(dirpath)
	dirpath.mkdir(parents=True, exist_ok=True)


# Test LMCatConfig - these tests remain largely unchanged
def test_lmcat_config_defaults():
	config = LMCatConfig()
	assert config.tree_divider == "│   "
	assert config.indent == " "
	assert config.file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_partial():
	data = {"tree_divider": "|---"}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "|---"
	assert config.indent == " "
	assert config.file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_all():
	data = {
		"tree_divider": "XX",
		"indent": "YY",
		"file_divider": "ZZ",
		"content_divider": "@@@",
	}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "XX"
	assert config.indent == "YY"
	assert config.file_divider == "ZZ"
	assert config.content_divider == "@@@"


# Test IgnoreHandler class
def test_ignore_handler_init():
	"""Test basic initialization of IgnoreHandler"""
	test_dir = TEMP_PATH / "test_ignore_handler_init"
	ensure_clean_dir(test_dir)
	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)
	assert handler.root_dir == test_dir
	assert handler.config == config


def test_ignore_handler_basic_ignore():
	"""Test basic ignore patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_basic_ignore"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / "file2.log").write_text("content2")
	(test_dir / ".lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches
	assert not handler.is_ignored(test_dir / "file1.txt")
	assert handler.is_ignored(test_dir / "file2.log")


def test_ignore_handler_directory_patterns():
	"""Test directory ignore patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_directory"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.txt").write_text("content2")
	(test_dir / ".lmignore").write_text("subdir2/\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches
	assert not handler.is_ignored(test_dir / "subdir1")
	assert handler.is_ignored(test_dir / "subdir2")
	assert not handler.is_ignored(test_dir / "subdir1/file1.txt")
	assert handler.is_ignored(test_dir / "subdir2/file2.txt")


def test_ignore_handler_negation():
	"""Test negation patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_negation"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / "file2.txt").write_text("content2")
	(test_dir / ".gitignore").write_text("*.txt\n")
	(test_dir / ".lmignore").write_text("!file2.txt\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches - file2.txt should be unignored by the negation
	assert handler.is_ignored(test_dir / "file1.txt")
	assert not handler.is_ignored(test_dir / "file2.txt")


def test_ignore_handler_nested_ignore_files():
	"""Test nested ignore files with different patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_nested"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir").mkdir()
	(test_dir / "subdir/file1.txt").write_text("content1")
	(test_dir / "subdir/file2.log").write_text("content2")

	# Root ignores .txt, subdir ignores .log
	(test_dir / ".lmignore").write_text("*.txt\n")
	(test_dir / "subdir/.lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test both patterns are active
	assert handler.is_ignored(test_dir / "subdir/file1.txt")
	assert handler.is_ignored(test_dir / "subdir/file2.log")


def test_ignore_handler_gitignore_disabled():
	"""Test that gitignore patterns are ignored when disabled"""
	test_dir = TEMP_PATH / "test_ignore_handler_gitignore_disabled"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / ".gitignore").write_text("*.txt\n")

	config = LMCatConfig(include_gitignore=False)
	handler = IgnoreHandler(test_dir, config)

	# File should not be ignored since gitignore is disabled
	assert not handler.is_ignored(test_dir / "file1.txt")


# Test walking functions with new IgnoreHandler
def test_walk_dir_basic():
	"""Test basic directory walking with no ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_dir_basic"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.txt").write_text("content2")
	(test_dir / "file3.txt").write_text("content3")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	tree_output, files = walk_dir(test_dir, handler, config)
	joined_output = "\n".join(tree_output)

	# Check output contains all entries
	assert "subdir1" in joined_output
	assert "subdir2" in joined_output
	assert "file1.txt" in joined_output
	assert "file2.txt" in joined_output
	assert "file3.txt" in joined_output

	# Check collected files
	assert len(files) == 3
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file2.txt", "file3.txt"}


def test_walk_dir_with_ignore():
	"""Test directory walking with ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_dir_with_ignore"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.log").write_text("content2")
	(test_dir / "file3.txt").write_text("content3")

	# Ignore .log files
	(test_dir / ".lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	tree_output, files = walk_dir(test_dir, handler, config)
	joined_output = "\n".join(tree_output)

	# Check output excludes .log file
	assert "file2.log" not in joined_output
	assert "file1.txt" in joined_output
	assert "file3.txt" in joined_output

	# Check collected files
	assert len(files) == 2
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file3.txt"}


def test_walk_and_collect_complex():
	"""Test full directory walking with multiple ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_and_collect_complex"
	ensure_clean_dir(test_dir)

	# Create complex directory structure
	(test_dir / "subdir1/nested").mkdir(parents=True)
	(test_dir / "subdir2/nested").mkdir(parents=True)
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir1/nested/file2.log").write_text("content2")
	(test_dir / "subdir2/file3.txt").write_text("content3")
	(test_dir / "subdir2/nested/file4.log").write_text("content4")

	# Root ignores .log files
	(test_dir / ".lmignore").write_text("*.log\n")
	# subdir2 ignores nested dir
	(test_dir / "subdir2/.lmignore").write_text("nested/\n")

	config = LMCatConfig()
	tree_output, files = walk_and_collect(test_dir, config)
	joined_output = "\n".join(tree_output)

	# Check correct files are excluded
	assert "file1.txt" in joined_output
	assert "file2.log" not in joined_output
	assert "file3.txt" in joined_output
	assert "file4.log" not in joined_output
	assert "nested" not in joined_output.split("\n")[-5:]  # Check last few lines

	# Check collected files
	assert len(files) == 2
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file3.txt"}


# Test CLI functionality
def test_cli_output_file():
	"""Test writing output to a file"""
	test_dir = TEMP_PATH / "test_cli_output_file"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	output_file = test_dir / "output.md"

	original_cwd = os.getcwd()
	try:
		os.chdir(test_dir)
		subprocess.run(
			["uv", "run", "python", "-m", "lmcat", "--output", str(output_file)],
			check=True,
		)

		# Check output file exists and contains expected content
		assert output_file.is_file()
		content = output_file.read_text()
		assert "# File Tree" in content
		assert "file1.txt" in content
		assert "content1" in content
	except subprocess.CalledProcessError as e:
		print(f"{e = }", file=sys.stderr)
		print(e.stdout, file=sys.stderr)
		print(e.stderr, file=sys.stderr)
		raise e
	finally:
		os.chdir(original_cwd)


def test_cli_tree_only():
	"""Test --tree-only option"""
	test_dir = TEMP_PATH / "test_cli_tree_only"
	ensure_clean_dir(test_dir)

	# Create test file
	(test_dir / "file1.txt").write_text("content1")

	original_cwd = os.getcwd()
	try:
		os.chdir(test_dir)
		result = subprocess.run(
			["uv", "run", "python", "-m", "lmcat", "--tree-only"],
			capture_output=True,
			text=True,
			check=True,
		)

		# Check output has tree but not content
		assert "# File Tree" in result.stdout
		assert "file1.txt" in result.stdout
		assert "# File Contents" not in result.stdout
		assert "content1" not in result.stdout
	except subprocess.CalledProcessError as e:
		print(f"{e = }", file=sys.stderr)
		print(e.stdout, file=sys.stderr)
		print(e.stderr, file=sys.stderr)
		raise e
	finally:
		os.chdir(original_cwd)

``````{ end_of_file: "tests/test_lmcat.py" }

``````{ path: "README.md" }
# lmcat

A Python tool for concatenating files and directory structures into a single document, perfect for sharing code with language models. It respects `.gitignore` and `.lmignore` patterns and provides configurable output formatting.

## Features

- Creates a tree view of your directory structure
- Includes file contents with clear delimiters
- Respects `.gitignore` patterns (can be disabled)
- Supports custom ignore patterns via `.lmignore`
- Configurable via `pyproject.toml`, `lmcat.toml`, or `lmcat.json`
- Python 3.11+ native, with fallback support for older versions

## Installation

Install from PyPI:

```bash
pip install lmcat
```

## Usage

Basic usage - concatenate current directory:

```bash
python -m lmcat
```

The output will include a directory tree and the contents of each non-ignored file.

### Command Line Options

- `-g`, `--no-include-gitignore`: Ignore `.gitignore` files (they are included by default)
- `-t`, `--tree-only`: Only print the directory tree, not file contents
- `-o`, `--output`: Specify an output file (defaults to stdout)
- `-h`, `--help`: Show help message

### Configuration

lmcat can be configured using any of these files (in order of precedence):

1. `pyproject.toml` (under `[tool.lmcat]`)
2. `lmcat.toml`
3. `lmcat.json`

Configuration options:

```toml
[tool.lmcat]
tree_divider = "│   "    # Used for vertical lines in the tree
indent = "    "          # Used for indentation
file_divider = "├── "    # Used for file/directory entries
content_divider = "``````" # Used to delimit file contents
include_gitignore = true # Whether to respect .gitignore files
tree_only = false       # Whether to only show the tree
```

### Ignore Patterns

lmcat supports two types of ignore files:

1. `.gitignore` - Standard Git ignore patterns (used by default)
2. `.lmignore` - Custom ignore patterns specific to lmcat

`.lmignore` follows the same pattern syntax as `.gitignore`. Patterns in `.lmignore` take precedence over `.gitignore`.

Example `.lmignore`:
```gitignore
# Ignore all .log files
*.log

# Ignore the build directory and its contents
build/

# Un-ignore a specific file (overrides previous patterns)
!important.log
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mivanit/lmcat
cd lmcat
```

2. Set up the development environment:
```bash
make setup
```

This will:
- Create a virtual environment
- Install development dependencies
- Set up pre-commit hooks

### Development Commands

The project uses `make` for common development tasks:

- `make dep`: Install/update dependencies
- `make format`: Format code using ruff and pycln
- `make test`: Run tests
- `make typing`: Run type checks
- `make check`: Run all checks (format, test, typing)
- `make clean`: Clean temporary files
- `make docs`: Generate documentation
- `make build`: Build the package
- `make publish`: Publish to PyPI (maintainers only)

Run `make help` to see all available commands.

### Running Tests

```bash
make test
```

For verbose output:
```bash
VERBOSE=1 make test
```

For test coverage:
```bash
make cov
```


### Roadmap

- better tests, I feel like gitignore/lmignore interaction is broken
- llm summarization and caching of those summaries in `.lmsummary/`
- reasonable defaults for file extensions to ignore
- web interface
``````{ end_of_file: "README.md" }

``````{ path: "example_output.md" }
# File Tree

```
lmcat
├── lmcat
│   ├── __init__.py
│   ├── __main__.py
│   ├── index.html
│   └── lmcat.py
├── tests
│   └── test_lmcat.py
├── README.md
├── example_output.md
├── makefile
├── pyproject.toml
```

# File Contents

``````{ path: "lmcat/__init__.py" }
"""
.. include:: ../README.md
"""

from lmcat.lmcat import main

__all__ = ["main"]

``````{ end_of_file: "lmcat/__init__.py" }

``````{ path: "lmcat/__main__.py" }
from lmcat import main

if __name__ == "__main__":
	main()

``````{ end_of_file: "lmcat/__main__.py" }

``````{ path: "lmcat/index.html" }
<!DOCTYPE html>
<html>
<head>
    <title>Minimal Git Browser</title>
    <script src="https://unpkg.com/@isomorphic-git/lightning-fs@4.6.0/dist/lightning-fs.min.js"></script>
    <script src="https://unpkg.com/isomorphic-git@1.24.0/index.umd.min.js"></script>
    <script src="https://unpkg.com/isomorphic-git@1.24.0/http/web/index.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
</head>
<body>
    <div style="margin: 20px;">
        <label for="url">Repository URL:</label>
        <input id="url" type="text" value="https://github.com/mivanit/lmcat" style="width: 300px; margin-right: 10px;">
        <button onclick="process()">Process</button>
        <div id="status" style="margin-top: 10px; color: gray;"></div>
    </div>
    <pre id="output" style="margin: 20px; padding: 10px; background: #f5f5f5;"></pre>

    <script>
        let fs, pfs, pyodide;

        // Debug function to check available objects
        function debugGlobals() {
            console.log('Available globals:');
            console.log('git:', typeof window.git);
            console.log('http:', typeof window.http);
            console.log('GitHttp:', typeof window.GitHttp);
            console.log('GitHttpClient:', typeof window.GitHttpClient);
        }

        async function init() {
            try {
                fs = new LightningFS('fs');
                pfs = fs.promises;
                
                // Initialize Pyodide
                pyodide = await loadPyodide();
                await pyodide.runPythonAsync(`
                    import os
                    def list_files(path):
                        try:
                            return str(list(os.listdir(path)))
                        except Exception as e:
                            return str(e)
                    def some_string():
                        return 'Hello from Python!'
                `);

                // Debug available objects
                debugGlobals();
                
                document.getElementById('status').textContent = 'Initialized successfully';
            } catch (err) {
                console.error('Init error:', err);
                document.getElementById('status').textContent = 'Initialization failed: ' + err.message;
            }
        }

        async function process() {
            const output = document.getElementById('output');
            const status = document.getElementById('status');
            status.textContent = 'Processing...';
            
            try {
                const dir = '/repo';
                await pfs.rmdir(dir, { recursive: true }).catch(() => {});
                await pfs.mkdir(dir).catch(() => {});
                
                                // Use the GitHttp object that's available globally
                if (!window.GitHttp) {
                    throw new Error('GitHttp is not available');
                }

                status.textContent = 'Cloning repository...';
                
                await git.clone({
                    fs,
                    http: GitHttp,
                    dir,
                    url: document.getElementById('url').value,
                    depth: 1,
                    singleBranch: true,
                    corsProxy: 'https://cors.isomorphic-git.org'
                });

                status.textContent = 'Listing files...';
                console.log('Listing files...');
                const result = await pyodide.runPythonAsync(`list_files('.')`);
                // const result = await pyodide.runPythonAsync(`some_string()`);
                console.log('result:', result);
                output.textContent = JSON.stringify(result, null, 2);
                status.textContent = 'Done!';
            } catch (err) {
                console.error('Process error:', err);
                status.textContent = 'Error: ' + err.message;
                output.textContent = err.stack || err.message;
            }
        }

        // Initialize on page load
        init();
    </script>
</body>
</html>
``````{ end_of_file: "lmcat/index.html" }

``````{ path: "lmcat/lmcat.py" }
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
	prefix: str = "",
) -> tuple[list[str], list[Path]]:
	"""Recursively walk a directory, building tree lines and collecting file paths"""
	tree_output: list[str] = []
	collected_files: list[Path] = []

	entries: list[Path] = sorted_entries(directory)
	for i, entry in enumerate(entries):
		if ignore_handler.is_ignored(entry):
			continue

		is_last: bool = i == len(entries) - 1
		connector: str = (
			config.file_divider
			if not is_last
			else config.file_divider.replace("├", "└")
		)

		if entry.is_dir():
			tree_output.append(f"{prefix}{connector}{entry.name}")
			extension: str = config.tree_divider if not is_last else config.indent
			sub_output: list[str]
			sub_files: list[Path]
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
	sub_output: list[str]
	sub_files: list[Path]
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
		Path(args.output).parent.mkdir(parents=True, exist_ok=True)
		with open(args.output, "w", encoding="utf-8") as f:
			f.write("\n".join(output))
	else:
		if sys.platform == "win32":
			sys.stdout = io.TextIOWrapper(
				sys.stdout.buffer, encoding="utf-8", errors="replace"
			)
			sys.stderr = io.TextIOWrapper(
				sys.stderr.buffer, encoding="utf-8", errors="replace"
			)

		print("\n".join(output))


if __name__ == "__main__":
	main()

``````{ end_of_file: "lmcat/lmcat.py" }

``````{ path: "tests/test_lmcat.py" }
import sys
import os
import shutil
import subprocess
from pathlib import Path

from lmcat.lmcat import (
	LMCatConfig,
	IgnoreHandler,
	walk_dir,
	walk_and_collect,
)

# We will store all test directories under this path:
TEMP_PATH: Path = Path("tests/_temp")


def ensure_clean_dir(dirpath: Path) -> None:
	"""Remove `dirpath` if it exists, then re-create it."""
	if dirpath.is_dir():
		shutil.rmtree(dirpath)
	dirpath.mkdir(parents=True, exist_ok=True)


# Test LMCatConfig - these tests remain largely unchanged
def test_lmcat_config_defaults():
	config = LMCatConfig()
	assert config.tree_divider == "│   "
	assert config.indent == " "
	assert config.file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_partial():
	data = {"tree_divider": "|---"}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "|---"
	assert config.indent == " "
	assert config.file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_all():
	data = {
		"tree_divider": "XX",
		"indent": "YY",
		"file_divider": "ZZ",
		"content_divider": "@@@",
	}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "XX"
	assert config.indent == "YY"
	assert config.file_divider == "ZZ"
	assert config.content_divider == "@@@"


# Test IgnoreHandler class
def test_ignore_handler_init():
	"""Test basic initialization of IgnoreHandler"""
	test_dir = TEMP_PATH / "test_ignore_handler_init"
	ensure_clean_dir(test_dir)
	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)
	assert handler.root_dir == test_dir
	assert handler.config == config


def test_ignore_handler_basic_ignore():
	"""Test basic ignore patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_basic_ignore"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / "file2.log").write_text("content2")
	(test_dir / ".lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches
	assert not handler.is_ignored(test_dir / "file1.txt")
	assert handler.is_ignored(test_dir / "file2.log")


def test_ignore_handler_directory_patterns():
	"""Test directory ignore patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_directory"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.txt").write_text("content2")
	(test_dir / ".lmignore").write_text("subdir2/\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches
	assert not handler.is_ignored(test_dir / "subdir1")
	assert handler.is_ignored(test_dir / "subdir2")
	assert not handler.is_ignored(test_dir / "subdir1/file1.txt")
	assert handler.is_ignored(test_dir / "subdir2/file2.txt")


def test_ignore_handler_negation():
	"""Test negation patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_negation"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / "file2.txt").write_text("content2")
	(test_dir / ".gitignore").write_text("*.txt\n")
	(test_dir / ".lmignore").write_text("!file2.txt\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches - file2.txt should be unignored by the negation
	assert handler.is_ignored(test_dir / "file1.txt")
	assert not handler.is_ignored(test_dir / "file2.txt")


def test_ignore_handler_nested_ignore_files():
	"""Test nested ignore files with different patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_nested"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir").mkdir()
	(test_dir / "subdir/file1.txt").write_text("content1")
	(test_dir / "subdir/file2.log").write_text("content2")

	# Root ignores .txt, subdir ignores .log
	(test_dir / ".lmignore").write_text("*.txt\n")
	(test_dir / "subdir/.lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test both patterns are active
	assert handler.is_ignored(test_dir / "subdir/file1.txt")
	assert handler.is_ignored(test_dir / "subdir/file2.log")


def test_ignore_handler_gitignore_disabled():
	"""Test that gitignore patterns are ignored when disabled"""
	test_dir = TEMP_PATH / "test_ignore_handler_gitignore_disabled"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / ".gitignore").write_text("*.txt\n")

	config = LMCatConfig(include_gitignore=False)
	handler = IgnoreHandler(test_dir, config)

	# File should not be ignored since gitignore is disabled
	assert not handler.is_ignored(test_dir / "file1.txt")


# Test walking functions with new IgnoreHandler
def test_walk_dir_basic():
	"""Test basic directory walking with no ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_dir_basic"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.txt").write_text("content2")
	(test_dir / "file3.txt").write_text("content3")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	tree_output, files = walk_dir(test_dir, handler, config)
	joined_output = "\n".join(tree_output)

	# Check output contains all entries
	assert "subdir1" in joined_output
	assert "subdir2" in joined_output
	assert "file1.txt" in joined_output
	assert "file2.txt" in joined_output
	assert "file3.txt" in joined_output

	# Check collected files
	assert len(files) == 3
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file2.txt", "file3.txt"}


def test_walk_dir_with_ignore():
	"""Test directory walking with ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_dir_with_ignore"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.log").write_text("content2")
	(test_dir / "file3.txt").write_text("content3")

	# Ignore .log files
	(test_dir / ".lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	tree_output, files = walk_dir(test_dir, handler, config)
	joined_output = "\n".join(tree_output)

	# Check output excludes .log file
	assert "file2.log" not in joined_output
	assert "file1.txt" in joined_output
	assert "file3.txt" in joined_output

	# Check collected files
	assert len(files) == 2
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file3.txt"}


def test_walk_and_collect_complex():
	"""Test full directory walking with multiple ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_and_collect_complex"
	ensure_clean_dir(test_dir)

	# Create complex directory structure
	(test_dir / "subdir1/nested").mkdir(parents=True)
	(test_dir / "subdir2/nested").mkdir(parents=True)
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir1/nested/file2.log").write_text("content2")
	(test_dir / "subdir2/file3.txt").write_text("content3")
	(test_dir / "subdir2/nested/file4.log").write_text("content4")

	# Root ignores .log files
	(test_dir / ".lmignore").write_text("*.log\n")
	# subdir2 ignores nested dir
	(test_dir / "subdir2/.lmignore").write_text("nested/\n")

	config = LMCatConfig()
	tree_output, files = walk_and_collect(test_dir, config)
	joined_output = "\n".join(tree_output)

	# Check correct files are excluded
	assert "file1.txt" in joined_output
	assert "file2.log" not in joined_output
	assert "file3.txt" in joined_output
	assert "file4.log" not in joined_output
	assert "nested" not in joined_output.split("\n")[-5:]  # Check last few lines

	# Check collected files
	assert len(files) == 2
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file3.txt"}


# Test CLI functionality
def test_cli_output_file():
	"""Test writing output to a file"""
	test_dir = TEMP_PATH / "test_cli_output_file"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	output_file = test_dir / "output.md"

	original_cwd = os.getcwd()
	try:
		os.chdir(test_dir)
		subprocess.run(
			["uv", "run", "python", "-m", "lmcat", "--output", str(output_file)],
			check=True,
		)

		# Check output file exists and contains expected content
		assert output_file.is_file()
		content = output_file.read_text()
		assert "# File Tree" in content
		assert "file1.txt" in content
		assert "content1" in content
	except subprocess.CalledProcessError as e:
		print(f"{e = }", file=sys.stderr)
		print(e.stdout, file=sys.stderr)
		print(e.stderr, file=sys.stderr)
		raise e
	finally:
		os.chdir(original_cwd)


def test_cli_tree_only():
	"""Test --tree-only option"""
	test_dir = TEMP_PATH / "test_cli_tree_only"
	ensure_clean_dir(test_dir)

	# Create test file
	(test_dir / "file1.txt").write_text("content1")

	original_cwd = os.getcwd()
	try:
		os.chdir(test_dir)
		result = subprocess.run(
			["uv", "run", "python", "-m", "lmcat", "--tree-only"],
			capture_output=True,
			text=True,
			check=True,
		)

		# Check output has tree but not content
		assert "# File Tree" in result.stdout
		assert "file1.txt" in result.stdout
		assert "# File Contents" not in result.stdout
		assert "content1" not in result.stdout
	except subprocess.CalledProcessError as e:
		print(f"{e = }", file=sys.stderr)
		print(e.stdout, file=sys.stderr)
		print(e.stderr, file=sys.stderr)
		raise e
	finally:
		os.chdir(original_cwd)

``````{ end_of_file: "tests/test_lmcat.py" }

``````{ path: "README.md" }
# lmcat

A Python tool for concatenating files and directory structures into a single document, perfect for sharing code with language models. It respects `.gitignore` and `.lmignore` patterns and provides configurable output formatting.

## Features

- Creates a tree view of your directory structure
- Includes file contents with clear delimiters
- Respects `.gitignore` patterns (can be disabled)
- Supports custom ignore patterns via `.lmignore`
- Configurable via `pyproject.toml`, `lmcat.toml`, or `lmcat.json`
- Python 3.11+ native, with fallback support for older versions

## Installation

Install from PyPI:

```bash
pip install lmcat
```

## Usage

Basic usage - concatenate current directory:

```bash
python -m lmcat
```

The output will include a directory tree and the contents of each non-ignored file.

### Command Line Options

- `-g`, `--no-include-gitignore`: Ignore `.gitignore` files (they are included by default)
- `-t`, `--tree-only`: Only print the directory tree, not file contents
- `-o`, `--output`: Specify an output file (defaults to stdout)
- `-h`, `--help`: Show help message

### Configuration

lmcat can be configured using any of these files (in order of precedence):

1. `pyproject.toml` (under `[tool.lmcat]`)
2. `lmcat.toml`
3. `lmcat.json`

Configuration options:

```toml
[tool.lmcat]
tree_divider = "│   "    # Used for vertical lines in the tree
indent = "    "          # Used for indentation
file_divider = "├── "    # Used for file/directory entries
content_divider = "``````" # Used to delimit file contents
include_gitignore = true # Whether to respect .gitignore files
tree_only = false       # Whether to only show the tree
```

### Ignore Patterns

lmcat supports two types of ignore files:

1. `.gitignore` - Standard Git ignore patterns (used by default)
2. `.lmignore` - Custom ignore patterns specific to lmcat

`.lmignore` follows the same pattern syntax as `.gitignore`. Patterns in `.lmignore` take precedence over `.gitignore`.

Example `.lmignore`:
```gitignore
# Ignore all .log files
*.log

# Ignore the build directory and its contents
build/

# Un-ignore a specific file (overrides previous patterns)
!important.log
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mivanit/lmcat
cd lmcat
```

2. Set up the development environment:
```bash
make setup
```

This will:
- Create a virtual environment
- Install development dependencies
- Set up pre-commit hooks

### Development Commands

The project uses `make` for common development tasks:

- `make dep`: Install/update dependencies
- `make format`: Format code using ruff and pycln
- `make test`: Run tests
- `make typing`: Run type checks
- `make check`: Run all checks (format, test, typing)
- `make clean`: Clean temporary files
- `make docs`: Generate documentation
- `make build`: Build the package
- `make publish`: Publish to PyPI (maintainers only)

Run `make help` to see all available commands.

### Running Tests

```bash
make test
```

For verbose output:
```bash
VERBOSE=1 make test
```

For test coverage:
```bash
make cov
```


### Roadmap

- better tests, I feel like gitignore/lmignore interaction is broken
- llm summarization and caching of those summaries in `.lmsummary/`
- reasonable defaults for file extensions to ignore
- web interface
``````{ end_of_file: "README.md" }

``````{ path: "example_output.md" }
# File Tree

```
lmcat
├── lmcat
│   ├── __init__.py
│   ├── __main__.py
│   ├── index.html
│   └── lmcat.py
├── tests
│   └── test_lmcat.py
├── README.md
├── example_output.md
├── makefile
├── pyproject.toml
```

# File Contents

``````{ path: "lmcat/__init__.py" }
"""
.. include:: ../README.md
"""

from lmcat.lmcat import main

__all__ = ["main"]

``````{ end_of_file: "lmcat/__init__.py" }

``````{ path: "lmcat/__main__.py" }
from lmcat import main

if __name__ == "__main__":
	main()

``````{ end_of_file: "lmcat/__main__.py" }

``````{ path: "lmcat/index.html" }
<!DOCTYPE html>
<html>
<head>
    <title>Minimal Git Browser</title>
    <script src="https://unpkg.com/@isomorphic-git/lightning-fs@4.6.0/dist/lightning-fs.min.js"></script>
    <script src="https://unpkg.com/isomorphic-git@1.24.0/index.umd.min.js"></script>
    <script src="https://unpkg.com/isomorphic-git@1.24.0/http/web/index.umd.js"></script>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"></script>
</head>
<body>
    <div style="margin: 20px;">
        <label for="url">Repository URL:</label>
        <input id="url" type="text" value="https://github.com/mivanit/lmcat" style="width: 300px; margin-right: 10px;">
        <button onclick="process()">Process</button>
        <div id="status" style="margin-top: 10px; color: gray;"></div>
    </div>
    <pre id="output" style="margin: 20px; padding: 10px; background: #f5f5f5;"></pre>

    <script>
        let fs, pfs, pyodide;

        // Debug function to check available objects
        function debugGlobals() {
            console.log('Available globals:');
            console.log('git:', typeof window.git);
            console.log('http:', typeof window.http);
            console.log('GitHttp:', typeof window.GitHttp);
            console.log('GitHttpClient:', typeof window.GitHttpClient);
        }

        async function init() {
            try {
                fs = new LightningFS('fs');
                pfs = fs.promises;
                
                // Initialize Pyodide
                pyodide = await loadPyodide();
                await pyodide.runPythonAsync(`
                    import os
                    def list_files(path):
                        try:
                            return str(list(os.listdir(path)))
                        except Exception as e:
                            return str(e)
                    def some_string():
                        return 'Hello from Python!'
                `);

                // Debug available objects
                debugGlobals();
                
                document.getElementById('status').textContent = 'Initialized successfully';
            } catch (err) {
                console.error('Init error:', err);
                document.getElementById('status').textContent = 'Initialization failed: ' + err.message;
            }
        }

        async function process() {
            const output = document.getElementById('output');
            const status = document.getElementById('status');
            status.textContent = 'Processing...';
            
            try {
                const dir = '/repo';
                await pfs.rmdir(dir, { recursive: true }).catch(() => {});
                await pfs.mkdir(dir).catch(() => {});
                
                                // Use the GitHttp object that's available globally
                if (!window.GitHttp) {
                    throw new Error('GitHttp is not available');
                }

                status.textContent = 'Cloning repository...';
                
                await git.clone({
                    fs,
                    http: GitHttp,
                    dir,
                    url: document.getElementById('url').value,
                    depth: 1,
                    singleBranch: true,
                    corsProxy: 'https://cors.isomorphic-git.org'
                });

                status.textContent = 'Listing files...';
                console.log('Listing files...');
                const result = await pyodide.runPythonAsync(`list_files('.')`);
                // const result = await pyodide.runPythonAsync(`some_string()`);
                console.log('result:', result);
                output.textContent = JSON.stringify(result, null, 2);
                status.textContent = 'Done!';
            } catch (err) {
                console.error('Process error:', err);
                status.textContent = 'Error: ' + err.message;
                output.textContent = err.stack || err.message;
            }
        }

        // Initialize on page load
        init();
    </script>
</body>
</html>
``````{ end_of_file: "lmcat/index.html" }

``````{ path: "lmcat/lmcat.py" }
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
	prefix: str = "",
) -> tuple[list[str], list[Path]]:
	"""Recursively walk a directory, building tree lines and collecting file paths"""
	tree_output: list[str] = []
	collected_files: list[Path] = []

	entries: list[Path] = sorted_entries(directory)
	for i, entry in enumerate(entries):
		if ignore_handler.is_ignored(entry):
			continue

		is_last: bool = i == len(entries) - 1
		connector: str = (
			config.file_divider
			if not is_last
			else config.file_divider.replace("├", "└")
		)

		if entry.is_dir():
			tree_output.append(f"{prefix}{connector}{entry.name}")
			extension: str = config.tree_divider if not is_last else config.indent
			sub_output: list[str]
			sub_files: list[Path]
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
	sub_output: list[str]
	sub_files: list[Path]
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
		Path(args.output).parent.mkdir(parents=True, exist_ok=True)
		with open(args.output, "w", encoding="utf-8") as f:
			f.write("\n".join(output))
	else:
		if sys.platform == "win32":
			sys.stdout = io.TextIOWrapper(
				sys.stdout.buffer, encoding="utf-8", errors="replace"
			)
			sys.stderr = io.TextIOWrapper(
				sys.stderr.buffer, encoding="utf-8", errors="replace"
			)

		print("\n".join(output))


if __name__ == "__main__":
	main()

``````{ end_of_file: "lmcat/lmcat.py" }

``````{ path: "tests/test_lmcat.py" }
import sys
import os
import shutil
import subprocess
from pathlib import Path

from lmcat.lmcat import (
	LMCatConfig,
	IgnoreHandler,
	walk_dir,
	walk_and_collect,
)

# We will store all test directories under this path:
TEMP_PATH: Path = Path("tests/_temp")


def ensure_clean_dir(dirpath: Path) -> None:
	"""Remove `dirpath` if it exists, then re-create it."""
	if dirpath.is_dir():
		shutil.rmtree(dirpath)
	dirpath.mkdir(parents=True, exist_ok=True)


# Test LMCatConfig - these tests remain largely unchanged
def test_lmcat_config_defaults():
	config = LMCatConfig()
	assert config.tree_divider == "│   "
	assert config.indent == " "
	assert config.file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_partial():
	data = {"tree_divider": "|---"}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "|---"
	assert config.indent == " "
	assert config.file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_all():
	data = {
		"tree_divider": "XX",
		"indent": "YY",
		"file_divider": "ZZ",
		"content_divider": "@@@",
	}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "XX"
	assert config.indent == "YY"
	assert config.file_divider == "ZZ"
	assert config.content_divider == "@@@"


# Test IgnoreHandler class
def test_ignore_handler_init():
	"""Test basic initialization of IgnoreHandler"""
	test_dir = TEMP_PATH / "test_ignore_handler_init"
	ensure_clean_dir(test_dir)
	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)
	assert handler.root_dir == test_dir
	assert handler.config == config


def test_ignore_handler_basic_ignore():
	"""Test basic ignore patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_basic_ignore"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / "file2.log").write_text("content2")
	(test_dir / ".lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches
	assert not handler.is_ignored(test_dir / "file1.txt")
	assert handler.is_ignored(test_dir / "file2.log")


def test_ignore_handler_directory_patterns():
	"""Test directory ignore patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_directory"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.txt").write_text("content2")
	(test_dir / ".lmignore").write_text("subdir2/\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches
	assert not handler.is_ignored(test_dir / "subdir1")
	assert handler.is_ignored(test_dir / "subdir2")
	assert not handler.is_ignored(test_dir / "subdir1/file1.txt")
	assert handler.is_ignored(test_dir / "subdir2/file2.txt")


def test_ignore_handler_negation():
	"""Test negation patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_negation"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / "file2.txt").write_text("content2")
	(test_dir / ".gitignore").write_text("*.txt\n")
	(test_dir / ".lmignore").write_text("!file2.txt\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test matches - file2.txt should be unignored by the negation
	assert handler.is_ignored(test_dir / "file1.txt")
	assert not handler.is_ignored(test_dir / "file2.txt")


def test_ignore_handler_nested_ignore_files():
	"""Test nested ignore files with different patterns"""
	test_dir = TEMP_PATH / "test_ignore_handler_nested"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir").mkdir()
	(test_dir / "subdir/file1.txt").write_text("content1")
	(test_dir / "subdir/file2.log").write_text("content2")

	# Root ignores .txt, subdir ignores .log
	(test_dir / ".lmignore").write_text("*.txt\n")
	(test_dir / "subdir/.lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	# Test both patterns are active
	assert handler.is_ignored(test_dir / "subdir/file1.txt")
	assert handler.is_ignored(test_dir / "subdir/file2.log")


def test_ignore_handler_gitignore_disabled():
	"""Test that gitignore patterns are ignored when disabled"""
	test_dir = TEMP_PATH / "test_ignore_handler_gitignore_disabled"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	(test_dir / ".gitignore").write_text("*.txt\n")

	config = LMCatConfig(include_gitignore=False)
	handler = IgnoreHandler(test_dir, config)

	# File should not be ignored since gitignore is disabled
	assert not handler.is_ignored(test_dir / "file1.txt")


# Test walking functions with new IgnoreHandler
def test_walk_dir_basic():
	"""Test basic directory walking with no ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_dir_basic"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.txt").write_text("content2")
	(test_dir / "file3.txt").write_text("content3")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	tree_output, files = walk_dir(test_dir, handler, config)
	joined_output = "\n".join(tree_output)

	# Check output contains all entries
	assert "subdir1" in joined_output
	assert "subdir2" in joined_output
	assert "file1.txt" in joined_output
	assert "file2.txt" in joined_output
	assert "file3.txt" in joined_output

	# Check collected files
	assert len(files) == 3
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file2.txt", "file3.txt"}


def test_walk_dir_with_ignore():
	"""Test directory walking with ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_dir_with_ignore"
	ensure_clean_dir(test_dir)

	# Create test structure
	(test_dir / "subdir1").mkdir()
	(test_dir / "subdir2").mkdir()
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir2/file2.log").write_text("content2")
	(test_dir / "file3.txt").write_text("content3")

	# Ignore .log files
	(test_dir / ".lmignore").write_text("*.log\n")

	config = LMCatConfig()
	handler = IgnoreHandler(test_dir, config)

	tree_output, files = walk_dir(test_dir, handler, config)
	joined_output = "\n".join(tree_output)

	# Check output excludes .log file
	assert "file2.log" not in joined_output
	assert "file1.txt" in joined_output
	assert "file3.txt" in joined_output

	# Check collected files
	assert len(files) == 2
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file3.txt"}


def test_walk_and_collect_complex():
	"""Test full directory walking with multiple ignore patterns"""
	test_dir = TEMP_PATH / "test_walk_and_collect_complex"
	ensure_clean_dir(test_dir)

	# Create complex directory structure
	(test_dir / "subdir1/nested").mkdir(parents=True)
	(test_dir / "subdir2/nested").mkdir(parents=True)
	(test_dir / "subdir1/file1.txt").write_text("content1")
	(test_dir / "subdir1/nested/file2.log").write_text("content2")
	(test_dir / "subdir2/file3.txt").write_text("content3")
	(test_dir / "subdir2/nested/file4.log").write_text("content4")

	# Root ignores .log files
	(test_dir / ".lmignore").write_text("*.log\n")
	# subdir2 ignores nested dir
	(test_dir / "subdir2/.lmignore").write_text("nested/\n")

	config = LMCatConfig()
	tree_output, files = walk_and_collect(test_dir, config)
	joined_output = "\n".join(tree_output)

	# Check correct files are excluded
	assert "file1.txt" in joined_output
	assert "file2.log" not in joined_output
	assert "file3.txt" in joined_output
	assert "file4.log" not in joined_output
	assert "nested" not in joined_output.split("\n")[-5:]  # Check last few lines

	# Check collected files
	assert len(files) == 2
	file_names = {f.name for f in files}
	assert file_names == {"file1.txt", "file3.txt"}


# Test CLI functionality
def test_cli_output_file():
	"""Test writing output to a file"""
	test_dir = TEMP_PATH / "test_cli_output_file"
	ensure_clean_dir(test_dir)

	# Create test files
	(test_dir / "file1.txt").write_text("content1")
	output_file = test_dir / "output.md"

	original_cwd = os.getcwd()
	try:
		os.chdir(test_dir)
		subprocess.run(
			["uv", "run", "python", "-m", "lmcat", "--output", str(output_file)],
			check=True,
		)

		# Check output file exists and contains expected content
		assert output_file.is_file()
		content = output_file.read_text()
		assert "# File Tree" in content
		assert "file1.txt" in content
		assert "content1" in content
	except subprocess.CalledProcessError as e:
		print(f"{e = }", file=sys.stderr)
		print(e.stdout, file=sys.stderr)
		print(e.stderr, file=sys.stderr)
		raise e
	finally:
		os.chdir(original_cwd)


def test_cli_tree_only():
	"""Test --tree-only option"""
	test_dir = TEMP_PATH / "test_cli_tree_only"
	ensure_clean_dir(test_dir)

	# Create test file
	(test_dir / "file1.txt").write_text("content1")

	original_cwd = os.getcwd()
	try:
		os.chdir(test_dir)
		result = subprocess.run(
			["uv", "run", "python", "-m", "lmcat", "--tree-only"],
			capture_output=True,
			text=True,
			check=True,
		)

		# Check output has tree but not content
		assert "# File Tree" in result.stdout
		assert "file1.txt" in result.stdout
		assert "# File Contents" not in result.stdout
		assert "content1" not in result.stdout
	except subprocess.CalledProcessError as e:
		print(f"{e = }", file=sys.stderr)
		print(e.stdout, file=sys.stderr)
		print(e.stderr, file=sys.stderr)
		raise e
	finally:
		os.chdir(original_cwd)

``````{ end_of_file: "tests/test_lmcat.py" }

``````{ path: "README.md" }
# lmcat

A Python tool for concatenating files and directory structures into a single document, perfect for sharing code with language models. It respects `.gitignore` and `.lmignore` patterns and provides configurable output formatting.

## Features

- Creates a tree view of your directory structure
- Includes file contents with clear delimiters
- Respects `.gitignore` patterns (can be disabled)
- Supports custom ignore patterns via `.lmignore`
- Configurable via `pyproject.toml`, `lmcat.toml`, or `lmcat.json`
- Python 3.11+ native, with fallback support for older versions

## Installation

Install from PyPI:

```bash
pip install lmcat
```

## Usage

Basic usage - concatenate current directory:

```bash
python -m lmcat
```

The output will include a directory tree and the contents of each non-ignored file.

### Command Line Options

- `-g`, `--no-include-gitignore`: Ignore `.gitignore` files (they are included by default)
- `-t`, `--tree-only`: Only print the directory tree, not file contents
- `-o`, `--output`: Specify an output file (defaults to stdout)
- `-h`, `--help`: Show help message

### Configuration

lmcat can be configured using any of these files (in order of precedence):

1. `pyproject.toml` (under `[tool.lmcat]`)
2. `lmcat.toml`
3. `lmcat.json`

Configuration options:

```toml
[tool.lmcat]
tree_divider = "│   "    # Used for vertical lines in the tree
indent = "    "          # Used for indentation
file_divider = "├── "    # Used for file/directory entries
content_divider = "``````" # Used to delimit file contents
include_gitignore = true # Whether to respect .gitignore files
tree_only = false       # Whether to only show the tree
```

### Ignore Patterns

lmcat supports two types of ignore files:

1. `.gitignore` - Standard Git ignore patterns (used by default)
2. `.lmignore` - Custom ignore patterns specific to lmcat

`.lmignore` follows the same pattern syntax as `.gitignore`. Patterns in `.lmignore` take precedence over `.gitignore`.

Example `.lmignore`:
```gitignore
# Ignore all .log files
*.log

# Ignore the build directory and its contents
build/

# Un-ignore a specific file (overrides previous patterns)
!important.log
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mivanit/lmcat
cd lmcat
```

2. Set up the development environment:
```bash
make setup
```

This will:
- Create a virtual environment
- Install development dependencies
- Set up pre-commit hooks

### Development Commands

The project uses `make` for common development tasks:

- `make dep`: Install/update dependencies
- `make format`: Format code using ruff and pycln
- `make test`: Run tests
- `make typing`: Run type checks
- `make check`: Run all checks (format, test, typing)
- `make clean`: Clean temporary files
- `make docs`: Generate documentation
- `make build`: Build the package
- `make publish`: Publish to PyPI (maintainers only)

Run `make help` to see all available commands.

### Running Tests

```bash
make test
```

For verbose output:
```bash
VERBOSE=1 make test
```

For test coverage:
```bash
make cov
```


### Roadmap

- better tests, I feel like gitignore/lmignore interaction is broken
- llm summarization and caching of those summaries in `.lmsummary/`
- reasonable defaults for file extensions to ignore
- web interface
``````{ end_of_file: "README.md" }

``````{ path: "example_output.md" }
# File Tree

```
lmcat
```

# File Contents
``````{ end_of_file: "example_output.md" }

``````{ path: "makefile" }
# ==================================================
# configuration & variables
# ==================================================

# !!! MODIFY AT LEAST THIS PART TO SUIT YOUR PROJECT !!!
# it assumes that the source is in a directory named the same as the package name
# this also gets passed to some other places
PACKAGE_NAME := lmcat

# for checking you are on the right branch when publishing
PUBLISH_BRANCH := main

# where to put docs
DOCS_DIR := docs

# where to put the coverage reports
# note that this will be published with the docs!
# modify the `docs` targets and `.gitignore` if you don't want that
COVERAGE_REPORTS_DIR := docs/coverage

# where the tests are, for pytest
TESTS_DIR := tests

# tests temp directory to clean up. will remove this in `make clean`
TESTS_TEMP_DIR := $(TESTS_DIR)/_temp/

# probably don't change these:
# --------------------------------------------------

# where the pyproject.toml file is. no idea why you would change this but just in case
PYPROJECT := pyproject.toml

# requirements.txt files for base package, all extras, dev, and all
REQ_LOCATION := .github/requirements

# local files (don't push this to git)
LOCAL_DIR := .github/local

# will print this token when publishing. make sure not to commit this file!!!
PYPI_TOKEN_FILE := $(LOCAL_DIR)/.pypi-token

# version files
VERSIONS_DIR := .github/versions

# the last version that was auto-uploaded. will use this to create a commit log for version tag
# see `gen-commit-log` target
LAST_VERSION_FILE := $(VERSIONS_DIR)/.lastversion

# current version (writing to file needed due to shell escaping issues)
VERSION_FILE := $(VERSIONS_DIR)/.version

# base python to use. Will add `uv run` in front of this if `RUN_GLOBAL` is not set to 1
PYTHON_BASE := python

# where the commit log will be stored
COMMIT_LOG_FILE := $(LOCAL_DIR)/.commit_log

# pandoc commands (for docs)
PANDOC ?= pandoc

# version vars - extracted automatically from `pyproject.toml`, `$(LAST_VERSION_FILE)`, and $(PYTHON)
# --------------------------------------------------

# assuming your `pyproject.toml` has a line that looks like `version = "0.0.1"`, `gen-version-info` will extract this
VERSION := NULL
# `gen-version-info` will read the last version from `$(LAST_VERSION_FILE)`, or `NULL` if it doesn't exist
LAST_VERSION := NULL
# get the python version, now that we have picked the python command
PYTHON_VERSION := NULL

# cuda version
# --------------------------------------------------
# 0 or 1
CUDA_PRESENT :=
# a version like "12.4" or "NULL"
CUDA_VERSION := NULL
# a version like "124" or "NULL"
CUDA_VERSION_SHORT := NULL


# python scripts we want to use inside the makefile
# --------------------------------------------------

# create commands for exporting requirements as specified in `pyproject.toml:tool.uv-exports.exports`
define EXPORT_SCRIPT
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from pathlib import Path
from typing import Union, List, Optional

pyproject_path: Path = Path(sys.argv[1])
output_dir: Path = Path(sys.argv[2])

with open(pyproject_path, 'rb') as f:
	pyproject_data: dict = tomllib.load(f)

# all available groups
all_groups: List[str] = list(pyproject_data.get('dependency-groups', {}).keys())
all_extras: List[str] = list(pyproject_data.get('project', {}).get('optional-dependencies', {}).keys())

# options for exporting
export_opts: dict = pyproject_data.get('tool', {}).get('uv-exports', {})

# what are we exporting?
exports: List[str] = export_opts.get('exports', [])
if not exports:
	exports = [{'name': 'all', 'groups': [], 'extras': [], 'options': []}]

# export each configuration
for export in exports:
	# get name and validate
	name = export.get('name')
	if not name or not name.isalnum():
		print(f"Export configuration missing valid 'name' field {export}", file=sys.stderr)
		continue

	# get other options with default fallbacks
	filename: str = export.get('filename') or f"requirements-{name}.txt"
	groups: Union[List[str], bool, None] = export.get('groups', None)
	extras: Union[List[str], bool] = export.get('extras', [])
	options: List[str] = export.get('options', [])

	# init command
	cmd: List[str] = ['uv', 'export'] + export_opts.get('args', [])

	# handle groups
	if groups is not None:
		groups_list: List[str] = []
		if isinstance(groups, bool):
			if groups:
				groups_list = all_groups.copy()
		else:
			groups_list = groups
		
		for group in all_groups:
			if group in groups_list:
				cmd.extend(['--group', group])
			else:
				cmd.extend(['--no-group', group])

	# handle extras
	extras_list: List[str] = []
	if isinstance(extras, bool):
		if extras:
			extras_list = all_extras.copy()
	else:
		extras_list = extras

	for extra in extras_list:
		cmd.extend(['--extra', extra])

	cmd.extend(options)

	output_path = output_dir / filename
	print(f"{' '.join(cmd)} > {output_path.as_posix()}")
endef

export EXPORT_SCRIPT

# get the version from `pyproject.toml:project.version`
define GET_VERSION_SCRIPT
import sys

try:
	if sys.version_info >= (3, 11):
		import tomllib
	else:
		import tomli as tomllib

	pyproject_path = '$(PYPROJECT)'

	with open(pyproject_path, 'rb') as f:
		pyproject_data = tomllib.load(f)

	print('v' + pyproject_data['project']['version'], end='')
except Exception as e:
	print('NULL', end='')
	sys.exit(1)
endef

export GET_VERSION_SCRIPT


# get the commit log since the last version from `$(LAST_VERSION_FILE)`
define GET_COMMIT_LOG_SCRIPT
import subprocess
import sys

last_version = sys.argv[1].strip()
commit_log_file = '$(COMMIT_LOG_FILE)'

if last_version == 'NULL':
    print('!!! ERROR !!!', file=sys.stderr)
    print('LAST_VERSION is NULL, can\'t get commit log!', file=sys.stderr)
    sys.exit(1)

try:
    log_cmd = ['git', 'log', f'{last_version}..HEAD', '--pretty=format:- %s (%h)']
    commits = subprocess.check_output(log_cmd).decode('utf-8').strip().split('\n')
    with open(commit_log_file, 'w') as f:
        f.write('\n'.join(reversed(commits)))
except subprocess.CalledProcessError as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
endef

export GET_COMMIT_LOG_SCRIPT

# get cuda information and whether torch sees it
define CHECK_TORCH_SCRIPT
import os
import sys
print(f'python version: {sys.version}')
print(f"\tpython executable path: {str(sys.executable)}")
print(f"\tsys_platform: {sys.platform}")
print(f'\tcurrent working directory: {os.getcwd()}')
print(f'\tHost name: {os.name}')
print(f'\tCPU count: {os.cpu_count()}')
print()

try:
	import torch
except Exception as e:
	print('ERROR: error importing torch, terminating        ')
	print('-'*50)
	raise e
	sys.exit(1)

print(f'torch version: {torch.__version__}')

print(f'\t{torch.cuda.is_available() = }')

if torch.cuda.is_available():
	# print('\tCUDA is available on torch')
	print(f'\tCUDA version via torch: {torch.version.cuda}')

	if torch.cuda.device_count() > 0:
		print(f"\tcurrent device: {torch.cuda.current_device() = }\n")
		n_devices: int = torch.cuda.device_count()
		print(f"detected {n_devices = }")
		for current_device in range(n_devices):
			try:
				# print(f'checking current device {current_device} of {torch.cuda.device_count()} devices')
				print(f'\tdevice {current_device}')
				dev_prop = torch.cuda.get_device_properties(torch.device(0))
				print(f'\t    name:                   {dev_prop.name}')
				print(f'\t    version:                {dev_prop.major}.{dev_prop.minor}')
				print(f'\t    total_memory:           {dev_prop.total_memory} ({dev_prop.total_memory:.1e})')
				print(f'\t    multi_processor_count:  {dev_prop.multi_processor_count}')
				print(f'\t    is_integrated:          {dev_prop.is_integrated}')
				print(f'\t    is_multi_gpu_board:     {dev_prop.is_multi_gpu_board}')
				print(f'\t')
			except Exception as e:
				print(f'Exception when trying to get properties of device {current_device}')
				raise e
		sys.exit(0)
	else:
		print(f'ERROR: {torch.cuda.device_count()} devices detected, invalid')
		print('-'*50)
		sys.exit(1)

else:
	print('ERROR: CUDA is NOT available, terminating')
	print('-'*50)
	sys.exit(1)
endef

export CHECK_TORCH_SCRIPT


# ==================================================
# reading command line options
# ==================================================

# for formatting or something, we might want to run python without uv
# RUN_GLOBAL=1 to use global `PYTHON_BASE` instead of `uv run $(PYTHON_BASE)`
RUN_GLOBAL ?= 0

ifeq ($(RUN_GLOBAL),0)
	PYTHON = uv run $(PYTHON_BASE)
else
	PYTHON = $(PYTHON_BASE)
endif

# if you want different behavior for different python versions
# --------------------------------------------------
# COMPATIBILITY_MODE := $(shell $(PYTHON) -c "import sys; print(1 if sys.version_info < (3, 10) else 0)")

# options we might want to pass to pytest
# --------------------------------------------------

# base options for pytest, will be appended to if `COV` or `VERBOSE` are 1.
# user can also set this when running make to add more options
PYTEST_OPTIONS ?=

# set to `1` to run pytest with `--cov=.` to get coverage reports in a `.coverage` file
COV ?= 1
# set to `1` to run pytest with `--verbose`
VERBOSE ?= 0

ifeq ($(VERBOSE),1)
	PYTEST_OPTIONS += --verbose
endif

ifeq ($(COV),1)
	PYTEST_OPTIONS += --cov=.
endif

# ==================================================
# default target (help)
# ==================================================

# first/default target is help
.PHONY: default
default: help

# ==================================================
# getting version info
# we do this in a separate target because it takes a bit of time
# ==================================================

# this recipe is weird. we need it because:
# - a one liner for getting the version with toml is unwieldy, and using regex is fragile
# - using $$GET_VERSION_SCRIPT within $(shell ...) doesn't work because of escaping issues
# - trying to write to the file inside the `gen-version-info` recipe doesn't work, 
# 	shell eval happens before our `python -c ...` gets run and `cat` doesn't see the new file
.PHONY: write-proj-version
write-proj-version:
	@mkdir -p $(VERSIONS_DIR)
	@$(PYTHON) -c "$$GET_VERSION_SCRIPT" > $(VERSION_FILE)

# gets version info from $(PYPROJECT), last version from $(LAST_VERSION_FILE), and python version
# uses just `python` for everything except getting the python version. no echo here, because this is "private"
.PHONY: gen-version-info
gen-version-info: write-proj-version
	@mkdir -p $(LOCAL_DIR)
	$(eval VERSION := $(shell cat $(VERSION_FILE)) )
	$(eval LAST_VERSION := $(shell [ -f $(LAST_VERSION_FILE) ] && cat $(LAST_VERSION_FILE) || echo NULL) )
	$(eval PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')") )

# getting commit log since the tag specified in $(LAST_VERSION_FILE)
# will write to $(COMMIT_LOG_FILE)
# when publishing, the contents of $(COMMIT_LOG_FILE) will be used as the tag description (but can be edited during the process)
# no echo here, because this is "private"
.PHONY: gen-commit-log
gen-commit-log: gen-version-info
	@if [ "$(LAST_VERSION)" = "NULL" ]; then \
		echo "!!! ERROR !!!"; \
		echo "LAST_VERSION is NULL, cant get commit log!"; \
		exit 1; \
	fi
	@mkdir -p $(LOCAL_DIR)
	@$(PYTHON) -c "$$GET_COMMIT_LOG_SCRIPT" "$(LAST_VERSION)"


# force the version info to be read, printing it out
# also force the commit log to be generated, and cat it out
.PHONY: version
version: gen-commit-log
	@echo "Current version is $(VERSION), last auto-uploaded version is $(LAST_VERSION)"
	@echo "Commit log since last version from '$(COMMIT_LOG_FILE)':"
	@cat $(COMMIT_LOG_FILE)
	@echo ""
	@if [ "$(VERSION)" = "$(LAST_VERSION)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Python package $(VERSION) is the same as last published version $(LAST_VERSION), exiting!"; \
		exit 1; \
	fi


# ==================================================
# dependencies and setup
# ==================================================

.PHONY: setup
setup: dep-check
	@echo "install and update via uv"
	@echo "To activate the virtual environment, run one of:"
	@echo "  source .venv/bin/activate"
	@echo "  source .venv/Scripts/activate"

.PHONY: get-cuda-info
get-cuda-info:
	$(eval CUDA_PRESENT := $(shell if command -v nvcc > /dev/null 2>&1; then echo 1; else echo 0; fi))
	$(eval CUDA_VERSION := $(if $(filter $(CUDA_PRESENT),1),$(shell nvcc --version 2>/dev/null | grep "release" | awk '{print $$5}' | sed 's/,//'),NULL))
	$(eval CUDA_VERSION_SHORT := $(if $(filter $(CUDA_PRESENT),1),$(shell echo $(CUDA_VERSION) | sed 's/\.//'),NULL))

.PHONY: dep-check-torch
dep-check-torch:
	@echo "see if torch is installed, and which CUDA version and devices it sees"
	$(PYTHON) -c "$$CHECK_TORCH_SCRIPT"

.PHONY: dep
dep: get-cuda-info
	@echo "Exporting dependencies as per $(PYPROJECT) section 'tool.uv-exports.exports'"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION) | sh -x
	
# @if [ "$(CUDA_PRESENT)" = "1" ]; then \
# 	echo "CUDA is present, installing torch with CUDA $(CUDA_VERSION)"; \
# 	uv pip install torch --upgrade --index https://download.pytorch.org/whl/cu$(CUDA_VERSION_SHORT); \
# fi
	

.PHONY: dep-check
dep-check:
	@echo "Checking that exported requirements are up to date"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)-TEMP
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION)-TEMP | sh -x
	diff -r $(REQ_LOCATION)-TEMP $(REQ_LOCATION)
	rm -rf $(REQ_LOCATION)-TEMP


.PHONY: dep-clean
dep-clean:
	@echo "clean up lock files, .venv, and requirements files"
	rm -rf .venv
	rm -rf uv.lock
	rm -rf $(REQ_LOCATION)/*.txt

# ==================================================
# checks (formatting/linting, typing, tests)
# ==================================================

# runs ruff and pycln to format the code
.PHONY: format
format:
	@echo "format the source code"
	$(PYTHON) -m ruff format --config $(PYPROJECT) .
	$(PYTHON) -m ruff check --fix --config $(PYPROJECT) .
	$(PYTHON) -m pycln --config $(PYPROJECT) --all .

# runs ruff and pycln to check if the code is formatted correctly
.PHONY: format-check
format-check:
	@echo "check if the source code is formatted correctly"
	$(PYTHON) -m ruff check --config $(PYPROJECT) .
	$(PYTHON) -m pycln --check --config $(PYPROJECT) .

# runs type checks with mypy
# at some point, need to add back --check-untyped-defs to mypy call
# but it complains when we specify arguments by keyword where positional is fine
# not sure how to fix this
.PHONY: typing
typing: clean
	@echo "running type checks"
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(PACKAGE_NAME)/
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(TESTS_DIR)/

.PHONY: test
test: clean
	@echo "running tests"
	$(PYTHON) -m pytest $(PYTEST_OPTIONS) $(TESTS_DIR)

.PHONY: check
check: clean format-check test typing
	@echo "run format checks, tests, and typing checks"

# ==================================================
# coverage & docs
# ==================================================

# generates a whole tree of documentation in html format.
# see `docs/make_docs.py` and the templates in `docs/templates/html/` for more info
.PHONY: docs-html
docs-html:
	@echo "generate html docs"
	$(PYTHON) docs/make_docs.py

# instead of a whole website, generates a single markdown file with all docs using the templates in `docs/templates/markdown/`.
# this is useful if you want to have a copy that you can grep/search, but those docs are much messier.
# docs-combined will use pandoc to convert them to other formats.
.PHONY: docs-md
docs-md:
	@echo "generate combined (single-file) docs in markdown"
	mkdir $(DOCS_DIR)/combined -p
	$(PYTHON) docs/make_docs.py --combined

# after running docs-md, this will convert the combined markdown file to other formats:
# gfm (github-flavored markdown), plain text, and html
# requires pandoc in path, pointed to by $(PANDOC)
# pdf output would be nice but requires other deps
.PHONY: docs-combined
docs-combined: docs-md
	@echo "generate combined (single-file) docs in markdown and convert to other formats"
	@echo "requires pandoc in path"
	$(PANDOC) -f markdown -t gfm $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME)_gfm.md
	$(PANDOC) -f markdown -t plain $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).txt
	$(PANDOC) -f markdown -t html $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).html

# generates coverage reports as html and text with `pytest-cov`, and a badge with `coverage-badge`
# if `.coverage` is not found, will run tests first
# also removes the `.gitignore` file that `coverage html` creates, since we count that as part of the docs
.PHONY: cov
cov:
	@echo "generate coverage reports"
	@if [ ! -f .coverage ]; then \
		echo ".coverage not found, running tests first..."; \
		$(MAKE) test; \
	fi
	mkdir $(COVERAGE_REPORTS_DIR) -p
	$(PYTHON) -m coverage report -m > $(COVERAGE_REPORTS_DIR)/coverage.txt
	$(PYTHON) -m coverage_badge -f -o $(COVERAGE_REPORTS_DIR)/coverage.svg
	$(PYTHON) -m coverage html --directory=$(COVERAGE_REPORTS_DIR)/html/
	rm -rf $(COVERAGE_REPORTS_DIR)/html/.gitignore

# runs the coverage report, then the docs, then the combined docs
.PHONY: docs
docs: cov docs-html docs-combined
	@echo "generate all documentation and coverage reports"

# removed all generated documentation files, but leaves the templates and the `docs/make_docs.py` script
# distinct from `make clean`
.PHONY: docs-clean
docs-clean:
	@echo "remove generated docs"
	rm -rf $(DOCS_DIR)/combined/
	rm -rf $(DOCS_DIR)/$(PACKAGE_NAME)/
	rm -rf $(COVERAGE_REPORTS_DIR)/
	rm $(DOCS_DIR)/$(PACKAGE_NAME).html
	rm $(DOCS_DIR)/index.html
	rm $(DOCS_DIR)/search.js
	rm $(DOCS_DIR)/package_map.dot
	rm $(DOCS_DIR)/package_map.html


# ==================================================
# build and publish
# ==================================================

# verifies that the current branch is $(PUBLISH_BRANCH) and that git is clean
# used before publishing
.PHONY: verify-git
verify-git: 
	@echo "checking git status"
	if [ "$(shell git branch --show-current)" != $(PUBLISH_BRANCH) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not on the $(PUBLISH_BRANCH) branch, exiting!"; \
		exit 1; \
	fi; \
	if [ -n "$(shell git status --porcelain)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not clean, exiting!"; \
		exit 1; \
	fi; \


.PHONY: build
build: 
	@echo "build the package"
	uv build

# gets the commit log, checks everything, builds, and then publishes with twine
# will ask the user to confirm the new version number (and this allows for editing the tag info)
# will also print the contents of $(PYPI_TOKEN_FILE) to the console for the user to copy and paste in when prompted by twine
.PHONY: publish
publish: gen-commit-log check build verify-git version gen-version-info
	@echo "run all checks, build, and then publish"

	@echo "Enter the new version number if you want to upload to pypi and create a new tag"
	@echo "Now would also be the time to edit $(COMMIT_LOG_FILE), as that will be used as the tag description"
	@read -p "Confirm: " NEW_VERSION; \
	if [ "$$NEW_VERSION" = $(VERSION) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Version confirmed. Proceeding with publish."; \
	else \
		echo "Version mismatch, exiting: you gave $$NEW_VERSION but expected $(VERSION)"; \
		exit 1; \
	fi;

	@echo "pypi username: __token__"
	@echo "pypi token from '$(PYPI_TOKEN_FILE)' :"
	echo $$(cat $(PYPI_TOKEN_FILE))

	echo "Uploading!"; \
	echo $(VERSION) > $(LAST_VERSION_FILE); \
	git add $(LAST_VERSION_FILE); \
	git commit -m "Auto update to $(VERSION)"; \
	git tag -a $(VERSION) -F $(COMMIT_LOG_FILE); \
	git push origin $(VERSION); \
	twine upload dist/* --verbose

# ==================================================
# cleanup of temp files
# ==================================================

# cleans up temp files from formatter, type checking, tests, coverage
# removes all built files
# removes $(TESTS_TEMP_DIR) to remove temporary test files
# recursively removes all `__pycache__` directories and `*.pyc` or `*.pyo` files
# distinct from `make docs-clean`, which only removes generated documentation files
.PHONY: clean
clean:
	@echo "clean up temporary files"
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf $(PACKAGE_NAME).egg-info
	rm -rf $(TESTS_TEMP_DIR)
	$(PYTHON_BASE) -Bc "import pathlib; [p.unlink() for path in ['$(PACKAGE_NAME)', '$(TESTS_DIR)', '$(DOCS_DIR)'] for pattern in ['*.py[co]', '__pycache__/*'] for p in pathlib.Path(path).rglob(pattern)]"

.PHONY: clean-all
clean-all: clean dep-clean docs-clean
	@echo "clean up all temporary files, dep files, venv, and generated docs"


# ==================================================
# smart help command
# ==================================================

# listing targets is from stackoverflow
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
# no .PHONY because this will only be run before `make help`
# it's a separate command because getting the versions takes a bit of time
help-targets:
	@echo -n "# make targets"
	@echo ":"
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 30


.PHONY: info
info: gen-version-info get-cuda-info
	@echo "# makefile variables"
	@echo "    PYTHON = $(PYTHON)"
	@echo "    PYTHON_VERSION = $(PYTHON_VERSION)"
	@echo "    PACKAGE_NAME = $(PACKAGE_NAME)"
	@echo "    VERSION = $(VERSION)"
	@echo "    LAST_VERSION = $(LAST_VERSION)"
	@echo "    PYTEST_OPTIONS = $(PYTEST_OPTIONS)"
	@echo "    CUDA_PRESENT = $(CUDA_PRESENT)"
	@if [ "$(CUDA_PRESENT)" = "1" ]; then \
		echo "    CUDA_VERSION = $(CUDA_VERSION)"; \
		echo "    CUDA_VERSION_SHORT = $(CUDA_VERSION_SHORT)"; \
	fi

.PHONY: info-long
info-long: info
	@echo "# other variables"
	@echo "    PUBLISH_BRANCH = $(PUBLISH_BRANCH)"
	@echo "    DOCS_DIR = $(DOCS_DIR)"
	@echo "    COVERAGE_REPORTS_DIR = $(COVERAGE_REPORTS_DIR)"
	@echo "    TESTS_DIR = $(TESTS_DIR)"
	@echo "    TESTS_TEMP_DIR = $(TESTS_TEMP_DIR)"
	@echo "    PYPROJECT = $(PYPROJECT)"
	@echo "    REQ_LOCATION = $(REQ_LOCATION)"
	@echo "    REQ_BASE = $(REQ_BASE)"
	@echo "    REQ_EXTRAS = $(REQ_EXTRAS)"
	@echo "    REQ_DEV = $(REQ_DEV)"
	@echo "    REQ_ALL = $(REQ_ALL)"
	@echo "    LOCAL_DIR = $(LOCAL_DIR)"
	@echo "    PYPI_TOKEN_FILE = $(PYPI_TOKEN_FILE)"
	@echo "    LAST_VERSION_FILE = $(LAST_VERSION_FILE)"
	@echo "    PYTHON_BASE = $(PYTHON_BASE)"
	@echo "    COMMIT_LOG_FILE = $(COMMIT_LOG_FILE)"
	@echo "    PANDOC = $(PANDOC)"
	@echo "    COV = $(COV)"
	@echo "    VERBOSE = $(VERBOSE)"
	@echo "    RUN_GLOBAL = $(RUN_GLOBAL)"
	@echo "    TYPECHECK_ARGS = $(TYPECHECK_ARGS)"

# immediately print out the help targets, and then local variables (but those take a bit longer)
.PHONY: help
help: help-targets info
	@echo -n ""

# ==================================================
# custom targets
# ==================================================
# (put them down here, or delimit with ~~~~~)


.PHONY: demo
demo:
	@echo "example of code output"
	$(PYTHON) -m lmcat -o example_output.md
``````{ end_of_file: "makefile" }

``````{ path: "pyproject.toml" }
[project]
name = "lmcat"
version = "0.0.1"
description = "concatenating files for tossing them into a language model"
authors = [
	{ name = "Michael Ivanitskiy", email = "mivanits@umich.edu" }
]
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
	"igittigitt>=2.1.5",
]


[dependency-groups]
dev = [
	# test
	"pytest>=8.2.2",
	# coverage
	"pytest-cov>=4.1.0",
	"coverage-badge>=1.1.0",
	# type checking
	"mypy>=1.0.1",
	# docs
	'pdoc>=14.6.0',
	# tomli since no tomlib in python < 3.11
	"tomli>=2.1.0; python_version < '3.11'",
]
lint = [
	# lint
	"pycln>=2.1.3",
	"ruff>=0.4.8",
]

[tool.uv]
default-groups = ["dev", "lint"]

[project.urls]
Homepage = "https://miv.name/lmcat"
Documentation = "https://miv.name/lmcat"
Repository = "https://github.com/mivanit/lmcat"
Issues = "https://github.com/mivanit/lmcat/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# ruff config
[tool.ruff]
exclude = ["__pycache__"]

[tool.ruff.format]
indent-style = "tab"
skip-magic-trailing-comma = false

# Custom export configurations
[tool.uv-exports]
args = [
	"--no-hashes"
]
exports = [
	# no groups, no extras, just the base dependencies
    { name = "base", groups = false, extras = false },
	# all extras but no groups
    { name = "extras", groups = false, extras = true },
	# include the dev group (this is the default behavior)
    { name = "dev", groups = ["dev"] },
	# only the lint group -- custom options for this
	{ name = "lint", options = ["--only-group", "lint"] },
	# all groups and extras
    { name = "all", filename="requirements.txt", groups = true, extras=true },
	# all groups and extras, a different way
	{ name = "all", groups = true, options = ["--all-extras"] },
]


``````{ end_of_file: "pyproject.toml" }
``````{ end_of_file: "example_output.md" }

``````{ path: "makefile" }
# ==================================================
# configuration & variables
# ==================================================

# !!! MODIFY AT LEAST THIS PART TO SUIT YOUR PROJECT !!!
# it assumes that the source is in a directory named the same as the package name
# this also gets passed to some other places
PACKAGE_NAME := lmcat

# for checking you are on the right branch when publishing
PUBLISH_BRANCH := main

# where to put docs
DOCS_DIR := docs

# where to put the coverage reports
# note that this will be published with the docs!
# modify the `docs` targets and `.gitignore` if you don't want that
COVERAGE_REPORTS_DIR := docs/coverage

# where the tests are, for pytest
TESTS_DIR := tests

# tests temp directory to clean up. will remove this in `make clean`
TESTS_TEMP_DIR := $(TESTS_DIR)/_temp/

# probably don't change these:
# --------------------------------------------------

# where the pyproject.toml file is. no idea why you would change this but just in case
PYPROJECT := pyproject.toml

# requirements.txt files for base package, all extras, dev, and all
REQ_LOCATION := .github/requirements

# local files (don't push this to git)
LOCAL_DIR := .github/local

# will print this token when publishing. make sure not to commit this file!!!
PYPI_TOKEN_FILE := $(LOCAL_DIR)/.pypi-token

# version files
VERSIONS_DIR := .github/versions

# the last version that was auto-uploaded. will use this to create a commit log for version tag
# see `gen-commit-log` target
LAST_VERSION_FILE := $(VERSIONS_DIR)/.lastversion

# current version (writing to file needed due to shell escaping issues)
VERSION_FILE := $(VERSIONS_DIR)/.version

# base python to use. Will add `uv run` in front of this if `RUN_GLOBAL` is not set to 1
PYTHON_BASE := python

# where the commit log will be stored
COMMIT_LOG_FILE := $(LOCAL_DIR)/.commit_log

# pandoc commands (for docs)
PANDOC ?= pandoc

# version vars - extracted automatically from `pyproject.toml`, `$(LAST_VERSION_FILE)`, and $(PYTHON)
# --------------------------------------------------

# assuming your `pyproject.toml` has a line that looks like `version = "0.0.1"`, `gen-version-info` will extract this
VERSION := NULL
# `gen-version-info` will read the last version from `$(LAST_VERSION_FILE)`, or `NULL` if it doesn't exist
LAST_VERSION := NULL
# get the python version, now that we have picked the python command
PYTHON_VERSION := NULL

# cuda version
# --------------------------------------------------
# 0 or 1
CUDA_PRESENT :=
# a version like "12.4" or "NULL"
CUDA_VERSION := NULL
# a version like "124" or "NULL"
CUDA_VERSION_SHORT := NULL


# python scripts we want to use inside the makefile
# --------------------------------------------------

# create commands for exporting requirements as specified in `pyproject.toml:tool.uv-exports.exports`
define EXPORT_SCRIPT
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from pathlib import Path
from typing import Union, List, Optional

pyproject_path: Path = Path(sys.argv[1])
output_dir: Path = Path(sys.argv[2])

with open(pyproject_path, 'rb') as f:
	pyproject_data: dict = tomllib.load(f)

# all available groups
all_groups: List[str] = list(pyproject_data.get('dependency-groups', {}).keys())
all_extras: List[str] = list(pyproject_data.get('project', {}).get('optional-dependencies', {}).keys())

# options for exporting
export_opts: dict = pyproject_data.get('tool', {}).get('uv-exports', {})

# what are we exporting?
exports: List[str] = export_opts.get('exports', [])
if not exports:
	exports = [{'name': 'all', 'groups': [], 'extras': [], 'options': []}]

# export each configuration
for export in exports:
	# get name and validate
	name = export.get('name')
	if not name or not name.isalnum():
		print(f"Export configuration missing valid 'name' field {export}", file=sys.stderr)
		continue

	# get other options with default fallbacks
	filename: str = export.get('filename') or f"requirements-{name}.txt"
	groups: Union[List[str], bool, None] = export.get('groups', None)
	extras: Union[List[str], bool] = export.get('extras', [])
	options: List[str] = export.get('options', [])

	# init command
	cmd: List[str] = ['uv', 'export'] + export_opts.get('args', [])

	# handle groups
	if groups is not None:
		groups_list: List[str] = []
		if isinstance(groups, bool):
			if groups:
				groups_list = all_groups.copy()
		else:
			groups_list = groups
		
		for group in all_groups:
			if group in groups_list:
				cmd.extend(['--group', group])
			else:
				cmd.extend(['--no-group', group])

	# handle extras
	extras_list: List[str] = []
	if isinstance(extras, bool):
		if extras:
			extras_list = all_extras.copy()
	else:
		extras_list = extras

	for extra in extras_list:
		cmd.extend(['--extra', extra])

	cmd.extend(options)

	output_path = output_dir / filename
	print(f"{' '.join(cmd)} > {output_path.as_posix()}")
endef

export EXPORT_SCRIPT

# get the version from `pyproject.toml:project.version`
define GET_VERSION_SCRIPT
import sys

try:
	if sys.version_info >= (3, 11):
		import tomllib
	else:
		import tomli as tomllib

	pyproject_path = '$(PYPROJECT)'

	with open(pyproject_path, 'rb') as f:
		pyproject_data = tomllib.load(f)

	print('v' + pyproject_data['project']['version'], end='')
except Exception as e:
	print('NULL', end='')
	sys.exit(1)
endef

export GET_VERSION_SCRIPT


# get the commit log since the last version from `$(LAST_VERSION_FILE)`
define GET_COMMIT_LOG_SCRIPT
import subprocess
import sys

last_version = sys.argv[1].strip()
commit_log_file = '$(COMMIT_LOG_FILE)'

if last_version == 'NULL':
    print('!!! ERROR !!!', file=sys.stderr)
    print('LAST_VERSION is NULL, can\'t get commit log!', file=sys.stderr)
    sys.exit(1)

try:
    log_cmd = ['git', 'log', f'{last_version}..HEAD', '--pretty=format:- %s (%h)']
    commits = subprocess.check_output(log_cmd).decode('utf-8').strip().split('\n')
    with open(commit_log_file, 'w') as f:
        f.write('\n'.join(reversed(commits)))
except subprocess.CalledProcessError as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
endef

export GET_COMMIT_LOG_SCRIPT

# get cuda information and whether torch sees it
define CHECK_TORCH_SCRIPT
import os
import sys
print(f'python version: {sys.version}')
print(f"\tpython executable path: {str(sys.executable)}")
print(f"\tsys_platform: {sys.platform}")
print(f'\tcurrent working directory: {os.getcwd()}')
print(f'\tHost name: {os.name}')
print(f'\tCPU count: {os.cpu_count()}')
print()

try:
	import torch
except Exception as e:
	print('ERROR: error importing torch, terminating        ')
	print('-'*50)
	raise e
	sys.exit(1)

print(f'torch version: {torch.__version__}')

print(f'\t{torch.cuda.is_available() = }')

if torch.cuda.is_available():
	# print('\tCUDA is available on torch')
	print(f'\tCUDA version via torch: {torch.version.cuda}')

	if torch.cuda.device_count() > 0:
		print(f"\tcurrent device: {torch.cuda.current_device() = }\n")
		n_devices: int = torch.cuda.device_count()
		print(f"detected {n_devices = }")
		for current_device in range(n_devices):
			try:
				# print(f'checking current device {current_device} of {torch.cuda.device_count()} devices')
				print(f'\tdevice {current_device}')
				dev_prop = torch.cuda.get_device_properties(torch.device(0))
				print(f'\t    name:                   {dev_prop.name}')
				print(f'\t    version:                {dev_prop.major}.{dev_prop.minor}')
				print(f'\t    total_memory:           {dev_prop.total_memory} ({dev_prop.total_memory:.1e})')
				print(f'\t    multi_processor_count:  {dev_prop.multi_processor_count}')
				print(f'\t    is_integrated:          {dev_prop.is_integrated}')
				print(f'\t    is_multi_gpu_board:     {dev_prop.is_multi_gpu_board}')
				print(f'\t')
			except Exception as e:
				print(f'Exception when trying to get properties of device {current_device}')
				raise e
		sys.exit(0)
	else:
		print(f'ERROR: {torch.cuda.device_count()} devices detected, invalid')
		print('-'*50)
		sys.exit(1)

else:
	print('ERROR: CUDA is NOT available, terminating')
	print('-'*50)
	sys.exit(1)
endef

export CHECK_TORCH_SCRIPT


# ==================================================
# reading command line options
# ==================================================

# for formatting or something, we might want to run python without uv
# RUN_GLOBAL=1 to use global `PYTHON_BASE` instead of `uv run $(PYTHON_BASE)`
RUN_GLOBAL ?= 0

ifeq ($(RUN_GLOBAL),0)
	PYTHON = uv run $(PYTHON_BASE)
else
	PYTHON = $(PYTHON_BASE)
endif

# if you want different behavior for different python versions
# --------------------------------------------------
# COMPATIBILITY_MODE := $(shell $(PYTHON) -c "import sys; print(1 if sys.version_info < (3, 10) else 0)")

# options we might want to pass to pytest
# --------------------------------------------------

# base options for pytest, will be appended to if `COV` or `VERBOSE` are 1.
# user can also set this when running make to add more options
PYTEST_OPTIONS ?=

# set to `1` to run pytest with `--cov=.` to get coverage reports in a `.coverage` file
COV ?= 1
# set to `1` to run pytest with `--verbose`
VERBOSE ?= 0

ifeq ($(VERBOSE),1)
	PYTEST_OPTIONS += --verbose
endif

ifeq ($(COV),1)
	PYTEST_OPTIONS += --cov=.
endif

# ==================================================
# default target (help)
# ==================================================

# first/default target is help
.PHONY: default
default: help

# ==================================================
# getting version info
# we do this in a separate target because it takes a bit of time
# ==================================================

# this recipe is weird. we need it because:
# - a one liner for getting the version with toml is unwieldy, and using regex is fragile
# - using $$GET_VERSION_SCRIPT within $(shell ...) doesn't work because of escaping issues
# - trying to write to the file inside the `gen-version-info` recipe doesn't work, 
# 	shell eval happens before our `python -c ...` gets run and `cat` doesn't see the new file
.PHONY: write-proj-version
write-proj-version:
	@mkdir -p $(VERSIONS_DIR)
	@$(PYTHON) -c "$$GET_VERSION_SCRIPT" > $(VERSION_FILE)

# gets version info from $(PYPROJECT), last version from $(LAST_VERSION_FILE), and python version
# uses just `python` for everything except getting the python version. no echo here, because this is "private"
.PHONY: gen-version-info
gen-version-info: write-proj-version
	@mkdir -p $(LOCAL_DIR)
	$(eval VERSION := $(shell cat $(VERSION_FILE)) )
	$(eval LAST_VERSION := $(shell [ -f $(LAST_VERSION_FILE) ] && cat $(LAST_VERSION_FILE) || echo NULL) )
	$(eval PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')") )

# getting commit log since the tag specified in $(LAST_VERSION_FILE)
# will write to $(COMMIT_LOG_FILE)
# when publishing, the contents of $(COMMIT_LOG_FILE) will be used as the tag description (but can be edited during the process)
# no echo here, because this is "private"
.PHONY: gen-commit-log
gen-commit-log: gen-version-info
	@if [ "$(LAST_VERSION)" = "NULL" ]; then \
		echo "!!! ERROR !!!"; \
		echo "LAST_VERSION is NULL, cant get commit log!"; \
		exit 1; \
	fi
	@mkdir -p $(LOCAL_DIR)
	@$(PYTHON) -c "$$GET_COMMIT_LOG_SCRIPT" "$(LAST_VERSION)"


# force the version info to be read, printing it out
# also force the commit log to be generated, and cat it out
.PHONY: version
version: gen-commit-log
	@echo "Current version is $(VERSION), last auto-uploaded version is $(LAST_VERSION)"
	@echo "Commit log since last version from '$(COMMIT_LOG_FILE)':"
	@cat $(COMMIT_LOG_FILE)
	@echo ""
	@if [ "$(VERSION)" = "$(LAST_VERSION)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Python package $(VERSION) is the same as last published version $(LAST_VERSION), exiting!"; \
		exit 1; \
	fi


# ==================================================
# dependencies and setup
# ==================================================

.PHONY: setup
setup: dep-check
	@echo "install and update via uv"
	@echo "To activate the virtual environment, run one of:"
	@echo "  source .venv/bin/activate"
	@echo "  source .venv/Scripts/activate"

.PHONY: get-cuda-info
get-cuda-info:
	$(eval CUDA_PRESENT := $(shell if command -v nvcc > /dev/null 2>&1; then echo 1; else echo 0; fi))
	$(eval CUDA_VERSION := $(if $(filter $(CUDA_PRESENT),1),$(shell nvcc --version 2>/dev/null | grep "release" | awk '{print $$5}' | sed 's/,//'),NULL))
	$(eval CUDA_VERSION_SHORT := $(if $(filter $(CUDA_PRESENT),1),$(shell echo $(CUDA_VERSION) | sed 's/\.//'),NULL))

.PHONY: dep-check-torch
dep-check-torch:
	@echo "see if torch is installed, and which CUDA version and devices it sees"
	$(PYTHON) -c "$$CHECK_TORCH_SCRIPT"

.PHONY: dep
dep: get-cuda-info
	@echo "Exporting dependencies as per $(PYPROJECT) section 'tool.uv-exports.exports'"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION) | sh -x
	
# @if [ "$(CUDA_PRESENT)" = "1" ]; then \
# 	echo "CUDA is present, installing torch with CUDA $(CUDA_VERSION)"; \
# 	uv pip install torch --upgrade --index https://download.pytorch.org/whl/cu$(CUDA_VERSION_SHORT); \
# fi
	

.PHONY: dep-check
dep-check:
	@echo "Checking that exported requirements are up to date"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)-TEMP
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION)-TEMP | sh -x
	diff -r $(REQ_LOCATION)-TEMP $(REQ_LOCATION)
	rm -rf $(REQ_LOCATION)-TEMP


.PHONY: dep-clean
dep-clean:
	@echo "clean up lock files, .venv, and requirements files"
	rm -rf .venv
	rm -rf uv.lock
	rm -rf $(REQ_LOCATION)/*.txt

# ==================================================
# checks (formatting/linting, typing, tests)
# ==================================================

# runs ruff and pycln to format the code
.PHONY: format
format:
	@echo "format the source code"
	$(PYTHON) -m ruff format --config $(PYPROJECT) .
	$(PYTHON) -m ruff check --fix --config $(PYPROJECT) .
	$(PYTHON) -m pycln --config $(PYPROJECT) --all .

# runs ruff and pycln to check if the code is formatted correctly
.PHONY: format-check
format-check:
	@echo "check if the source code is formatted correctly"
	$(PYTHON) -m ruff check --config $(PYPROJECT) .
	$(PYTHON) -m pycln --check --config $(PYPROJECT) .

# runs type checks with mypy
# at some point, need to add back --check-untyped-defs to mypy call
# but it complains when we specify arguments by keyword where positional is fine
# not sure how to fix this
.PHONY: typing
typing: clean
	@echo "running type checks"
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(PACKAGE_NAME)/
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(TESTS_DIR)/

.PHONY: test
test: clean
	@echo "running tests"
	$(PYTHON) -m pytest $(PYTEST_OPTIONS) $(TESTS_DIR)

.PHONY: check
check: clean format-check test typing
	@echo "run format checks, tests, and typing checks"

# ==================================================
# coverage & docs
# ==================================================

# generates a whole tree of documentation in html format.
# see `docs/make_docs.py` and the templates in `docs/templates/html/` for more info
.PHONY: docs-html
docs-html:
	@echo "generate html docs"
	$(PYTHON) docs/make_docs.py

# instead of a whole website, generates a single markdown file with all docs using the templates in `docs/templates/markdown/`.
# this is useful if you want to have a copy that you can grep/search, but those docs are much messier.
# docs-combined will use pandoc to convert them to other formats.
.PHONY: docs-md
docs-md:
	@echo "generate combined (single-file) docs in markdown"
	mkdir $(DOCS_DIR)/combined -p
	$(PYTHON) docs/make_docs.py --combined

# after running docs-md, this will convert the combined markdown file to other formats:
# gfm (github-flavored markdown), plain text, and html
# requires pandoc in path, pointed to by $(PANDOC)
# pdf output would be nice but requires other deps
.PHONY: docs-combined
docs-combined: docs-md
	@echo "generate combined (single-file) docs in markdown and convert to other formats"
	@echo "requires pandoc in path"
	$(PANDOC) -f markdown -t gfm $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME)_gfm.md
	$(PANDOC) -f markdown -t plain $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).txt
	$(PANDOC) -f markdown -t html $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).html

# generates coverage reports as html and text with `pytest-cov`, and a badge with `coverage-badge`
# if `.coverage` is not found, will run tests first
# also removes the `.gitignore` file that `coverage html` creates, since we count that as part of the docs
.PHONY: cov
cov:
	@echo "generate coverage reports"
	@if [ ! -f .coverage ]; then \
		echo ".coverage not found, running tests first..."; \
		$(MAKE) test; \
	fi
	mkdir $(COVERAGE_REPORTS_DIR) -p
	$(PYTHON) -m coverage report -m > $(COVERAGE_REPORTS_DIR)/coverage.txt
	$(PYTHON) -m coverage_badge -f -o $(COVERAGE_REPORTS_DIR)/coverage.svg
	$(PYTHON) -m coverage html --directory=$(COVERAGE_REPORTS_DIR)/html/
	rm -rf $(COVERAGE_REPORTS_DIR)/html/.gitignore

# runs the coverage report, then the docs, then the combined docs
# ~~~~~~~~~~~~~~~~~~~~
# demo also created for docs
# ~~~~~~~~~~~~~~~~~~~~
.PHONY: docs
docs: demo cov docs-html docs-combined
	@echo "generate all documentation and coverage reports"

# removed all generated documentation files, but leaves the templates and the `docs/make_docs.py` script
# distinct from `make clean`
.PHONY: docs-clean
docs-clean:
	@echo "remove generated docs"
	rm -rf $(DOCS_DIR)/combined/
	rm -rf $(DOCS_DIR)/$(PACKAGE_NAME)/
	rm -rf $(COVERAGE_REPORTS_DIR)/
	rm $(DOCS_DIR)/$(PACKAGE_NAME).html
	rm $(DOCS_DIR)/index.html
	rm $(DOCS_DIR)/search.js
	rm $(DOCS_DIR)/package_map.dot
	rm $(DOCS_DIR)/package_map.html


# ==================================================
# build and publish
# ==================================================

# verifies that the current branch is $(PUBLISH_BRANCH) and that git is clean
# used before publishing
.PHONY: verify-git
verify-git: 
	@echo "checking git status"
	if [ "$(shell git branch --show-current)" != $(PUBLISH_BRANCH) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not on the $(PUBLISH_BRANCH) branch, exiting!"; \
		exit 1; \
	fi; \
	if [ -n "$(shell git status --porcelain)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not clean, exiting!"; \
		exit 1; \
	fi; \


.PHONY: build
build: 
	@echo "build the package"
	uv build

# gets the commit log, checks everything, builds, and then publishes with twine
# will ask the user to confirm the new version number (and this allows for editing the tag info)
# will also print the contents of $(PYPI_TOKEN_FILE) to the console for the user to copy and paste in when prompted by twine
.PHONY: publish
publish: gen-commit-log check build verify-git version gen-version-info
	@echo "run all checks, build, and then publish"

	@echo "Enter the new version number if you want to upload to pypi and create a new tag"
	@echo "Now would also be the time to edit $(COMMIT_LOG_FILE), as that will be used as the tag description"
	@read -p "Confirm: " NEW_VERSION; \
	if [ "$$NEW_VERSION" = $(VERSION) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Version confirmed. Proceeding with publish."; \
	else \
		echo "Version mismatch, exiting: you gave $$NEW_VERSION but expected $(VERSION)"; \
		exit 1; \
	fi;

	@echo "pypi username: __token__"
	@echo "pypi token from '$(PYPI_TOKEN_FILE)' :"
	echo $$(cat $(PYPI_TOKEN_FILE))

	echo "Uploading!"; \
	echo $(VERSION) > $(LAST_VERSION_FILE); \
	git add $(LAST_VERSION_FILE); \
	git commit -m "Auto update to $(VERSION)"; \
	git tag -a $(VERSION) -F $(COMMIT_LOG_FILE); \
	git push origin $(VERSION); \
	twine upload dist/* --verbose

# ==================================================
# cleanup of temp files
# ==================================================

# cleans up temp files from formatter, type checking, tests, coverage
# removes all built files
# removes $(TESTS_TEMP_DIR) to remove temporary test files
# recursively removes all `__pycache__` directories and `*.pyc` or `*.pyo` files
# distinct from `make docs-clean`, which only removes generated documentation files
.PHONY: clean
clean:
	@echo "clean up temporary files"
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf $(PACKAGE_NAME).egg-info
	rm -rf $(TESTS_TEMP_DIR)
	$(PYTHON_BASE) -Bc "import pathlib; [p.unlink() for path in ['$(PACKAGE_NAME)', '$(TESTS_DIR)', '$(DOCS_DIR)'] for pattern in ['*.py[co]', '__pycache__/*'] for p in pathlib.Path(path).rglob(pattern)]"

.PHONY: clean-all
clean-all: clean dep-clean docs-clean
	@echo "clean up all temporary files, dep files, venv, and generated docs"


# ==================================================
# smart help command
# ==================================================

# listing targets is from stackoverflow
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
# no .PHONY because this will only be run before `make help`
# it's a separate command because getting the versions takes a bit of time
help-targets:
	@echo -n "# make targets"
	@echo ":"
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 30


.PHONY: info
info: gen-version-info get-cuda-info
	@echo "# makefile variables"
	@echo "    PYTHON = $(PYTHON)"
	@echo "    PYTHON_VERSION = $(PYTHON_VERSION)"
	@echo "    PACKAGE_NAME = $(PACKAGE_NAME)"
	@echo "    VERSION = $(VERSION)"
	@echo "    LAST_VERSION = $(LAST_VERSION)"
	@echo "    PYTEST_OPTIONS = $(PYTEST_OPTIONS)"
	@echo "    CUDA_PRESENT = $(CUDA_PRESENT)"
	@if [ "$(CUDA_PRESENT)" = "1" ]; then \
		echo "    CUDA_VERSION = $(CUDA_VERSION)"; \
		echo "    CUDA_VERSION_SHORT = $(CUDA_VERSION_SHORT)"; \
	fi

.PHONY: info-long
info-long: info
	@echo "# other variables"
	@echo "    PUBLISH_BRANCH = $(PUBLISH_BRANCH)"
	@echo "    DOCS_DIR = $(DOCS_DIR)"
	@echo "    COVERAGE_REPORTS_DIR = $(COVERAGE_REPORTS_DIR)"
	@echo "    TESTS_DIR = $(TESTS_DIR)"
	@echo "    TESTS_TEMP_DIR = $(TESTS_TEMP_DIR)"
	@echo "    PYPROJECT = $(PYPROJECT)"
	@echo "    REQ_LOCATION = $(REQ_LOCATION)"
	@echo "    REQ_BASE = $(REQ_BASE)"
	@echo "    REQ_EXTRAS = $(REQ_EXTRAS)"
	@echo "    REQ_DEV = $(REQ_DEV)"
	@echo "    REQ_ALL = $(REQ_ALL)"
	@echo "    LOCAL_DIR = $(LOCAL_DIR)"
	@echo "    PYPI_TOKEN_FILE = $(PYPI_TOKEN_FILE)"
	@echo "    LAST_VERSION_FILE = $(LAST_VERSION_FILE)"
	@echo "    PYTHON_BASE = $(PYTHON_BASE)"
	@echo "    COMMIT_LOG_FILE = $(COMMIT_LOG_FILE)"
	@echo "    PANDOC = $(PANDOC)"
	@echo "    COV = $(COV)"
	@echo "    VERBOSE = $(VERBOSE)"
	@echo "    RUN_GLOBAL = $(RUN_GLOBAL)"
	@echo "    TYPECHECK_ARGS = $(TYPECHECK_ARGS)"

# immediately print out the help targets, and then local variables (but those take a bit longer)
.PHONY: help
help: help-targets info
	@echo -n ""

# ==================================================
# custom targets
# ==================================================
# (put them down here, or delimit with ~~~~~)


.PHONY: demo
demo:
	@echo "example of code output"
	$(PYTHON) -m lmcat -o example_output.md
``````{ end_of_file: "makefile" }

``````{ path: "pyproject.toml" }
[project]
name = "lmcat"
version = "0.0.1"
description = "concatenating files for tossing them into a language model"
authors = [
	{ name = "Michael Ivanitskiy", email = "mivanits@umich.edu" }
]
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
	"igittigitt>=2.1.5",
]


[dependency-groups]
dev = [
	# test
	"pytest>=8.2.2",
	# coverage
	"pytest-cov>=4.1.0",
	"coverage-badge>=1.1.0",
	# type checking
	"mypy>=1.0.1",
	# docs
	'pdoc>=14.6.0',
	# tomli since no tomlib in python < 3.11
	"tomli>=2.1.0; python_version < '3.11'",
]
lint = [
	# lint
	"pycln>=2.1.3",
	"ruff>=0.4.8",
]

[tool.uv]
default-groups = ["dev", "lint"]

[project.urls]
Homepage = "https://miv.name/lmcat"
Documentation = "https://miv.name/lmcat"
Repository = "https://github.com/mivanit/lmcat"
Issues = "https://github.com/mivanit/lmcat/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# ruff config
[tool.ruff]
exclude = ["__pycache__"]

[tool.ruff.format]
indent-style = "tab"
skip-magic-trailing-comma = false

# Custom export configurations
[tool.uv-exports]
args = [
	"--no-hashes"
]
exports = [
	# no groups, no extras, just the base dependencies
    { name = "base", groups = false, extras = false },
	# all extras but no groups
    { name = "extras", groups = false, extras = true },
	# include the dev group (this is the default behavior)
    { name = "dev", groups = ["dev"] },
	# only the lint group -- custom options for this
	{ name = "lint", options = ["--only-group", "lint"] },
	# all groups and extras
    { name = "all", filename="requirements.txt", groups = true, extras=true },
	# all groups and extras, a different way
	{ name = "all", groups = true, options = ["--all-extras"] },
]


``````{ end_of_file: "pyproject.toml" }
``````{ end_of_file: "example_output.md" }

``````{ path: "makefile" }
# ==================================================
# configuration & variables
# ==================================================

# !!! MODIFY AT LEAST THIS PART TO SUIT YOUR PROJECT !!!
# it assumes that the source is in a directory named the same as the package name
# this also gets passed to some other places
PACKAGE_NAME := lmcat

# for checking you are on the right branch when publishing
PUBLISH_BRANCH := main

# where to put docs
DOCS_DIR := docs

# where to put the coverage reports
# note that this will be published with the docs!
# modify the `docs` targets and `.gitignore` if you don't want that
COVERAGE_REPORTS_DIR := docs/coverage

# where the tests are, for pytest
TESTS_DIR := tests

# tests temp directory to clean up. will remove this in `make clean`
TESTS_TEMP_DIR := $(TESTS_DIR)/_temp/

# probably don't change these:
# --------------------------------------------------

# where the pyproject.toml file is. no idea why you would change this but just in case
PYPROJECT := pyproject.toml

# requirements.txt files for base package, all extras, dev, and all
REQ_LOCATION := .github/requirements

# local files (don't push this to git)
LOCAL_DIR := .github/local

# will print this token when publishing. make sure not to commit this file!!!
PYPI_TOKEN_FILE := $(LOCAL_DIR)/.pypi-token

# version files
VERSIONS_DIR := .github/versions

# the last version that was auto-uploaded. will use this to create a commit log for version tag
# see `gen-commit-log` target
LAST_VERSION_FILE := $(VERSIONS_DIR)/.lastversion

# current version (writing to file needed due to shell escaping issues)
VERSION_FILE := $(VERSIONS_DIR)/.version

# base python to use. Will add `uv run` in front of this if `RUN_GLOBAL` is not set to 1
PYTHON_BASE := python

# where the commit log will be stored
COMMIT_LOG_FILE := $(LOCAL_DIR)/.commit_log

# pandoc commands (for docs)
PANDOC ?= pandoc

# version vars - extracted automatically from `pyproject.toml`, `$(LAST_VERSION_FILE)`, and $(PYTHON)
# --------------------------------------------------

# assuming your `pyproject.toml` has a line that looks like `version = "0.0.1"`, `gen-version-info` will extract this
VERSION := NULL
# `gen-version-info` will read the last version from `$(LAST_VERSION_FILE)`, or `NULL` if it doesn't exist
LAST_VERSION := NULL
# get the python version, now that we have picked the python command
PYTHON_VERSION := NULL

# cuda version
# --------------------------------------------------
# 0 or 1
CUDA_PRESENT :=
# a version like "12.4" or "NULL"
CUDA_VERSION := NULL
# a version like "124" or "NULL"
CUDA_VERSION_SHORT := NULL


# python scripts we want to use inside the makefile
# --------------------------------------------------

# create commands for exporting requirements as specified in `pyproject.toml:tool.uv-exports.exports`
define EXPORT_SCRIPT
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from pathlib import Path
from typing import Union, List, Optional

pyproject_path: Path = Path(sys.argv[1])
output_dir: Path = Path(sys.argv[2])

with open(pyproject_path, 'rb') as f:
	pyproject_data: dict = tomllib.load(f)

# all available groups
all_groups: List[str] = list(pyproject_data.get('dependency-groups', {}).keys())
all_extras: List[str] = list(pyproject_data.get('project', {}).get('optional-dependencies', {}).keys())

# options for exporting
export_opts: dict = pyproject_data.get('tool', {}).get('uv-exports', {})

# what are we exporting?
exports: List[str] = export_opts.get('exports', [])
if not exports:
	exports = [{'name': 'all', 'groups': [], 'extras': [], 'options': []}]

# export each configuration
for export in exports:
	# get name and validate
	name = export.get('name')
	if not name or not name.isalnum():
		print(f"Export configuration missing valid 'name' field {export}", file=sys.stderr)
		continue

	# get other options with default fallbacks
	filename: str = export.get('filename') or f"requirements-{name}.txt"
	groups: Union[List[str], bool, None] = export.get('groups', None)
	extras: Union[List[str], bool] = export.get('extras', [])
	options: List[str] = export.get('options', [])

	# init command
	cmd: List[str] = ['uv', 'export'] + export_opts.get('args', [])

	# handle groups
	if groups is not None:
		groups_list: List[str] = []
		if isinstance(groups, bool):
			if groups:
				groups_list = all_groups.copy()
		else:
			groups_list = groups
		
		for group in all_groups:
			if group in groups_list:
				cmd.extend(['--group', group])
			else:
				cmd.extend(['--no-group', group])

	# handle extras
	extras_list: List[str] = []
	if isinstance(extras, bool):
		if extras:
			extras_list = all_extras.copy()
	else:
		extras_list = extras

	for extra in extras_list:
		cmd.extend(['--extra', extra])

	cmd.extend(options)

	output_path = output_dir / filename
	print(f"{' '.join(cmd)} > {output_path.as_posix()}")
endef

export EXPORT_SCRIPT

# get the version from `pyproject.toml:project.version`
define GET_VERSION_SCRIPT
import sys

try:
	if sys.version_info >= (3, 11):
		import tomllib
	else:
		import tomli as tomllib

	pyproject_path = '$(PYPROJECT)'

	with open(pyproject_path, 'rb') as f:
		pyproject_data = tomllib.load(f)

	print('v' + pyproject_data['project']['version'], end='')
except Exception as e:
	print('NULL', end='')
	sys.exit(1)
endef

export GET_VERSION_SCRIPT


# get the commit log since the last version from `$(LAST_VERSION_FILE)`
define GET_COMMIT_LOG_SCRIPT
import subprocess
import sys

last_version = sys.argv[1].strip()
commit_log_file = '$(COMMIT_LOG_FILE)'

if last_version == 'NULL':
    print('!!! ERROR !!!', file=sys.stderr)
    print('LAST_VERSION is NULL, can\'t get commit log!', file=sys.stderr)
    sys.exit(1)

try:
    log_cmd = ['git', 'log', f'{last_version}..HEAD', '--pretty=format:- %s (%h)']
    commits = subprocess.check_output(log_cmd).decode('utf-8').strip().split('\n')
    with open(commit_log_file, 'w') as f:
        f.write('\n'.join(reversed(commits)))
except subprocess.CalledProcessError as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
endef

export GET_COMMIT_LOG_SCRIPT

# get cuda information and whether torch sees it
define CHECK_TORCH_SCRIPT
import os
import sys
print(f'python version: {sys.version}')
print(f"\tpython executable path: {str(sys.executable)}")
print(f"\tsys_platform: {sys.platform}")
print(f'\tcurrent working directory: {os.getcwd()}')
print(f'\tHost name: {os.name}')
print(f'\tCPU count: {os.cpu_count()}')
print()

try:
	import torch
except Exception as e:
	print('ERROR: error importing torch, terminating        ')
	print('-'*50)
	raise e
	sys.exit(1)

print(f'torch version: {torch.__version__}')

print(f'\t{torch.cuda.is_available() = }')

if torch.cuda.is_available():
	# print('\tCUDA is available on torch')
	print(f'\tCUDA version via torch: {torch.version.cuda}')

	if torch.cuda.device_count() > 0:
		print(f"\tcurrent device: {torch.cuda.current_device() = }\n")
		n_devices: int = torch.cuda.device_count()
		print(f"detected {n_devices = }")
		for current_device in range(n_devices):
			try:
				# print(f'checking current device {current_device} of {torch.cuda.device_count()} devices')
				print(f'\tdevice {current_device}')
				dev_prop = torch.cuda.get_device_properties(torch.device(0))
				print(f'\t    name:                   {dev_prop.name}')
				print(f'\t    version:                {dev_prop.major}.{dev_prop.minor}')
				print(f'\t    total_memory:           {dev_prop.total_memory} ({dev_prop.total_memory:.1e})')
				print(f'\t    multi_processor_count:  {dev_prop.multi_processor_count}')
				print(f'\t    is_integrated:          {dev_prop.is_integrated}')
				print(f'\t    is_multi_gpu_board:     {dev_prop.is_multi_gpu_board}')
				print(f'\t')
			except Exception as e:
				print(f'Exception when trying to get properties of device {current_device}')
				raise e
		sys.exit(0)
	else:
		print(f'ERROR: {torch.cuda.device_count()} devices detected, invalid')
		print('-'*50)
		sys.exit(1)

else:
	print('ERROR: CUDA is NOT available, terminating')
	print('-'*50)
	sys.exit(1)
endef

export CHECK_TORCH_SCRIPT


# ==================================================
# reading command line options
# ==================================================

# for formatting or something, we might want to run python without uv
# RUN_GLOBAL=1 to use global `PYTHON_BASE` instead of `uv run $(PYTHON_BASE)`
RUN_GLOBAL ?= 0

ifeq ($(RUN_GLOBAL),0)
	PYTHON = uv run $(PYTHON_BASE)
else
	PYTHON = $(PYTHON_BASE)
endif

# if you want different behavior for different python versions
# --------------------------------------------------
# COMPATIBILITY_MODE := $(shell $(PYTHON) -c "import sys; print(1 if sys.version_info < (3, 10) else 0)")

# options we might want to pass to pytest
# --------------------------------------------------

# base options for pytest, will be appended to if `COV` or `VERBOSE` are 1.
# user can also set this when running make to add more options
PYTEST_OPTIONS ?=

# set to `1` to run pytest with `--cov=.` to get coverage reports in a `.coverage` file
COV ?= 1
# set to `1` to run pytest with `--verbose`
VERBOSE ?= 0

ifeq ($(VERBOSE),1)
	PYTEST_OPTIONS += --verbose
endif

ifeq ($(COV),1)
	PYTEST_OPTIONS += --cov=.
endif

# ==================================================
# default target (help)
# ==================================================

# first/default target is help
.PHONY: default
default: help

# ==================================================
# getting version info
# we do this in a separate target because it takes a bit of time
# ==================================================

# this recipe is weird. we need it because:
# - a one liner for getting the version with toml is unwieldy, and using regex is fragile
# - using $$GET_VERSION_SCRIPT within $(shell ...) doesn't work because of escaping issues
# - trying to write to the file inside the `gen-version-info` recipe doesn't work, 
# 	shell eval happens before our `python -c ...` gets run and `cat` doesn't see the new file
.PHONY: write-proj-version
write-proj-version:
	@mkdir -p $(VERSIONS_DIR)
	@$(PYTHON) -c "$$GET_VERSION_SCRIPT" > $(VERSION_FILE)

# gets version info from $(PYPROJECT), last version from $(LAST_VERSION_FILE), and python version
# uses just `python` for everything except getting the python version. no echo here, because this is "private"
.PHONY: gen-version-info
gen-version-info: write-proj-version
	@mkdir -p $(LOCAL_DIR)
	$(eval VERSION := $(shell cat $(VERSION_FILE)) )
	$(eval LAST_VERSION := $(shell [ -f $(LAST_VERSION_FILE) ] && cat $(LAST_VERSION_FILE) || echo NULL) )
	$(eval PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')") )

# getting commit log since the tag specified in $(LAST_VERSION_FILE)
# will write to $(COMMIT_LOG_FILE)
# when publishing, the contents of $(COMMIT_LOG_FILE) will be used as the tag description (but can be edited during the process)
# no echo here, because this is "private"
.PHONY: gen-commit-log
gen-commit-log: gen-version-info
	@if [ "$(LAST_VERSION)" = "NULL" ]; then \
		echo "!!! ERROR !!!"; \
		echo "LAST_VERSION is NULL, cant get commit log!"; \
		exit 1; \
	fi
	@mkdir -p $(LOCAL_DIR)
	@$(PYTHON) -c "$$GET_COMMIT_LOG_SCRIPT" "$(LAST_VERSION)"


# force the version info to be read, printing it out
# also force the commit log to be generated, and cat it out
.PHONY: version
version: gen-commit-log
	@echo "Current version is $(VERSION), last auto-uploaded version is $(LAST_VERSION)"
	@echo "Commit log since last version from '$(COMMIT_LOG_FILE)':"
	@cat $(COMMIT_LOG_FILE)
	@echo ""
	@if [ "$(VERSION)" = "$(LAST_VERSION)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Python package $(VERSION) is the same as last published version $(LAST_VERSION), exiting!"; \
		exit 1; \
	fi


# ==================================================
# dependencies and setup
# ==================================================

.PHONY: setup
setup: dep-check
	@echo "install and update via uv"
	@echo "To activate the virtual environment, run one of:"
	@echo "  source .venv/bin/activate"
	@echo "  source .venv/Scripts/activate"

.PHONY: get-cuda-info
get-cuda-info:
	$(eval CUDA_PRESENT := $(shell if command -v nvcc > /dev/null 2>&1; then echo 1; else echo 0; fi))
	$(eval CUDA_VERSION := $(if $(filter $(CUDA_PRESENT),1),$(shell nvcc --version 2>/dev/null | grep "release" | awk '{print $$5}' | sed 's/,//'),NULL))
	$(eval CUDA_VERSION_SHORT := $(if $(filter $(CUDA_PRESENT),1),$(shell echo $(CUDA_VERSION) | sed 's/\.//'),NULL))

.PHONY: dep-check-torch
dep-check-torch:
	@echo "see if torch is installed, and which CUDA version and devices it sees"
	$(PYTHON) -c "$$CHECK_TORCH_SCRIPT"

.PHONY: dep
dep: get-cuda-info
	@echo "Exporting dependencies as per $(PYPROJECT) section 'tool.uv-exports.exports'"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION) | sh -x
	
# @if [ "$(CUDA_PRESENT)" = "1" ]; then \
# 	echo "CUDA is present, installing torch with CUDA $(CUDA_VERSION)"; \
# 	uv pip install torch --upgrade --index https://download.pytorch.org/whl/cu$(CUDA_VERSION_SHORT); \
# fi
	

.PHONY: dep-check
dep-check:
	@echo "Checking that exported requirements are up to date"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)-TEMP
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION)-TEMP | sh -x
	diff -r $(REQ_LOCATION)-TEMP $(REQ_LOCATION)
	rm -rf $(REQ_LOCATION)-TEMP


.PHONY: dep-clean
dep-clean:
	@echo "clean up lock files, .venv, and requirements files"
	rm -rf .venv
	rm -rf uv.lock
	rm -rf $(REQ_LOCATION)/*.txt

# ==================================================
# checks (formatting/linting, typing, tests)
# ==================================================

# runs ruff and pycln to format the code
.PHONY: format
format:
	@echo "format the source code"
	$(PYTHON) -m ruff format --config $(PYPROJECT) .
	$(PYTHON) -m ruff check --fix --config $(PYPROJECT) .
	$(PYTHON) -m pycln --config $(PYPROJECT) --all .

# runs ruff and pycln to check if the code is formatted correctly
.PHONY: format-check
format-check:
	@echo "check if the source code is formatted correctly"
	$(PYTHON) -m ruff check --config $(PYPROJECT) .
	$(PYTHON) -m pycln --check --config $(PYPROJECT) .

# runs type checks with mypy
# at some point, need to add back --check-untyped-defs to mypy call
# but it complains when we specify arguments by keyword where positional is fine
# not sure how to fix this
.PHONY: typing
typing: clean
	@echo "running type checks"
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(PACKAGE_NAME)/
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(TESTS_DIR)/

.PHONY: test
test: clean
	@echo "running tests"
	$(PYTHON) -m pytest $(PYTEST_OPTIONS) $(TESTS_DIR)

.PHONY: check
check: clean format-check test typing
	@echo "run format checks, tests, and typing checks"

# ==================================================
# coverage & docs
# ==================================================

# generates a whole tree of documentation in html format.
# see `docs/make_docs.py` and the templates in `docs/templates/html/` for more info
.PHONY: docs-html
docs-html:
	@echo "generate html docs"
	$(PYTHON) docs/make_docs.py

# instead of a whole website, generates a single markdown file with all docs using the templates in `docs/templates/markdown/`.
# this is useful if you want to have a copy that you can grep/search, but those docs are much messier.
# docs-combined will use pandoc to convert them to other formats.
.PHONY: docs-md
docs-md:
	@echo "generate combined (single-file) docs in markdown"
	mkdir $(DOCS_DIR)/combined -p
	$(PYTHON) docs/make_docs.py --combined

# after running docs-md, this will convert the combined markdown file to other formats:
# gfm (github-flavored markdown), plain text, and html
# requires pandoc in path, pointed to by $(PANDOC)
# pdf output would be nice but requires other deps
.PHONY: docs-combined
docs-combined: docs-md
	@echo "generate combined (single-file) docs in markdown and convert to other formats"
	@echo "requires pandoc in path"
	$(PANDOC) -f markdown -t gfm $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME)_gfm.md
	$(PANDOC) -f markdown -t plain $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).txt
	$(PANDOC) -f markdown -t html $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).html

# generates coverage reports as html and text with `pytest-cov`, and a badge with `coverage-badge`
# if `.coverage` is not found, will run tests first
# also removes the `.gitignore` file that `coverage html` creates, since we count that as part of the docs
.PHONY: cov
cov:
	@echo "generate coverage reports"
	@if [ ! -f .coverage ]; then \
		echo ".coverage not found, running tests first..."; \
		$(MAKE) test; \
	fi
	mkdir $(COVERAGE_REPORTS_DIR) -p
	$(PYTHON) -m coverage report -m > $(COVERAGE_REPORTS_DIR)/coverage.txt
	$(PYTHON) -m coverage_badge -f -o $(COVERAGE_REPORTS_DIR)/coverage.svg
	$(PYTHON) -m coverage html --directory=$(COVERAGE_REPORTS_DIR)/html/
	rm -rf $(COVERAGE_REPORTS_DIR)/html/.gitignore

# runs the coverage report, then the docs, then the combined docs
# ~~~~~~~~~~~~~~~~~~~~
# demo also created for docs
# ~~~~~~~~~~~~~~~~~~~~
.PHONY: docs
docs: demo cov docs-html docs-combined
	@echo "generate all documentation and coverage reports"

# removed all generated documentation files, but leaves the templates and the `docs/make_docs.py` script
# distinct from `make clean`
.PHONY: docs-clean
docs-clean:
	@echo "remove generated docs"
	rm -rf $(DOCS_DIR)/combined/
	rm -rf $(DOCS_DIR)/$(PACKAGE_NAME)/
	rm -rf $(COVERAGE_REPORTS_DIR)/
	rm $(DOCS_DIR)/$(PACKAGE_NAME).html
	rm $(DOCS_DIR)/index.html
	rm $(DOCS_DIR)/search.js
	rm $(DOCS_DIR)/package_map.dot
	rm $(DOCS_DIR)/package_map.html


# ==================================================
# build and publish
# ==================================================

# verifies that the current branch is $(PUBLISH_BRANCH) and that git is clean
# used before publishing
.PHONY: verify-git
verify-git: 
	@echo "checking git status"
	if [ "$(shell git branch --show-current)" != $(PUBLISH_BRANCH) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not on the $(PUBLISH_BRANCH) branch, exiting!"; \
		exit 1; \
	fi; \
	if [ -n "$(shell git status --porcelain)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not clean, exiting!"; \
		exit 1; \
	fi; \


.PHONY: build
build: 
	@echo "build the package"
	uv build

# gets the commit log, checks everything, builds, and then publishes with twine
# will ask the user to confirm the new version number (and this allows for editing the tag info)
# will also print the contents of $(PYPI_TOKEN_FILE) to the console for the user to copy and paste in when prompted by twine
.PHONY: publish
publish: gen-commit-log check build verify-git version gen-version-info
	@echo "run all checks, build, and then publish"

	@echo "Enter the new version number if you want to upload to pypi and create a new tag"
	@echo "Now would also be the time to edit $(COMMIT_LOG_FILE), as that will be used as the tag description"
	@read -p "Confirm: " NEW_VERSION; \
	if [ "$$NEW_VERSION" = $(VERSION) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Version confirmed. Proceeding with publish."; \
	else \
		echo "Version mismatch, exiting: you gave $$NEW_VERSION but expected $(VERSION)"; \
		exit 1; \
	fi;

	@echo "pypi username: __token__"
	@echo "pypi token from '$(PYPI_TOKEN_FILE)' :"
	echo $$(cat $(PYPI_TOKEN_FILE))

	echo "Uploading!"; \
	echo $(VERSION) > $(LAST_VERSION_FILE); \
	git add $(LAST_VERSION_FILE); \
	git commit -m "Auto update to $(VERSION)"; \
	git tag -a $(VERSION) -F $(COMMIT_LOG_FILE); \
	git push origin $(VERSION); \
	twine upload dist/* --verbose

# ==================================================
# cleanup of temp files
# ==================================================

# cleans up temp files from formatter, type checking, tests, coverage
# removes all built files
# removes $(TESTS_TEMP_DIR) to remove temporary test files
# recursively removes all `__pycache__` directories and `*.pyc` or `*.pyo` files
# distinct from `make docs-clean`, which only removes generated documentation files
.PHONY: clean
clean:
	@echo "clean up temporary files"
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf $(PACKAGE_NAME).egg-info
	rm -rf $(TESTS_TEMP_DIR)
	$(PYTHON_BASE) -Bc "import pathlib; [p.unlink() for path in ['$(PACKAGE_NAME)', '$(TESTS_DIR)', '$(DOCS_DIR)'] for pattern in ['*.py[co]', '__pycache__/*'] for p in pathlib.Path(path).rglob(pattern)]"

.PHONY: clean-all
clean-all: clean dep-clean docs-clean
	@echo "clean up all temporary files, dep files, venv, and generated docs"


# ==================================================
# smart help command
# ==================================================

# listing targets is from stackoverflow
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
# no .PHONY because this will only be run before `make help`
# it's a separate command because getting the versions takes a bit of time
help-targets:
	@echo -n "# make targets"
	@echo ":"
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 30


.PHONY: info
info: gen-version-info get-cuda-info
	@echo "# makefile variables"
	@echo "    PYTHON = $(PYTHON)"
	@echo "    PYTHON_VERSION = $(PYTHON_VERSION)"
	@echo "    PACKAGE_NAME = $(PACKAGE_NAME)"
	@echo "    VERSION = $(VERSION)"
	@echo "    LAST_VERSION = $(LAST_VERSION)"
	@echo "    PYTEST_OPTIONS = $(PYTEST_OPTIONS)"
	@echo "    CUDA_PRESENT = $(CUDA_PRESENT)"
	@if [ "$(CUDA_PRESENT)" = "1" ]; then \
		echo "    CUDA_VERSION = $(CUDA_VERSION)"; \
		echo "    CUDA_VERSION_SHORT = $(CUDA_VERSION_SHORT)"; \
	fi

.PHONY: info-long
info-long: info
	@echo "# other variables"
	@echo "    PUBLISH_BRANCH = $(PUBLISH_BRANCH)"
	@echo "    DOCS_DIR = $(DOCS_DIR)"
	@echo "    COVERAGE_REPORTS_DIR = $(COVERAGE_REPORTS_DIR)"
	@echo "    TESTS_DIR = $(TESTS_DIR)"
	@echo "    TESTS_TEMP_DIR = $(TESTS_TEMP_DIR)"
	@echo "    PYPROJECT = $(PYPROJECT)"
	@echo "    REQ_LOCATION = $(REQ_LOCATION)"
	@echo "    REQ_BASE = $(REQ_BASE)"
	@echo "    REQ_EXTRAS = $(REQ_EXTRAS)"
	@echo "    REQ_DEV = $(REQ_DEV)"
	@echo "    REQ_ALL = $(REQ_ALL)"
	@echo "    LOCAL_DIR = $(LOCAL_DIR)"
	@echo "    PYPI_TOKEN_FILE = $(PYPI_TOKEN_FILE)"
	@echo "    LAST_VERSION_FILE = $(LAST_VERSION_FILE)"
	@echo "    PYTHON_BASE = $(PYTHON_BASE)"
	@echo "    COMMIT_LOG_FILE = $(COMMIT_LOG_FILE)"
	@echo "    PANDOC = $(PANDOC)"
	@echo "    COV = $(COV)"
	@echo "    VERBOSE = $(VERBOSE)"
	@echo "    RUN_GLOBAL = $(RUN_GLOBAL)"
	@echo "    TYPECHECK_ARGS = $(TYPECHECK_ARGS)"

# immediately print out the help targets, and then local variables (but those take a bit longer)
.PHONY: help
help: help-targets info
	@echo -n ""

# ==================================================
# custom targets
# ==================================================
# (put them down here, or delimit with ~~~~~)


.PHONY: demo
demo:
	@echo "example of code output"
	$(PYTHON) -m lmcat -o example_output.md
``````{ end_of_file: "makefile" }

``````{ path: "pyproject.toml" }
[project]
name = "lmcat"
version = "0.0.1"
description = "concatenating files for tossing them into a language model"
authors = [
	{ name = "Michael Ivanitskiy", email = "mivanits@umich.edu" }
]
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
	"igittigitt>=2.1.5",
]


[dependency-groups]
dev = [
	# test
	"pytest>=8.2.2",
	# coverage
	"pytest-cov>=4.1.0",
	"coverage-badge>=1.1.0",
	# type checking
	"mypy>=1.0.1",
	# docs
	'pdoc>=14.6.0',
	# tomli since no tomlib in python < 3.11
	"tomli>=2.1.0; python_version < '3.11'",
]
lint = [
	# lint
	"pycln>=2.1.3",
	"ruff>=0.4.8",
]

[tool.uv]
default-groups = ["dev", "lint"]

[project.urls]
Homepage = "https://miv.name/lmcat"
Documentation = "https://miv.name/lmcat"
Repository = "https://github.com/mivanit/lmcat"
Issues = "https://github.com/mivanit/lmcat/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# ruff config
[tool.ruff]
exclude = ["__pycache__"]

[tool.ruff.format]
indent-style = "tab"
skip-magic-trailing-comma = false

# Custom export configurations
[tool.uv-exports]
args = [
	"--no-hashes"
]
exports = [
	# no groups, no extras, just the base dependencies
    { name = "base", groups = false, extras = false },
	# all extras but no groups
    { name = "extras", groups = false, extras = true },
	# include the dev group (this is the default behavior)
    { name = "dev", groups = ["dev"] },
	# only the lint group -- custom options for this
	{ name = "lint", options = ["--only-group", "lint"] },
	# all groups and extras
    { name = "all", filename="requirements.txt", groups = true, extras=true },
	# all groups and extras, a different way
	{ name = "all", groups = true, options = ["--all-extras"] },
]


``````{ end_of_file: "pyproject.toml" }
``````{ end_of_file: "example_output.md" }

``````{ path: "makefile" }
# ==================================================
# configuration & variables
# ==================================================

# !!! MODIFY AT LEAST THIS PART TO SUIT YOUR PROJECT !!!
# it assumes that the source is in a directory named the same as the package name
# this also gets passed to some other places
PACKAGE_NAME := lmcat

# for checking you are on the right branch when publishing
PUBLISH_BRANCH := main

# where to put docs
DOCS_DIR := docs

# where to put the coverage reports
# note that this will be published with the docs!
# modify the `docs` targets and `.gitignore` if you don't want that
COVERAGE_REPORTS_DIR := docs/coverage

# where the tests are, for pytest
TESTS_DIR := tests

# tests temp directory to clean up. will remove this in `make clean`
TESTS_TEMP_DIR := $(TESTS_DIR)/_temp/

# probably don't change these:
# --------------------------------------------------

# where the pyproject.toml file is. no idea why you would change this but just in case
PYPROJECT := pyproject.toml

# requirements.txt files for base package, all extras, dev, and all
REQ_LOCATION := .github/requirements

# local files (don't push this to git)
LOCAL_DIR := .github/local

# will print this token when publishing. make sure not to commit this file!!!
PYPI_TOKEN_FILE := $(LOCAL_DIR)/.pypi-token

# version files
VERSIONS_DIR := .github/versions

# the last version that was auto-uploaded. will use this to create a commit log for version tag
# see `gen-commit-log` target
LAST_VERSION_FILE := $(VERSIONS_DIR)/.lastversion

# current version (writing to file needed due to shell escaping issues)
VERSION_FILE := $(VERSIONS_DIR)/.version

# base python to use. Will add `uv run` in front of this if `RUN_GLOBAL` is not set to 1
PYTHON_BASE := python

# where the commit log will be stored
COMMIT_LOG_FILE := $(LOCAL_DIR)/.commit_log

# pandoc commands (for docs)
PANDOC ?= pandoc

# version vars - extracted automatically from `pyproject.toml`, `$(LAST_VERSION_FILE)`, and $(PYTHON)
# --------------------------------------------------

# assuming your `pyproject.toml` has a line that looks like `version = "0.0.1"`, `gen-version-info` will extract this
VERSION := NULL
# `gen-version-info` will read the last version from `$(LAST_VERSION_FILE)`, or `NULL` if it doesn't exist
LAST_VERSION := NULL
# get the python version, now that we have picked the python command
PYTHON_VERSION := NULL

# cuda version
# --------------------------------------------------
# 0 or 1
CUDA_PRESENT :=
# a version like "12.4" or "NULL"
CUDA_VERSION := NULL
# a version like "124" or "NULL"
CUDA_VERSION_SHORT := NULL


# python scripts we want to use inside the makefile
# --------------------------------------------------

# create commands for exporting requirements as specified in `pyproject.toml:tool.uv-exports.exports`
define EXPORT_SCRIPT
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from pathlib import Path
from typing import Union, List, Optional

pyproject_path: Path = Path(sys.argv[1])
output_dir: Path = Path(sys.argv[2])

with open(pyproject_path, 'rb') as f:
	pyproject_data: dict = tomllib.load(f)

# all available groups
all_groups: List[str] = list(pyproject_data.get('dependency-groups', {}).keys())
all_extras: List[str] = list(pyproject_data.get('project', {}).get('optional-dependencies', {}).keys())

# options for exporting
export_opts: dict = pyproject_data.get('tool', {}).get('uv-exports', {})

# what are we exporting?
exports: List[str] = export_opts.get('exports', [])
if not exports:
	exports = [{'name': 'all', 'groups': [], 'extras': [], 'options': []}]

# export each configuration
for export in exports:
	# get name and validate
	name = export.get('name')
	if not name or not name.isalnum():
		print(f"Export configuration missing valid 'name' field {export}", file=sys.stderr)
		continue

	# get other options with default fallbacks
	filename: str = export.get('filename') or f"requirements-{name}.txt"
	groups: Union[List[str], bool, None] = export.get('groups', None)
	extras: Union[List[str], bool] = export.get('extras', [])
	options: List[str] = export.get('options', [])

	# init command
	cmd: List[str] = ['uv', 'export'] + export_opts.get('args', [])

	# handle groups
	if groups is not None:
		groups_list: List[str] = []
		if isinstance(groups, bool):
			if groups:
				groups_list = all_groups.copy()
		else:
			groups_list = groups
		
		for group in all_groups:
			if group in groups_list:
				cmd.extend(['--group', group])
			else:
				cmd.extend(['--no-group', group])

	# handle extras
	extras_list: List[str] = []
	if isinstance(extras, bool):
		if extras:
			extras_list = all_extras.copy()
	else:
		extras_list = extras

	for extra in extras_list:
		cmd.extend(['--extra', extra])

	cmd.extend(options)

	output_path = output_dir / filename
	print(f"{' '.join(cmd)} > {output_path.as_posix()}")
endef

export EXPORT_SCRIPT

# get the version from `pyproject.toml:project.version`
define GET_VERSION_SCRIPT
import sys

try:
	if sys.version_info >= (3, 11):
		import tomllib
	else:
		import tomli as tomllib

	pyproject_path = '$(PYPROJECT)'

	with open(pyproject_path, 'rb') as f:
		pyproject_data = tomllib.load(f)

	print('v' + pyproject_data['project']['version'], end='')
except Exception as e:
	print('NULL', end='')
	sys.exit(1)
endef

export GET_VERSION_SCRIPT


# get the commit log since the last version from `$(LAST_VERSION_FILE)`
define GET_COMMIT_LOG_SCRIPT
import subprocess
import sys

last_version = sys.argv[1].strip()
commit_log_file = '$(COMMIT_LOG_FILE)'

if last_version == 'NULL':
    print('!!! ERROR !!!', file=sys.stderr)
    print('LAST_VERSION is NULL, can\'t get commit log!', file=sys.stderr)
    sys.exit(1)

try:
    log_cmd = ['git', 'log', f'{last_version}..HEAD', '--pretty=format:- %s (%h)']
    commits = subprocess.check_output(log_cmd).decode('utf-8').strip().split('\n')
    with open(commit_log_file, 'w') as f:
        f.write('\n'.join(reversed(commits)))
except subprocess.CalledProcessError as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
endef

export GET_COMMIT_LOG_SCRIPT

# get cuda information and whether torch sees it
define CHECK_TORCH_SCRIPT
import os
import sys
print(f'python version: {sys.version}')
print(f"\tpython executable path: {str(sys.executable)}")
print(f"\tsys_platform: {sys.platform}")
print(f'\tcurrent working directory: {os.getcwd()}')
print(f'\tHost name: {os.name}')
print(f'\tCPU count: {os.cpu_count()}')
print()

try:
	import torch
except Exception as e:
	print('ERROR: error importing torch, terminating        ')
	print('-'*50)
	raise e
	sys.exit(1)

print(f'torch version: {torch.__version__}')

print(f'\t{torch.cuda.is_available() = }')

if torch.cuda.is_available():
	# print('\tCUDA is available on torch')
	print(f'\tCUDA version via torch: {torch.version.cuda}')

	if torch.cuda.device_count() > 0:
		print(f"\tcurrent device: {torch.cuda.current_device() = }\n")
		n_devices: int = torch.cuda.device_count()
		print(f"detected {n_devices = }")
		for current_device in range(n_devices):
			try:
				# print(f'checking current device {current_device} of {torch.cuda.device_count()} devices')
				print(f'\tdevice {current_device}')
				dev_prop = torch.cuda.get_device_properties(torch.device(0))
				print(f'\t    name:                   {dev_prop.name}')
				print(f'\t    version:                {dev_prop.major}.{dev_prop.minor}')
				print(f'\t    total_memory:           {dev_prop.total_memory} ({dev_prop.total_memory:.1e})')
				print(f'\t    multi_processor_count:  {dev_prop.multi_processor_count}')
				print(f'\t    is_integrated:          {dev_prop.is_integrated}')
				print(f'\t    is_multi_gpu_board:     {dev_prop.is_multi_gpu_board}')
				print(f'\t')
			except Exception as e:
				print(f'Exception when trying to get properties of device {current_device}')
				raise e
		sys.exit(0)
	else:
		print(f'ERROR: {torch.cuda.device_count()} devices detected, invalid')
		print('-'*50)
		sys.exit(1)

else:
	print('ERROR: CUDA is NOT available, terminating')
	print('-'*50)
	sys.exit(1)
endef

export CHECK_TORCH_SCRIPT


# ==================================================
# reading command line options
# ==================================================

# for formatting or something, we might want to run python without uv
# RUN_GLOBAL=1 to use global `PYTHON_BASE` instead of `uv run $(PYTHON_BASE)`
RUN_GLOBAL ?= 0

ifeq ($(RUN_GLOBAL),0)
	PYTHON = uv run $(PYTHON_BASE)
else
	PYTHON = $(PYTHON_BASE)
endif

# if you want different behavior for different python versions
# --------------------------------------------------
# COMPATIBILITY_MODE := $(shell $(PYTHON) -c "import sys; print(1 if sys.version_info < (3, 10) else 0)")

# options we might want to pass to pytest
# --------------------------------------------------

# base options for pytest, will be appended to if `COV` or `VERBOSE` are 1.
# user can also set this when running make to add more options
PYTEST_OPTIONS ?=

# set to `1` to run pytest with `--cov=.` to get coverage reports in a `.coverage` file
COV ?= 1
# set to `1` to run pytest with `--verbose`
VERBOSE ?= 0

ifeq ($(VERBOSE),1)
	PYTEST_OPTIONS += --verbose
endif

ifeq ($(COV),1)
	PYTEST_OPTIONS += --cov=.
endif

# ==================================================
# default target (help)
# ==================================================

# first/default target is help
.PHONY: default
default: help

# ==================================================
# getting version info
# we do this in a separate target because it takes a bit of time
# ==================================================

# this recipe is weird. we need it because:
# - a one liner for getting the version with toml is unwieldy, and using regex is fragile
# - using $$GET_VERSION_SCRIPT within $(shell ...) doesn't work because of escaping issues
# - trying to write to the file inside the `gen-version-info` recipe doesn't work, 
# 	shell eval happens before our `python -c ...` gets run and `cat` doesn't see the new file
.PHONY: write-proj-version
write-proj-version:
	@mkdir -p $(VERSIONS_DIR)
	@$(PYTHON) -c "$$GET_VERSION_SCRIPT" > $(VERSION_FILE)

# gets version info from $(PYPROJECT), last version from $(LAST_VERSION_FILE), and python version
# uses just `python` for everything except getting the python version. no echo here, because this is "private"
.PHONY: gen-version-info
gen-version-info: write-proj-version
	@mkdir -p $(LOCAL_DIR)
	$(eval VERSION := $(shell cat $(VERSION_FILE)) )
	$(eval LAST_VERSION := $(shell [ -f $(LAST_VERSION_FILE) ] && cat $(LAST_VERSION_FILE) || echo NULL) )
	$(eval PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')") )

# getting commit log since the tag specified in $(LAST_VERSION_FILE)
# will write to $(COMMIT_LOG_FILE)
# when publishing, the contents of $(COMMIT_LOG_FILE) will be used as the tag description (but can be edited during the process)
# no echo here, because this is "private"
.PHONY: gen-commit-log
gen-commit-log: gen-version-info
	@if [ "$(LAST_VERSION)" = "NULL" ]; then \
		echo "!!! ERROR !!!"; \
		echo "LAST_VERSION is NULL, cant get commit log!"; \
		exit 1; \
	fi
	@mkdir -p $(LOCAL_DIR)
	@$(PYTHON) -c "$$GET_COMMIT_LOG_SCRIPT" "$(LAST_VERSION)"


# force the version info to be read, printing it out
# also force the commit log to be generated, and cat it out
.PHONY: version
version: gen-commit-log
	@echo "Current version is $(VERSION), last auto-uploaded version is $(LAST_VERSION)"
	@echo "Commit log since last version from '$(COMMIT_LOG_FILE)':"
	@cat $(COMMIT_LOG_FILE)
	@echo ""
	@if [ "$(VERSION)" = "$(LAST_VERSION)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Python package $(VERSION) is the same as last published version $(LAST_VERSION), exiting!"; \
		exit 1; \
	fi


# ==================================================
# dependencies and setup
# ==================================================

.PHONY: setup
setup: dep-check
	@echo "install and update via uv"
	@echo "To activate the virtual environment, run one of:"
	@echo "  source .venv/bin/activate"
	@echo "  source .venv/Scripts/activate"

.PHONY: get-cuda-info
get-cuda-info:
	$(eval CUDA_PRESENT := $(shell if command -v nvcc > /dev/null 2>&1; then echo 1; else echo 0; fi))
	$(eval CUDA_VERSION := $(if $(filter $(CUDA_PRESENT),1),$(shell nvcc --version 2>/dev/null | grep "release" | awk '{print $$5}' | sed 's/,//'),NULL))
	$(eval CUDA_VERSION_SHORT := $(if $(filter $(CUDA_PRESENT),1),$(shell echo $(CUDA_VERSION) | sed 's/\.//'),NULL))

.PHONY: dep-check-torch
dep-check-torch:
	@echo "see if torch is installed, and which CUDA version and devices it sees"
	$(PYTHON) -c "$$CHECK_TORCH_SCRIPT"

.PHONY: dep
dep: get-cuda-info
	@echo "Exporting dependencies as per $(PYPROJECT) section 'tool.uv-exports.exports'"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION) | sh -x
	
# @if [ "$(CUDA_PRESENT)" = "1" ]; then \
# 	echo "CUDA is present, installing torch with CUDA $(CUDA_VERSION)"; \
# 	uv pip install torch --upgrade --index https://download.pytorch.org/whl/cu$(CUDA_VERSION_SHORT); \
# fi
	

.PHONY: dep-check
dep-check:
	@echo "Checking that exported requirements are up to date"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)-TEMP
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION)-TEMP | sh -x
	diff -r $(REQ_LOCATION)-TEMP $(REQ_LOCATION)
	rm -rf $(REQ_LOCATION)-TEMP


.PHONY: dep-clean
dep-clean:
	@echo "clean up lock files, .venv, and requirements files"
	rm -rf .venv
	rm -rf uv.lock
	rm -rf $(REQ_LOCATION)/*.txt

# ==================================================
# checks (formatting/linting, typing, tests)
# ==================================================

# runs ruff and pycln to format the code
.PHONY: format
format:
	@echo "format the source code"
	$(PYTHON) -m ruff format --config $(PYPROJECT) .
	$(PYTHON) -m ruff check --fix --config $(PYPROJECT) .
	$(PYTHON) -m pycln --config $(PYPROJECT) --all .

# runs ruff and pycln to check if the code is formatted correctly
.PHONY: format-check
format-check:
	@echo "check if the source code is formatted correctly"
	$(PYTHON) -m ruff check --config $(PYPROJECT) .
	$(PYTHON) -m pycln --check --config $(PYPROJECT) .

# runs type checks with mypy
# at some point, need to add back --check-untyped-defs to mypy call
# but it complains when we specify arguments by keyword where positional is fine
# not sure how to fix this
.PHONY: typing
typing: clean
	@echo "running type checks"
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(PACKAGE_NAME)/
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(TESTS_DIR)/

.PHONY: test
test: clean
	@echo "running tests"
	$(PYTHON) -m pytest $(PYTEST_OPTIONS) $(TESTS_DIR)

.PHONY: check
check: clean format-check test typing
	@echo "run format checks, tests, and typing checks"

# ==================================================
# coverage & docs
# ==================================================

# generates a whole tree of documentation in html format.
# see `docs/make_docs.py` and the templates in `docs/templates/html/` for more info
.PHONY: docs-html
docs-html:
	@echo "generate html docs"
	$(PYTHON) docs/make_docs.py

# instead of a whole website, generates a single markdown file with all docs using the templates in `docs/templates/markdown/`.
# this is useful if you want to have a copy that you can grep/search, but those docs are much messier.
# docs-combined will use pandoc to convert them to other formats.
.PHONY: docs-md
docs-md:
	@echo "generate combined (single-file) docs in markdown"
	mkdir $(DOCS_DIR)/combined -p
	$(PYTHON) docs/make_docs.py --combined

# after running docs-md, this will convert the combined markdown file to other formats:
# gfm (github-flavored markdown), plain text, and html
# requires pandoc in path, pointed to by $(PANDOC)
# pdf output would be nice but requires other deps
.PHONY: docs-combined
docs-combined: docs-md
	@echo "generate combined (single-file) docs in markdown and convert to other formats"
	@echo "requires pandoc in path"
	$(PANDOC) -f markdown -t gfm $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME)_gfm.md
	$(PANDOC) -f markdown -t plain $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).txt
	$(PANDOC) -f markdown -t html $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).html

# generates coverage reports as html and text with `pytest-cov`, and a badge with `coverage-badge`
# if `.coverage` is not found, will run tests first
# also removes the `.gitignore` file that `coverage html` creates, since we count that as part of the docs
.PHONY: cov
cov:
	@echo "generate coverage reports"
	@if [ ! -f .coverage ]; then \
		echo ".coverage not found, running tests first..."; \
		$(MAKE) test; \
	fi
	mkdir $(COVERAGE_REPORTS_DIR) -p
	$(PYTHON) -m coverage report -m > $(COVERAGE_REPORTS_DIR)/coverage.txt
	$(PYTHON) -m coverage_badge -f -o $(COVERAGE_REPORTS_DIR)/coverage.svg
	$(PYTHON) -m coverage html --directory=$(COVERAGE_REPORTS_DIR)/html/
	rm -rf $(COVERAGE_REPORTS_DIR)/html/.gitignore

# runs the coverage report, then the docs, then the combined docs
# ~~~~~~~~~~~~~~~~~~~~
# demo also created for docs
# ~~~~~~~~~~~~~~~~~~~~
.PHONY: docs
docs: demo cov docs-html docs-combined
	@echo "generate all documentation and coverage reports"

# removed all generated documentation files, but leaves the templates and the `docs/make_docs.py` script
# distinct from `make clean`
.PHONY: docs-clean
docs-clean:
	@echo "remove generated docs"
	rm -rf $(DOCS_DIR)/combined/
	rm -rf $(DOCS_DIR)/$(PACKAGE_NAME)/
	rm -rf $(COVERAGE_REPORTS_DIR)/
	rm $(DOCS_DIR)/$(PACKAGE_NAME).html
	rm $(DOCS_DIR)/index.html
	rm $(DOCS_DIR)/search.js
	rm $(DOCS_DIR)/package_map.dot
	rm $(DOCS_DIR)/package_map.html


# ==================================================
# build and publish
# ==================================================

# verifies that the current branch is $(PUBLISH_BRANCH) and that git is clean
# used before publishing
.PHONY: verify-git
verify-git: 
	@echo "checking git status"
	if [ "$(shell git branch --show-current)" != $(PUBLISH_BRANCH) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not on the $(PUBLISH_BRANCH) branch, exiting!"; \
		exit 1; \
	fi; \
	if [ -n "$(shell git status --porcelain)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not clean, exiting!"; \
		exit 1; \
	fi; \


.PHONY: build
build: 
	@echo "build the package"
	uv build

# gets the commit log, checks everything, builds, and then publishes with twine
# will ask the user to confirm the new version number (and this allows for editing the tag info)
# will also print the contents of $(PYPI_TOKEN_FILE) to the console for the user to copy and paste in when prompted by twine
.PHONY: publish
publish: gen-commit-log check build verify-git version gen-version-info
	@echo "run all checks, build, and then publish"

	@echo "Enter the new version number if you want to upload to pypi and create a new tag"
	@echo "Now would also be the time to edit $(COMMIT_LOG_FILE), as that will be used as the tag description"
	@read -p "Confirm: " NEW_VERSION; \
	if [ "$$NEW_VERSION" = $(VERSION) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Version confirmed. Proceeding with publish."; \
	else \
		echo "Version mismatch, exiting: you gave $$NEW_VERSION but expected $(VERSION)"; \
		exit 1; \
	fi;

	@echo "pypi username: __token__"
	@echo "pypi token from '$(PYPI_TOKEN_FILE)' :"
	echo $$(cat $(PYPI_TOKEN_FILE))

	echo "Uploading!"; \
	echo $(VERSION) > $(LAST_VERSION_FILE); \
	git add $(LAST_VERSION_FILE); \
	git commit -m "Auto update to $(VERSION)"; \
	git tag -a $(VERSION) -F $(COMMIT_LOG_FILE); \
	git push origin $(VERSION); \
	twine upload dist/* --verbose

# ==================================================
# cleanup of temp files
# ==================================================

# cleans up temp files from formatter, type checking, tests, coverage
# removes all built files
# removes $(TESTS_TEMP_DIR) to remove temporary test files
# recursively removes all `__pycache__` directories and `*.pyc` or `*.pyo` files
# distinct from `make docs-clean`, which only removes generated documentation files
.PHONY: clean
clean:
	@echo "clean up temporary files"
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf $(PACKAGE_NAME).egg-info
	rm -rf $(TESTS_TEMP_DIR)
	$(PYTHON_BASE) -Bc "import pathlib; [p.unlink() for path in ['$(PACKAGE_NAME)', '$(TESTS_DIR)', '$(DOCS_DIR)'] for pattern in ['*.py[co]', '__pycache__/*'] for p in pathlib.Path(path).rglob(pattern)]"

.PHONY: clean-all
clean-all: clean dep-clean docs-clean
	@echo "clean up all temporary files, dep files, venv, and generated docs"


# ==================================================
# smart help command
# ==================================================

# listing targets is from stackoverflow
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
# no .PHONY because this will only be run before `make help`
# it's a separate command because getting the versions takes a bit of time
help-targets:
	@echo -n "# make targets"
	@echo ":"
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 30


.PHONY: info
info: gen-version-info get-cuda-info
	@echo "# makefile variables"
	@echo "    PYTHON = $(PYTHON)"
	@echo "    PYTHON_VERSION = $(PYTHON_VERSION)"
	@echo "    PACKAGE_NAME = $(PACKAGE_NAME)"
	@echo "    VERSION = $(VERSION)"
	@echo "    LAST_VERSION = $(LAST_VERSION)"
	@echo "    PYTEST_OPTIONS = $(PYTEST_OPTIONS)"
	@echo "    CUDA_PRESENT = $(CUDA_PRESENT)"
	@if [ "$(CUDA_PRESENT)" = "1" ]; then \
		echo "    CUDA_VERSION = $(CUDA_VERSION)"; \
		echo "    CUDA_VERSION_SHORT = $(CUDA_VERSION_SHORT)"; \
	fi

.PHONY: info-long
info-long: info
	@echo "# other variables"
	@echo "    PUBLISH_BRANCH = $(PUBLISH_BRANCH)"
	@echo "    DOCS_DIR = $(DOCS_DIR)"
	@echo "    COVERAGE_REPORTS_DIR = $(COVERAGE_REPORTS_DIR)"
	@echo "    TESTS_DIR = $(TESTS_DIR)"
	@echo "    TESTS_TEMP_DIR = $(TESTS_TEMP_DIR)"
	@echo "    PYPROJECT = $(PYPROJECT)"
	@echo "    REQ_LOCATION = $(REQ_LOCATION)"
	@echo "    REQ_BASE = $(REQ_BASE)"
	@echo "    REQ_EXTRAS = $(REQ_EXTRAS)"
	@echo "    REQ_DEV = $(REQ_DEV)"
	@echo "    REQ_ALL = $(REQ_ALL)"
	@echo "    LOCAL_DIR = $(LOCAL_DIR)"
	@echo "    PYPI_TOKEN_FILE = $(PYPI_TOKEN_FILE)"
	@echo "    LAST_VERSION_FILE = $(LAST_VERSION_FILE)"
	@echo "    PYTHON_BASE = $(PYTHON_BASE)"
	@echo "    COMMIT_LOG_FILE = $(COMMIT_LOG_FILE)"
	@echo "    PANDOC = $(PANDOC)"
	@echo "    COV = $(COV)"
	@echo "    VERBOSE = $(VERBOSE)"
	@echo "    RUN_GLOBAL = $(RUN_GLOBAL)"
	@echo "    TYPECHECK_ARGS = $(TYPECHECK_ARGS)"

# immediately print out the help targets, and then local variables (but those take a bit longer)
.PHONY: help
help: help-targets info
	@echo -n ""

# ==================================================
# custom targets
# ==================================================
# (put them down here, or delimit with ~~~~~)


.PHONY: demo
demo:
	@echo "example of code output"
	$(PYTHON) -m lmcat -o example_output.md

.PHONY: demo-tree
demo-tree:
	@echo "example of code output, tree direct to console"
	$(PYTHON) -m lmcat -t
``````{ end_of_file: "makefile" }

``````{ path: "pyproject.toml" }
[project]
name = "lmcat"
version = "0.0.1"
description = "concatenating files for tossing them into a language model"
authors = [
	{ name = "Michael Ivanitskiy", email = "mivanits@umich.edu" }
]
readme = "README.md"
requires-python = ">=3.11"

dependencies = [
	"igittigitt>=2.1.5",
	"muutils>=0.6.21",
]

[project.optional-dependencies]
tokenizers = [
    "tokenizers>=0.21.0",
]

[dependency-groups]
dev = [
	# test
	"pytest>=8.2.2",
	# coverage
	"pytest-cov>=4.1.0",
	"coverage-badge>=1.1.0",
	# type checking
	"mypy>=1.0.1",
	# docs
	'pdoc>=14.6.0',
	# tomli since no tomlib in python < 3.11
	"tomli>=2.1.0; python_version < '3.11'",
]
lint = [
	# lint
	"pycln>=2.1.3",
	"ruff>=0.4.8",
]


[tool.uv]
default-groups = ["dev", "lint"]

[project.urls]
Homepage = "https://miv.name/lmcat"
Documentation = "https://miv.name/lmcat"
Repository = "https://github.com/mivanit/lmcat"
Issues = "https://github.com/mivanit/lmcat/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# ruff config
[tool.ruff]
exclude = ["__pycache__"]

[tool.ruff.format]
indent-style = "tab"
skip-magic-trailing-comma = false

# Custom export configurations
[tool.uv-exports]
args = [
	"--no-hashes"
]
exports = [
	# no groups, no extras, just the base dependencies
    { name = "base", groups = false, extras = false },
	# all extras but no groups
    { name = "extras", groups = false, extras = true },
	# include the dev group (this is the default behavior)
    { name = "dev", groups = ["dev"] },
	# only the lint group -- custom options for this
	{ name = "lint", options = ["--only-group", "lint"] },
	# all groups and extras
    { name = "all", filename="requirements.txt", groups = true, extras=true },
	# all groups and extras, a different way
	{ name = "all", groups = true, options = ["--all-extras"] },
]


``````{ end_of_file: "pyproject.toml" }