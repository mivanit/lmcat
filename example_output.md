# Stats
- 10 files
- 57 lines
- 39463 (39K) chars
- 15431 (15K) `gpt2` tokens

# File Tree

```
lmcat                    
├── lmcat                
│   ├── __init__.py      [  7L     84C    35T]
│   ├── __main__.py      [  4L     59C    23T]
│   ├── file_stats.py    [ 84L  2,032C   721T]
│   ├── index.html       [104L  4,125C 2,152T]
│   ├── lmcat.py         [420L 11,906C 4,511T]
│   └── processors.py    [ 27L    758C   248T]
├── tests                
│   ├── test_lmcat.py    [327L  9,778C 3,605T]
│   └── test_lmcat_2.py  [148L  4,192C 1,586T]
├── README.md            [138L  3,320C 1,021T]
├── pyproject.toml       [ 82L  1,824C   744T]
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

``````{ path: "lmcat/file_stats.py" }
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional

# Handle Python 3.11+ vs older Python for TOML parsing
try:
	import tomllib
except ImportError:
	try:
		import tomli as tomllib  # type: ignore
	except ImportError:
		tomllib = None  # type: ignore[assignment]


# tokenizers (optional dep)
TOKENIZERS_PRESENT: bool = False
try:
	import tokenizers  # type: ignore[import-untyped]

	TOKENIZERS_PRESENT = True
except ImportError:
	pass


class TokenizerWrapper:
	"""tokenizer wrapper. stores name and provides `n_tokens` method.

	uses splitting by whitespace as a fallback -- `whitespace-split`"""

	def __init__(self, name: str = "whitespace-split") -> None:
		self.name: str = name
		self.use_fallback: bool = name == "whitespace-split"
		self.tokenizer: Optional[tokenizers.Tokenizer] = (
			None if self.use_fallback else tokenizers.Tokenizer.from_pretrained(name)
		)

	def n_tokens(self, text: str) -> int:
		"""Return number of tokens in text"""
		if self.use_fallback:
			return len(text.split())
		else:
			assert self.tokenizer is not None
			return len(self.tokenizer.encode(text).tokens)


@dataclass
class FileStats:
	"""Statistics for a single file"""

	lines: int
	chars: int
	tokens: Optional[int] = None

	@classmethod
	def from_file(
		cls,
		path: Path,
		tokenizer: TokenizerWrapper,
	) -> "FileStats":
		"""Get statistics for a single file

		# Parameters:
		- `path : Path`
			Path to the file to analyze
		- `tokenizer : Optional[tokenizers.Tokenizer]`
			Tokenizer to use for counting tokens, if any

		# Returns:
		- `FileStats`
			Statistics for the file
		"""
		with path.open("r", encoding="utf-8", errors="ignore") as f:
			content: str = f.read()
			lines: int = len(content.splitlines())
			chars: int = len(content)
			tokens: int = tokenizer.n_tokens(content)
			return FileStats(lines=lines, chars=chars, tokens=tokens)


class TreeEntry(NamedTuple):
	"""Entry in the tree output with optional stats"""

	line: str
	stats: Optional[FileStats] = None

``````{ end_of_file: "lmcat/file_stats.py" }

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
import argparse
import io
import json

# from dataclasses import dataclass, field
from pathlib import Path
import sys


# Handle Python 3.11+ vs older Python for TOML parsing
try:
	import tomllib
except ImportError:
	try:
		import tomli as tomllib  # type: ignore
	except ImportError:
		tomllib = None  # type: ignore[assignment]

import igittigitt  # noqa: E402

from muutils.json_serialize import (
	SerializableDataclass,
	serializable_dataclass,
	serializable_field,
)
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
	tokenizer: str = serializable_field(
		default="gpt2" if TOKENIZERS_PRESENT else "whitespace-split"
	)
	"Tokenizer to use for tokenizing the output. `gpt2` by default. passed to `tokenizers.Tokenizer.from_pretrained()`. If specified and `tokenizers` not installed, will throw exception. fallback `whitespace-split` used to avoid exception when `tokenizers` not installed."

	# tree formatting
	tree_divider: str = serializable_field(default="│   ")
	tree_file_divider: str = serializable_field(default="├── ")
	tree_indent: str = serializable_field(default=" ")

	def get_tokenizer_obj(self) -> TokenizerWrapper:
		"""Get the tokenizer object"""
		return TokenizerWrapper(self.tokenizer)

	@classmethod
	def read(cls, root_dir: Path) -> "LMCatConfig":
		"""Attempt to read config from pyproject.toml, lmcat.toml, or lmcat.json."""
		pyproject_path: Path = root_dir / "pyproject.toml"
		lmcat_toml_path: Path = root_dir / "lmcat.toml"
		lmcat_json_path: Path = root_dir / "lmcat.json"

		if (
			sum(
				int(p.is_file())
				for p in (pyproject_path, lmcat_toml_path, lmcat_json_path)
			)
			> 1
		):
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
			self.parser.add_rule(pattern=pattern, base_path=self.root_dir)

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

	# Add file contents if not suppressed
	if not config.tree_only:
		output.append("# File Contents")

		for fpath in collected_files:
			relpath_posix = fpath.relative_to(root_dir).as_posix()
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

``````{ end_of_file: "lmcat/lmcat.py" }

``````{ path: "lmcat/processors.py" }
from typing import Callable
from pathlib import Path


PATH_PROCESSORS: dict[str, Callable[[Path], str]] = dict()

TEXT_PROCESSORS: dict[str, Callable[[str], str]] = dict()

DECIDERS: dict[str, Callable[[Path], bool]] = dict()


def register_path_processor(func: Callable[[Path], str]) -> Callable[[Path], str]:
	"""Register a function as a path processor"""
	PATH_PROCESSORS[func.__name__] = func
	return func


def register_text_processor(func: Callable[[str], str]) -> Callable[[str], str]:
	"""Register a function as a text processor"""
	TEXT_PROCESSORS[func.__name__] = func
	return func


def register_decider(func: Callable[[Path], bool]) -> Callable[[Path], bool]:
	"""Register a function as a decider"""
	DECIDERS[func.__name__] = func
	return func

``````{ end_of_file: "lmcat/processors.py" }

``````{ path: "tests/test_lmcat.py" }
import sys
import os
import shutil
import subprocess
from pathlib import Path

from lmcat.file_stats import TokenizerWrapper
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
	assert config.tree_indent == " "
	assert config.tree_file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_partial():
	data = {"tree_divider": "|---"}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "|---"
	assert config.tree_indent == " "
	assert config.tree_file_divider == "├── "
	assert config.content_divider == "``````"


def test_lmcat_config_load_all():
	data = {
		"tree_divider": "XX",
		"tree_indent": "YY",
		"tree_file_divider": "ZZ",
		"content_divider": "@@@",
	}
	config = LMCatConfig.load(data)
	assert config.tree_divider == "XX"
	assert config.tree_indent == "YY"
	assert config.tree_file_divider == "ZZ"
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

	config = LMCatConfig(ignore_patterns_files=list())
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

	tree_output, files = walk_dir(test_dir, handler, config, TokenizerWrapper())
	joined_output = "\n".join([x.line for x in tree_output])

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

	tree_output, files = walk_dir(test_dir, handler, config, TokenizerWrapper())
	joined_output = "\n".join([x.line for x in tree_output])

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

``````{ path: "tests/test_lmcat_2.py" }
import os
from pathlib import Path

import pytest

from lmcat.lmcat import (
	LMCatConfig,
	walk_and_collect,
	assemble_summary,
)

# Base test directory
TEMP_PATH: Path = Path("tests/_temp")


def test_unicode_file_handling():
	"""Test handling of Unicode in filenames and content"""
	test_dir = TEMP_PATH / "unicode_test"
	test_dir.mkdir(parents=True, exist_ok=True)

	# Create directories
	(test_dir / "привет").mkdir()
	(test_dir / "emoji_📁").mkdir()

	# Create files
	(test_dir / "hello_世界.txt").write_text(
		"Hello 世界\nこんにちは\n", encoding="utf-8"
	)
	(test_dir / "привет/мир.txt").write_text("Привет мир!\n", encoding="utf-8")
	(test_dir / "emoji_📁/test_🔧.txt").write_text(
		"Test with emojis 🎉\n", encoding="utf-8"
	)
	(test_dir / ".gitignore").write_text("*.tmp\n")
	(test_dir / "unicode_temp_⚡.tmp").write_text("should be ignored", encoding="utf-8")

	config = LMCatConfig()

	# Test walking
	tree_output, file_list = walk_and_collect(test_dir, config)
	tree_str = "\n".join(tree_output)

	# Check filenames in tree
	assert "hello_世界.txt" in tree_str
	assert "мир.txt" in tree_str
	assert "test_🔧.txt" in tree_str
	assert "unicode_temp_⚡.tmp" not in tree_str  # Should be ignored

	# Check content handling
	summary = assemble_summary(test_dir, config)
	assert "Hello 世界" in summary
	assert "Привет мир!" in summary
	assert "Test with emojis 🎉" in summary


def test_large_file_handling():
	"""Test handling of large files"""
	test_dir = TEMP_PATH / "large_file_test"
	test_dir.mkdir(parents=True, exist_ok=True)

	# Create regular files
	(test_dir / "small.txt").write_text("small content\n")
	(test_dir / "medium.txt").write_text("medium " * 1000)

	# Create large file
	with (test_dir / "large.txt").open("w") as f:
		f.write("x" * (1024 * 1024))

	config = LMCatConfig()
	tree_output, file_list = walk_and_collect(test_dir, config)

	# Check stats in tree output
	tree_str = "\n".join(tree_output)
	assert "small.txt" in tree_str
	assert "medium.txt" in tree_str
	assert "large.txt" in tree_str

	# Check that files are readable in summary
	summary = assemble_summary(test_dir, config)
	assert "small content" in summary
	assert "medium " * 10 in summary  # Check start of medium file
	assert "x" * 100 in summary  # Check start of large file


def test_symlink_handling():
	"""Test handling of symlinks in directory structure"""
	test_dir = TEMP_PATH / "symlink_test"
	test_dir.mkdir(parents=True, exist_ok=True)

	# Create directories and files
	(test_dir / "src").mkdir()
	(test_dir / "docs").mkdir()
	(test_dir / "src/module.py").write_text("print('original')\n")
	(test_dir / "docs/readme.md").write_text("# Documentation\n")

	try:
		# Create symlinks
		(test_dir / "src/linked.py").symlink_to(test_dir / "src/module.py")
		(test_dir / "docs_link").symlink_to(test_dir / "docs")

		config = LMCatConfig()
		tree_output, file_list = walk_and_collect(test_dir, config)
		tree_str = "\n".join(tree_output)

		# Check if symlinks are handled
		assert "linked.py" in tree_str
		assert "docs_link" in tree_str

		# Verify symlink contents are included
		summary = assemble_summary(test_dir, config)
		assert "print('original')" in summary
		assert "# Documentation" in summary

	except OSError:
		pytest.skip("Symlink creation not supported")


def test_error_handling():
	"""Test error handling for various filesystem conditions"""
	test_dir = TEMP_PATH / "error_test"
	test_dir.mkdir(parents=True, exist_ok=True)

	# Create test files
	(test_dir / "readable.txt").write_text("can read this\n")
	(test_dir / "binary.bin").write_bytes(b"\x00\x01\x02\x03")
	(test_dir / "unreadable.txt").write_text("secret")

	try:
		os.chmod(test_dir / "unreadable.txt", 0o000)
	except PermissionError:
		pytest.skip("Cannot create unreadable file")

	config = LMCatConfig()
	tree_output, file_list = walk_and_collect(test_dir, config)
	tree_str = "\n".join(tree_output)

	# Check that readable files are included
	assert "readable.txt" in tree_str
	assert "binary.bin" in tree_str

	# Check content
	summary = assemble_summary(test_dir, config)
	assert "can read this" in summary

	# Restore permissions for cleanup
	try:
		os.chmod(test_dir / "unreadable.txt", 0o666)
	except OSError:
		pass

``````{ end_of_file: "tests/test_lmcat_2.py" }

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