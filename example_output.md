# Stats
- 12 files
- 67 lines
- 54972 (55K) chars
- 20734 (21K) `gpt2` tokens

# File Tree

```
lmcat                           
├── lmcat                       
│   ├── __init__.py             [  7L     84C    35T]
│   ├── __main__.py             [  4L     59C    23T]
│   ├── file_stats.py           [ 84L  2,032C   721T]
│   ├── index.html              [104L  4,125C 2,152T]
│   ├── lmcat.py                [463L 13,493C 5,027T]
│   ├── processing_pipeline.py  [183L  5,120C 1,855T]
│   └── processors.py           [117L  2,987C 1,038T]
├── tests                       
│   ├── test_lmcat.py           [327L  9,778C 3,605T]
│   └── test_lmcat_2.py         [148L  4,192C 1,586T]
├── README.md                   [138L  3,320C 1,021T]
├── makefile                    [691L 23,553C 8,387T]
├── pyproject.toml              [ 90L  1,985C   814T]
```

# File Contents

``````{ path="lmcat/__init__.py"  }
"""
.. include:: ../README.md
"""

from lmcat.lmcat import main

__all__ = ["main"]

``````{ end_of_file="lmcat/__init__.py" }

``````{ path="lmcat/__main__.py"  }
from lmcat import main

if __name__ == "__main__":
	main()

``````{ end_of_file="lmcat/__main__.py" }

``````{ path="lmcat/file_stats.py"  }
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

``````{ end_of_file="lmcat/file_stats.py" }

``````{ path="lmcat/index.html"  }
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
``````{ end_of_file="lmcat/index.html" }

``````{ path="lmcat/lmcat.py"  }
import argparse
import io
import json

# from dataclasses import dataclass, field
from pathlib import Path
import sys

from lmcat.processing_pipeline import ProcessingPipeline


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
from lmcat.processing_pipeline import OnMultipleProcessors


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
	# --allow-plugins is a command line only option and must be set to true for this to work
	plugins_file: Path | None = serializable_field(
		default=None,
		serialization_fn=lambda x: x.as_posix() if x else None,
		deserialize_fn=lambda x: Path(x) if x else None,
	)
	allow_plugins: bool = serializable_field(
		default=False,
		deserialize_fn=lambda x: False,  # this can only be overriden through the command line
	)

	# processing pipeline
	glob_process: dict[str, str] = serializable_field(default_factory=dict)
	decider_process: dict[str, str] = serializable_field(default_factory=dict)
	on_multiple_processors: OnMultipleProcessors = serializable_field(
		default="except",
		assert_type=False,
	)

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

	def get_processing_pipeline(self) -> ProcessingPipeline:
		"""Get the processing pipeline object"""
		plugins_file: Path | None = self.plugins_file if self.allow_plugins else None
		return ProcessingPipeline(
			plugins_file=plugins_file,
			glob_process_keys=self.glob_process,
			decider_process_keys=self.decider_process,
			on_multiple_processors=self.on_multiple_processors,
		)

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

	processing_pipeline: ProcessingPipeline = config.get_processing_pipeline()

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
			# get the path
			relpath_posix: str = fpath.relative_to(root_dir).as_posix()

			# process the contents
			f_contents: str
			p_name: str | None
			f_contents, p_name = processing_pipeline.process_file(fpath)
			processed_with: str = f'processed_with="{p_name}"' if p_name else ""

			# start of file marker
			pathspec_start: str = f'{{ path="{relpath_posix}" {processed_with} }}'
			pathspec_end: str = f'{{ end_of_file="{relpath_posix}" }}'
			output.append("")
			output.append(config.content_divider + pathspec_start)

			# process the actual contents of the file with the pipeline, and append
			output.append(f_contents)

			# add the end of file marker
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
	arg_parser.add_argument(
		"--allow-plugins",
		action="store_true",
		default=False,
		help="Allow plugins to be loaded from the plugins file. WARNING: this will execute arbitrary code found in the file pointed to by `config.plugins_file`, and **is a security risk**.",
	)

	args: argparse.Namespace = arg_parser.parse_known_args()[0]
	root_dir: Path = Path(".").resolve()
	config: LMCatConfig = LMCatConfig.read(root_dir)

	# CLI overrides
	config.tree_only = args.tree_only
	config.allow_plugins = args.allow_plugins

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

``````{ end_of_file="lmcat/lmcat.py" }

``````{ path="lmcat/processing_pipeline.py"  }
from importlib.util import spec_from_file_location, module_from_spec
import sys
from pathlib import Path
from typing import Literal
import re
import warnings

from lmcat.processors import (
	ProcessorName,
	DeciderName,
	ProcessorFunc,
	DeciderFunc,
	PROCESSORS,
	DECIDERS,
)

OnMultipleProcessors = Literal["warn", "except", "do_first", "do_last", "skip"]


def _compile_glob(pattern: str) -> re.Pattern:
	"""Convert a glob pattern to a regex pattern.

	# Parameters:
		- `pattern : str`
		Glob pattern to compile

	# Returns:
		- `re.Pattern`
		Compiled regex pattern
	"""
	regex_str: str = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
	return re.compile(f"^{regex_str}$")


def load_plugins(plugins_file: Path) -> None:
	"""Load plugins from a Python file.

	# Parameters:
	 - `plugins_file : Path`
	    Path to plugins file
	"""
	if not plugins_file.exists():
		return

	try:
		# Load module
		spec = spec_from_file_location("lmcat_plugins", plugins_file)
		if spec is None or spec.loader is None:
			return

		module = module_from_spec(spec)
		# Add to sys.modules so imports work properly
		sys.modules["lmcat_plugins"] = module
		spec.loader.exec_module(module)
	except Exception as e:
		print(f"Error loading plugins: {e}", file=sys.stderr)


class ProcessingPipeline:
	"""Manages the processing pipeline for files.

	# Attributes:
	 - `glob_process : dict[str, ProcessorName]`
		Maps glob patterns to processor names
	 - `decider_process : dict[DeciderName, ProcessorName]`
		Maps decider names to processor names
	 - `_compiled_globs : dict[str, re.Pattern]`
		Cached compiled glob patterns for performance
	"""

	def __init__(
		self,
		plugins_file: Path | None,
		glob_process_keys: dict[str, ProcessorName],
		decider_process_keys: dict[DeciderName, ProcessorName],
		on_multiple_processors: OnMultipleProcessors,
	):
		# store the vars
		self.plugins_file: Path | None = plugins_file
		self.glob_process_keys: dict[str, ProcessorName] = glob_process_keys
		self.decider_process_keys: dict[DeciderName, ProcessorName] = (
			decider_process_keys
		)
		self.on_multiple_processors: OnMultipleProcessors = on_multiple_processors

		# load the plugins file
		if self.plugins_file is not None:
			load_plugins(self.plugins_file)

		# try to get the glob and decider processor functions
		try:
			self.glob_process: dict[re.Pattern, ProcessorFunc] = {
				_compile_glob(glob_pattern): PROCESSORS[processor_name]
				for glob_pattern, processor_name in self.glob_process_keys.items()
			}
		except KeyError as e:
			raise ValueError(
				f"Invalid glob processor:\n{e}\n{PROCESSORS.keys() = }\n{self.glob_process_keys = }"
			) from e

		try:
			self.decider_process: dict[DeciderFunc, ProcessorFunc] = {
				DECIDERS[decider_name]: PROCESSORS[processor_name]
				for decider_name, processor_name in self.decider_process_keys.items()
			}
		except KeyError as e:
			raise ValueError(
				f"Invalid decider or decider processor:\n{e}\n{DECIDERS.keys() = }\n{PROCESSORS.keys() = }\n{self.decider_process_keys = }"
			) from e

	def get_processors_for_path(self, path: Path) -> list[ProcessorFunc]:
		"""Get all applicable processors for a given path.

		# Parameters:
		 - `path : Path`
			Path to get processors for

		# Returns:
		 - `list[ProcessorFunc]`
			List of applicable path processors
		"""
		processors: list[ProcessorFunc] = []

		# Check glob patterns
		for glob_pattern, processor in self.glob_process.items():
			if glob_pattern.match(path.name):
				processors.append(processor)

		# Check deciders
		for decider, processor in self.decider_process.items():
			if decider(path):
				processors.append(processor)

		return processors

	def process_file(self, path: Path) -> tuple[str, str | None]:
		"""Process a file through the pipeline.

		# Parameters:
		 - `path : Path`
			Path to process the content of

		# Returns:
		 - `tuple[str, str]`
			Processed content and the processor name
			if no processor is found, will be `(path.read_text(), None)`
		"""
		# Get all applicable processors
		processors: list[ProcessorFunc] = self.get_processors_for_path(path)

		# Early return if no processors
		selected_processor: ProcessorFunc | None

		if len(processors) == 0:
			selected_processor = None
		elif len(processors) == 1:
			# Apply single processor
			selected_processor = processors[0]
		else:
			match self.on_multiple_processors:
				case "warn":
					warnings.warn(f"Multiple processors for {path.name}: {processors}")
					selected_processor = processors[0]
				case "except":
					raise ValueError(
						f"Multiple processors for {path.name}: {processors}"
					)
				case "do_first":
					selected_processor = processors[0]
				case "do_last":
					selected_processor = processors[-1]
				case "skip":
					selected_processor = None
				case _:
					raise ValueError(
						f"Invalid on_multiple_processors: {self.on_multiple_processors = }"
					)

		# Process the file and return
		if selected_processor is None:
			return path.read_text(encoding="utf-8", errors="surrogateescape"), None
		else:
			return selected_processor(path), selected_processor.__name__

``````{ end_of_file="lmcat/processing_pipeline.py" }

``````{ path="lmcat/processors.py"  }
from typing import Callable, Sequence
from pathlib import Path

ProcessorName = str
DeciderName = str


ProcessorFunc = Callable[[Path], str]
DeciderFunc = Callable[[Path], bool]

PROCESSORS: dict[ProcessorName, ProcessorFunc] = dict()

DECIDERS: dict[DeciderName, DeciderFunc] = dict()


def register_processor(func: ProcessorFunc) -> ProcessorFunc:
	"""Register a function as a path processor"""
	PROCESSORS[ProcessorName(func.__name__)] = func
	return func


def register_decider(func: DeciderFunc) -> DeciderFunc:
	"""Register a function as a decider"""
	DECIDERS[DeciderName(func.__name__)] = func
	return func


@register_processor
def remove_comments(path: Path) -> str:
	"""Remove single-line comments from code."""
	lines = path.read_text().splitlines()
	processed = [line for line in lines if not line.strip().startswith("#")]
	return "\n".join(processed)


@register_processor
def compress_whitespace(path: Path) -> str:
	"""Compress multiple whitespace characters into single spaces."""
	return " ".join(path.read_text().split())


@register_processor
def to_relative_path(path: Path) -> str:
	"""return the path to the file as a string"""
	return path.as_posix()


@register_decider
def is_python_file(path: Path) -> bool:
	"""Check if file is a Python source file."""
	return path.suffix == ".py"


@register_decider
def is_documentation(path: Path) -> bool:
	"""Check if file is documentation."""
	return path.suffix in {".md", ".rst", ".txt"}


@register_processor
def makefile_processor(path: Path) -> str:
	"""Process a Makefile to show only target descriptions and basic structure.

	Preserves:
	- Comments above .PHONY targets up to first empty line
	- The .PHONY line and target line
	- First line after target if it starts with @echo

	# Parameters:
	 - `path : Path`
		Path to the Makefile to process

	# Returns:
	 - `str`
		Processed Makefile content
	"""
	lines: Sequence[str] = path.read_text().splitlines()
	output_lines: list[str] = []

	i: int = 0
	while i < len(lines):
		line: str = lines[i]

		# Look for .PHONY lines
		if line.strip().startswith(".PHONY:"):
			# Store target name for later matching
			target_name: str = line.split(":")[1].strip()

			# Collect comments above until empty line
			comment_lines: list[str] = []
			look_back: int = i - 1
			while look_back >= 0 and lines[look_back].strip():
				if lines[look_back].strip().startswith("#"):
					comment_lines.insert(0, lines[look_back])
				look_back -= 1

			# Add collected comments
			output_lines.extend(comment_lines)

			# Add .PHONY line
			output_lines.append(line)

			# Add target line (should be next)
			if i + 1 < len(lines) and lines[i + 1].startswith(f"{target_name}:"):
				output_lines.append(lines[i + 1])
				i += 1

				# Check for @echo on next line
				if i + 1 < len(lines) and lines[i + 1].strip().startswith("@echo"):
					output_lines.append(lines[i + 1])

				output_lines.append("	...")
				output_lines.append("")

		i += 1

	return "\n".join(output_lines)

``````{ end_of_file="lmcat/processors.py" }

``````{ path="tests/test_lmcat.py"  }
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

``````{ end_of_file="tests/test_lmcat.py" }

``````{ path="tests/test_lmcat_2.py"  }
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

``````{ end_of_file="tests/test_lmcat_2.py" }

``````{ path="README.md"  }
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
``````{ end_of_file="README.md" }

``````{ path="makefile" processed_with="makefile_processor" }
# first/default target is help
.PHONY: default
default: help
	...

# this recipe is weird. we need it because:
# - a one liner for getting the version with toml is unwieldy, and using regex is fragile
# - using $$GET_VERSION_SCRIPT within $(shell ...) doesn't work because of escaping issues
# - trying to write to the file inside the `gen-version-info` recipe doesn't work, 
# 	shell eval happens before our `python -c ...` gets run and `cat` doesn't see the new file
.PHONY: write-proj-version
write-proj-version:
	...

# gets version info from $(PYPROJECT), last version from $(LAST_VERSION_FILE), and python version
# uses just `python` for everything except getting the python version. no echo here, because this is "private"
.PHONY: gen-version-info
gen-version-info: write-proj-version
	...

# getting commit log since the tag specified in $(LAST_VERSION_FILE)
# will write to $(COMMIT_LOG_FILE)
# when publishing, the contents of $(COMMIT_LOG_FILE) will be used as the tag description (but can be edited during the process)
# no echo here, because this is "private"
.PHONY: gen-commit-log
gen-commit-log: gen-version-info
	...

# force the version info to be read, printing it out
# also force the commit log to be generated, and cat it out
.PHONY: version
version: gen-commit-log
	@echo "Current version is $(VERSION), last auto-uploaded version is $(LAST_VERSION)"
	...

.PHONY: setup
setup: dep-check
	@echo "install and update via uv"
	...

.PHONY: get-cuda-info
get-cuda-info:
	...

.PHONY: dep-check-torch
dep-check-torch:
	@echo "see if torch is installed, and which CUDA version and devices it sees"
	...

.PHONY: dep
dep: get-cuda-info
	@echo "Exporting dependencies as per $(PYPROJECT) section 'tool.uv-exports.exports'"
	...

.PHONY: dep-check
dep-check:
	@echo "Checking that exported requirements are up to date"
	...

.PHONY: dep-clean
dep-clean:
	@echo "clean up lock files, .venv, and requirements files"
	...

# runs ruff and pycln to format the code
.PHONY: format
format:
	@echo "format the source code"
	...

# runs ruff and pycln to check if the code is formatted correctly
.PHONY: format-check
format-check:
	@echo "check if the source code is formatted correctly"
	...

# runs type checks with mypy
# at some point, need to add back --check-untyped-defs to mypy call
# but it complains when we specify arguments by keyword where positional is fine
# not sure how to fix this
.PHONY: typing
typing: clean
	@echo "running type checks"
	...

.PHONY: test
test: clean
	@echo "running tests"
	...

.PHONY: check
check: clean format-check test typing
	@echo "run format checks, tests, and typing checks"
	...

# generates a whole tree of documentation in html format.
# see `docs/make_docs.py` and the templates in `docs/templates/html/` for more info
.PHONY: docs-html
docs-html:
	@echo "generate html docs"
	...

# instead of a whole website, generates a single markdown file with all docs using the templates in `docs/templates/markdown/`.
# this is useful if you want to have a copy that you can grep/search, but those docs are much messier.
# docs-combined will use pandoc to convert them to other formats.
.PHONY: docs-md
docs-md:
	@echo "generate combined (single-file) docs in markdown"
	...

# after running docs-md, this will convert the combined markdown file to other formats:
# gfm (github-flavored markdown), plain text, and html
# requires pandoc in path, pointed to by $(PANDOC)
# pdf output would be nice but requires other deps
.PHONY: docs-combined
docs-combined: docs-md
	@echo "generate combined (single-file) docs in markdown and convert to other formats"
	...

# generates coverage reports as html and text with `pytest-cov`, and a badge with `coverage-badge`
# if `.coverage` is not found, will run tests first
# also removes the `.gitignore` file that `coverage html` creates, since we count that as part of the docs
.PHONY: cov
cov:
	@echo "generate coverage reports"
	...

# runs the coverage report, then the docs, then the combined docs
# ~~~~~~~~~~~~~~~~~~~~
# demo also created for docs
# ~~~~~~~~~~~~~~~~~~~~
.PHONY: docs
docs: demo cov docs-html docs-combined
	@echo "generate all documentation and coverage reports"
	...

# removed all generated documentation files, but leaves the templates and the `docs/make_docs.py` script
# distinct from `make clean`
.PHONY: docs-clean
docs-clean:
	@echo "remove generated docs"
	...

# verifies that the current branch is $(PUBLISH_BRANCH) and that git is clean
# used before publishing
.PHONY: verify-git
verify-git: 
	@echo "checking git status"
	...

.PHONY: build
build: 
	@echo "build the package"
	...

# gets the commit log, checks everything, builds, and then publishes with twine
# will ask the user to confirm the new version number (and this allows for editing the tag info)
# will also print the contents of $(PYPI_TOKEN_FILE) to the console for the user to copy and paste in when prompted by twine
.PHONY: publish
publish: gen-commit-log check build verify-git version gen-version-info
	@echo "run all checks, build, and then publish"
	...

# cleans up temp files from formatter, type checking, tests, coverage
# removes all built files
# removes $(TESTS_TEMP_DIR) to remove temporary test files
# recursively removes all `__pycache__` directories and `*.pyc` or `*.pyo` files
# distinct from `make docs-clean`, which only removes generated documentation files
.PHONY: clean
clean:
	@echo "clean up temporary files"
	...

.PHONY: clean-all
clean-all: clean dep-clean docs-clean
	@echo "clean up all temporary files, dep files, venv, and generated docs"
	...

.PHONY: info
info: gen-version-info get-cuda-info
	@echo "# makefile variables"
	...

.PHONY: info-long
info-long: info
	@echo "# other variables"
	...

# immediately print out the help targets, and then local variables (but those take a bit longer)
.PHONY: help
help: help-targets info
	@echo -n ""
	...

.PHONY: demo
demo:
	@echo "example of code output"
	...

.PHONY: demo-tree
demo-tree:
	@echo "example of code output, tree direct to console"
	...

``````{ end_of_file="makefile" }

``````{ path="pyproject.toml"  }
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


[tool.lmcat]
ignore_patterns_files = [".lmignore", ".gitignore"]

[tool.lmcat.glob_process]
"[mM]akefile" = "makefile_processor"

# [tool.lmcat.decider_process]
``````{ end_of_file="pyproject.toml" }