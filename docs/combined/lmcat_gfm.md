> docs for [`lmcat`](https://github.com/mivanit/lmcat) v0.1.0

## Contents

# lmcat

A Python tool for concatenating files and directory structures into a
single document, perfect for sharing code with language models. It
respects `.gitignore` and `.lmignore` patterns and provides configurable
output formatting.

## Features

- Tree view of directory structure with file statistics (lines,
  characters, tokens)
- Includes file contents with clear delimiters
- Respects `.gitignore` patterns (can be disabled)
- Supports custom ignore patterns via `.lmignore`
- Configurable via `pyproject.toml`, `lmcat.toml`, or `lmcat.json`
  - you can specify `glob_process` or `decider_process` to run on files,
    like if you want to convert a notebook to a markdown file

## Installation

Install from PyPI:

``` bash
pip install lmcat
```

or, install with support for counting tokens:

``` bash
pip install lmcat[tokenizers]
```

## Usage

Basic usage - concatenate current directory:

``` bash
# Only show directory tree
python -m lmcat --tree-only

# Write output to file
python -m lmcat --output summary.md

# Print current configuration
python -m lmcat --print-cfg
```

The output will include a directory tree and the contents of each
non-ignored file.

### Command Line Options

- `-t`, `--tree-only`: Only print the directory tree, not file contents
- `-o`, `--output`: Specify an output file (defaults to stdout)
- `-h`, `--help`: Show help message

### Configuration

lmcat is best configured via a `tool.lmcat` section in `pyproject.toml`:

``` toml
[tool.lmcat]
# Tree formatting
tree_divider = "│   "    # Vertical lines in tree
tree_indent = " "        # Indentation
tree_file_divider = "├── "  # File/directory entries
content_divider = "``````"  # File content delimiters

# Processing pipeline
tokenizer = "gpt2"  # or "whitespace-split"
tree_only = false   # Only show tree structure
on_multiple_processors = "except"  # Behavior when multiple processors match

# File handling
ignore_patterns = ["*.tmp", "*.log"]  # Additional patterns to ignore
ignore_patterns_files = [".gitignore", ".lmignore"]

# processors
[tool.lmcat.glob_process]
"[mM]akefile" = "makefile_recipes"
"*.ipynb" = "ipynb_to_md"
```

## Development

### Setup

1.  Clone the repository:

``` bash
git clone https://github.com/mivanit/lmcat
cd lmcat
```

2.  Set up the development environment:

``` bash
make setup
```

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

``` bash
make test
```

For verbose output:

``` bash
VERBOSE=1 make test
```

### Roadmap

- more processors and deciders, like:
  - only first `n` lines if file is too large
  - first few lines of a csv file
  - json schema of a big json/toml/yaml file
  - metadata extraction from images
- better tests, I feel like gitignore/lmignore interaction is broken
- llm summarization and caching of those summaries in `.lmsummary/`
- reasonable defaults for file extensions to ignore
- web interface

## Submodules

- [`lmcat`](#lmcat)
- [`file_stats`](#file_stats)
- [`processing_pipeline`](#processing_pipeline)
- [`processors`](#processors)

## API Documentation

- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/__init__.py)

# `lmcat`

### lmcat

A Python tool for concatenating files and directory structures into a
single document, perfect for sharing code with language models. It
respects `.gitignore` and `.lmignore` patterns and provides configurable
output formatting.

#### Features

- Tree view of directory structure with file statistics (lines,
  characters, tokens)
- Includes file contents with clear delimiters
- Respects `.gitignore` patterns (can be disabled)
- Supports custom ignore patterns via `.lmignore`
- Configurable via `pyproject.toml`, `lmcat.toml`, or `lmcat.json`
  - you can specify `glob_process` or `decider_process` to run on files,
    like if you want to convert a notebook to a markdown file

#### Installation

Install from PyPI:

``` bash
pip install lmcat
```

or, install with support for counting tokens:

``` bash
pip install lmcat[tokenizers]
```

#### Usage

Basic usage - concatenate current directory:

``` bash
### Only show directory tree
python -m lmcat --tree-only

### Write output to file
python -m lmcat --output summary.md

### Print current configuration
python -m lmcat --print-cfg
```

The output will include a directory tree and the contents of each
non-ignored file.

##### Command Line Options

- `-t`, `--tree-only`: Only print the directory tree, not file contents
- `-o`, `--output`: Specify an output file (defaults to stdout)
- `-h`, `--help`: Show help message

##### Configuration

lmcat is best configured via a `tool.lmcat` section in `pyproject.toml`:

``` toml
[tool.lmcat]
### Tree formatting
tree_divider = "│   "    # Vertical lines in tree
tree_indent = " "        # Indentation
tree_file_divider = "├── "  # File/directory entries
content_divider = "``````"  # File content delimiters

### Processing pipeline
tokenizer = "gpt2"  # or "whitespace-split"
tree_only = false   # Only show tree structure
on_multiple_processors = "except"  # Behavior when multiple processors match

### File handling
ignore_patterns = ["*.tmp", "*.log"]  # Additional patterns to ignore
ignore_patterns_files = [".gitignore", ".lmignore"]

### processors
[tool.lmcat.glob_process]
"[mM]akefile" = "makefile_recipes"
"*.ipynb" = "ipynb_to_md"
```

#### Development

##### Setup

1.  Clone the repository:

``` bash
git clone https://github.com/mivanit/lmcat
cd lmcat
```

2.  Set up the development environment:

``` bash
make setup
```

##### Development Commands

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

##### Running Tests

``` bash
make test
```

For verbose output:

``` bash
VERBOSE=1 make test
```

##### Roadmap

- more processors and deciders, like:
  - only first `n` lines if file is too large
  - first few lines of a csv file
  - json schema of a big json/toml/yaml file
  - metadata extraction from images
- better tests, I feel like gitignore/lmignore interaction is broken
- llm summarization and caching of those summaries in `.lmsummary/`
- reasonable defaults for file extensions to ignore
- web interface

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/__init__.py#L0-L14)

### `def main`

``` python
() -> None
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/__init__.py#L393-L459)

Main entry point for the script

> docs for [`lmcat`](https://github.com/mivanit/lmcat) v0.1.0

## API Documentation

- [`TOKENIZERS_PRESENT`](#TOKENIZERS_PRESENT)
- [`TokenizerWrapper`](#TokenizerWrapper)
- [`FileStats`](#FileStats)
- [`TreeEntry`](#TreeEntry)

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/file_stats.py)

# `lmcat.file_stats`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/file_stats.py#L0-L83)

- `TOKENIZERS_PRESENT: bool = True`

### `class TokenizerWrapper:`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/file_stats.py#L25-L43)

tokenizer wrapper. stores name and provides `n_tokens` method.

uses splitting by whitespace as a fallback – `whitespace-split`

### `TokenizerWrapper`

``` python
(name: str = 'whitespace-split')
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/file_stats.py#L30-L35)

- `name: str`

- `use_fallback: bool`

- `tokenizer: Optional[tokenizers.Tokenizer]`

### `def n_tokens`

``` python
(self, text: str) -> int
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/file_stats.py#L37-L43)

Return number of tokens in text

### `class FileStats:`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/file_stats.py#L46-L77)

Statistics for a single file

### `FileStats`

``` python
(lines: int, chars: int, tokens: Optional[int] = None)
```

- `lines: int`

- `chars: int`

- `tokens: Optional[int] = None`

### `def from_file`

``` python
(
    cls,
    path: pathlib.Path,
    tokenizer: lmcat.file_stats.TokenizerWrapper
) -> lmcat.file_stats.FileStats
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/file_stats.py#L54-L77)

Get statistics for a single file

### Parameters:

- `path : Path` Path to the file to analyze
- `tokenizer : Optional[tokenizers.Tokenizer]` Tokenizer to use for
  counting tokens, if any

### Returns:

- `FileStats` Statistics for the file

### `class TreeEntry(typing.NamedTuple):`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/file_stats.py#L80-L84)

Entry in the tree output with optional stats

### `TreeEntry`

``` python
(line: str, stats: Optional[lmcat.file_stats.FileStats] = None)
```

Create new instance of TreeEntry(line, stats)

- `line: str`

Alias for field number 0

- `stats: Optional[lmcat.file_stats.FileStats]`

Alias for field number 1

### Inherited Members

- [`index`](#TreeEntry.index)
- [`count`](#TreeEntry.count)

> docs for [`lmcat`](https://github.com/mivanit/lmcat) v0.1.0

## API Documentation

- [`LMCatConfig`](#LMCatConfig)
- [`IgnoreHandler`](#IgnoreHandler)
- [`sorted_entries`](#sorted_entries)
- [`walk_dir`](#walk_dir)
- [`format_tree_with_stats`](#format_tree_with_stats)
- [`walk_and_collect`](#walk_and_collect)
- [`assemble_summary`](#assemble_summary)
- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py)

# `lmcat.lmcat`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L0-L462)

### `class LMCatConfig(muutils.json_serialize.serializable_dataclass.SerializableDataclass):`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L35-L144)

Configuration dataclass for lmcat

### Parameters:

- `tree_divider: str`
- `tree_indent: str`
- `tree_file_divider: str`
- `content_divider: str`
- `include_gitignore: bool` (default True)
- `tree_only: bool` (default False)

### `LMCatConfig`

``` python
(
    *,
    content_divider: str = '``````',
    tree_only: bool = False,
    ignore_patterns: list[str] = <factory>,
    ignore_patterns_files: list[pathlib.Path] = <factory>,
    plugins_file: pathlib.Path | None = None,
    allow_plugins: bool = False,
    glob_process: dict[str, str] = <factory>,
    decider_process: dict[str, str] = <factory>,
    on_multiple_processors: Literal['warn', 'except', 'do_first', 'do_last', 'skip'] = 'except',
    tokenizer: str = 'gpt2',
    tree_divider: str = '│   ',
    tree_file_divider: str = '├── ',
    tree_indent: str = ' '
)
```

- ``````` content_divider: str = '``````' ```````

- `tree_only: bool = False`

- `ignore_patterns: list[str]`

- `ignore_patterns_files: list[pathlib.Path]`

- `plugins_file: pathlib.Path | None = None`

- `allow_plugins: bool = False`

- `glob_process: dict[str, str]`

- `decider_process: dict[str, str]`

- `on_multiple_processors: Literal['warn', 'except', 'do_first', 'do_last', 'skip'] = 'except'`

- `tokenizer: str = 'gpt2'`

Tokenizer to use for tokenizing the output. `gpt2` by default. passed to
`tokenizers.Tokenizer.from_pretrained()`. If specified and `tokenizers`
not installed, will throw exception. fallback `whitespace-split` used to
avoid exception when `tokenizers` not installed.

- `tree_divider: str = '│   '`

- `tree_file_divider: str = '├── '`

- `tree_indent: str = ' '`

### `def get_tokenizer_obj`

``` python
(self) -> lmcat.file_stats.TokenizerWrapper
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L92-L94)

Get the tokenizer object

### `def get_processing_pipeline`

``` python
(self) -> lmcat.processing_pipeline.ProcessingPipeline
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L96-L104)

Get the processing pipeline object

### `def read`

``` python
(cls, root_dir: pathlib.Path) -> lmcat.lmcat.LMCatConfig
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L106-L144)

Attempt to read config from pyproject.toml, lmcat.toml, or lmcat.json.

### `def serialize`

``` python
(self) -> dict[str, typing.Any]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L703-L759)

returns the class as a dict, implemented by using
`@serializable_dataclass` decorator

### `def load`

``` python
(cls, data: Union[dict[str, Any], ~T]) -> Type[~T]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L766-L852)

takes in an appropriately structured dict and returns an instance of the
class, implemented by using `@serializable_dataclass` decorator

### `def validate_fields_types`

``` python
(
    self: muutils.json_serialize.serializable_dataclass.SerializableDataclass,
    on_typecheck_error: muutils.errormode.ErrorMode = ErrorMode.Except
) -> bool
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L303-L312)

validate the types of all the fields on a `SerializableDataclass`. calls
`SerializableDataclass__validate_field_type` for each field

### Inherited Members

- [`validate_field_type`](#LMCatConfig.validate_field_type)
- [`diff`](#LMCatConfig.diff)
- [`update_from_nested_dict`](#LMCatConfig.update_from_nested_dict)

### `class IgnoreHandler:`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L147-L172)

Handles all ignore pattern matching using igittigitt

### `IgnoreHandler`

``` python
(root_dir: pathlib.Path, config: lmcat.lmcat.LMCatConfig)
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L150-L163)

- `root_dir: pathlib.Path`

- `config: lmcat.lmcat.LMCatConfig`

- `parser: igittigitt.igittigitt.IgnoreParser`

### `def is_ignored`

``` python
(self, path: pathlib.Path) -> bool
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L165-L172)

Check if a path should be ignored

### `def sorted_entries`

``` python
(directory: pathlib.Path) -> list[pathlib.Path]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L175-L183)

Return directory contents sorted: directories first, then files

### `def walk_dir`

``` python
(
    directory: pathlib.Path,
    ignore_handler: lmcat.lmcat.IgnoreHandler,
    config: lmcat.lmcat.LMCatConfig,
    tokenizer: lmcat.file_stats.TokenizerWrapper,
    prefix: str = ''
) -> tuple[list[lmcat.file_stats.TreeEntry], list[pathlib.Path]]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L186-L228)

Recursively walk a directory, building tree lines and collecting file
paths

### `def format_tree_with_stats`

``` python
(
    entries: list[lmcat.file_stats.TreeEntry],
    show_tokens: bool = False
) -> list[str]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L231-L282)

Format tree entries with aligned statistics

### Parameters:

- `entries : list[TreeEntry]` List of tree entries with optional stats
- `show_tokens : bool` Whether to show token counts

### Returns:

- `list[str]` Formatted tree lines with aligned stats

### `def walk_and_collect`

``` python
(
    root_dir: pathlib.Path,
    config: lmcat.lmcat.LMCatConfig
) -> tuple[list[str], list[pathlib.Path]]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L285-L316)

Walk filesystem from root_dir and gather tree listing plus file paths

### `def assemble_summary`

``` python
(root_dir: pathlib.Path, config: lmcat.lmcat.LMCatConfig) -> str
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L319-L390)

Assemble the summary output and return

### `def main`

``` python
() -> None
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/lmcat.py#L393-L459)

Main entry point for the script

> docs for [`lmcat`](https://github.com/mivanit/lmcat) v0.1.0

## API Documentation

- [`OnMultipleProcessors`](#OnMultipleProcessors)
- [`load_plugins`](#load_plugins)
- [`ProcessingPipeline`](#ProcessingPipeline)

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processing_pipeline.py)

# `lmcat.processing_pipeline`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processing_pipeline.py#L0-L182)

- `OnMultipleProcessors = typing.Literal['warn', 'except', 'do_first', 'do_last', 'skip']`

### `def load_plugins`

``` python
(plugins_file: pathlib.Path) -> None
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processing_pipeline.py#L35-L56)

Load plugins from a Python file.

### Parameters:

- `plugins_file : Path` Path to plugins file

### `class ProcessingPipeline:`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processing_pipeline.py#L59-L183)

Manages the processing pipeline for files.

### Attributes:

- `glob_process : dict[str, ProcessorName]` Maps glob patterns to
  processor names
- `decider_process : dict[DeciderName, ProcessorName]` Maps decider
  names to processor names
- `_compiled_globs : dict[str, re.Pattern]` Cached compiled glob
  patterns for performance

### `ProcessingPipeline`

``` python
(
    plugins_file: pathlib.Path | None,
    decider_process_keys: dict[str, str],
    glob_process_keys: dict[str, str],
    on_multiple_processors: Literal['warn', 'except', 'do_first', 'do_last', 'skip']
)
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processing_pipeline.py#L71-L109)

- `plugins_file: pathlib.Path | None`

- `decider_process_keys: dict[str, str]`

- `glob_process_keys: dict[str, str]`

- `on_multiple_processors: Literal['warn', 'except', 'do_first', 'do_last', 'skip']`

### `def get_processors_for_path`

``` python
(self, path: pathlib.Path) -> list[typing.Callable[[pathlib.Path], str]]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processing_pipeline.py#L111-L134)

Get all applicable processors for a given path.

### Parameters:

- `path : Path` Path to get processors for

### Returns:

- `list[ProcessorFunc]` List of applicable path processors

### `def process_file`

``` python
(self, path: pathlib.Path) -> tuple[str, str | None]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processing_pipeline.py#L136-L183)

Process a file through the pipeline.

### Parameters:

- `path : Path` Path to process the content of

### Returns:

- `tuple[str, str]` Processed content and the processor name if no
  processor is found, will be `(path.read_text(), None)`

> docs for [`lmcat`](https://github.com/mivanit/lmcat) v0.1.0

## API Documentation

- [`ProcessorName`](#ProcessorName)
- [`DeciderName`](#DeciderName)
- [`ProcessorFunc`](#ProcessorFunc)
- [`DeciderFunc`](#DeciderFunc)
- [`PROCESSORS`](#PROCESSORS)
- [`DECIDERS`](#DECIDERS)
- [`register_processor`](#register_processor)
- [`register_decider`](#register_decider)
- [`is_over_10kb`](#is_over_10kb)
- [`is_documentation`](#is_documentation)
- [`remove_comments`](#remove_comments)
- [`compress_whitespace`](#compress_whitespace)
- [`to_relative_path`](#to_relative_path)
- [`ipynb_to_md`](#ipynb_to_md)
- [`makefile_recipes`](#makefile_recipes)
- [`csv_preview_5_lines`](#csv_preview_5_lines)

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py)

# `lmcat.processors`

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L0-L180)

- `ProcessorName = <class 'str'>`

- `DeciderName = <class 'str'>`

- `ProcessorFunc = typing.Callable[[pathlib.Path], str]`

- `DeciderFunc = typing.Callable[[pathlib.Path], bool]`

- `PROCESSORS: dict[str, typing.Callable[[pathlib.Path], str]] = {'remove_comments': <function remove_comments>, 'compress_whitespace': <function compress_whitespace>, 'to_relative_path': <function to_relative_path>, 'ipynb_to_md': <function ipynb_to_md>, 'makefile_recipes': <function makefile_recipes>, 'csv_preview_5_lines': <function csv_preview_5_lines>}`

- `DECIDERS: dict[str, typing.Callable[[pathlib.Path], bool]] = {'is_over_10kb': <function is_over_10kb>, 'is_documentation': <function is_documentation>}`

### `def register_processor`

``` python
(func: Callable[[pathlib.Path], str]) -> Callable[[pathlib.Path], str]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L28-L31)

Register a function as a path processor

### `def register_decider`

``` python
(func: Callable[[pathlib.Path], bool]) -> Callable[[pathlib.Path], bool]
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L34-L37)

Register a function as a decider

### `def is_over_10kb`

``` python
(path: pathlib.Path) -> bool
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L42-L45)

Check if file is over 10KB.

### `def is_documentation`

``` python
(path: pathlib.Path) -> bool
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L48-L51)

Check if file is documentation.

### `def remove_comments`

``` python
(path: pathlib.Path) -> str
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L58-L63)

Remove single-line comments from code.

### `def compress_whitespace`

``` python
(path: pathlib.Path) -> str
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L66-L69)

Compress multiple whitespace characters into single spaces.

### `def to_relative_path`

``` python
(path: pathlib.Path) -> str
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L72-L75)

return the path to the file as a string

### `def ipynb_to_md`

``` python
(path: pathlib.Path) -> str
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L78-L94)

Convert an IPython notebook to markdown.

### `def makefile_recipes`

``` python
(path: pathlib.Path) -> str
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L97-L154)

Process a Makefile to show only target descriptions and basic structure.

Preserves: - Comments above .PHONY targets up to first empty line - The
.PHONY line and target line - First line after target if it starts with
@echo

### Parameters:

- `path : Path` Path to the Makefile to process

### Returns:

- `str` Processed Makefile content

### `def csv_preview_5_lines`

``` python
(path: pathlib.Path) -> str
```

[View Source on
GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/processors.py#L157-L181)

Preview first few lines of a CSV file (up to 5)

Reads only first 1024 bytes and splits into lines. Does not attempt to
parse CSV structure.

### Parameters:

- `path : Path` Path to CSV file

### Returns:

- `str` First few lines of the file
