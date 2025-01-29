> docs for [`lmcat`](https://github.com/mivanit/lmcat) v0.1.0


## Contents
# lmcat

A Python tool for concatenating files and directory structures into a single document, perfect for sharing code with language models. It respects `.gitignore` and `.lmignore` patterns and provides configurable output formatting.

## Features

- Tree view of directory structure with file statistics (lines, characters, tokens)
- Includes file contents with clear delimiters
- Respects `.gitignore` patterns (can be disabled)
- Supports custom ignore patterns via `.lmignore`
- Configurable via `pyproject.toml`, `lmcat.toml`, or `lmcat.json`
	- you can specify `glob_process` or `decider_process` to run on files, like if you want to convert a notebook to a markdown file

## Installation

Install from PyPI:

```bash
pip install lmcat
```

or, install with support for counting tokens:
```bash
pip install lmcat[tokenizers]
```

## Usage

Basic usage - concatenate current directory:

```bash
# Only show directory tree
python -m lmcat --tree-only

# Write output to file
python -m lmcat --output summary.md

# Print current configuration
python -m lmcat --print-cfg
```

The output will include a directory tree and the contents of each non-ignored file.

### Command Line Options

- `-t`, `--tree-only`: Only print the directory tree, not file contents
- `-o`, `--output`: Specify an output file (defaults to stdout)
- `-h`, `--help`: Show help message

### Configuration

lmcat is best configured via a `tool.lmcat` section in `pyproject.toml`:

```toml
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

1. Clone the repository:
```bash
git clone https://github.com/mivanit/lmcat
cd lmcat
```

2. Set up the development environment:
```bash
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

```bash
make test
```

For verbose output:
```bash
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



## API Documentation

 - [`main`](#main)




[View Source on GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/__init__.py)

# `lmcat` { #lmcat }

### lmcat

A Python tool for concatenating files and directory structures into a single document, perfect for sharing code with language models. It respects `.gitignore` and `.lmignore` patterns and provides configurable output formatting.

#### Features

- Tree view of directory structure with file statistics (lines, characters, tokens)
- Includes file contents with clear delimiters
- Respects `.gitignore` patterns (can be disabled)
- Supports custom ignore patterns via `.lmignore`
- Configurable via `pyproject.toml`, `lmcat.toml`, or `lmcat.json`
	- you can specify `glob_process` or `decider_process` to run on files, like if you want to convert a notebook to a markdown file

#### Installation

Install from PyPI:

```bash
pip install lmcat
```

or, install with support for counting tokens:
```bash
pip install lmcat[tokenizers]
```

#### Usage

Basic usage - concatenate current directory:

```bash
### Only show directory tree
python -m lmcat --tree-only

### Write output to file
python -m lmcat --output summary.md

### Print current configuration
python -m lmcat --print-cfg
```

The output will include a directory tree and the contents of each non-ignored file.

##### Command Line Options

- `-t`, `--tree-only`: Only print the directory tree, not file contents
- `-o`, `--output`: Specify an output file (defaults to stdout)
- `-h`, `--help`: Show help message

##### Configuration

lmcat is best configured via a `tool.lmcat` section in `pyproject.toml`:

```toml
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

1. Clone the repository:
```bash
git clone https://github.com/mivanit/lmcat
cd lmcat
```

2. Set up the development environment:
```bash
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

```bash
make test
```

For verbose output:
```bash
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


[View Source on GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/__init__.py#L0-L6)



### `def main` { #main }
```python
() -> None
```


[View Source on GitHub](https://github.com/mivanit/lmcat/blob/0.1.0/__init__.py#L393-L459)


Main entry point for the script



