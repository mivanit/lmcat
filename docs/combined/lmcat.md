> docs for [`lmcat`](https://github.com/mivanit/lmcat) v0.0.1


## Contents
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



## API Documentation

 - [`main`](#main)




[View Source on GitHub](https://github.com/mivanit/lmcat/blob/0.0.1/__init__.py)

# `lmcat` { #lmcat }

### lmcat

A Python tool for concatenating files and directory structures into a single document, perfect for sharing code with language models. It respects `.gitignore` and `.lmignore` patterns and provides configurable output formatting.

#### Features

- Creates a tree view of your directory structure
- Includes file contents with clear delimiters
- Respects `.gitignore` patterns (can be disabled)
- Supports custom ignore patterns via `.lmignore`
- Configurable via `pyproject.toml`, `lmcat.toml`, or `lmcat.json`
- Python 3.11+ native, with fallback support for older versions

#### Installation

Install from PyPI:

```bash
pip install lmcat
```

#### Usage

Basic usage - concatenate current directory:

```bash
python -m lmcat
```

The output will include a directory tree and the contents of each non-ignored file.

##### Command Line Options

- `-g`, `--no-include-gitignore`: Ignore `.gitignore` files (they are included by default)
- `-t`, `--tree-only`: Only print the directory tree, not file contents
- `-o`, `--output`: Specify an output file (defaults to stdout)
- `-h`, `--help`: Show help message

##### Configuration

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

##### Ignore Patterns

lmcat supports two types of ignore files:

1. `.gitignore` - Standard Git ignore patterns (used by default)
2. `.lmignore` - Custom ignore patterns specific to lmcat

`.lmignore` follows the same pattern syntax as `.gitignore`. Patterns in `.lmignore` take precedence over `.gitignore`.

Example `.lmignore`:
```gitignore
### Ignore all .log files
*.log

### Ignore the build directory and its contents
build/

### Un-ignore a specific file (overrides previous patterns)
!important.log
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

This will:
- Create a virtual environment
- Install development dependencies
- Set up pre-commit hooks

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

For test coverage:
```bash
make cov
```


##### Roadmap

- better tests, I feel like gitignore/lmignore interaction is broken
- llm summarization and caching of those summaries in `.lmsummary/`
- reasonable defaults for file extensions to ignore
- web interface


[View Source on GitHub](https://github.com/mivanit/lmcat/blob/0.0.1/__init__.py#L0-L6)



### `def main` { #main }
```python
() -> None
```


[View Source on GitHub](https://github.com/mivanit/lmcat/blob/0.0.1/__init__.py#L194-L273)


Main entry point for the script



