import json
from typing import Callable, Sequence
from pathlib import Path


# type defs
# ==================================================

ProcessorName = str
DeciderName = str

ProcessorFunc = Callable[[Path], str]
DeciderFunc = Callable[[Path], bool]


# global dicts of processors and deciders
# ==================================================

PROCESSORS: dict[ProcessorName, ProcessorFunc] = dict()

DECIDERS: dict[DeciderName, DeciderFunc] = dict()


# register functions
# ==================================================


def register_processor(func: ProcessorFunc) -> ProcessorFunc:
	"""Register a function as a path processor"""
	PROCESSORS[ProcessorName(func.__name__)] = func
	return func


def register_decider(func: DeciderFunc) -> DeciderFunc:
	"""Register a function as a decider"""
	DECIDERS[DeciderName(func.__name__)] = func
	return func


# default deciders
# ==================================================
@register_decider
def is_over_10kb(path: Path) -> bool:
	"""Check if file is over 10KB."""
	return path.stat().st_size > 2**1


@register_decider
def is_documentation(path: Path) -> bool:
	"""Check if file is documentation."""
	return path.suffix in {".md", ".rst", ".txt"}


# default processors
# ==================================================


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


@register_processor
def ipynb_to_md(path: Path) -> str:
	"""Convert an IPython notebook to markdown."""
	nb_contents: dict = json.loads(path.read_text(encoding="utf-8"))

	output: list[str] = []

	for cell in nb_contents["cells"]:
		if cell["cell_type"] == "markdown":
			output.extend(cell["source"])
			output.append("\n\n")
		elif cell["cell_type"] == "code":
			output.append("```python\n")
			output.extend(cell["source"])
			output.append("\n```\n\n")

	return "".join(output)


@register_processor
def makefile_recipes(path: Path) -> str:
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


@register_processor
def csv_preview_5_lines(path: Path) -> str:
	"""Preview first few lines of a CSV file (up to 5)

	Reads only first 1024 bytes and splits into lines.
	Does not attempt to parse CSV structure.

	# Parameters:
	- `path : Path`
	    Path to CSV file

	# Returns:
	- `str`
	    First few lines of the file"""
	try:
		with path.open("r", encoding="utf-8") as f:
			content = f.read(1024)

		lines = content.splitlines()[:5]
		if len(content) == 1024:
			lines.append("... (truncated)")

		return "\n".join(lines)
	except Exception as e:
		return f"Error previewing CSV: {str(e)}"
