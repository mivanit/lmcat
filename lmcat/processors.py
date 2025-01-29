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
	processed = [line for line in lines if not line.strip().startswith('#')]
	return '\n'.join(processed)

@register_processor
def compress_whitespace(path: Path) -> str:
	"""Compress multiple whitespace characters into single spaces."""
	return ' '.join(path.read_text().split())

@register_processor
def to_relative_path(path: Path) -> str:
	"""return the path to the file as a string"""
	return path.as_posix()

@register_decider
def is_python_file(path: Path) -> bool:
	"""Check if file is a Python source file."""
	return path.suffix == '.py'

@register_decider
def is_documentation(path: Path) -> bool:
	"""Check if file is documentation."""
	return path.suffix in {'.md', '.rst', '.txt'}


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
		if line.strip().startswith('.PHONY:'):
			# Store target name for later matching
			target_name: str = line.split(':')[1].strip()
			
			# Collect comments above until empty line
			comment_lines: list[str] = []
			look_back: int = i - 1
			while look_back >= 0 and lines[look_back].strip():
				if lines[look_back].strip().startswith('#'):
					comment_lines.insert(0, lines[look_back])
				look_back -= 1
			
			# Add collected comments
			output_lines.extend(comment_lines)
			
			# Add .PHONY line
			output_lines.append(line)
			
			# Add target line (should be next)
			if i + 1 < len(lines) and lines[i + 1].startswith(f'{target_name}:'):
				output_lines.append(lines[i + 1])
				i += 1
				
				# Check for @echo on next line
				if i + 1 < len(lines) and lines[i + 1].strip().startswith('@echo'):
					output_lines.append(lines[i + 1])

				output_lines.append('	...')
				output_lines.append('')
			
		i += 1
	
	return '\n'.join(output_lines)