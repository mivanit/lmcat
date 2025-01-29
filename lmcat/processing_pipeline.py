from importlib.util import spec_from_file_location, module_from_spec
import sys
from pathlib import Path
from typing import Callable, Literal, Optional
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
			plugins_file: Path|None,
			glob_process_keys: dict[str, ProcessorName],
			decider_process_keys: dict[DeciderName, ProcessorName],
			on_multiple_processors: OnMultipleProcessors,
		):
		# store the vars
		self.plugins_file: Path|None = plugins_file
		self.glob_process_keys: dict[str, ProcessorName] = glob_process_keys
		self.decider_process_keys: dict[DeciderName, ProcessorName] = decider_process_keys
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

	def process_file(self, path: Path) -> tuple[str, str|None]:
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
		selected_processor: ProcessorFunc|None

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
					raise ValueError(f"Multiple processors for {path.name}: {processors}")
				case "do_first":
					selected_processor = processors[0]
				case "do_last":
					selected_processor = processors[-1]
				case "skip":
					selected_processor = None
				case _:
					raise ValueError(f"Invalid on_multiple_processors: {self.on_multiple_processors = }")
				

		# Process the file and return
		if selected_processor is None:
			return path.read_text(encoding="utf-8", errors="surrogateescape"), None
		else:
			return selected_processor(path), selected_processor.__name__
	
