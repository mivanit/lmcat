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
