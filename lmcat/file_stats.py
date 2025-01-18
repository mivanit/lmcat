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
