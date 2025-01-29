from pathlib import Path
import pytest
from typing import Any

from lmcat.file_stats import FileStats, TokenizerWrapper
from lmcat.lmcat import LMCatConfig
from lmcat.processing_pipeline import OnMultipleProcessors
from lmcat.processors import register_processor, register_decider

# Use same temp path as other tests
TEMP_PATH: Path = Path("tests/_temp")


def test_tokenizer_wrapper_gpt2():
	"""Test TokenizerWrapper with GPT2 tokenizer if available"""
	try:
		tokenizer = TokenizerWrapper("gpt2")
		assert tokenizer.name == "gpt2"
		assert not tokenizer.use_fallback

		# Test token counting
		assert tokenizer.n_tokens("Hello world") == 2
		assert tokenizer.n_tokens("Hello   world") == 4  # Multiple spaces
	except ImportError:
		pytest.skip("tokenizers package not installed")


def test_tokenizer_wrapper_fallback():
	"""Test TokenizerWrapper fallback whitespace tokenization"""
	tokenizer = TokenizerWrapper("whitespace-split")
	assert tokenizer.name == "whitespace-split"
	assert tokenizer.use_fallback

	assert tokenizer.n_tokens("Hello world") == 2
	assert tokenizer.n_tokens("Hello   world") == 2
	assert tokenizer.n_tokens("abc") == 1


def test_processing_pipeline_multiple_matches():
	"""Test different behaviors when multiple processors match"""
	test_dir = TEMP_PATH / "pipeline_test"
	test_dir.mkdir(parents=True, exist_ok=True)
	test_file = test_dir / "test.txt"
	test_file.write_text("original content")

	# Register test processors
	@register_processor
	def processor1(path: Path) -> str:
		return "processor1 output"

	@register_processor
	def processor2(path: Path) -> str:
		return "processor2 output"

	@register_decider
	def always_true(path: Path) -> bool:
		return True

	# Test different OnMultipleProcessors behaviors
	configs: dict[OnMultipleProcessors, Any] = {
		"do_first": "processor1 output",
		"do_last": "processor2 output",
		"skip": "original content",
	}

	for mode, expected in configs.items():
		print(f"{mode = }, {expected = }")
		config = LMCatConfig(
			decider_process={"always_true": "processor1"},
			glob_process={"*.txt": "processor2"},
			on_multiple_processors=mode,
		)
		pipeline = config.get_processing_pipeline()
		result, p_used = pipeline.process_file(test_file)
		if mode == "skip":
			assert p_used is None
		elif mode == "do_first":
			assert p_used == "processor1"
		elif mode == "do_last":
			assert p_used == "processor2"

		assert result == expected

	# Test "except" mode raises error
	config_except = LMCatConfig(
		decider_process={"always_true": "processor1"},
		glob_process={"*.txt": "processor2"},
		on_multiple_processors="except",
	)
	pipeline = config_except.get_processing_pipeline()
	with pytest.raises(ValueError):
		pipeline.process_file(test_file)


def test_filestats_large_file():
	"""Test FileStats handling of large files"""
	test_dir = TEMP_PATH / "large_file_stats"
	test_dir.mkdir(parents=True, exist_ok=True)
	large_file = test_dir / "large.txt"

	# Create 5MB file
	chunk = "x" * 1024  # 1KB chunk
	with large_file.open("w") as f:
		for _ in range(5 * 1024):  # Write 5MB
			f.write(chunk)

	tokenizer = TokenizerWrapper()
	stats = FileStats.from_file(large_file, tokenizer)

	assert stats.lines == 1
	assert stats.chars == 5 * 1024 * 1024
	assert stats.tokens is not None
	assert stats.tokens > 0


def test_config_plugins():
	"""Test plugin loading functionality"""
	test_dir = TEMP_PATH / "plugins_test"
	test_dir.mkdir(parents=True, exist_ok=True)

	# Create test plugin file
	plugin_file = test_dir / "test_plugin.py"
	plugin_file.write_text("""
from pathlib import Path
from lmcat.processors import register_processor, register_decider

@register_processor
def custom_processor(path: Path) -> str:
    return "custom processed"

@register_decider
def custom_decider(path: Path) -> bool:
    return path.suffix == '.custom'
""")

	# Test with plugins enabled
	config = LMCatConfig(
		plugins_file=plugin_file,
		allow_plugins=True,
		decider_process={"custom_decider": "custom_processor"},
	)

	pipeline = config.get_processing_pipeline()

	# Create test file
	test_file = test_dir / "test.custom"
	test_file.write_text("original content")

	# Test custom processor
	result, processor_name = pipeline.process_file(test_file)
	assert result == "custom processed"
	assert processor_name == "custom_processor"


def test_error_files():
	"""Test handling of files with various error conditions"""
	test_dir = TEMP_PATH / "error_files"
	test_dir.mkdir(parents=True, exist_ok=True)

	# Create a directory that looks like a file
	dir_file = test_dir / "dir.txt"
	dir_file.mkdir()

	# Create an empty file
	empty_file = test_dir / "empty.txt"
	empty_file.touch()

	# Create file with invalid UTF-8
	invalid_utf8 = test_dir / "invalid.txt"
	invalid_utf8.write_bytes(b"Hello\xff\xfeWorld")

	tokenizer = TokenizerWrapper()

	# Test empty file
	stats = FileStats.from_file(empty_file, tokenizer)
	assert stats.lines == 0
	assert stats.chars == 0
	assert stats.tokens == 0

	# Test invalid UTF-8 file
	stats = FileStats.from_file(invalid_utf8, tokenizer)
	assert stats.lines >= 0  # Should handle without crashing
	assert stats.chars >= 0
	assert stats.tokens is not None
