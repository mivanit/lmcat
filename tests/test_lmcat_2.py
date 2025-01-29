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
