import os
import json
import pytest
import shutil
from pathlib import Path
from typing import Generator, Callable

from lmcat.lmcat import (
    LMCatConfig,
    IgnoreHandler,
    walk_dir,
    walk_and_collect,
    assemble_summary,
    format_tree_with_stats,
)
from lmcat.file_stats import FileStats, TokenizerWrapper, TreeEntry

# Base test directory
TEMP_PATH: Path = Path("tests/_temp")

def ensure_clean_dir(dirpath: Path) -> None:
    """Remove dirpath if it exists, then re-create it."""
    if dirpath.is_dir():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def make_test_dir() -> Callable[[str, dict[str, str]], Path]:
    """Fixture that returns a function to create test directories with files
    
    # Parameters:
    - name: str - name of the test directory under TEMP_PATH
    - files: dict[str, str] - mapping of relative paths to file contents
    
    # Returns:
    - Path to the created test directory
    """
    def _make_test_dir(name: str, files: dict[str, str]) -> Path:
        test_dir = TEMP_PATH / name
        ensure_clean_dir(test_dir)
        
        for rel_path, content in files.items():
            file_path = test_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # Handle binary content (e.g., .pyc files)
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content, encoding='utf-8')
        
        return test_dir
    
    return _make_test_dir

def test_nested_ignore_patterns(make_test_dir: Callable[[str, dict[str, str]], Path]) -> None:
    """Test handling of nested .gitignore and .lmignore files"""
    files = {
        ".gitignore": "*.pyc\n__pycache__/\n",
        ".lmignore": "*.log\n!important.log\n",
        "src/.gitignore": "*.tmp\n",
        "src/.lmignore": "!debug.tmp\n",
        "src/app.py": "print('hello')\n",
        "src/debug.tmp": "debug data",
        "src/other.tmp": "temp data",
        "src/debug.log": "debug log",
        "src/important.log": "important log",
        "src/__pycache__/app.cpython-39.pyc": b"dummy pyc content",
    }
    
    test_dir = make_test_dir("nested_ignore_test", files)
    config = LMCatConfig()
    handler = IgnoreHandler(test_dir, config)
    
    # Test ignore patterns
    assert not handler.is_ignored(test_dir / "src" / "app.py")
    assert not handler.is_ignored(test_dir / "src" / "debug.tmp")  # Unignored by nested .lmignore
    assert handler.is_ignored(test_dir / "src" / "other.tmp")
    assert handler.is_ignored(test_dir / "src" / "debug.log")
    assert not handler.is_ignored(test_dir / "src" / "important.log")
    assert handler.is_ignored(test_dir / "src" / "__pycache__" / "app.cpython-39.pyc")

def test_unicode_file_handling(make_test_dir: Callable[[str, dict[str, str]], Path]) -> None:
    """Test handling of Unicode in filenames and content"""
    files = {
        "hello_世界.txt": "Hello 世界\nこんにちは\n",
        "привет/мир.txt": "Привет мир!\n",
        "emoji_📁/test_🔧.txt": "Test with emojis 🎉\n",
        ".gitignore": "*.tmp\n",
        "unicode_temp_⚡.tmp": "should be ignored",
    }
    
    test_dir = make_test_dir("unicode_test", files)
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

def test_symlink_handling(make_test_dir: Callable[[str, dict[str, str]], Path]) -> None:
    """Test handling of symlinks in directory structure"""
    # Create initial files
    files = {
        "src/module.py": "print('original')\n",
        "docs/readme.md": "# Documentation\n",
    }
    
    test_dir = make_test_dir("symlink_test", files)
    
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
        
    except OSError as e:
        pytest.skip(f"Symlink creation failed: {e}")

def test_large_file_handling(make_test_dir: Callable[[str, dict[str, str]], Path]) -> None:
    """Test handling of large files"""
    # Create a mix of large and small files
    files = {
        "small.txt": "small content\n",
        "medium.txt": "medium " * 1000,
        # Large file created separately
    }
    
    test_dir = make_test_dir("large_file_test", files)
    
    # Create a 1MB file
    large_file = test_dir / "large.txt"
    with large_file.open("w") as f:
        f.write("x" * (1024 * 1024))
    
    config = LMCatConfig()
    tree_output, file_list = walk_and_collect(test_dir, config)
    
    # Check stats in tree output
    tree_str = "\n".join(tree_output)
    assert "small.txt" in tree_str
    assert "medium.txt" in tree_str
    assert "large.txt" in tree_str
    
    # Check that large file is readable
    summary = assemble_summary(test_dir, config)
    assert "small content" in summary
    assert "medium " * 10 in summary  # Check start of medium file
    assert "x" * 100 in summary  # Check start of large file

def test_empty_directories(make_test_dir: Callable[[str, dict[str, str]], Path]) -> None:
    """Test handling of empty directories and nested empty directories"""
    files = {
        "src/module.py": "print('hello')\n",
        # Empty directories created after
    }
    
    test_dir = make_test_dir("empty_dir_test", files)
    
    # Create empty directories
    (test_dir / "empty1").mkdir()
    (test_dir / "empty2" / "nested").mkdir(parents=True)
    (test_dir / "src" / "empty3").mkdir()
    
    config = LMCatConfig()
    tree_output, file_list = walk_and_collect(test_dir, config)
    tree_str = "\n".join(tree_output)
    
    # Check empty directories in tree
    assert "empty1" in tree_str
    assert "empty2" in tree_str
    assert "nested" in tree_str
    assert "empty3" in tree_str
    
    # Only non-empty files should be in file_list
    assert len(file_list) == 1
    assert file_list[0].name == "module.py"

def test_error_handling(make_test_dir: Callable[[str, dict[str, str]], Path]) -> None:
    """Test error handling for various filesystem conditions"""
    files = {
        "readable.txt": "can read this\n",
        "binary.bin": b"\x00\x01\x02\x03",
    }
    
    test_dir = make_test_dir("error_test", files)
    
    # Create unreadable file
    unreadable = test_dir / "unreadable.txt"
    unreadable.write_text("secret")
    try:
        os.chmod(unreadable, 0o000)
    except PermissionError:
        pytest.skip("Cannot create unreadable file")
    
    config = LMCatConfig()
    try:
        tree_output, file_list = walk_and_collect(test_dir, config)
        tree_str = "\n".join(tree_output)
        
        # Check that readable files are included
        assert "readable.txt" in tree_str
        assert "binary.bin" in tree_str
        
        # Unreadable file might still appear in tree but not in content
        summary = assemble_summary(test_dir, config)
        assert "can read this" in summary
        
    finally:
        # Restore permissions so cleanup can work
        try:
            os.chmod(unreadable, 0o666)
        except OSError:
            pass