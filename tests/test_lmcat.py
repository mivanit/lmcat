import pytest
from pathlib import Path
import os
import shutil
import textwrap
import subprocess

from lmcat.lmcat import (
    LMCatConfig,
    load_ignore_patterns,
    is_ignored,
    walk_dir,
    walk_and_collect,
    main
)

def test_lmcat_config_defaults():
    """Test that LMCatConfig defaults are as expected."""
    config = LMCatConfig()
    assert config.tree_divider == "│   "
    assert config.indent == "    "
    assert config.file_divider == "├── "
    assert config.content_divider == "``````"

def test_lmcat_config_load_partial():
    """Test partial override from dict."""
    data = {
        "tree_divider": "|---",
    }
    config = LMCatConfig.load(data)
    assert config.tree_divider == "|---"
    # Other defaults remain
    assert config.indent == "    "
    assert config.file_divider == "├── "
    assert config.content_divider == "``````"

def test_lmcat_config_load_all():
    """Test loading a dict that overrides all keys."""
    data = {
        "tree_divider": "XX",
        "indent": "YY",
        "file_divider": "ZZ",
        "content_divider": "@@@"
    }
    config = LMCatConfig.load(data)
    assert config.tree_divider == "XX"
    assert config.indent == "YY"
    assert config.file_divider == "ZZ"
    assert config.content_divider == "@@@"

@pytest.fixture
def sample_root(tmp_path: Path):
    """Create a sample directory structure with optional config files."""
    # Dir structure:
    # tmp_path/
    #   subdir1/
    #     file1.txt
    #   subdir2/
    #     file2.md
    #   file3.log
    # We'll let tests add .lmignore or config files as needed
    (tmp_path / "subdir1").mkdir()
    (tmp_path / "subdir2").mkdir()

    (tmp_path / "subdir1" / "file1.txt").write_text("content of file1", encoding="utf-8")
    (tmp_path / "subdir2" / "file2.md").write_text("content of file2", encoding="utf-8")
    (tmp_path / "file3.log").write_text("content of file3", encoding="utf-8")

    return tmp_path

def test_load_config_no_files(sample_root: Path):
    """Test read() with no config files: fallback to defaults."""
    config = LMCatConfig.read(sample_root)
    assert config.tree_divider == "│   "
    assert config.indent == "    "
    assert config.file_divider == "├── "
    assert config.content_divider == "``````"

@pytest.mark.parametrize("config_name,contents", [
    ("pyproject.toml", textwrap.dedent("""\
        [tool.lmcat]
        tree_divider = ">>>"
        indent = "___"
        file_divider = "+++"
        content_divider = "@@@@@@"
    """)),
    ("lmcat.toml", textwrap.dedent("""\
        tree_divider = ">>>"
        indent = "___"
        file_divider = "+++"
        content_divider = "@@@@@@"
    """)),
    ("lmcat.json", textwrap.dedent("""\
        {
            "tree_divider": ">>>",
            "indent": "___",
            "file_divider": "+++",
            "content_divider": "@@@@@@"
        }
    """)),
])
def test_load_config_variants(sample_root: Path, config_name: str, contents: str):
    """Test read() from pyproject.toml, lmcat.toml, or lmcat.json."""
    (sample_root / config_name).write_text(contents, encoding="utf-8")
    config = LMCatConfig.read(sample_root)
    assert config.tree_divider == ">>>"
    assert config.indent == "___"
    assert config.file_divider == "+++"
    assert config.content_divider == "@@@@@@"

def test_load_ignore_patterns_no_lmignore(sample_root: Path):
    """No .lmignore files => empty dict."""
    ignore_dict = load_ignore_patterns(sample_root)
    assert ignore_dict == {}

def test_load_ignore_patterns_one_file(sample_root: Path):
    """Single .lmignore in root."""
    (sample_root / ".lmignore").write_text(textwrap.dedent("""\
        # This is a comment
        *.md
        subdir1/
    """))
    ignore_dict = load_ignore_patterns(sample_root)
    assert len(ignore_dict) == 1
    assert sample_root in ignore_dict
    patterns = ignore_dict[sample_root]
    assert len(patterns) == 2
    assert "*.md" in patterns
    assert "subdir1/" in patterns

def test_is_ignored_basic(sample_root: Path):
    """Test is_ignored with basic patterns in .lmignore."""
    (sample_root / ".lmignore").write_text(textwrap.dedent("""\
        *.log
        subdir2/
    """))
    ignore_dict = load_ignore_patterns(sample_root)

    # file3.log => should be ignored
    path_log = sample_root / "file3.log"
    assert is_ignored(path_log, sample_root, ignore_dict) is True

    # subdir2 => ignored
    path_subdir2 = sample_root / "subdir2"
    assert is_ignored(path_subdir2, sample_root, ignore_dict) is True
    # file2 within subdir2 => also ignored
    path_file2 = sample_root / "subdir2" / "file2.md"
    assert is_ignored(path_file2, sample_root, ignore_dict) is True

    # subdir1 => not ignored
    path_subdir1 = sample_root / "subdir1"
    assert is_ignored(path_subdir1, sample_root, ignore_dict) is False
    # file1 => not ignored
    path_file1 = sample_root / "subdir1" / "file1.txt"
    assert is_ignored(path_file1, sample_root, ignore_dict) is False

def test_is_ignored_child_dir_has_own_lmignore(sample_root: Path):
    """Child directory has its own .lmignore, distinct from root's."""
    # Root .lmignore => ignores *.md
    # child .lmignore => ignores file1*
    (sample_root / ".lmignore").write_text("*.md\n")
    subdir1_lmignore = sample_root / "subdir1" / ".lmignore"
    subdir1_lmignore.write_text("file1*\n")

    ignore_dict = load_ignore_patterns(sample_root)

    # subdir2 => contains file2.md => should be ignored by root-lmignore pattern
    path_file2 = sample_root / "subdir2" / "file2.md"
    assert is_ignored(path_file2, sample_root, ignore_dict) is True

    # subdir1 => has file1.txt => should be ignored by subdir1-lmignore pattern
    path_file1 = sample_root / "subdir1" / "file1.txt"
    assert is_ignored(path_file1, sample_root, ignore_dict) is True

    # subdir1 => is not itself ignored by root-lmignore
    path_subdir1 = sample_root / "subdir1"
    assert is_ignored(path_subdir1, sample_root, ignore_dict) is False

def test_is_ignored_stop_at_root(sample_root: Path):
    """Ensure we don't go above root when checking .lmignore ancestry."""
    (sample_root / ".lmignore").write_text("*.log\n")
    ignore_dict = load_ignore_patterns(sample_root)

    outside_path = sample_root.parent  # one level above
    # Even if there's a .lmignore above, we won't consider it if root_dir is sample_root
    # (We won't test that fully here, but let's confirm no crash)
    assert is_ignored(outside_path, sample_root, ignore_dict) is False

def test_walk_dir_no_lmignore(sample_root: Path):
    """Check that walk_dir returns everything when no .lmignore is present."""
    config = LMCatConfig()
    ignore_dict = {}
    tree_out, files = walk_dir(sample_root, sample_root, ignore_dict, config)
    # We expect lines for subdir1, file1, subdir2, file2, file3
    # subdir1 and subdir2 appear in alphabetical order: subdir1, subdir2
    # file3 appears after them
    assert "subdir1" in tree_out[0]
    assert "subdir2" in tree_out[1] or "subdir2" in tree_out[2]
    assert "file3.log" in tree_out[-1]
    # We also expect the correct number of total lines
    # Explanation: each directory line, then we expand subdir1 -> file1, subdir2 -> file2
    # e.g. something like:
    # subdir1
    #     file1.txt
    # subdir2
    #     file2.md
    # file3.log
    assert len(files) == 3
    # check the exact file set
    assert set(files) == {
        sample_root / "subdir1" / "file1.txt",
        sample_root / "subdir2" / "file2.md",
        sample_root / "file3.log",
    }

def test_walk_dir_with_ignore(sample_root: Path):
    """Check ignoring subdir2 and *.log."""
    (sample_root / ".lmignore").write_text(textwrap.dedent("""\
        subdir2/
        *.log
    """))
    ignore_dict = load_ignore_patterns(sample_root)
    config = LMCatConfig()
    tree_out, files = walk_dir(sample_root, sample_root, ignore_dict, config)
    # subdir2 + file3.log => ignored
    # only subdir1/file1.txt remains
    assert any("subdir2" in line for line in tree_out) is False
    assert any("file3.log" in line for line in tree_out) is False
    assert any("subdir1" in line for line in tree_out) is True
    assert any("file1.txt" in line for line in tree_out) is True
    assert len(files) == 1
    assert files[0] == sample_root / "subdir1" / "file1.txt"

def test_walk_and_collect(sample_root: Path):
    """Integration test for walk_and_collect with no ignores."""
    config = LMCatConfig()
    tree_out, files = walk_and_collect(sample_root, config)
    # tree_out[0] is the name of the root directory
    assert tree_out[0] == sample_root.name
    # subsequent lines, we check that subdirs & files appear
    assert any("subdir1" in line for line in tree_out)
    assert any("subdir2" in line for line in tree_out)
    assert any("file1.txt" in line for line in tree_out)
    assert any("file2.md" in line for line in tree_out)
    assert any("file3.log" in line for line in tree_out)
    assert len(files) == 3

def test_walk_and_collect_with_ignore(sample_root: Path):
    """Integration test with .lmignore ignoring subdir1, *.md."""
    (sample_root / ".lmignore").write_text("subdir1/\n*.md\n")
    config = LMCatConfig()
    tree_out, files = walk_and_collect(sample_root, config)
    # root name
    assert tree_out[0] == sample_root.name
    # subdir1 => ignored
    assert not any("subdir1" in line for line in tree_out)
    # file2.md => ignored
    assert not any("file2.md" in line for line in tree_out)
    # only file3.log remains
    assert any("file3.log" in line for line in tree_out)
    assert len(files) == 1
    assert files[0].name == "file3.log"

def test_main_no_config_no_ignore(tmp_path: Path, capsys):
    """Test main() in a simple directory with no config, no ignores."""
    # We'll replicate a minimal script usage
    (tmp_path / "fileA.txt").write_text("AAA", encoding="utf-8")
    (tmp_path / "fileB.txt").write_text("BBB", encoding="utf-8")
    # We'll cd into tmp_path and run main
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # We expect the default config. We'll call main() directly,
        # capturing the output
        main()
        captured = capsys.readouterr()
        out = captured.out
        # The first line is the root dir name (the tmp dir's name)
        assert tmp_path.name in out
        # Then some lines for fileA.txt, fileB.txt
        assert "fileA.txt" in out
        assert "fileB.txt" in out
        # Then the content divider and contents for each file
        assert "``````" in out
        assert "AAA" in out
        assert "BBB" in out
    finally:
        os.chdir(original_cwd)

def test_main_with_ignore_and_toml_config(tmp_path: Path, capsys):
    """Test main() with a .lmignore that excludes files, and a toml config that changes dividers."""
    (tmp_path / "pyproject.toml").write_text(textwrap.dedent("""\
        [tool.lmcat]
        tree_divider = ">>>"
        indent = "---"
        file_divider = "+++"
        content_divider = "@@@@"
    """))
    (tmp_path / "fileA.log").write_text("AAA", encoding="utf-8")
    (tmp_path / "fileB.txt").write_text("BBB", encoding="utf-8")
    (tmp_path / ".lmignore").write_text("*.log\n", encoding="utf-8")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        main()
        captured = capsys.readouterr()
        out = captured.out
        # The first line is the root dir name
        assert tmp_path.name in out
        # We used "+++" as a file divider, ">>>" as a tree divider, etc
        assert "+++" in out
        assert ">>>" in out
        # fileA.log => ignored
        assert "fileA.log" not in out
        # fileB.txt => included
        assert "fileB.txt" in out
        # content divider => "@@@@"
        # content => "BBB"
        assert "@@@@" in out
        assert "BBB" in out
    finally:
        os.chdir(original_cwd)

def test_main_with_nested_lmignore(tmp_path: Path, capsys):
    """Test main() with nested .lmignore in subfolders, ensuring we ignore subdir2, but partial subdir1 is allowed."""
    subdir1 = tmp_path / "subdir1"
    subdir1.mkdir()
    subdir2 = tmp_path / "subdir2"
    subdir2.mkdir()

    # Root ignore: ignore subdir2 entirely
    (tmp_path / ".lmignore").write_text("subdir2\n")
    # subdir1 ignore: ignore anything that matches "secret*"
    (subdir1 / ".lmignore").write_text("secret*\n")

    (subdir1 / "normal.txt").write_text("normal content", encoding="utf-8")
    (subdir1 / "secret.txt").write_text("secret content", encoding="utf-8")
    (subdir2 / "whatever.dat").write_text("some data", encoding="utf-8")

    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        main()
        captured = capsys.readouterr()
        out = captured.out
        # Root name
        assert tmp_path.name in out
        # subdir2 => fully ignored
        assert "subdir2" not in out
        assert "whatever.dat" not in out
        # subdir1 => included, but "secret.txt" ignored
        assert "subdir1" in out
        assert "normal.txt" in out
        assert "secret.txt" not in out
        # content of normal.txt => present
        assert "normal content" in out
    finally:
        os.chdir(original_cwd)