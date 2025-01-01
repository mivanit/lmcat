import pytest
import os
import textwrap
import shutil
import subprocess
from pathlib import Path

from lmcat.lmcat import (
    LMCatConfig,
    load_ignore_patterns,
    is_ignored,
    walk_dir,
    walk_and_collect,
    main,
)

# We will store all test directories under this path:
TEMP_PATH: Path = Path("tests/_temp")


def ensure_clean_dir(dirpath: Path) -> None:
    """Remove `dirpath` if it exists, then re-create it."""
    if dirpath.is_dir():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)


def test_lmcat_config_defaults():
    config = LMCatConfig()
    assert config.tree_divider == "│   "
    assert config.indent == " "
    assert config.file_divider == "├── "
    assert config.content_divider == "``````"


def test_lmcat_config_load_partial():
    data = {"tree_divider": "|---"}
    config = LMCatConfig.load(data)
    assert config.tree_divider == "|---"
    assert config.indent == " "
    assert config.file_divider == "├── "
    assert config.content_divider == "``````"


def test_lmcat_config_load_all():
    data = {
        "tree_divider": "XX",
        "indent": "YY",
        "file_divider": "ZZ",
        "content_divider": "@@@",
    }
    config = LMCatConfig.load(data)
    assert config.tree_divider == "XX"
    assert config.indent == "YY"
    assert config.file_divider == "ZZ"
    assert config.content_divider == "@@@"


def test_load_config_no_files():
    test_dir = TEMP_PATH / "test_load_config_no_files"
    ensure_clean_dir(test_dir)
    # No config files => fallback to defaults
    config = LMCatConfig.read(test_dir)
    assert config.tree_divider == "│   "
    assert config.indent == " "
    assert config.file_divider == "├── "
    assert config.content_divider == "``````"


@pytest.mark.parametrize(
    "config_name,contents",
    [
        (
            "pyproject.toml",
            textwrap.dedent(
                """\
                [tool.lmcat]
                tree_divider = ">>>"
                indent = "___"
                file_divider = "+++"
                content_divider = "@@@@@@"
                """
            ),
        ),
        (
            "lmcat.toml",
            textwrap.dedent(
                """\
                tree_divider = ">>>"
                indent = "___"
                file_divider = "+++"
                content_divider = "@@@@@@"
                """
            ),
        ),
        (
            "lmcat.json",
            textwrap.dedent(
                """\
                {
                    "tree_divider": ">>>",
                    "indent": "___",
                    "file_divider": "+++",
                    "content_divider": "@@@@@@"
                }
                """
            ),
        ),
    ],
)
def test_load_config_variants(config_name: str, contents: str):
    test_dir = TEMP_PATH / f"test_load_config_variants_{config_name}"
    ensure_clean_dir(test_dir)
    (test_dir / config_name).write_text(contents, encoding="utf-8")

    config = LMCatConfig.read(test_dir)
    assert config.tree_divider == ">>>"
    assert config.indent == "___"
    assert config.file_divider == "+++"
    assert config.content_divider == "@@@@@@"


def test_load_ignore_patterns_no_lmignore():
    test_dir = TEMP_PATH / "test_load_ignore_patterns_no_lmignore"
    ensure_clean_dir(test_dir)
    # No .lmignore => empty dict
    ignore_dict = load_ignore_patterns(test_dir)
    assert ignore_dict == {}


def test_load_ignore_patterns_one_file():
    test_dir = TEMP_PATH / "test_load_ignore_patterns_one_file"
    ensure_clean_dir(test_dir)
    (test_dir / ".lmignore").write_text(
        textwrap.dedent(
            """\
            # This is a comment
            *.md
            subdir1/
            """
        ),
        encoding="utf-8",
    )
    ignore_dict = load_ignore_patterns(test_dir)
    assert len(ignore_dict) == 1
    assert test_dir in ignore_dict
    patterns = ignore_dict[test_dir]
    assert len(patterns) == 2
    assert "*.md" in patterns
    assert "subdir1/" in patterns


def test_is_ignored_basic():
    test_dir = TEMP_PATH / "test_is_ignored_basic"
    ensure_clean_dir(test_dir)
    # Populate some sample structure
    (test_dir / "subdir1").mkdir()
    (test_dir / "subdir2").mkdir()
    (test_dir / "subdir1" / "file1.txt").write_text("content1", encoding="utf-8")
    (test_dir / "subdir2" / "file2.md").write_text("content2", encoding="utf-8")
    (test_dir / "file3.log").write_text("log content", encoding="utf-8")

    (test_dir / ".lmignore").write_text(
        textwrap.dedent(
            """\
            *.log
            subdir2/
            """
        ),
        encoding="utf-8",
    )

    ignore_dict = load_ignore_patterns(test_dir)

    # file3.log => should be ignored
    path_log = test_dir / "file3.log"
    assert is_ignored(path_log, test_dir, ignore_dict) is True

    # subdir2 => ignored
    path_subdir2 = test_dir / "subdir2"
    assert is_ignored(path_subdir2, test_dir, ignore_dict) is True
    # file2 within subdir2 => also ignored
    path_file2 = test_dir / "subdir2" / "file2.md"
    assert is_ignored(path_file2, test_dir, ignore_dict) is True

    # subdir1 => not ignored
    path_subdir1 = test_dir / "subdir1"
    assert is_ignored(path_subdir1, test_dir, ignore_dict) is False
    # file1 => not ignored
    path_file1 = test_dir / "subdir1" / "file1.txt"
    assert is_ignored(path_file1, test_dir, ignore_dict) is False


def test_is_ignored_child_dir_has_own_lmignore():
    test_dir = TEMP_PATH / "test_is_ignored_child_dir_has_own_lmignore"
    ensure_clean_dir(test_dir)
    (test_dir / "subdir1").mkdir()
    (test_dir / "subdir2").mkdir()
    (test_dir / "subdir2" / "file2.md").write_text("content2", encoding="utf-8")
    (test_dir / "subdir1" / "file1.txt").write_text("content1", encoding="utf-8")

    # Root .lmignore => ignore *.md
    (test_dir / ".lmignore").write_text("*.md\n", encoding="utf-8")
    # subdir1 => has own .lmignore ignoring file1*
    (test_dir / "subdir1" / ".lmignore").write_text("file1*\n", encoding="utf-8")

    ignore_dict = load_ignore_patterns(test_dir)

    # subdir2 => file2.md => should be ignored by root-lmignore pattern
    path_file2 = test_dir / "subdir2" / "file2.md"
    assert is_ignored(path_file2, test_dir, ignore_dict) is True

    # subdir1 => file1.txt => ignored by subdir1-lmignore
    path_file1 = test_dir / "subdir1" / "file1.txt"
    assert is_ignored(path_file1, test_dir, ignore_dict) is True

    # subdir1 => itself is not ignored
    path_subdir1 = test_dir / "subdir1"
    assert is_ignored(path_subdir1, test_dir, ignore_dict) is False


def test_is_ignored_stop_at_root():
    test_dir = TEMP_PATH / "test_is_ignored_stop_at_root"
    ensure_clean_dir(test_dir)
    (test_dir / ".lmignore").write_text("*.log\n", encoding="utf-8")

    outside_path = test_dir.parent  # e.g. tests/_temp
    ignore_dict = load_ignore_patterns(test_dir)
    # Should not ignore outside_path, since we never climb above test_dir
    assert is_ignored(outside_path, test_dir, ignore_dict) is False


def test_walk_dir_no_lmignore():
    test_dir = TEMP_PATH / "test_walk_dir_no_lmignore"
    ensure_clean_dir(test_dir)
    (test_dir / "subdir1").mkdir()
    (test_dir / "subdir2").mkdir()
    (test_dir / "subdir1" / "file1.txt").write_text("content1", encoding="utf-8")
    (test_dir / "subdir2" / "file2.md").write_text("content2", encoding="utf-8")
    (test_dir / "file3.log").write_text("log content", encoding="utf-8")

    config = LMCatConfig()
    ignore_dict = {}
    tree_out, files = walk_dir(test_dir, test_dir, ignore_dict, config)

    joined_out = "\n".join(tree_out)
    # subdir1, subdir2, file3.log should appear
    assert "subdir1" in joined_out
    assert "subdir2" in joined_out
    assert "file3.log" in joined_out

    assert len(files) == 3
    assert set(files) == {
        test_dir / "subdir1" / "file1.txt",
        test_dir / "subdir2" / "file2.md",
        test_dir / "file3.log",
    }


def test_walk_dir_with_ignore():
    test_dir = TEMP_PATH / "test_walk_dir_with_ignore"
    ensure_clean_dir(test_dir)
    (test_dir / "subdir1").mkdir()
    (test_dir / "subdir2").mkdir()
    (test_dir / "subdir1" / "file1.txt").write_text("content1", encoding="utf-8")
    (test_dir / "subdir2" / "file2.md").write_text("content2", encoding="utf-8")
    (test_dir / "file3.log").write_text("log content", encoding="utf-8")

    (test_dir / ".lmignore").write_text(
        textwrap.dedent(
            """\
            subdir2/
            *.log
            """
        ),
        encoding="utf-8",
    )
    ignore_dict = load_ignore_patterns(test_dir)
    config = LMCatConfig()
    tree_out, files = walk_dir(test_dir, test_dir, ignore_dict, config)

    joined_out = "\n".join(tree_out)
    # Expect subdir2 + file3.log to be hidden
    assert "subdir2" not in joined_out
    assert "file3.log" not in joined_out
    assert "subdir1" in joined_out
    assert "file1.txt" in joined_out

    assert len(files) == 1
    assert files[0] == test_dir / "subdir1" / "file1.txt"


def test_walk_and_collect():
    test_dir = TEMP_PATH / "test_walk_and_collect"
    ensure_clean_dir(test_dir)
    (test_dir / "subdir1").mkdir()
    (test_dir / "subdir2").mkdir()
    (test_dir / "subdir1" / "file1.txt").write_text("content1", encoding="utf-8")
    (test_dir / "subdir2" / "file2.md").write_text("content2", encoding="utf-8")
    (test_dir / "file3.log").write_text("log content", encoding="utf-8")

    config = LMCatConfig()
    tree_out, files = walk_and_collect(test_dir, config)

    assert tree_out[0] == test_dir.name
    joined_out = "\n".join(tree_out)
    assert "subdir1" in joined_out
    assert "subdir2" in joined_out
    assert "file1.txt" in joined_out
    assert "file2.md" in joined_out
    assert "file3.log" in joined_out
    assert len(files) == 3


def test_walk_and_collect_with_ignore():
    test_dir = TEMP_PATH / "test_walk_and_collect_with_ignore"
    ensure_clean_dir(test_dir)
    (test_dir / "subdir1").mkdir()
    (test_dir / "subdir2").mkdir()
    (test_dir / "subdir1" / "file1.txt").write_text("content1", encoding="utf-8")
    (test_dir / "subdir2" / "file2.md").write_text("content2", encoding="utf-8")
    (test_dir / "file3.log").write_text("log content", encoding="utf-8")

    (test_dir / ".lmignore").write_text("subdir1/\n*.md\n", encoding="utf-8")
    config = LMCatConfig()
    tree_out, files = walk_and_collect(test_dir, config)

    assert tree_out[0] == test_dir.name
    joined_out = "\n".join(tree_out)
    # subdir1 => hidden
    assert "subdir1" not in joined_out
    assert "file2.md" not in joined_out
    assert "file3.log" in joined_out
    assert len(files) == 1
    assert files[0].name == "file3.log"


def test_main_no_config_no_ignore(capsys):
    test_dir = TEMP_PATH / "test_main_no_config_no_ignore"
    ensure_clean_dir(test_dir)
    (test_dir / "fileA.txt").write_text("AAA", encoding="utf-8")
    (test_dir / "fileB.txt").write_text("BBB", encoding="utf-8")

    original_cwd = os.getcwd()
    try:
        os.chdir(test_dir)
        main()
        captured = capsys.readouterr()
        out = captured.out
        # The first line is the root dir name
        assert test_dir.name in out
        # Then lines for fileA.txt, fileB.txt
        assert "fileA.txt" in out
        assert "fileB.txt" in out
        # Then the content divider and contents
        assert "``````" in out
        assert "AAA" in out
        assert "BBB" in out
    finally:
        os.chdir(original_cwd)


def test_main_with_ignore_and_toml_config(capsys):
    test_dir = TEMP_PATH / "test_main_with_ignore_and_toml_config"
    ensure_clean_dir(test_dir)
    (test_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [tool.lmcat]
            tree_divider = ">>>"
            indent = "---"
            file_divider = "+++"
            content_divider = "@@@@"
            """
        ),
        encoding="utf-8",
    )
    (test_dir / "fileA.log").write_text("AAA", encoding="utf-8")
    (test_dir / "fileB.txt").write_text("BBB", encoding="utf-8")
    (test_dir / ".lmignore").write_text("*.log\n", encoding="utf-8")

    original_cwd = os.getcwd()
    try:
        os.chdir(test_dir)
        main()
        captured = capsys.readouterr()
        out = captured.out
        # The first line is the root dir name
        assert test_dir.name in out
        # We used "+++" as a file divider, ">>>" as a tree divider
        assert "+++" in out
        assert ">>>" in out
        # fileA.log => ignored
        assert "fileA.log" not in out
        # fileB.txt => included
        assert "fileB.txt" in out
        # content divider => "@@@@"
        assert "@@@@" in out
        # content => "BBB"
        assert "BBB" in out
    finally:
        os.chdir(original_cwd)


def test_main_with_nested_lmignore(capsys):
    test_dir = TEMP_PATH / "test_main_with_nested_lmignore"
    ensure_clean_dir(test_dir)
    subdir1 = test_dir / "subdir1"
    subdir2 = test_dir / "subdir2"
    subdir1.mkdir()
    subdir2.mkdir()

    # Root ignore: ignore subdir2 entirely
    (test_dir / ".lmignore").write_text("subdir2\n", encoding="utf-8")
    # subdir1: ignore anything matching "secret*"
    (subdir1 / ".lmignore").write_text("secret*\n", encoding="utf-8")

    (subdir1 / "normal.txt").write_text("normal content", encoding="utf-8")
    (subdir1 / "secret.txt").write_text("secret content", encoding="utf-8")
    (subdir2 / "whatever.dat").write_text("some data", encoding="utf-8")

    original_cwd = os.getcwd()
    try:
        os.chdir(test_dir)
        main()
        captured = capsys.readouterr()
        out = captured.out
        # Root name
        assert test_dir.name in out
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


def test_gitignore_only_included():
    test_dir = TEMP_PATH / "test_gitignore_only_included"
    ensure_clean_dir(test_dir)
    # subdirA => keep.txt, remove.txt
    subdirA = test_dir / "subdirA"
    subdirA.mkdir(parents=True, exist_ok=True)
    (subdirA / "keep.txt").write_text("KEEP A", encoding="utf-8")
    (subdirA / "remove.txt").write_text("REMOVE A", encoding="utf-8")

    (test_dir / ".gitignore").write_text("*.txt\n", encoding="utf-8")

    config = LMCatConfig(include_gitignore=True)
    ignore_dict = load_ignore_patterns(test_dir, config)

    path_keep = subdirA / "keep.txt"
    # According to the test docstring, we want keep.txt to be ignored by .gitignore
    assert is_ignored(path_keep, test_dir, ignore_dict) is True

    path_remove = test_dir / "subdirA" / "remove.txt"
    # no .lmignore => we haven't un-ignored remove.txt,
    # so it still matches *.txt => also ignored
    assert is_ignored(path_remove, test_dir, ignore_dict) is True


def test_gitignore_lmignore_negation():
    test_dir = TEMP_PATH / "test_gitignore_lmignore_negation"
    ensure_clean_dir(test_dir)
    subdirA = test_dir / "subdirA"
    subdirA.mkdir(parents=True, exist_ok=True)
    (subdirA / "keep.txt").write_text("KEEP A", encoding="utf-8")
    (subdirA / "remove.txt").write_text("REMOVE A", encoding="utf-8")

    (test_dir / ".gitignore").write_text("*.txt\n", encoding="utf-8")
    (test_dir / ".lmignore").write_text("!remove.txt\n", encoding="utf-8")

    config = LMCatConfig(include_gitignore=True)
    ignore_dict = load_ignore_patterns(test_dir, config)

    path_keep = subdirA / "keep.txt"
    # matched by .gitignore => *.txt => ignored
    assert is_ignored(path_keep, test_dir, ignore_dict) is True

    path_remove = subdirA / "remove.txt"
    # .gitignore => ignore *.txt, but .lmignore => "!remove.txt"
    # => final match => unignored
    assert is_ignored(path_remove, test_dir, ignore_dict) is False


def test_gitignore_lmignore_directory():
    test_dir = TEMP_PATH / "test_gitignore_lmignore_directory"
    ensure_clean_dir(test_dir)
    subdirB = test_dir / "subdirB"
    subdirB.mkdir(parents=True, exist_ok=True)
    (subdirB / "keep.dat").write_text("KEEP B", encoding="utf-8")
    (subdirB / "remove.dat").write_text("REMOVE B", encoding="utf-8")

    (test_dir / ".gitignore").write_text("subdirB/\n", encoding="utf-8")
    (test_dir / ".lmignore").write_text("!remove.dat\n", encoding="utf-8")

    config = LMCatConfig()
    ignore_dict = load_ignore_patterns(test_dir, config)

    # subdirB => initially ignored by .gitignore
    assert is_ignored(subdirB, test_dir, ignore_dict) is True

    # remove.dat => final rule is "!remove.dat", so unignored
    path_remove = subdirB / "remove.dat"
    assert is_ignored(path_remove, test_dir, ignore_dict) is False


def test_cli_override_no_gitignore(capsys):
    test_dir = TEMP_PATH / "test_cli_override_no_gitignore"
    ensure_clean_dir(test_dir)
    (test_dir / ".gitignore").write_text("*.txt\n", encoding="utf-8")

    subdirA = test_dir / "subdirA"
    subdirA.mkdir(parents=True, exist_ok=True)
    (subdirA / "keep.txt").write_text("KEEP A", encoding="utf-8")

    # Minimal script runner
    script_path = test_dir / "runner.py"
    script_path.write_text(
        textwrap.dedent(
            """\
            import sys
            from pathlib import Path
            from lmcat.lmcat import main

            if __name__ == "__main__":
                sys.exit(main())
            """
        ),
        encoding="utf-8",
    )

    original_cwd = os.getcwd()
    try:
        os.chdir(test_dir)
        proc = subprocess.run(
            ["python", "runner.py", "--no-include-gitignore"],
            capture_output=True,
            text=True,
        )
        out = proc.stdout
        print(out)  # debugging
        # We expect subdirA in the tree output, because .gitignore is disabled
        assert "subdirA" in out
        assert "keep.txt" in out
        assert proc.returncode == 0
    finally:
        os.chdir(original_cwd)


def test_cli_override_suppress_contents():
    test_dir = TEMP_PATH / "test_cli_override_suppress_contents"
    ensure_clean_dir(test_dir)
    (test_dir / ".gitignore").write_text("", encoding="utf-8")

    subdirA = test_dir / "subdirA"
    subdirA.mkdir(parents=True, exist_ok=True)
    (subdirA / "keep.txt").write_text("KEEP A", encoding="utf-8")
    (subdirA / "remove.txt").write_text("REMOVE A", encoding="utf-8")

    script_path = test_dir / "runner.py"
    script_path.write_text(
        textwrap.dedent(
            """\
            import sys
            from pathlib import Path
            from lmcat.lmcat import main

            if __name__ == "__main__":
                sys.exit(main())
            """
        ),
        encoding="utf-8",
    )

    original_cwd = os.getcwd()
    try:
        os.chdir(test_dir)
        proc = subprocess.run(
            ["python", "runner.py", "--suppress-contents"],
            capture_output=True,
            text=True,
        )
        out = proc.stdout
        print(out)  # debugging
        # We see subdirA/keep.txt in the tree
        assert "subdirA" in out
        assert "keep.txt" in out
        # But no file content
        assert "KEEP A" not in out
        assert "REMOVE A" not in out
        assert proc.returncode == 0
    finally:
        os.chdir(original_cwd)
