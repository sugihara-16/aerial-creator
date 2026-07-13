from __future__ import annotations

from amsrr.utils.hashing import hash_directory_manifest


def test_directory_manifest_binds_relative_paths_and_contents(tmp_path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    first = root / "a.usda"
    first.write_text("one", encoding="utf-8")
    nested = root / "payloads"
    nested.mkdir()
    second = nested / "physics.usda"
    second.write_text("two", encoding="utf-8")

    initial = hash_directory_manifest(root)
    second.write_text("changed", encoding="utf-8")
    assert hash_directory_manifest(root) != initial

    changed_content = hash_directory_manifest(root)
    second.rename(nested / "renamed.usda")
    assert hash_directory_manifest(root) != changed_content
