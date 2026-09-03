from pathlib import Path

from scripts.consolidate_wan22_artifacts import repair_references


def test_repair_references_rewrites_text_and_preserves_layout_provenance(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        '{"path":"/old/root/run/receipt.json", "media":"/old/root/run/video.mp4"}\n',
        encoding="utf-8",
    )
    binary = tmp_path / "video.mp4"
    binary.write_bytes(b"\x00/old/root/run/video.mp4")
    result = repair_references(
        tmp_path,
        [("/old/root", "/new/root")],
    )
    assert result["changed_file_count"] == 1
    assert "/new/root/run/receipt.json" in receipt.read_text(encoding="utf-8")
    assert binary.read_bytes() == b"\x00/old/root/run/video.mp4"
