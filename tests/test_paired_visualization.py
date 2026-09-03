from pathlib import Path

import numpy as np

from wmloop.execute.paired_visualization import create_paired_visualization


def test_paired_visualization_writes_inspection_artifacts(tmp_path: Path):
    import imageio.v2 as imageio

    generated = tmp_path / "generated.mp4"
    ground_truth = tmp_path / "ground_truth.mp4"
    frames = [np.full((8, 12, 3), index * 30, dtype=np.uint8) for index in range(3)]
    imageio.mimsave(str(generated), frames, fps=5, codec="libx264", macro_block_size=1)
    imageio.mimsave(str(ground_truth), list(reversed(frames)), fps=5, codec="libx264", macro_block_size=1)

    result = create_paired_visualization(
        generated_video=generated,
        ground_truth_video=ground_truth,
        output_root=tmp_path / "visualization",
        metadata={"seed": 4101, "training_steps": 512},
    )

    assert result["state"] == "ready"
    assert Path(result["artifacts"]["comparison_video"]).is_file()
    assert Path(result["artifacts"]["contact_sheet"]).is_file()
    assert result["inputs"]["frames"] == 3
