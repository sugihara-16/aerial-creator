from __future__ import annotations

from pathlib import Path

from amsrr.morphology.grasp_carry_designs import GRASP_CARRY_VARIANT_ORDER
from amsrr.visualization.p2_morphology import render_p2_morphology_visualizations


def test_p2_morphology_visualization_outputs_graph_and_layout_svgs(tmp_path: Path) -> None:
    output_dir = tmp_path / "visualization"

    manifest = render_p2_morphology_visualizations(output_dir=output_dir, seed=0, sample_index=0)

    assert set(manifest.graph_files) == {variant.value for variant in GRASP_CARRY_VARIANT_ORDER}
    assert set(manifest.layout_files) == {variant.value for variant in GRASP_CARRY_VARIANT_ORDER}
    for variant in GRASP_CARRY_VARIANT_ORDER:
        graph_path = Path(manifest.graph_files[variant.value])
        layout_path = Path(manifest.layout_files[variant.value])
        assert graph_path.exists()
        assert layout_path.exists()
        graph_text = graph_path.read_text(encoding="utf-8")
        layout_text = layout_path.read_text(encoding="utf-8")
        assert "<svg" in graph_text
        assert "<svg" in layout_text
        assert variant.value in graph_text
        assert "RobotAnchors" in graph_text
        assert "Slot-anchor binding prior" in graph_text
        assert "Control groups" in graph_text
        assert "base module" in layout_text
        assert "Dock edges" in layout_text

