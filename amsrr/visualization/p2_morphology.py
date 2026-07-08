from __future__ import annotations

import argparse
import html
import math
from dataclasses import dataclass
from pathlib import Path

from amsrr.morphology.grasp_carry_designs import GRASP_CARRY_VARIANT_ORDER
from amsrr.policies.design_policy_p2 import P2DesignCandidateEvaluation
from amsrr.schemas.morphology import DesignOutput, MorphologyGraph
from amsrr.training.p2_inspection_context import build_p2_inspection_context


@dataclass(frozen=True)
class P2MorphologyVisualizationManifest:
    output_dir: str
    graph_files: dict[str, str]
    layout_files: dict[str, str]


def render_p2_morphology_visualizations(
    *,
    config_path: str | Path = "configs/training/p2_design_grasp_carry.yaml",
    output_dir: str | Path = "outputs/p2_5/visualization",
    seed: int = 0,
    sample_index: int = 0,
) -> P2MorphologyVisualizationManifest:
    context = build_p2_inspection_context(
        config_path=config_path,
        seed=seed,
        sample_index=sample_index,
    )
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    by_variant = {candidate.variant: candidate for candidate in context.selection.candidates}
    graph_files: dict[str, str] = {}
    layout_files: dict[str, str] = {}
    for variant in GRASP_CARRY_VARIANT_ORDER:
        candidate = by_variant[variant.value]
        graph_path = target_dir / f"{variant.value}_graph.svg"
        layout_path = target_dir / f"{variant.value}_layout.svg"
        graph_path.write_text(_graph_svg(candidate), encoding="utf-8")
        layout_path.write_text(_layout_svg(candidate), encoding="utf-8")
        graph_files[variant.value] = str(graph_path)
        layout_files[variant.value] = str(layout_path)
    return P2MorphologyVisualizationManifest(
        output_dir=str(target_dir),
        graph_files=graph_files,
        layout_files=layout_files,
    )


def _graph_svg(candidate: P2DesignCandidateEvaluation) -> str:
    design = candidate.design_output
    morphology = design.target_morphology
    positions = _graph_positions(morphology)
    width = 980
    height = 700
    lines = _svg_header(width, height)
    lines.append(_title_block(candidate, x=24, y=34))
    for edge in morphology.dock_edges:
        x1, y1 = positions[edge.src_module_id]
        x2, y2 = positions[edge.dst_module_id]
        lines.append(
            _line(
                x1,
                y1,
                x2,
                y2,
                stroke="#60738a",
                width=3,
                extra=f"<title>DockEdge {edge.edge_id}: {edge.src_module_id} -> {edge.dst_module_id}, {edge.edge_role}</title>",
            )
        )
        lines.append(_text((x1 + x2) / 2.0, (y1 + y2) / 2.0 - 8, f"e{edge.edge_id}:{edge.edge_role}", size=11))
    for module in sorted(morphology.modules, key=lambda item: item.module_id):
        x, y = positions[module.module_id]
        fill = "#ffe08a" if module.is_base else "#d9edf7"
        stroke = "#9a7412" if module.is_base else "#3a6f8f"
        lines.append(_circle(x, y, 34, fill=fill, stroke=stroke, width=3))
        lines.append(_text(x, y - 5, f"M{module.module_id}", size=14, weight="700", anchor="middle"))
        lines.append(_text(x, y + 13, module.role_id, size=10, anchor="middle"))
    _append_anchor_annotations(lines, morphology, positions, start_x=650, start_y=120)
    _append_binding_annotations(lines, design, start_x=650, start_y=330)
    _append_control_group_annotations(lines, morphology, start_x=650, start_y=500)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _layout_svg(candidate: P2DesignCandidateEvaluation) -> str:
    design = candidate.design_output
    morphology = design.target_morphology
    positions = _layout_positions(morphology)
    width = 980
    height = 700
    lines = _svg_header(width, height)
    lines.append(_title_block(candidate, x=24, y=34, subtitle="simple 3D layout projection: x -> horizontal, y/z -> vertical/isometric"))
    lines.append(_line(80, 570, 460, 570, stroke="#909090", width=1))
    lines.append(_line(80, 570, 120, 520, stroke="#909090", width=1))
    lines.append(_text(466, 574, "+x", size=11))
    lines.append(_text(122, 516, "+y/+z", size=11))
    for edge in morphology.dock_edges:
        x1, y1 = positions[edge.src_module_id]
        x2, y2 = positions[edge.dst_module_id]
        lines.append(_line(x1, y1, x2, y2, stroke="#526a7c", width=3))
    for module in sorted(morphology.modules, key=lambda item: item.module_id):
        x, y = positions[module.module_id]
        pose = module.pose_in_design_frame
        fill = "#ffe08a" if module.is_base else "#c8e6c9"
        lines.append(_rect(x - 30, y - 22, 60, 44, fill=fill, stroke="#345", stroke_width=2, rx=4))
        lines.append(_text(x, y - 3, f"M{module.module_id}", size=13, weight="700", anchor="middle"))
        lines.append(_text(x, y + 14, f"({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f})", size=9, anchor="middle"))
    for anchor in morphology.robot_anchors:
        module_x, module_y = positions[anchor.module_id]
        ax = module_x + 42
        ay = module_y - 25 + 13 * (anchor.anchor_id % 4)
        lines.append(_circle(ax, ay, 8, fill="#f6b26b", stroke="#9c5700", width=1))
        lines.append(_text(ax + 12, ay + 4, f"A{anchor.anchor_id}: slots {','.join(str(s) for s in anchor.associated_contact_slot_ids)}", size=10))
    _append_layout_table(lines, morphology, start_x=600, start_y=120)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _title_block(
    candidate: P2DesignCandidateEvaluation,
    *,
    x: float,
    y: float,
    subtitle: str | None = None,
) -> str:
    status = "selected" if candidate.design_output.design_scores.get("p2_design_policy_selected") == 1.0 else "candidate"
    accepted = "accepted" if candidate.accepted else "rejected"
    reason = "" if candidate.rejection_reason is None else f", reason={candidate.rejection_reason}"
    text = [
        _text(x, y, f"{candidate.variant} - {status}, {accepted}, score={candidate.soft_score:.3f}{reason}", size=18, weight="700"),
    ]
    if subtitle:
        text.append(_text(x, y + 22, subtitle, size=12))
    return "\n".join(text)


def _graph_positions(morphology: MorphologyGraph) -> dict[int, tuple[float, float]]:
    module_ids = [module.module_id for module in sorted(morphology.modules, key=lambda item: item.module_id)]
    count = max(1, len(module_ids))
    center_x = 330
    center_y = 330
    radius = 165
    positions: dict[int, tuple[float, float]] = {}
    for index, module_id in enumerate(module_ids):
        if count == 1:
            positions[module_id] = (center_x, center_y)
            continue
        angle = -math.pi / 2.0 + (2.0 * math.pi * float(index) / float(count))
        positions[module_id] = (center_x + radius * math.cos(angle), center_y + radius * math.sin(angle))
    return positions


def _layout_positions(morphology: MorphologyGraph) -> dict[int, tuple[float, float]]:
    modules = sorted(morphology.modules, key=lambda item: item.module_id)
    xs = [module.pose_in_design_frame[0] for module in modules]
    ys = [module.pose_in_design_frame[1] for module in modules]
    zs = [module.pose_in_design_frame[2] for module in modules]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    x_span = max(0.1, max_x - min_x)
    yz_values = [ys[idx] + 0.45 * zs[idx] for idx in range(len(modules))]
    min_yz = min(min_y, min(yz_values), min_z)
    max_yz = max(max_y, max(yz_values), max_z)
    yz_span = max(0.1, max_yz - min_yz)
    positions: dict[int, tuple[float, float]] = {}
    for module in modules:
        pose = module.pose_in_design_frame
        x = 110 + ((pose[0] - min_x) / x_span) * 380
        y_projected = pose[1] + 0.45 * pose[2]
        y = 500 - ((y_projected - min_yz) / yz_span) * 300
        positions[module.module_id] = (x, y)
    return positions


def _append_anchor_annotations(
    lines: list[str],
    morphology: MorphologyGraph,
    positions: dict[int, tuple[float, float]],
    *,
    start_x: float,
    start_y: float,
) -> None:
    lines.append(_text(start_x, start_y, "RobotAnchors", size=14, weight="700"))
    y = start_y + 22
    for anchor in sorted(morphology.robot_anchors, key=lambda item: item.anchor_id):
        module_x, module_y = positions[anchor.module_id]
        ax = module_x + 42
        ay = module_y + 18 * ((anchor.anchor_id % 3) - 1)
        lines.append(_line(module_x, module_y, ax, ay, stroke="#cc7a00", width=1))
        lines.append(_circle(ax, ay, 8, fill="#f6b26b", stroke="#9c5700", width=1))
        slot_label = ",".join(str(slot_id) for slot_id in anchor.associated_contact_slot_ids)
        lines.append(
            _text(
                start_x,
                y,
                f"A{anchor.anchor_id}: module M{anchor.module_id}, type={anchor.anchor_type}, slots=[{slot_label}]",
                size=11,
            )
        )
        y += 17


def _append_binding_annotations(lines: list[str], design: DesignOutput, *, start_x: float, start_y: float) -> None:
    lines.append(_text(start_x, start_y, "Slot-anchor binding prior", size=14, weight="700"))
    y = start_y + 22
    if not design.slot_anchor_binding_prior:
        lines.append(_text(start_x, y, "none", size=11))
        return
    for prior in sorted(design.slot_anchor_binding_prior, key=lambda item: (item.slot_id, item.anchor_id)):
        lines.append(
            _text(
                start_x,
                y,
                f"slot {prior.slot_id} -> anchor A{prior.anchor_id}, score={prior.score:.2f}, {prior.reason_code}",
                size=11,
            )
        )
        y += 17


def _append_control_group_annotations(
    lines: list[str],
    morphology: MorphologyGraph,
    *,
    start_x: float,
    start_y: float,
) -> None:
    lines.append(_text(start_x, start_y, "Control groups", size=14, weight="700"))
    y = start_y + 22
    for group in sorted(morphology.control_groups, key=lambda item: item.group_id):
        module_label = ",".join(str(module_id) for module_id in group.module_ids)
        lines.append(_text(start_x, y, f"{group.group_id}: role={group.role}, modules=[{module_label}]", size=11))
        y += 17


def _append_layout_table(lines: list[str], morphology: MorphologyGraph, *, start_x: float, start_y: float) -> None:
    lines.append(_text(start_x, start_y, "Modules and roles", size=14, weight="700"))
    y = start_y + 24
    lines.append(_text(start_x, y, f"base module: M{morphology.base_module_id}", size=11))
    y += 22
    for module in sorted(morphology.modules, key=lambda item: item.module_id):
        pose = module.pose_in_design_frame
        lines.append(
            _text(
                start_x,
                y,
                f"M{module.module_id}: {module.role_id}, pose=({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f})",
                size=11,
            )
        )
        y += 17
    y += 12
    lines.append(_text(start_x, y, "Dock edges", size=14, weight="700"))
    y += 22
    for edge in sorted(morphology.dock_edges, key=lambda item: item.edge_id):
        lines.append(_text(start_x, y, f"e{edge.edge_id}: M{edge.src_module_id} -> M{edge.dst_module_id}, {edge.edge_role}", size=11))
        y += 17


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        '<style>text { font-family: Arial, sans-serif; fill: #1f2933; }</style>',
    ]


def _line(x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: int, extra: str = "") -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round">{extra}</line>'
    )


def _circle(x: float, y: float, radius: float, *, fill: str, stroke: str, width: int) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def _rect(
    x: float,
    y: float,
    rect_width: float,
    rect_height: float,
    *,
    fill: str,
    stroke: str,
    stroke_width: int,
    rx: int = 0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{rect_width:.1f}" height="{rect_height:.1f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def _text(
    x: float,
    y: float,
    text: str,
    *,
    size: int,
    weight: str = "400",
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render P2 grasp/carry morphology variant visualizations.")
    parser.add_argument("--config", default="configs/training/p2_design_grasp_carry.yaml")
    parser.add_argument("--output-dir", default="outputs/p2_5/visualization")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = render_p2_morphology_visualizations(
        config_path=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
        sample_index=args.sample_index,
    )
    for variant, path in sorted(manifest.graph_files.items()):
        print(f"{variant} graph: {path}")
        print(f"{variant} layout: {manifest.layout_files[variant]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
