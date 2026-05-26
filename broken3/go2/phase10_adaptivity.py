"""Per-arch (φ, α) adaptivity comparison across V2 architectures.

For each scene × DR axis × arch, prints:
  - φ_mean and α_mean at the 3 sweep values
  - span = max-min across the sweep (the actual adaptation)
  - within-cell std (how much the policy varies within one DR cell)

The question: does the policy actually shift (φ, α) with the DR axis it
*should* be sensitive to? Big span = it's adapting. Tiny span = it
converged to a fixed setting regardless of state.

Spans printed in [0..1] units (raw) and as a % of bound width:
  φ bound width = 1.0 (so raw span == fraction)
  α bound width = 3.8 (so raw span / 3.8 = fraction)
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


PHI_W = 1.0
ALPHA_W = 3.8

ARCHS = ["V2Full", "V2NoPriv", "V2NoProprio", "V2RMAClassic", "V2History"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    args = ap.parse_args()

    # by_scene[scene][arch] = dict(cells: {(axis,val): cell_dict})
    by_scene = defaultdict(lambda: defaultdict(dict))
    for name in sorted(os.listdir(args.eval_dir)):
        if not (name.startswith("eval_V2") and name.endswith(".json")):
            continue
        d = json.load(open(os.path.join(args.eval_dir, name)))
        scene = d["scene"]
        arch = d.get("arch", "?")
        for c in d["cells"]:
            by_scene[scene][arch][(c["dr_axis"], c["dr_value"])] = c

    scene_order = ["E1Gap", "E2Slalom", "E3Wall", "E4Field"]
    for scene in scene_order:
        if scene not in by_scene:
            continue
        arch_data = by_scene[scene]
        # group cells by axis to print sweeps neatly
        all_cells = set()
        for arch in ARCHS:
            if arch in arch_data:
                all_cells |= set(arch_data[arch].keys())
        by_axis = defaultdict(list)
        for (ax, v) in sorted(all_cells, key=lambda k: (k[0], k[1])):
            by_axis[ax].append(v)
        for ax in by_axis:
            by_axis[ax] = sorted(set(by_axis[ax]))

        print("\n" + "=" * 110)
        print(f"  SCENE: {scene}")
        print("=" * 110)

        for axis, values in by_axis.items():
            print(f"\n  --- sweep axis: {axis}  values: {values} ---")
            # header
            hdr_phi   = "  " + f"{'arch':<14}" + "   " + "  ".join(f"φ@{v:<4.2f}" for v in values) + f"   {'φ span':>7} {'% bd':>5}"
            hdr_alpha = "  " + f"{'arch':<14}" + "   " + "  ".join(f"α@{v:<4.2f}" for v in values) + f"   {'α span':>7} {'% bd':>5}"
            print(hdr_phi)
            print("  " + "-" * (len(hdr_phi)-2))
            for arch in ARCHS:
                if arch not in arch_data:
                    continue
                phis = []
                for v in values:
                    c = arch_data[arch].get((axis, v))
                    phis.append(c["phi_mean"] if c else None)
                phi_strs = "  ".join(f"{p:+.3f}" if p is not None else "  ?   " for p in phis)
                valid_phis = [p for p in phis if p is not None]
                phi_span = (max(valid_phis) - min(valid_phis)) if len(valid_phis) >= 2 else 0.0
                print(f"  {arch:<14}   {phi_strs}   {phi_span:>7.3f} {100*phi_span/PHI_W:>4.1f}%")
            print()
            print(hdr_alpha)
            print("  " + "-" * (len(hdr_alpha)-2))
            for arch in ARCHS:
                if arch not in arch_data:
                    continue
                alphas = []
                for v in values:
                    c = arch_data[arch].get((axis, v))
                    alphas.append(c["alpha_mean"] if c else None)
                a_strs = "  ".join(f"{a:.3f}" if a is not None else "  ?  " for a in alphas)
                valid_a = [a for a in alphas if a is not None]
                a_span = (max(valid_a) - min(valid_a)) if len(valid_a) >= 2 else 0.0
                print(f"  {arch:<14}   {a_strs}   {a_span:>7.3f} {100*a_span/ALPHA_W:>4.1f}%")

    # ===== summary: average span across all scenes/axes per arch =====
    print("\n" + "=" * 110)
    print("  CROSS-SCENE SUMMARY: mean span across all (scene, axis) for each arch")
    print("  Higher = more adaptive. 0 = pegged at a single value regardless of DR.")
    print("=" * 110)
    arch_spans_phi = defaultdict(list)
    arch_spans_alpha = defaultdict(list)
    for scene, arch_data in by_scene.items():
        # axes per scene
        all_cells = set()
        for arch in ARCHS:
            if arch in arch_data:
                all_cells |= set(arch_data[arch].keys())
        by_axis = defaultdict(list)
        for (ax, v) in all_cells:
            by_axis[ax].append(v)
        for ax in by_axis:
            by_axis[ax] = sorted(set(by_axis[ax]))
        for axis, values in by_axis.items():
            for arch in ARCHS:
                if arch not in arch_data:
                    continue
                phis = [arch_data[arch].get((axis, v), {}).get("phi_mean") for v in values]
                alphas = [arch_data[arch].get((axis, v), {}).get("alpha_mean") for v in values]
                valid_p = [p for p in phis if p is not None]
                valid_a = [a for a in alphas if a is not None]
                if len(valid_p) >= 2:
                    arch_spans_phi[arch].append(max(valid_p) - min(valid_p))
                if len(valid_a) >= 2:
                    arch_spans_alpha[arch].append(max(valid_a) - min(valid_a))
    print(f"  {'arch':<14}   {'mean φ span':>12} {'% bd':>6}    {'mean α span':>12} {'% bd':>6}")
    print("  " + "-" * 60)
    for arch in ARCHS:
        if arch not in arch_spans_phi:
            continue
        mp = sum(arch_spans_phi[arch]) / len(arch_spans_phi[arch])
        ma = sum(arch_spans_alpha[arch]) / len(arch_spans_alpha[arch])
        print(f"  {arch:<14}   {mp:>12.3f} {100*mp/PHI_W:>5.1f}%    {ma:>12.3f} {100*ma/ALPHA_W:>5.1f}%")


if __name__ == "__main__":
    main()
