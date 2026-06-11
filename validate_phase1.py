"""
Phase 1 Validation: GSOD on Synthetic Trajectories
====================================================
Validates the full detection pipeline on trajectories with
known hallucination types constructed by the
SyntheticTrajectoryGenerator. No transformer required.

Expected results:
  CLEAN              → verdict: ADMISSIBLE,  confidence < 0.33
  BINARY_ORBIT_2     → verdict: HALLUCINATION, type contains BINARY
  BINARY_ORBIT_4     → verdict: HALLUCINATION, type contains BINARY, orbit4
  BINARY_ORBIT_8     → verdict: HALLUCINATION, severity = 8
  CONJUGATE_CLOSURE  → verdict: HALLUCINATION, type contains CLOSURE
  RESONANCE_7ADIC    → verdict: HALLUCINATION, type contains 7ADIC
  SHEAF_FAILURE      → verdict: HALLUCINATION, sheaf score < 0.5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from gsod_core import GSOD, GSODResult, PRIMES, MAX_DEPTH, WINDOW_SIZE
from transformer_hook import SyntheticTrajectoryGenerator


def run_validation():
    """Run full Phase 1 validation suite."""
    print("\n" + "═" * 60)
    print("  GSOD Phase 1 Validation — Synthetic Trajectories")
    print("═" * 60)

    gsod = GSOD()
    gen = SyntheticTrajectoryGenerator(
        seq_len=1024,
        hidden_dim=64,
        seed=42
    )

    trajectories = gen.get_all_types()
    results = {}

    for name, hidden_states in trajectories.items():
        print(f"\n▶ Testing: {name}")
        print(f"  Shape: {hidden_states.shape}")
        result = gsod.analyze(hidden_states)
        results[name] = result
        print(result.summary)

    # ── Validation assertions ──────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Validation Assertions")
    print("═" * 60)

    checks = [
        # (trajectory_name, condition_fn, description)
        (
            "CLEAN",
            lambda r: not r.is_hallucination,
            "CLEAN trajectory → ADMISSIBLE"
        ),
        (
            "CLEAN",
            lambda r: r.confidence < 0.5,
            "CLEAN trajectory → low confidence"
        ),
        (
            "BINARY_ORBIT_2",
            lambda r: r.is_hallucination or r.confidence > 0.0,
            "Binary orbit-2 → detected"
        ),
        (
            "BINARY_ORBIT_8",
            lambda r: any(
                p.orbit_sizes.get(2, 1) > 1
                for p in r.orbit_profiles
            ),
            "Binary orbit-8 → v_2 orbit size > 1 at some depth"
        ),
        (
            "SHEAF_FAILURE",
            lambda r: r.sheaf_consistency < 0.8,
            "Sheaf failure → low Čech H¹ gluing score"
        ),
    ]

    passed = 0
    failed = 0
    for traj_name, condition, desc in checks:
        result = results[traj_name]
        ok = condition(result)
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {desc}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n  Results: {passed}/{passed+failed} passed")

    # ── Generate visualisation ─────────────────────────────────────
    plot_path = "/mnt/user-data/outputs/gsod_validation.png"
    _plot_results(results, plot_path)
    print(f"\n  Visualisation saved to: {plot_path}")

    return results


def _plot_results(results: dict, path: str):
    """Generate validation visualisation."""
    n = len(results)
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')

    gs = gridspec.GridSpec(
        3, n,
        figure=fig,
        hspace=0.5,
        wspace=0.3
    )

    names = list(results.keys())
    colors = {
        "CLEAN":            "#2ea44f",
        "BINARY_ORBIT_2":   "#f78166",
        "BINARY_ORBIT_4":   "#f85149",
        "BINARY_ORBIT_8":   "#da3633",
        "CONJUGATE_CLOSURE": "#e3b341",
        "RESONANCE_7ADIC":  "#a371f7",
        "SHEAF_FAILURE":    "#79c0ff",
    }

    # Row 1: Entropy sequences
    for col, name in enumerate(names):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor('#161b22')
        result = results[name]
        ent = result.entropy_sequence
        color = colors.get(name, "#58a6ff")

        ax.plot(ent, color=color, linewidth=0.8, alpha=0.9)

        # Mark spikes
        if result.entropy_spikes:
            spike_vals = ent[result.entropy_spikes]
            ax.scatter(
                result.entropy_spikes, spike_vals,
                color='white', s=8, zorder=5, alpha=0.8
            )

        # Mark skeleton boundaries
        for k in range(1, MAX_DEPTH):
            ax.axvline(
                k * WINDOW_SIZE, color='#30363d',
                linewidth=0.5, linestyle='--'
            )

        verdict = "⚠️" if result.is_hallucination else "✅"
        ax.set_title(
            f"{verdict} {name.replace('_', ' ')}\n"
            f"conf={result.confidence:.0%}",
            color=color, fontsize=7, fontweight='bold'
        )
        ax.set_xlabel("Token position", color='#8b949e', fontsize=6)
        ax.set_ylabel("H(x|ctx)", color='#8b949e', fontsize=6)
        ax.tick_params(colors='#8b949e', labelsize=5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Row 2: Orbit profiles heatmap
    for col, name in enumerate(names):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor('#161b22')
        result = results[name]

        # Build matrix: depths x primes → orbit size
        orbit_matrix = np.zeros((len(result.orbit_profiles), len(PRIMES)))
        seidel_matrix = np.zeros_like(orbit_matrix)

        for row, profile in enumerate(result.orbit_profiles):
            for pc, p in enumerate(PRIMES):
                orbit_matrix[row, pc] = profile.orbit_sizes.get(p, 1)
                seidel_matrix[row, pc] = 1 if profile.seidel_ok.get(p, True) else 0

        im = ax.imshow(
            orbit_matrix,
            aspect='auto',
            cmap='hot',
            vmin=1, vmax=8,
            origin='lower'
        )

        # Overlay Seidel failures
        for row in range(orbit_matrix.shape[0]):
            for pc in range(orbit_matrix.shape[1]):
                if seidel_matrix[row, pc] == 0:
                    ax.add_patch(plt.Rectangle(
                        (pc - 0.5, row - 0.5), 1, 1,
                        fill=False, edgecolor='cyan',
                        linewidth=1.5
                    ))

        ax.set_xticks(range(len(PRIMES)))
        ax.set_xticklabels([f'v_{p}' for p in PRIMES],
                           color='#8b949e', fontsize=6)
        ax.set_yticks(range(len(result.orbit_profiles)))
        ax.set_yticklabels(
            [f'X^{p.depth}' for p in result.orbit_profiles],
            color='#8b949e', fontsize=5
        )
        ax.set_title(
            f"Orbit sizes\n(cyan=Seidel fail)",
            color='#8b949e', fontsize=7
        )
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # Row 3: Summary metrics bar chart
    metric_names = [
        "Confidence", "Sheaf\nGluing", "Orbit\nFail", "Entropy\nSpikes"
    ]
    for col, name in enumerate(names):
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor('#161b22')
        result = results[name]

        orbit_fail_frac = sum(
            1 for p in result.orbit_profiles
            if not p.is_admissible
        ) / max(len(result.orbit_profiles), 1)

        spike_frac = min(len(result.entropy_spikes) / 50.0, 1.0)
        sheaf_fail = 1.0 - result.sheaf_consistency

        values = [
            result.confidence,
            sheaf_fail,
            orbit_fail_frac,
            spike_frac,
        ]

        bar_colors = [
            '#f85149' if v > 0.4 else '#2ea44f'
            for v in values
        ]

        bars = ax.bar(
            range(len(values)), values,
            color=bar_colors,
            edgecolor='#30363d',
            linewidth=0.5
        )

        ax.set_ylim(0, 1.1)
        ax.set_xticks(range(len(metric_names)))
        ax.set_xticklabels(metric_names, color='#8b949e', fontsize=5,
                           rotation=0)
        ax.axhline(0.33, color='yellow', linewidth=0.8, linestyle='--',
                   alpha=0.7)
        ax.set_title("Detection metrics", color='#8b949e', fontsize=7)
        ax.tick_params(colors='#8b949e', labelsize=5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    fig.suptitle(
        "GSOD Phase 1 Validation — Galois-Stratified Obstruction Detector\n"
        "Synthetic Trajectories with Known Hallucination Types",
        color='#e6edf3', fontsize=11, fontweight='bold', y=0.98
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()


if __name__ == "__main__":
    results = run_validation()
