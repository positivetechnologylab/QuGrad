"""
CMA-ES Classical Optimization Scaling Benchmark
================================================

Isolates the classical optimization cost of QuGrad (CMA-ES) from quantum
circuit simulation.  For n qubits the output state space is 2^n, which is
intractable to simulate classically beyond ~30 qubits.  However, the
classical optimization only needs:

  1. A parameter vector of dimension d = depth × (3n − 1)   [Sixteen ansatz]
  2. A scalar cost function (TVD between 20-bin CE histograms)

This script replaces circuit execution with a lightweight synthetic model
that maps parameters → NZP values, preserving exactly the same histogram-
binning and TVD computation used in real training (dists.py / ML.py).

Output: table of (n_qubits, 2^n, d, mean_time, std_time) ready for the paper.
"""

import numpy as np
import time
import cma
import sys
import warnings
warnings.filterwarnings("ignore")


# ── TVD computation (identical to dists.py) ─────────────────────────────────

def get_bins(samples, n_bins, lo, hi):
    """Bin samples into histogram counts — mirrors classes.getBinsList."""
    delta = (hi - lo) / n_bins
    bins = []
    for i in range(n_bins):
        lower = lo + i * delta
        upper = lower + delta
        count = int(np.sum((samples >= lower) & (samples <= upper)))
        bins.append(count)
    return bins


def tvd2(boxes1, boxes2):
    """Normalised TVD — identical to dists.TVD2."""
    b1 = np.array(boxes1, dtype=float)
    b2 = np.array(boxes2, dtype=float)
    s1, s2 = b1.sum(), b2.sum()
    if s1 == 0 or s2 == 0:
        return 1.0
    b1 /= s1
    b2 /= s2
    return 0.5 * np.sum(np.abs(b1 - b2))


# ── Ansatz parameter counts ─────────────────────────────────────────────────

def param_count_sixteen(n_qubits, depth=1):
    """Sixteen (A1): depth × (3n − 1) parameters."""
    return depth * (3 * n_qubits - 1)


# ── Synthetic cost function ─────────────────────────────────────────────────

def make_synthetic_cost(n_params, n_samples=100, n_bins=20, ce_range=(0.0, 0.6),
                        seed=None):
    """
    Build a cost function that mirrors the real QMill / QuGrad pipeline:

        θ  →  [synthetic NZP values]  →  histogram  →  TVD vs target

    The synthetic model replaces circuit execution with:
        NZP_j = sigmoid(s_j · θ / √d) × 0.6

    where s_j are fixed random 'state descriptors' (one per product state).
    This preserves the dimensionality and smoothness of the real landscape
    while being O(n_samples × d) per evaluation — no 2^n cost.

    The target distribution is a Left Weibull (matches the paper's hardest case).
    """
    rng = np.random.default_rng(seed)

    # Fixed product-state descriptors (analogous to Haar-random input states)
    S = rng.standard_normal((n_samples, n_params))

    # Target histogram: Left Weibull, 20 bins in [0, 0.6]
    target_samples = 0.05 * rng.weibull(1.2, size=100_000)
    target_bins = get_bins(target_samples, n_bins, *ce_range)

    lo, hi = ce_range

    def cost(theta):
        theta = np.asarray(theta)
        # Synthetic NZP: smooth, θ-dependent, bounded in [0, 0.6]
        logits = S @ theta / np.sqrt(n_params)
        nzp = hi / (1.0 + np.exp(-logits))          # sigmoid → [0, 0.6]
        gen_bins = get_bins(nzp, n_bins, lo, hi)
        return tvd2(gen_bins, target_bins)

    return cost


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(qubit_counts, depth=1, n_trials=10, max_fevals=500,
                  n_samples=100):
    """
    For each qubit count, run CMA-ES n_trials times and collect wall-clock
    times.  Returns a list of result dicts.
    """
    results = []

    for n_q in qubit_counts:
        d = param_count_sixteen(n_q, depth)
        output_states = 2 ** n_q

        trial_times  = []
        trial_fevals = []
        trial_costs  = []

        for t in range(n_trials):
            cost_fn = make_synthetic_cost(d, n_samples=n_samples, seed=t + 1)

            x0 = np.random.default_rng(t).uniform(0, 2 * np.pi, d)

            opts = {
                'maxfevals': max_fevals,
                'verbose':   -9,
                'seed':      t + 1,
                'bounds':    [0, 2 * np.pi],
                'tolfun':    1e-11,       # don't stop early on flat landscape
                'tolx':      1e-12,
            }

            start = time.perf_counter()
            es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
            es.optimize(cost_fn)
            elapsed = time.perf_counter() - start

            trial_times.append(elapsed)
            trial_fevals.append(es.result.evaluations)
            trial_costs.append(es.result.fbest)

        row = {
            'n_qubits':      n_q,
            'output_states':  output_states,
            'n_params':       d,
            'mean_time_s':    np.mean(trial_times),
            'std_time_s':     np.std(trial_times),
            'mean_fevals':    np.mean(trial_fevals),
            'mean_final_tvd': np.mean(trial_costs),
        }
        results.append(row)

        # Live progress
        exp_str = f"2^{n_q}" if n_q <= 40 else f"~10^{n_q * 0.301:.0f}"
        print(f"  n={n_q:3d}  |  2^n={exp_str:>8s}  |  d={d:5d}  |  "
              f"time = {row['mean_time_s']:.4f} ± {row['std_time_s']:.4f} s  |  "
              f"fevals = {row['mean_fevals']:.0f}  |  TVD = {row['mean_final_tvd']:.4f}")
        sys.stdout.flush()

    return results


def print_latex_table(results):
    """Print a LaTeX-ready table."""
    print("\n" + "=" * 80)
    print("LaTeX table:")
    print("=" * 80)
    print(r"\begin{tabular}{r r r r r}")
    print(r"\toprule")
    print(r"$n$ & $2^n$ & $d$ & Time (s) & Evaluations \\")
    print(r"\midrule")
    for r in results:
        n = r['n_qubits']
        states = f"$2^{{{n}}}$"
        print(f"  {n} & {states} & {r['n_params']} & "
              f"${r['mean_time_s']:.3f} \\pm {r['std_time_s']:.3f}$ & "
              f"{r['mean_fevals']:.0f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_markdown_table(results):
    """Print a Markdown table."""
    print("\n" + "=" * 80)
    print("Markdown table:")
    print("=" * 80)
    print(f"| Qubits (n) | Output States (2^n) | Parameters (d) | "
          f"CMA-ES Time (s) | Func Evals |")
    print(f"|{'-'*12}|{'-'*21}|{'-'*16}|{'-'*17}|{'-'*12}|")
    for r in results:
        n = r['n_qubits']
        print(f"| {n:>10} | {'2^' + str(n):>19} | {r['n_params']:>14} | "
              f"{r['mean_time_s']:.3f} ± {r['std_time_s']:.3f} | "
              f"{r['mean_fevals']:>10.0f} |")


# ── Proportional-budget benchmark ────────────────────────────────────────────

def run_proportional_benchmark(qubit_counts, depth=1, n_trials=10,
                               fevals_per_param=50, n_samples=100):
    """
    Give each dimension a convergence-appropriate budget: fevals = fevals_per_param × d.
    This shows how CMA-ES time scales when the problem is given enough budget to
    meaningfully optimise at each dimension.
    """
    results = []

    for n_q in qubit_counts:
        d = param_count_sixteen(n_q, depth)
        max_fevals = fevals_per_param * d
        output_states = 2 ** n_q

        trial_times  = []
        trial_fevals = []
        trial_costs  = []

        for t in range(n_trials):
            cost_fn = make_synthetic_cost(d, n_samples=n_samples, seed=t + 1)
            x0 = np.random.default_rng(t).uniform(0, 2 * np.pi, d)

            opts = {
                'maxfevals': max_fevals,
                'verbose':   -9,
                'seed':      t + 1,
                'bounds':    [0, 2 * np.pi],
                'tolfun':    1e-11,
                'tolx':      1e-12,
            }

            start = time.perf_counter()
            es = cma.CMAEvolutionStrategy(x0, 1.0, opts)
            es.optimize(cost_fn)
            elapsed = time.perf_counter() - start

            trial_times.append(elapsed)
            trial_fevals.append(es.result.evaluations)
            trial_costs.append(es.result.fbest)

        row = {
            'n_qubits':      n_q,
            'output_states':  output_states,
            'n_params':       d,
            'max_fevals':     max_fevals,
            'mean_time_s':    np.mean(trial_times),
            'std_time_s':     np.std(trial_times),
            'mean_fevals':    np.mean(trial_fevals),
            'mean_final_tvd': np.mean(trial_costs),
        }
        results.append(row)

        exp_str = f"2^{n_q}" if n_q <= 40 else f"~10^{n_q * 0.301:.0f}"
        print(f"  n={n_q:3d}  |  2^n={exp_str:>8s}  |  d={d:5d}  |  "
              f"budget={max_fevals:6d}  |  "
              f"time = {row['mean_time_s']:.4f} ± {row['std_time_s']:.4f} s  |  "
              f"fevals = {row['mean_fevals']:.0f}  |  TVD = {row['mean_final_tvd']:.4f}")
        sys.stdout.flush()

    return results


def print_proportional_latex(results):
    """Print LaTeX table for proportional-budget run."""
    print("\n" + "=" * 80)
    print("LaTeX table (proportional budget: 50 × d evaluations):")
    print("=" * 80)
    print(r"\begin{tabular}{r r r r r r}")
    print(r"\toprule")
    print(r"$n$ & $2^n$ & $d$ & Budget & Time (s) & Best TVD \\")
    print(r"\midrule")
    for r in results:
        n = r['n_qubits']
        states = f"$2^{{{n}}}$"
        print(f"  {n} & {states} & {r['n_params']} & {r['max_fevals']} & "
              f"${r['mean_time_s']:.2f} \\pm {r['std_time_s']:.2f}$ & "
              f"${r['mean_final_tvd']:.4f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_proportional_markdown(results):
    """Print Markdown table for proportional-budget run."""
    print("\n" + "=" * 80)
    print("Markdown table (proportional budget: 50 × d evaluations):")
    print("=" * 80)
    print(f"| Qubits (n) | 2^n | Params (d) | Budget (50d) | "
          f"CMA-ES Time (s) | Best TVD |")
    print(f"|{'-'*12}|{'-'*10}|{'-'*12}|{'-'*14}|{'-'*17}|{'-'*10}|")
    for r in results:
        n = r['n_qubits']
        print(f"| {n:>10} | 2^{n:<5} | {r['n_params']:>10} | "
              f"{r['max_fevals']:>12} | "
              f"{r['mean_time_s']:.2f} ± {r['std_time_s']:.2f} | "
              f"{r['mean_final_tvd']:.4f} |")


# ── Dual Annealing comparison ────────────────────────────────────────────────

def run_dual_annealing_benchmark(qubit_counts, depth=1, n_trials=10,
                                  maxiter=50, maxfun=500, n_samples=100):
    """
    Benchmark scipy.optimize.dual_annealing (original QMill optimizer)
    with the same synthetic cost function for fair comparison.
    """
    from scipy.optimize import dual_annealing

    results = []

    for n_q in qubit_counts:
        d = param_count_sixteen(n_q, depth)

        trial_times  = []
        trial_fevals = []
        trial_costs  = []

        for t in range(n_trials):
            cost_fn = make_synthetic_cost(d, n_samples=n_samples, seed=t + 1)
            x0 = np.random.default_rng(t).uniform(0, 2 * np.pi, d)

            eval_count = [0]
            def counted_cost(x):
                eval_count[0] += 1
                return cost_fn(x)

            start = time.perf_counter()
            res = dual_annealing(
                counted_cost,
                bounds=[(0, 2 * np.pi)] * d,
                x0=x0,
                maxiter=maxiter,
                maxfun=maxfun,
                seed=t + 1,
            )
            elapsed = time.perf_counter() - start

            trial_times.append(elapsed)
            trial_fevals.append(eval_count[0])
            trial_costs.append(res.fun)

        row = {
            'n_qubits':      n_q,
            'n_params':       d,
            'mean_time_s':    np.mean(trial_times),
            'std_time_s':     np.std(trial_times),
            'mean_fevals':    np.mean(trial_fevals),
            'mean_final_tvd': np.mean(trial_costs),
        }
        results.append(row)

        print(f"  n={n_q:3d}  |  d={d:5d}  |  "
              f"time = {row['mean_time_s']:.4f} ± {row['std_time_s']:.4f} s  |  "
              f"fevals = {row['mean_fevals']:.0f}  |  TVD = {row['mean_final_tvd']:.4f}")
        sys.stdout.flush()

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    qubit_counts = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100]

    # ── Table 1: Fixed budget (500 fevals) ───────────────────────────────────
    print("=" * 80)
    print("TABLE 1: CMA-ES — Fixed budget (500 function evaluations)")
    print("=" * 80)
    print("  Ansatz: Sixteen (A1),  depth=1,  100 product states,  10 trials")
    print()

    results_fixed = run_benchmark(
        qubit_counts=qubit_counts,
        depth=1,
        n_trials=10,
        max_fevals=500,
        n_samples=100,
    )
    print_markdown_table(results_fixed)
    print_latex_table(results_fixed)

    # ── Table 2: Proportional budget (50 × d fevals) ────────────────────────
    print("\n\n" + "=" * 80)
    print("TABLE 2: CMA-ES — Proportional budget (50 × d function evaluations)")
    print("=" * 80)
    print("  Gives each dimension a convergence-appropriate budget.")
    print()

    results_prop = run_proportional_benchmark(
        qubit_counts=qubit_counts,
        depth=1,
        n_trials=10,
        fevals_per_param=50,
        n_samples=100,
    )
    print_proportional_markdown(results_prop)
    print_proportional_latex(results_prop)

    # ── Table 3: Dual Annealing comparison ──────────────────────────────────
    print("\n\n" + "=" * 80)
    print("TABLE 3: Dual Annealing (original QMill) — same synthetic cost")
    print("=" * 80)
    print("  maxiter=50, maxfun=500  (matches original QMill settings)")
    print()

    results_da = run_dual_annealing_benchmark(
        qubit_counts=qubit_counts,
        depth=1,
        n_trials=10,
        maxiter=50,
        maxfun=500,
        n_samples=100,
    )

    print(f"\n| Qubits (n) | Params (d) | DA Time (s) | Func Evals | Best TVD |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*13}|{'-'*12}|{'-'*10}|")
    for r in results_da:
        print(f"| {r['n_qubits']:>10} | {r['n_params']:>10} | "
              f"{r['mean_time_s']:.3f} ± {r['std_time_s']:.3f} | "
              f"{r['mean_fevals']:>10.0f} | {r['mean_final_tvd']:.4f} |")

    # ── Save all results ────────────────────────────────────────────────────
    np.save('cmaes_scaling_results.npy', {
        'fixed_500':      results_fixed,
        'proportional_50d': results_prop,
        'dual_annealing':  results_da,
    }, allow_pickle=True)
    print(f"\nAll results saved to cmaes_scaling_results.npy")

