"""
NZP vs CE Validation
====================
Provides evidence that optimizing the NZP surrogate reliably yields
CE-distribution matching.

For each trained configuration (ansatz × target distribution):
  1. Loads the trained circuit parameters
  2. Generates 1000 fresh Haar-random product states
  3. For each state, builds the n-qubit circuit (product state + ansatz),
     computes the exact statevector, and extracts:
       - TRUE CE  = 1 - (1/n) Σ_i Tr(ρ_i²)       [sum of purities]
       - NZP      = 1 - Π_i (1 + Tr(ρ_i²))/2      [product, what SWAP test gives]
  4. Histograms both quantities and compares to the target distribution

Output:
  - Console table:  TVD(NZP, target) vs TVD(CE, target) + Pearson r(NZP, CE)
  - Figure 1:       NZP vs CE scatter (proves strong monotonic relationship)
  - Figure 2:       Per-distribution histograms (CE vs NZP vs Target)
  - LaTeX table ready for the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from scipy.stats import pearsonr, spearmanr, ks_2samp
import os, sys, time

# ── Configuration ────────────────────────────────────────────────────────────

N_QUBITS = 5
DEPTH    = 1
N_STATES = 1000     # fresh product states for validation
SEED     = 12345    # different from any training seed
N_BINS   = 20
CE_RANGE = (0.0, 0.6)

# ── Ansatz circuit builders (copied from circuits.py for self-containment) ───

def ansatz_Sixteen(circuit, thetas, depth):
    num = circuit.num_qubits
    for d in range(depth):
        i = 0
        for j in range(num):
            circuit.rx(thetas[d][i], j)
            i += 1
        for j in range(num):
            circuit.rz(thetas[d][i], j)
            i += 1
        for j in range(num - 1):
            if j % 2 == 1:
                pass
            else:
                circuit.crz(thetas[d][i], j+1, j)
                i += 1
        for j in range(num - 1):
            if j % 2 == 0:
                pass
            else:
                circuit.crz(thetas[d][i], j+1, j)
                i += 1


def ansatz_FiveDeprecated(circuit, thetas, depth):
    num = circuit.num_qubits
    for d in range(depth):
        for i in range(num):
            circuit.rx(thetas[d][i][0], i)
            circuit.rz(thetas[d][i][1], i)
        for i in range(num-1, -1, -1):
            for j in range(num-1, -1, -1):
                if i == j:
                    pass
                elif i > j:
                    circuit.crz(thetas[d][i][j+2], i, j)
                else:
                    circuit.crz(thetas[d][i][j+1], i, j)
        circuit.barrier()
        for i in range(num):
            circuit.rx(thetas[d][i][num + 1], i)
            circuit.rz(thetas[d][i][num + 2], i)


def ansatz_Custom_One(circuit, thetas, depth):
    num = circuit.num_qubits
    for d in range(depth):
        i = 0
        for j in range(num):
            circuit.h(j)
        for j in range(num):
            circuit.ry(thetas[d][i], j)
            i += 1
        for j in range(num-1):
            circuit.crx(thetas[d][i], j, j+1)
            i += 1
        for j in range(num):
            circuit.rz(thetas[d][i], j)
            i += 1
        circuit.barrier()


def ansatz_Custom_Two(circuit, thetas, depth):
    num = circuit.num_qubits
    for d in range(depth):
        i = 0
        for j in range(num):
            circuit.rx(thetas[d][i], j)
            i += 1
        for j in range(num-1):
            circuit.cx(j, j+1)
            circuit.ry(thetas[d][i], j+1)
            i += 1
        for j in range(num-1, 0, -1):
            circuit.crz(thetas[d][i], j, j-1)
            i += 1
        circuit.barrier()


# Ansatz registry: name -> (builder_fn, param_shape_fn)
ANSATZE = {
    'Sixteen':    (ansatz_Sixteen,        lambda n, d: (d, 3*n - 1)),
    'Five':       (ansatz_FiveDeprecated,  lambda n, d: (d, n, n + 3)),
    'Custom_One': (ansatz_Custom_One,      lambda n, d: (d, 3*n - 1)),
    'Custom_Two': (ansatz_Custom_Two,      lambda n, d: (d, 3*n - 1)),
}

# ── Target distribution sampling ─────────────────────────────────────────────

def get_target_samples(name, n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    if name == 'Uniform':
        return rng.uniform(0, 0.4, n)
    elif name == 'Normal':
        return rng.normal(0.2, 0.05, n)
    elif name == 'Left Weibull':
        return 0.05 * rng.weibull(1.2, n)
    elif name == 'Right Weibull':
        return 0.4 * np.ones(n) - 0.05 * rng.weibull(1.2, n)
    else:
        filemap = {
            'MNIST':         'mnist_dist_scaled.npy',
            'Fashion MNIST': 'fashionmnist_dist_scaled.npy',
            'CIFAR':         'cifar_dist_scaled.npy',
            'QCHEM':         'qchem_dist_scaled.npy',
            'Soillow':       'soillow_scaled.npy',
            'Soilhigh':      'soilhigh_scaled.npy',
            'dmlow':         'dmlow_scaled.npy',
            'dmhigh':        'dmhigh_scaled.npy',
        }
        return np.load(filemap[name])


# ── Haar-random product-state sampling (matches ML.sampleParamsDict) ─────────

def sample_product_states(n_qubits, n_states, seed=SEED):
    """
    Sample product states uniformly from the Haar measure on single qubits.
    Matches sampleParamsDict: theta = 2*arccos(1-2u), phi & lambda uniform.
    """
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_states):
        params = {}
        for i in range(n_qubits):
            u = rng.random()
            params[f'theta_{i}']  = 2.0 * np.arccos(1.0 - 2.0 * u)
            params[f'phi_{i}']    = rng.uniform(0, 2 * np.pi)
            params[f'lmbda_{i}']  = rng.uniform(0, 2 * np.pi)
        states.append(params)
    return states


# ── Core: compute exact CE and NZP via statevector ───────────────────────────

def compute_ce_nzp(ansatz_fn, trained_params, product_state, n_qubits, depth):
    """
    Build |psi> = U(theta) U_ref |0^n>, get statevector, compute:
      CE  = 1 - (1/n) sum_i Tr(rho_i^2)
      NZP = 1 - prod_i (1 + Tr(rho_i^2))/2
    """
    qc = QuantumCircuit(n_qubits)

    # Product state preparation
    for i in range(n_qubits):
        qc.u(product_state[f'theta_{i}'],
             product_state[f'phi_{i}'],
             product_state[f'lmbda_{i}'], i)

    # Apply trained ansatz
    ansatz_fn(qc, trained_params, depth)

    # Exact statevector
    sv = Statevector(qc)

    # Per-qubit purities
    purities = np.empty(n_qubits)
    for i in range(n_qubits):
        others = [j for j in range(n_qubits) if j != i]
        rho_i = partial_trace(sv, others)
        purities[i] = np.real(np.trace(rho_i.data @ rho_i.data))

    ce  = 1.0 - np.mean(purities)
    nzp = 1.0 - np.prod((1.0 + purities) / 2.0)

    return ce, nzp


# ── TVD on histograms ────────────────────────────────────────────────────────

def tvd_hist(s1, s2, n_bins=N_BINS, lo=CE_RANGE[0], hi=CE_RANGE[1]):
    edges = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(s1, bins=edges, density=True)
    h2, _ = np.histogram(s2, bins=edges, density=True)
    w = np.diff(edges)
    return 0.5 * np.sum(np.abs(h1 - h2) * w)


# ── Determine best ansatz per distribution (from existing results) ───────────

DIST_NAMES = [
    'Uniform', 'Normal', 'Left Weibull', 'Right Weibull',
    'MNIST', 'Fashion MNIST', 'CIFAR', 'QCHEM',
    'Soillow', 'Soilhigh', 'dmlow', 'dmhigh',
]

def find_best_ansatze():
    """Pick the ansatz with lowest TVD(trained NZP, target) per distribution."""
    edges = np.linspace(*CE_RANGE, N_BINS + 1)
    w = np.diff(edges)
    best = {}

    for dname in DIST_NAMES:
        target = get_target_samples(dname)
        h_tgt, _ = np.histogram(target, bins=edges, density=True)

        best_tvd, best_a = float('inf'), None
        for aname in ANSATZE:
            fp = f'runs_qmill/{aname}/{dname}/5/1/1/{aname}_5_1_results.npy'
            if not os.path.exists(fp):
                continue
            trained = np.load(fp).flatten()
            h_tr, _ = np.histogram(trained, bins=edges, density=True)
            tvd = 0.5 * np.sum(np.abs(h_tr - h_tgt) * w)
            if tvd < best_tvd:
                best_tvd, best_a = tvd, aname

        best[dname] = best_a
    return best


# ── Main validation ──────────────────────────────────────────────────────────

def run_validation():
    print("=" * 85)
    print("NZP vs CE Validation — Statevector Ground Truth")
    print("=" * 85)
    print(f"  {N_QUBITS} qubits, depth {DEPTH}, {N_STATES} fresh product states, seed {SEED}")
    print()

    # Generate one shared set of fresh product states
    print("Sampling product states ... ", end="", flush=True)
    states = sample_product_states(N_QUBITS, N_STATES, seed=SEED)
    print("done.")

    best_ansatze = find_best_ansatze()

    # Storage for aggregate scatter plot
    all_ce  = []
    all_nzp = []

    results = []

    for dname in DIST_NAMES:
        aname = best_ansatze[dname]
        ansatz_fn, shape_fn = ANSATZE[aname]
        shape = shape_fn(N_QUBITS, DEPTH)

        # Load trained parameters
        fp = f'runs_qmill/{aname}/{dname}/5/1/1/{aname}_5_1.npy'
        raw_params = np.load(fp)
        trained_params = raw_params.reshape(shape)

        # Compute CE and NZP for every product state
        t0 = time.perf_counter()
        ce_vals  = np.empty(N_STATES)
        nzp_vals = np.empty(N_STATES)

        for k, ps in enumerate(states):
            ce_vals[k], nzp_vals[k] = compute_ce_nzp(
                ansatz_fn, trained_params, ps, N_QUBITS, DEPTH
            )
            if (k + 1) % 200 == 0:
                print(f"    {dname}/{aname}: {k+1}/{N_STATES}", flush=True)

        elapsed = time.perf_counter() - t0

        all_ce.extend(ce_vals)
        all_nzp.extend(nzp_vals)

        # Target samples
        target = get_target_samples(dname)

        # TVDs
        tvd_ce  = tvd_hist(ce_vals, target)
        tvd_nzp = tvd_hist(nzp_vals, target)

        # Correlation between CE and NZP for this circuit
        r_pearson, _  = pearsonr(ce_vals, nzp_vals)
        r_spearman, _ = spearmanr(ce_vals, nzp_vals)

        # KS test: CE distribution vs target
        ks_stat, ks_pval = ks_2samp(ce_vals, target)

        results.append({
            'dist':       dname,
            'ansatz':     aname,
            'tvd_nzp':    tvd_nzp,
            'tvd_ce':     tvd_ce,
            'pearson_r':  r_pearson,
            'spearman_r': r_spearman,
            'ks_stat':    ks_stat,
            'ks_pval':    ks_pval,
            'ce_vals':    ce_vals,
            'nzp_vals':   nzp_vals,
            'time_s':     elapsed,
        })

        print(f"  {dname:<16} ({aname:<12})  "
              f"TVD_nzp={tvd_nzp:.4f}  TVD_ce={tvd_ce:.4f}  "
              f"r={r_pearson:.4f}  [{elapsed:.1f}s]")

    # ── Print summary table ──────────────────────────────────────────────────
    all_ce  = np.array(all_ce)
    all_nzp = np.array(all_nzp)
    r_global, _ = pearsonr(all_ce, all_nzp)
    rho_global, _ = spearmanr(all_ce, all_nzp)

    print()
    print("=" * 85)
    print("SUMMARY TABLE")
    print("=" * 85)
    print(f"{'Distribution':<16} {'Ansatz':<12} {'TVD(NZP)':>9} {'TVD(CE)':>9} "
          f"{'Pearson r':>10} {'Spearman':>9} {'KS stat':>8}")
    print("-" * 85)
    for r in results:
        print(f"{r['dist']:<16} {r['ansatz']:<12} {r['tvd_nzp']:>9.4f} {r['tvd_ce']:>9.4f} "
              f"{r['pearson_r']:>10.4f} {r['spearman_r']:>9.4f} {r['ks_stat']:>8.4f}")
    print("-" * 85)
    print(f"{'Global correlation:':<28} "
          f"{'':>9} {'':>9} {r_global:>10.4f} {rho_global:>9.4f}")

    # ── LaTeX table ──────────────────────────────────────────────────────────
    ansatz_labels = {
        'Sixteen': 'A1', 'Five': 'A2',
        'Custom_One': 'A3', 'Custom_Two': 'A4',
    }

    print("\n" + "=" * 85)
    print("LaTeX table:")
    print("=" * 85)
    print(r"\begin{tabular}{l c r r r r}")
    print(r"\toprule")
    print(r"Distribution & Ansatz & TVD$_{\mathrm{NZP}}$ & TVD$_{\mathrm{CE}}$ "
          r"& Pearson $r$ & Spearman $\rho$ \\")
    print(r"\midrule")
    for r in results:
        label = ansatz_labels.get(r['ansatz'], r['ansatz'])
        print(f"  {r['dist']:<16} & {label} & {r['tvd_nzp']:.4f} & {r['tvd_ce']:.4f} "
              f"& {r['pearson_r']:.4f} & {r['spearman_r']:.4f} \\\\")
    print(r"\midrule")
    print(f"  \\multicolumn{{4}}{{l}}{{Global (all states pooled)}} "
          f"& {r_global:.4f} & {rho_global:.4f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ── Figure 1: NZP vs CE scatter ──────────────────────────────────────────
    os.makedirs('Results', exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(all_nzp, all_ce, s=1, alpha=0.15, color='#1d4670')
    lo = min(all_nzp.min(), all_ce.min())
    hi = max(all_nzp.max(), all_ce.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, label='y = x')
    ax.set_xlabel('NZP (SWAP-test surrogate)')
    ax.set_ylabel('CE (exact statevector)')
    ax.set_title(f'NZP vs CE  —  Pearson r = {r_global:.4f}, '
                 f'Spearman ρ = {rho_global:.4f}')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig('Results/nzp_vs_ce_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Results/nzp_vs_ce_scatter.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("\nSaved: Results/nzp_vs_ce_scatter.pdf")

    # ── Figure 2: Per-distribution histogram comparison ──────────────────────
    title_map = {
        'Soillow': 'Soil Low', 'Soilhigh': 'Soil High',
        'dmlow': 'DM Low', 'dmhigh': 'DM High',
    }

    fig = plt.figure(figsize=(24, 16))
    gs  = gridspec.GridSpec(3, 4, figure=fig, wspace=0.25, hspace=0.35)

    edges = np.linspace(*CE_RANGE, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    for idx, r in enumerate(results):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])

        target = get_target_samples(r['dist'])
        h_tgt, _ = np.histogram(target,      bins=edges, density=True)
        h_ce,  _ = np.histogram(r['ce_vals'], bins=edges, density=True)
        h_nzp, _ = np.histogram(r['nzp_vals'], bins=edges, density=True)

        ax.step(centers, h_tgt, where='mid', color='k',       linestyle='--',
                linewidth=2.0, label='Target')
        ax.step(centers, h_ce,  where='mid', color='#1d4670', linewidth=1.6,
                label=f'CE (TVD={r["tvd_ce"]:.3f})')
        ax.step(centers, h_nzp, where='mid', color='#e07b39', linewidth=1.6,
                linestyle=':', label=f'NZP (TVD={r["tvd_nzp"]:.3f})')

        display = title_map.get(r['dist'], r['dist'])
        label   = {'Sixteen': 'A1', 'Five': 'A2',
                    'Custom_One': 'A3', 'Custom_Two': 'A4'}
        ax.set_title(f"{display} ({label.get(r['ansatz'], r['ansatz'])})")
        ax.set_xlim(*CE_RANGE)
        ax.grid(True, alpha=0.3, linestyle='--')

        if idx == 0:
            ax.legend(fontsize=8)

    fig.supxlabel('Value', y=0.04, fontsize=14)
    fig.supylabel('Density', x=0.04, fontsize=14)
    fig.suptitle('NZP vs CE Distribution Comparison (Statevector Ground Truth)',
                 fontsize=16, y=0.98)
    plt.savefig('Results/nzp_ce_histograms.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('Results/nzp_ce_histograms.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("Saved: Results/nzp_ce_histograms.pdf")

    # ── Save raw data ────────────────────────────────────────────────────────
    save_data = {}
    for r in results:
        key = f"{r['dist']}_{r['ansatz']}"
        save_data[key] = {
            'ce_vals':    r['ce_vals'],
            'nzp_vals':   r['nzp_vals'],
            'tvd_ce':     r['tvd_ce'],
            'tvd_nzp':    r['tvd_nzp'],
            'pearson_r':  r['pearson_r'],
            'spearman_r': r['spearman_r'],
        }
    np.save('Results/nzp_ce_validation_data.npy', save_data, allow_pickle=True)
    print("Saved: Results/nzp_ce_validation_data.npy")

    print("\nDone.")


if __name__ == "__main__":
    run_validation()