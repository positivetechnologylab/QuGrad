import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18

# Only A1 (Sixteen) ansatz
ansatz = 'Sixteen'

distributions = [
    'Uniform', 'Normal', 'Left Weibull', 'Right Weibull',
    'MNIST', 'Fashion MNIST', 'CIFAR', 'QCHEM',
    'Soillow', 'Soilhigh', 'dmlow', 'dmhigh'
]

display_names = {
    'Uniform': 'Uniform',
    'Normal': 'Normal',
    'Left Weibull': 'L. Weibull',
    'Right Weibull': 'R. Weibull',
    'MNIST': 'MNIST',
    'Fashion MNIST': 'F. MNIST',
    'CIFAR': 'CIFAR',
    'QCHEM': 'QCHEM',
    'Soillow': 'Soil Low',
    'Soilhigh': 'Soil High',
    'dmlow': 'DM Low',
    'dmhigh': 'DM High'
}

bar_colors = {
    'QuGrad (Ideal)': '#FAA0A0',
    'QuGrad (Noisy)': '#FF2400'
}
methods = ['QuGrad (Ideal)', 'QuGrad (Noisy)']


def read_purity_from_txt(txt_path):
    """Read Final Purity from the .txt file."""
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if 'Final Purity' in line:
                    # Format: "Final Purity:       0.8563 +/- 0.0000"
                    match = re.search(r'Final Purity:\s+([0-9.]+)', line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"    Error reading {txt_path}: {e}")
    return np.nan


def read_entropy_from_txt(txt_path):
    """Read Final Entropy from the .txt file."""
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if 'Final Entropy' in line:
                    # Format: "Final Entropy (bits): 0.5948 +/- 0.0000"
                    match = re.search(r'Final Entropy.*:\s+([0-9.]+)', line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        print(f"    Error reading {txt_path}: {e}")
    return np.nan


def collect_purity_results():
    """Collect purity results for QuGrad Ideal and Noisy."""
    results = {dist: {} for dist in distributions}
    print("Collecting purity results...")

    for dist in distributions:
        print(f"  Processing {dist}...")

        # QuGrad Ideal - always purity = 1.0 (pure states by definition)
        results[dist]['QuGrad (Ideal)'] = 1.0
        print(f"    QuGrad (Ideal): 1.0000 (ideal)")

        # QuGrad Noisy (depth 1) - try both possible paths
        noisy_txt = Path('runs_qugrad') / ansatz / dist / '5' / '1' / '1' / f'{ansatz}_5_1.txt'
        if not noisy_txt.exists():
            noisy_txt = Path('runs_qugrad') / ansatz / dist / '5' / '1' / '1' / f'{ansatz}_5_1.txt'

        if noisy_txt.exists():
            results[dist]['QuGrad (Noisy)'] = read_purity_from_txt(noisy_txt)
            print(f"    QuGrad (Noisy): {results[dist]['QuGrad (Noisy)']:.4f}" if not np.isnan(results[dist]['QuGrad (Noisy)']) else "    QuGrad (Noisy): N/A")
        else:
            print(f"    QuGrad (Noisy): File not found")
            results[dist]['QuGrad (Noisy)'] = np.nan

    return results


def create_purity_bar_plot(results, distributions, y_min=0.5, y_max=1.05):
    """Create grouped bar chart for purity comparison."""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.set_ylim(y_min, y_max)
    tick_interval = 0.1
    y_ticks = np.arange(y_min, y_max + 1e-9, tick_interval)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=0)

    x = np.arange(len(distributions))
    width = 0.35
    num_methods = len(methods)

    for i, method in enumerate(methods):
        purities = [results.get(dist, {}).get(method, np.nan) for dist in distributions]
        plot_purities = [p if p is not None and not np.isnan(p) else 0 for p in purities]

        bar_pos = x + i * width - (num_methods - 1) * width / 2
        ax.bar(bar_pos, plot_purities, width,
               label=method,
               color=bar_colors[method],
               edgecolor='black',
               linewidth=1,
               zorder=3)

    ax.set_ylabel('Purity')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(dist, dist) for dist in distributions], rotation=45, ha='right')

    ax.legend(loc='lower right',
              frameon=True,
              edgecolor='black',
              fancybox=False)

    plt.tight_layout()
    return fig


# Main execution
os.makedirs('Paper Plots', exist_ok=True)

purity_results = collect_purity_results()

print("\nCreating purity plot...")
fig = create_purity_bar_plot(purity_results, distributions)
save_path = 'Paper Plots/purity_comparison.pdf'
plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)
print(f"Saved plot: {save_path}")

# Print numerical results
print("\n--- Purity Results ---")
for dist in distributions:
    print(f"\n{display_names.get(dist, dist)}:")
    for method in methods:
        purity = purity_results[dist].get(method, np.nan)
        purity_str = f"{purity:.4f}" if not np.isnan(purity) else "N/A"
        print(f"  {method}: {purity_str}")

print("\n--- Script Finished ---")
