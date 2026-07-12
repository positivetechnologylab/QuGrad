import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22

# Only A1 (Sixteen) ansatz
ansatz = 'Sixteen'

# All distributions
distributions = [
    'Uniform', 'Normal', 'Left Weibull', 'Right Weibull',
    'MNIST', 'Fashion MNIST', 'CIFAR', 'QCHEM',
    'Soillow', 'Soilhigh', 'dmlow', 'dmhigh'
]

# Collect paired data across all distributions
ce_ideal_list = []
ce_noisy_list = []
ce_qmill_list = []

print("Loading data...")
for dist in distributions:
    # QuGrad Ideal (depth 2)
    ideal_path = Path('runs_qugrad') / 'Ideal' / ansatz / dist / '5' / '2' / '1' / f'{ansatz}_5_2_results.npy'
    # QuGrad Noisy (depth 1)
    noisy_path = Path('runs_qugrad') / ansatz / dist / '5' / '1' / '1' / f'{ansatz}_5_1_results.npy'
    # QMill (depth 1)
    qmill_path = Path('runs_qmill') / ansatz / dist / '5' / '1' / '1' / f'{ansatz}_5_1_results.npy'

    # Load if all three exist
    if ideal_path.exists() and noisy_path.exists() and qmill_path.exists():
        ideal_data = np.load(ideal_path).flatten()
        noisy_data = np.load(noisy_path).flatten()
        qmill_data = np.load(qmill_path).flatten()

        # Truncate to minimum length to create pseudo-paired data
        min_len = min(len(ideal_data), len(noisy_data), len(qmill_data))

        # Randomly sample to match lengths
        np.random.seed(42)  # For reproducibility
        ideal_idx = np.random.choice(len(ideal_data), min_len, replace=False) if len(ideal_data) > min_len else np.arange(len(ideal_data))
        noisy_idx = np.random.choice(len(noisy_data), min_len, replace=False) if len(noisy_data) > min_len else np.arange(len(noisy_data))
        qmill_idx = np.random.choice(len(qmill_data), min_len, replace=False) if len(qmill_data) > min_len else np.arange(len(qmill_data))

        ce_ideal_list.append(ideal_data[ideal_idx])
        ce_noisy_list.append(noisy_data[noisy_idx])
        ce_qmill_list.append(qmill_data[qmill_idx])

        print(f"  Loaded {dist}: {min_len} paired samples (from Ideal={len(ideal_data)}, Noisy={len(noisy_data)}, QMill={len(qmill_data)})")
    else:
        missing = []
        if not ideal_path.exists():
            missing.append("Ideal")
        if not noisy_path.exists():
            missing.append("Noisy")
        if not qmill_path.exists():
            missing.append("QMill")
        print(f"  Skipping {dist} - missing: {', '.join(missing)}")

if not ce_ideal_list:
    print("No data found!")
    exit()

# Concatenate all distributions
ce_ideal_raw = np.concatenate(ce_ideal_list)
ce_noisy_raw = np.concatenate(ce_noisy_list)
ce_qmill_raw = np.concatenate(ce_qmill_list)

print(f"\nTotal paired samples: {len(ce_ideal_raw)}")

# Normalize everything into [0,1]
def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn)

ce_ideal = normalize(ce_ideal_raw)
ce_noisy = normalize(ce_noisy_raw)
ce_qmill = normalize(ce_qmill_raw)

# Define five uniform bins on ideal CE
n_bins = 5
bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
labels = [f"{bin_edges[i]:.1f}--{bin_edges[i+1]:.1f}" for i in range(n_bins)]

# Digitize ideal CE into those bins
idx_ideal = np.clip(np.digitize(ce_ideal, bin_edges, right=False), 1, n_bins)

# Group all three by the ideal bin assignments (paired indexing)
ideal_by_bin = [ce_ideal[idx_ideal == i] for i in range(1, n_bins + 1)]
noisy_by_ideal = [ce_noisy[idx_ideal == i] for i in range(1, n_bins + 1)]
qmill_by_ideal = [ce_qmill[idx_ideal == i] for i in range(1, n_bins + 1)]

print("\nSamples per ideal CE bin:")
for i, label in enumerate(labels):
    print(f"  {label}: Ideal={len(ideal_by_bin[i])}, Noisy={len(noisy_by_ideal[i])}, QMill={len(qmill_by_ideal[i])}")

# Plot three boxplots per bin
x = np.arange(1, n_bins + 1)
width = 0.25
gap = 0.02

color_qmill = "#702963"  # Maroon
color_ideal = "#FAA0A0"  # Light pink
color_noisy = "#FF2400"  # Scarlet

fig, ax = plt.subplots(figsize=(10, 5))

# QMill boxplot (left)
bp0 = ax.boxplot(
    qmill_by_ideal,
    positions=x - width - gap,
    widths=width,
    patch_artist=True,
    showfliers=False
)

# QuGrad Ideal boxplot (center)
bp1 = ax.boxplot(
    ideal_by_bin,
    positions=x,
    widths=width,
    patch_artist=True,
    showfliers=False
)

# QuGrad Noisy boxplot (right)
bp2 = ax.boxplot(
    noisy_by_ideal,
    positions=x + width + gap,
    widths=width,
    patch_artist=True,
    showfliers=False
)

for bp, col in ((bp0, color_qmill), (bp1, color_ideal), (bp2, color_noisy)):
    for box in bp['boxes']:
        box.set_facecolor(col)
        box.set_edgecolor('black')
    for whisker in bp['whiskers']:
        whisker.set_color(col)
    for cap in bp['caps']:
        cap.set_color(col)
    for median in bp['medians']:
        median.set_color('black')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Normalized Ideal CE')
ax.set_ylabel('Normalized CE')
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.grid(True, alpha=0.3, linestyle='--')

handles = [bp0["boxes"][0], bp1["boxes"][0], bp2["boxes"][0]]
legend_labels = ['QMill', 'QuGrad (Ideal)', 'QuGrad (Noisy)']

fig.legend(
    handles,
    legend_labels,
    ncol=3,
    edgecolor='black',
    columnspacing=1.8,
    bbox_to_anchor=(0.985, 1.06),
)

plt.tight_layout()
plt.savefig('noisyideal_qugrad.pdf', bbox_inches='tight')
print("\nSaved: noisyideal_qugrad.pdf")
