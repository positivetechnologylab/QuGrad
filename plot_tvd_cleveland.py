import os
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

# Configuration
ANSATZ = 'Sixteen'
NUM_BINS = 30
DATA_RANGE = (0, 0.6)

# Lists of distributions
STRESS_DISTS = ['Uniform', 'Normal', 'Left Weibull', 'Right Weibull']
REAL_DISTS = ['MNIST', 'Fashion MNIST', 'CIFAR', 'QCHEM']
SENSOR_DISTS = ['Soillow', 'Soilhigh', 'dmlow', 'dmhigh']

ALL_DISTS = STRESS_DISTS + REAL_DISTS + SENSOR_DISTS

DISPLAY_NAMES = {
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

PANEL_MAPPING = {}
for d in STRESS_DISTS: PANEL_MAPPING[d] = 'stress'
for d in REAL_DISTS: PANEL_MAPPING[d] = 'real'
for d in SENSOR_DISTS: PANEL_MAPPING[d] = 'sensors'

METHODS = ['QMill', 'QuGrad (Ideal)', 'QuGrad (Noisy)', 'QMill (Noisy)']
METHOD_COLORS = {
    'QMill': '#702963',        # Maroon
    'QuGrad (Ideal)': '#FAA0A0', # Light pink
    'QuGrad (Noisy)': '#FF2400',  # Scarlet
    'QMill (Noisy)': '#FFB347' # Pastel Orange
}

# Try to import distribution constructors
try:
    from dists import Uniform, Normal, WeibullLeft, WeibullRight, MNIST, FashionMNIST, CIFAR, QCHEM, soil, dm
    DIST_CTORS = {
        'Uniform':       lambda: Uniform(100),
        'Normal':        lambda: Normal(100),
        'Left Weibull':  lambda: WeibullLeft(100),
        'Right Weibull': lambda: WeibullRight(100),
        'MNIST':         lambda: MNIST(100),
        'Fashion MNIST': lambda: FashionMNIST(100),
        'CIFAR':         lambda: CIFAR(100),
        'QCHEM':         lambda: QCHEM(100),
        'Soillow':       lambda: soil(100,'low'),
        'Soilhigh':      lambda: soil(100,'high'),
        'dmlow':         lambda: dm(100,'low'),
        'dmhigh':        lambda: dm(100,'high'),
    }
except ImportError:
    print("Warning: 'dists.py' not found or import failed. Using random target samples.")
    DIST_CTORS = {}

def tvd_on_bins(p, q, bins=30, data_range=(0,0.6)):
    """Calculate Total Variation Distance based on histograms."""
    p_hist, edges = np.histogram(p, bins=bins, range=data_range, density=True)
    q_hist, _     = np.histogram(q, bins=edges, density=True)
    widths        = np.diff(edges)
    return 0.5 * np.sum(np.abs(p_hist - q_hist) * widths)

def read_best_cost_from_txt(txt_path):
    """Read Best Cost from the .txt file saved during training."""
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if 'Best Cost' in line:
                    match = re.search(r'Best Cost[:\s]+([0-9.]+)', line)
                    if match:
                        return float(match.group(1))
    except Exception as e:
        pass
    return np.nan

def generate_target_samples(n_samples=1000):
    samples = {}
    print("Generating target samples...")
    for name in ALL_DISTS:
        if name in DIST_CTORS:
            try:
                d = DIST_CTORS[name]()
                d.createSampleDistributions(n_samples)
                samples[name] = np.array(d.samples).flatten()
            except Exception as e:
                print(f"  Error generating {name}: {e}")
                samples[name] = np.random.rand(n_samples * 100) * 0.6
        else:
            samples[name] = np.random.rand(n_samples * 100) * 0.6
    return samples

def collect_data(target_samples):
    records = []
    print("\nCollecting TVD data...")
    
    for dist in ALL_DISTS:
        panel = PANEL_MAPPING[dist]
        tgt = target_samples.get(dist)
        
        # 1. QMill
        qmill_path = Path('runs_qmill') / ANSATZ / dist / '5' / '1' / '1' / f'{ANSATZ}_5_1_results.npy'
        tvd_qmill = np.nan
        if qmill_path.exists() and tgt is not None:
            try:
                q = np.load(qmill_path).flatten()
                tvd_qmill = tvd_on_bins(tgt, q, bins=NUM_BINS, data_range=DATA_RANGE)
            except Exception:
                pass
        records.append({
            'panel': panel, 'target': dist, 'method': 'QMill', 'tvd': tvd_qmill
        })

        # 2. QuGrad (Ideal) - from TXT
        qugrad_ideal_txt = Path('runs_qugrad') / 'Ideal' / ANSATZ / dist / '5' / '1' / '1' / f'{ANSATZ}_5_1.txt'
        tvd_ideal = np.nan
        if qugrad_ideal_txt.exists():
            tvd_ideal = read_best_cost_from_txt(qugrad_ideal_txt)
        records.append({
            'panel': panel, 'target': dist, 'method': 'QuGrad (Ideal)', 'tvd': tvd_ideal
        })

        # 3. QuGrad (Noisy) - from TXT
        qugrad_noisy_txt = Path('runs_qugrad') / ANSATZ / dist / '5' / '1' / '1' / f'{ANSATZ}_5_1.txt'
        tvd_noisy = np.nan
        if qugrad_noisy_txt.exists():
            tvd_noisy = read_best_cost_from_txt(qugrad_noisy_txt)
        records.append({
            'panel': panel, 'target': dist, 'method': 'QuGrad (Noisy)', 'tvd': tvd_noisy
        })

        # 4. QMill (Noisy) - from CSV in qmill_noisy_eval/NoisySim
        # Path structure: qmill_noisy_eval/NoisySim/{dist}/5/1/1/*.csv
        qmill_noisy_dir = Path('qmill_noisy_eval') / 'NoisySim' / dist / '5' / '1' / '1'
        tvd_qmill_noisy = np.nan
        if qmill_noisy_dir.exists():
            csv_files = list(qmill_noisy_dir.glob("*.csv"))
            if csv_files and tgt is not None:
                try:
                    df = pd.read_csv(csv_files[0])
                    if 'ce' in df.columns:
                        q = df['ce'].values
                        tvd_qmill_noisy = tvd_on_bins(tgt, q, bins=20, data_range=DATA_RANGE) # Using 20 bins as per calc_all_tvds.py
                except Exception as e:
                    print(f"  Error reading QMill Noisy CSV for {dist}: {e}")
        
        records.append({
            'panel': panel, 'target': dist, 'method': 'QMill (Noisy)', 'tvd': tvd_qmill_noisy
        })
        
        print(f"  {dist}: QMill={tvd_qmill:.4f}, Ideal={tvd_ideal:.4f}, Noisy={tvd_noisy:.4f}, QMill(Noisy)={tvd_qmill_noisy:.4f}")

    return pd.DataFrame(records)

def plot_single_panel(df, panel_key, title, include_legend=False, xmax=1.0):
    fig, ax = plt.subplots(figsize=(6, 3.33), constrained_layout=True)
    
    markers = {'QMill': 'o', 'QuGrad (Ideal)': '^', 'QuGrad (Noisy)': 's', 'QMill (Noisy)': 'D'}
    
    panel_data = df[df['panel'] == panel_key].copy()
    
    if panel_key == 'stress':
        order = STRESS_DISTS[::-1]
    elif panel_key == 'real':
        order = REAL_DISTS[::-1]
    else:
        order = SENSOR_DISTS[::-1]
        
    order = [d for d in order if d in panel_data['target'].unique()]
    
    y_pos = range(len(order))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([DISPLAY_NAMES.get(d, d) for d in order])
    ax.set_ylim(-0.5, len(order) - 0.5)
    
    for y, target in zip(y_pos, order):
        subset = panel_data[panel_data['target'] == target]
        
        val_qmill = subset[subset['method'] == 'QMill']['tvd'].values
        val_ideal = subset[subset['method'] == 'QuGrad (Ideal)']['tvd'].values
        val_noisy = subset[subset['method'] == 'QuGrad (Noisy)']['tvd'].values
        val_qmill_noisy = subset[subset['method'] == 'QMill (Noisy)']['tvd'].values
        
        val_qmill = val_qmill[0] if len(val_qmill) > 0 else np.nan
        val_ideal = val_ideal[0] if len(val_ideal) > 0 else np.nan
        val_noisy = val_noisy[0] if len(val_noisy) > 0 else np.nan
        val_qmill_noisy = val_qmill_noisy[0] if len(val_qmill_noisy) > 0 else np.nan
        
        # Connect QMill to Ideal
        if not np.isnan(val_qmill) and not np.isnan(val_ideal):
            ax.plot([val_qmill, val_ideal], [y, y], color='gray', linestyle='-', linewidth=1, zorder=1)
        # Connect QMill to Noisy
        if not np.isnan(val_qmill) and not np.isnan(val_noisy):
            ax.plot([val_qmill, val_noisy], [y, y], color='gray', linestyle='-', linewidth=1, zorder=1)
        # Connect QMill to QMill Noisy
        if not np.isnan(val_qmill) and not np.isnan(val_qmill_noisy):
            ax.plot([val_qmill, val_qmill_noisy], [y, y], color='gray', linestyle=':', linewidth=1, zorder=1)
            
        for method in METHODS:
            val = subset[subset['method'] == method]['tvd'].values
            if len(val) > 0 and not np.isnan(val[0]):
                ax.scatter(val[0], y, 
                           color=METHOD_COLORS[method], 
                           marker=markers[method], 
                           s=300,  # Increased marker size
                           zorder=2,
                           edgecolor='black',
                           linewidth=0.5)

    # ax.set_title(title)
    ax.set_xlabel('TVD')
    ax.set_xlim(0, xmax)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    # Y-axis label is implicitly 'Target' but we don't necessarily need to write it if it's obvious from ticks
    # But let's keep it for clarity
    # ax.set_ylabel('Target') 

    if include_legend:
        from matplotlib.lines import Line2D
        # Increase legend marker size as well
        legend_elements = [
            Line2D([0], [0], marker=markers['QMill'], color='w', markerfacecolor=METHOD_COLORS['QMill'], 
                   markersize=18, markeredgecolor='black', label='QMill (Ideal)'),
            Line2D([0], [0], marker=markers['QuGrad (Ideal)'], color='w', markerfacecolor=METHOD_COLORS['QuGrad (Ideal)'], 
                   markersize=18, markeredgecolor='black', label='QuGrad (Ideal)'),
            Line2D([0], [0], marker=markers['QMill (Noisy)'], color='w', markerfacecolor=METHOD_COLORS['QMill (Noisy)'], 
                   markersize=18, markeredgecolor='black', label='QMill (Noisy)'),
            Line2D([0], [0], marker=markers['QuGrad (Noisy)'], color='w', markerfacecolor=METHOD_COLORS['QuGrad (Noisy)'], 
                   markersize=18, markeredgecolor='black', label='QuGrad (Noisy)'),
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, edgecolor='black', fancybox=False)

    return fig

def main():
    # 1. Generate target samples
    target_samples = generate_target_samples()
    
    # 2. Collect Data
    df = collect_data(target_samples)
    
    # Save raw data for inspection
    df.to_csv('tvd_results.csv', index=False)
    print("\nSaved TVD results to tvd_results.csv")
    
    # Determine global x-limit
    valid_tvd = df['tvd'].dropna()
    if len(valid_tvd) > 0:
        xmax = valid_tvd.max() * 1.1
    else:
        xmax = 1.0

    # 3. Create and Save Plots
    # Panel (a) Stress
    fig1 = plot_single_panel(df, 'stress', '(a) Stress-testing Distributions', include_legend=True, xmax=xmax)
    fig1.savefig('tvd_cleveland_stress.pdf', bbox_inches='tight', pad_inches=0.15)
    fig1.savefig('tvd_cleveland_stress.png', bbox_inches='tight', pad_inches=0.15, dpi=300)
    print("Saved tvd_cleveland_stress.pdf")

    # Panel (b) Real
    fig2 = plot_single_panel(df, 'real', '(b) Real-dataset Distributions', include_legend=False, xmax=xmax)
    fig2.savefig('tvd_cleveland_real.pdf', bbox_inches='tight', pad_inches=0.15)
    fig2.savefig('tvd_cleveland_real.png', bbox_inches='tight', pad_inches=0.15, dpi=300)
    print("Saved tvd_cleveland_real.pdf")

    # Panel (c) Sensors
    fig3 = plot_single_panel(df, 'sensors', '(c) Quantum Sensors', include_legend=True, xmax=xmax)
    fig3.savefig('tvd_cleveland_sensors.pdf', bbox_inches='tight', pad_inches=0.15)
    fig3.savefig('tvd_cleveland_sensors.png', bbox_inches='tight', pad_inches=0.15, dpi=300)
    print("Saved tvd_cleveland_sensors.pdf")

if __name__ == "__main__":
    main()
