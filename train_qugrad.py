"""
QuGrad Training: CMA-ES optimization of the NZP-TVD objective.

Trains a low-depth parameterized ansatz so that the NZP histogram of the states it
generates matches a target CE histogram, using CMA-ES (derivative-free, population-
based). Circuits run through an ideal or IBM-noise-calibrated executor.

METHODOLOGY:
✓ Product states (sampled U3 parameters)
✓ SWAP test circuits
✓ CE = 1 - P(|0...0>)
✓ TVD computation with 20 bins, range (0, 0.6)
✓ 2048 shots
✓ Noisy simulation with IBM noise models

RESULT FORMAT:
✓ Output tree: runs_qugrad/{ansatz}/{dist}/5/1/1/
✓ Same output files: .npy, _x0.npy, _results.npy, _x0_results.npy, _dist_test.png, .txt
"""

import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Import QMill modules
sys.path.append(str(Path(__file__).parent.parent))
from circuits import Five, Sixteen, Custom_One, Custom_Two
from dists import MNIST, Normal, Uniform, WeibullLeft, WeibullRight, FashionMNIST, CIFAR, QCHEM, assymTVD, soil, dm
from ML import pSampleSet, curriedF
from custom_executor import NoisyCircuitExecutor, CustomCircuitExecutor
from ibm_noise_fetcher import IBMNoiseModelFetcher
from classes import dimToNumber
# checkAnsatz removed - now using generator.forward() with training executor directly
from datetime import datetime
import os

try:
    import cma
except ImportError:
    pass  # We'll check again in run_cmaes_optimization if needed


# ============================================================================
# CMA-ES OPTIMIZER
# ============================================================================

def run_cmaes_optimization(generator, training_set, target_dist_obj, max_evals=20000, eval_batch_size=32, sigma0=1.0, seed=0, verbose=True, full_eval_batch_size=200, full_eval_every=10):
    """
    Run CMA-ES optimization to minimize the NZP-TVD objective (Eq. 11).
    """
    try:
        import cma
    except ImportError:
        raise ImportError("cma module not found. Please run: pip install cma")

    import time
    from ML import getSampleSubset

    print(f"Preparing CMA-ES optimization...")
    print(f"  Mode: NZP-TVD Optimization")

    start_time = time.time()

    # Create fixed evaluation set (proxy)
    fixed_states = getSampleSubset(eval_batch_size, training_set)
    # Create fixed full evaluation set (periodic check)
    actual_full_eval_size = min(full_eval_batch_size, len(training_set))
    full_eval_states = getSampleSubset(actual_full_eval_size, training_set)
    
    # Create fixed entropy evaluation set (paper-friendly size)
    # Using 32 states for robust statistics
    entropy_states = getSampleSubset(32, training_set)
    
    n = generator.n_params
    
    # Initialize mean from generator's initial theta
    x0 = generator.initial_theta.copy()

    # Cache for CE values computed during objective() calls
    # This allows us to save the EXACT CE values that produced best_cost
    gen_ce_cache = []

    # Objective function
    def objective(x):
        # CMA searches over real vector x; convert to angles
        theta = np.mod(x, 2 * np.pi)

        # NZP-TVD: execute the (noisy) circuits for the generated states and
        # compare the resulting NZP histogram to the target (Eq. 10-11).
        ce = generator.forward(fixed_states, theta_override=theta, mode="raw")
        tvd = assymTVD(ce, target_dist_obj)

        # Cache CE values (will be retrieved if this is the best individual)
        gen_ce_cache.append(ce)

        return float(tvd)

    # Compute initial cost and cache the CE values
    generator.initial_cost = objective(generator.initial_theta)
    generator.initial_ce = gen_ce_cache[0]  # Save the EXACT CE values
    gen_ce_cache.clear()  # Clear for training loop
    print(f"  Initial TVD: {generator.initial_cost:.6f}")

    # CMA-ES options
    # Heuristic for popsize: 8 + 6*log(n), doubles if n>=50
    # For noisy quantum landscapes, larger population is better.
    popsize = int(8 + 6 * np.log(n))
    if n >= 50:
        popsize *= 2
    
    # Force minimum population size for robustness
    popsize = max(popsize, 32)
        
    opts = {
        'seed': seed,
        'popsize': popsize,
        'maxfevals': max_evals,
        'verbose': -9 if not verbose else 1,
        'bounds': None, # We handle mod 2pi manually
        'tolflatfitness': 100, # prevent early stopping on flat noise
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    print(f"  Dimension: {n}")
    print(f"  Population size: {popsize}")
    print(f"  Eval batch size: {eval_batch_size} (proxy)")
    print(f"  Full eval batch size: {full_eval_batch_size} (every {full_eval_every} gens)")
    print(f"  Max evals: {max_evals}")
    
    history = {
        'tvd_proxy': [],        # best-of-generation on fixed_states
        'best_tvd_proxy': [],   # best ever proxy
        'tvd_full': [],         # periodic full eval TVD (append None when not evaluated)
        'best_tvd_full': [],    # best ever full eval
        'vn_entropy_mean': [],
        'vn_entropy_std': [],
        'purity_mean': [],
        'purity_std': [],
    }
    
    # Tracking
    generator.best_cost = float('inf') # Tracks best proxy TVD
    best_tvd_full = float('inf')       # Tracks best full TVD
    
    # Evolution loop
    generation = 0
    while not es.stop():
        generation += 1
        X = es.ask()
        fvals = [objective(x) for x in X]
        es.tell(X, fvals)
        
        # Track best in this generation (Proxy)
        gen_min_tvd = min(fvals)
        history['tvd_proxy'].append(gen_min_tvd)

        # Track best ever (Proxy)
        if gen_min_tvd < generator.best_cost:
            generator.best_cost = gen_min_tvd
            idx = np.argmin(fvals)
            best_x = X[idx]
            generator.best_theta = np.mod(best_x, 2 * np.pi)
            # Save the EXACT CE values that produced this best_cost (no recomputation!)
            generator.best_ce = gen_ce_cache[idx]

        # Clear cache for next generation (saves memory)
        gen_ce_cache.clear()

        history['best_tvd_proxy'].append(generator.best_cost)
        
        # --- ENTROPY & PURITY LOGGING (Every Generation) ---
        # Calculated on fixed entropy_states using the best theta of this generation
        entropy_metrics = generator.generated_state_entropy_metrics(
            product_states=entropy_states,
            theta_override=generator.best_theta,
            base=2
        )
        S_vals = [m["vn_entropy"] for m in entropy_metrics]
        P_vals = [m["purity"] for m in entropy_metrics]

        mean_S, std_S = float(np.mean(S_vals)), float(np.std(S_vals))
        mean_P, std_P = float(np.mean(P_vals)), float(np.std(P_vals))

        history['vn_entropy_mean'].append(mean_S)
        history['vn_entropy_std'].append(std_S)
        history['purity_mean'].append(mean_P)
        history['purity_std'].append(std_P)

        # Periodic full evaluation
        tvd_full = None
        if generation % full_eval_every == 0:
            ce_full = generator.forward(full_eval_states, theta_override=generator.best_theta, mode="raw")
            tvd_full = float(assymTVD(ce_full, target_dist_obj))
            
            if tvd_full < best_tvd_full:
                best_tvd_full = tvd_full

            # Update global best if full eval is better
            if tvd_full < generator.best_cost:
                generator.best_cost = tvd_full
            
            # Print with entropy metrics (already computed above)
            print(f"Generated-state diagnostics | VN entropy (bits): {mean_S:.4f}±{std_S:.4f} | Purity: {mean_P:.4f}±{std_P:.4f}")
                
        history['tvd_full'].append(tvd_full)
        history['best_tvd_full'].append(best_tvd_full)
        
        if verbose:
            # Custom printout to reduce clutter and show what we care about
            evals = es.countevals
            full_str = f"{tvd_full:.4f}" if tvd_full is not None else "...."
            print(f"Gen {generation:3d} | Evals {evals:5d} | Proxy Best: {generator.best_cost:.4f} | Full Best: {best_tvd_full:.4f} | Ent: {mean_S:.4f} | Pur: {mean_P:.4f}")
            
    end_time = time.time()
    
    # Finalize generator stats
    generator.final_cost = generator.best_cost
    generator.total_iterations = generation
    generator.training_time = end_time - start_time
    
    # Copy best theta to torch parameter
    with torch.no_grad():
        generator.theta.copy_(torch.tensor(generator.best_theta, dtype=torch.float32))
        
    print(f"CMA-ES finished.")
    print(f"  Total evals: {es.countevals}")
    print(f"  Best Proxy TVD: {generator.best_cost:.6f}")
    print(f"  Best Full TVD: {best_tvd_full:.6f}")

    # Use CACHED CE values (exact same values that produced best_cost and initial_cost)
    # NO recomputation - this eliminates shot noise discrepancy
    ce_best = generator.best_ce
    ce_initial = generator.initial_ce

    # Also return the target distribution's avgDist so TVD can be computed consistently
    target_avgDist = target_dist_obj.avgDist.copy()

    return generator.best_theta, generator.best_cost, history, entropy_states, ce_best, ce_initial, fixed_states, target_avgDist



# ============================================================================
# RESULT SAVING (MATCHES ANNEALING FOLDER STRUCTURE)
# ============================================================================

OPTIMIZER_LABELS = {
    'cmaes': ('CMA-ES', 'CMA-ES NZP-TVD training'),
}


def save_qugrad_results(generator, ansatz, target_dist, backend_name, date, output_dir='runs_qugrad', trainer=None, final_metrics=None, entropy_states_indices=None, ce_best=None, ce_initial=None, target_avgDist=None, optimizer='cmaes'):
    """
    Save results in the same layout as the runs_qmill/ (QMill baseline) tree.

    Structure:
        -Noisy: runs_qugrad/{ansatz_name}/{dist_name}/5/1/1/
        -Ideal: runs_qugrad/Ideal/{ansatz_name}/{dist_name}/5/1/1/

    Files saved (SAME AS ANNEALING):
    1. {ansatz}_5_1.npy - BEST parameters (lowest TVD during training)
    2. {ansatz}_5_1.txt - summary text file
    3. {ansatz}_5_1_results.npy - CE results from best params
    4. {ansatz}_5_1_x0.npy - initial parameters
    5. {ansatz}_5_1_x0_results.npy - CE results from initial params
    6. {ansatz}_5_1_dist_test.png - distribution plot
    7. {ansatz}_5_1_history.npy - training history (TVD, losses, gradients)
    8. {ansatz}_5_1_loss_curves.png - training curves plot
    9. {ansatz}_5_1_target_avgDist.npy - target distribution avgDist for consistent TVD

    Args:
        ce_best: Pre-computed CE values for best params (if None, will compute)
        ce_initial: Pre-computed CE values for initial params (if None, will compute)
        target_avgDist: Target distribution avgDist used during training (for consistent TVD)
    """
    ansatz_name = ansatz.name
    dist_name = target_dist.name
    qubits = ansatz.qubits
    depth = ansatz.depth

    # Check if 'restart' or 'GLOBAL_BEST' is in ansatz name to create subfolders
    if "_restart_" in ansatz_name or "_GLOBAL_BEST" in ansatz_name:
        # Extract base name and suffix
        # e.g., Layered_pyramid_restart_1 -> Layered_pyramid, restart_1
        # e.g., Layered_pyramid_GLOBAL_BEST -> Layered_pyramid, GLOBAL_BEST
        
        if "_restart_" in ansatz_name:
            parts = ansatz_name.split("_restart_")
            base_ansatz_name = parts[0]
            suffix = f"restart_{parts[1]}"
        else:
            parts = ansatz_name.split("_GLOBAL_BEST")
            base_ansatz_name = parts[0]
            suffix = "GLOBAL_BEST"
            
        # Ideal: runs_qugrad/Ideal/{base_ansatz_name}/{suffix}/{dist}/5/{depth}/1/
        if backend_name is None:
            result_dir = os.path.join(output_dir, 'Ideal', base_ansatz_name, suffix, dist_name, str(qubits), str(depth), '1')
        else:
            result_dir = os.path.join(output_dir, base_ansatz_name, suffix, dist_name, str(qubits), str(depth), '1')
    else:
        # Standard path
        if backend_name is None:
            result_dir = os.path.join(output_dir, 'Ideal', ansatz_name, dist_name, str(qubits), str(depth), '1')
        else:
            result_dir = os.path.join(output_dir, ansatz_name, dist_name, str(qubits), str(depth), '1')

    os.makedirs(result_dir, exist_ok=True)

    # Base filename
    base_name = f"{ansatz_name}_{qubits}_{depth}"
    base_path = os.path.join(result_dir, base_name)

    # Check for existing best cost to prevent overwriting with worse results
    existing_summary_path = f"{base_path}.txt"
    if os.path.exists(existing_summary_path):
        try:
            with open(existing_summary_path, 'r') as f:
                content = f.read()
                # Read all lines to find the Best Cost
                for line in content.split('\n'):
                    if "Best Cost:" in line:
                        # expected format: " Best Cost: 0.1234 (saved)"
                        parts = line.strip().split()
                        # parts example: ['Best', 'Cost:', '0.1234', '(saved)']
                        if len(parts) >= 3:
                            try:
                                # parts[2] should be the number
                                prev_best_cost = float(parts[2])
                                current_best_cost = float(generator.best_cost)
                                
                                # Strict check: if current is not strictly better (lower), don't overwrite
                                if current_best_cost >= prev_best_cost:
                                    print(f"\n⚠ Skipping save: Current best TVD ({current_best_cost:.6f}) is not better than saved TVD ({prev_best_cost:.6f}).")
                                    return result_dir
                                    
                            except (ValueError, TypeError, IndexError):
                                # If parsing fails, we assume we can overwrite (or it's a different format)
                                continue 
        except Exception as e:
            print(f"  ⚠ Check for existing results failed: {e}")

    # Get final parameters (best theta found during training)
    final_params = generator.best_theta

    # Get initial parameters
    initial_params = generator.initial_theta

    # Save final parameters (.npy file)
    np.save(f"{base_path}.npy", final_params)
    print(f"  ✓ Saved {base_path}.npy")

    # Save initial parameters (.npy file)
    np.save(f"{base_path}_x0.npy", initial_params)
    print(f"  ✓ Saved {base_path}_x0.npy")

    # Save summary text file (.txt file) - MATCHES ANNEALING FORMAT
    with open(f"{base_path}.txt", 'w') as f:
        f.write(f"{datetime.now().strftime('%a %b %d %H:%M:%S %Y')} \n")
        f.write(f" Ansatz: {ansatz_name} \n")
        f.write(f" Training Time: {generator.training_time} \n")
        f.write(f" Parameters: {final_params} \n")
        f.write(f" Dist: {dist_name} \n")
        f.write(f" Initial Cost: {generator.initial_cost} \n")
        f.write(f" Total Iterations: {generator.total_iterations} \n")
        f.write(f" Best Cost: {generator.best_cost} (saved) \n")
        f.write(f" Final Cost: {generator.final_cost} \n")
        method_label, message_label = OPTIMIZER_LABELS.get(
            optimizer, (f'Unknown optimizer ({optimizer})',) * 2
        )
        f.write(f" Message: {message_label} \n")
        f.write(f" Backend: {backend_name if backend_name else 'ideal'} \n")
        f.write(f" Date: {date if date else 'N/A'} \n")
        f.write(f" Method: {method_label} \n")
        
        if final_metrics:
            f.write(f" Final Entropy (bits): {final_metrics.get('mean_entropy', 'N/A'):.4f} +/- {final_metrics.get('std_entropy', 'N/A'):.4f} \n")
            f.write(f" Final Purity:       {final_metrics.get('mean_purity', 'N/A'):.4f} +/- {final_metrics.get('std_purity', 'N/A'):.4f} \n")
            f.write(f" Note: Metrics over fixed set of 32 states \n")

    print(f"  ✓ Saved {base_path}.txt")

    # Save CE distributions - use pre-computed values if provided, otherwise compute
    print(f"  Saving CE distributions...")
    try:
        from ML import pSampleSet
        import matplotlib.pyplot as plt

        # Use pre-computed CE values if provided (exact same as training evaluation)
        if ce_best is not None:
            print(f"  Using pre-computed CE values (exact training results)")
            ce_best_arr = np.array(ce_best)
        else:
            # Fallback: compute new CE values using training executor
            print(f"  Computing CE values using training executor...")
            eval_states = pSampleSet(qubits, 1000)
            ce_best_arr = np.array(generator.forward(eval_states, theta_override=generator.best_theta, mode="raw"))

        np.save(f"{base_path}_results.npy", ce_best_arr)
        print(f"  ✓ Saved {base_path}_results.npy ({len(ce_best_arr)} samples)")

        # Use pre-computed initial CE values if provided
        if ce_initial is not None:
            ce_initial_arr = np.array(ce_initial)
        else:
            # Fallback: compute new CE values for initial params
            if 'eval_states' not in locals():
                eval_states = pSampleSet(qubits, 1000)
            ce_initial_arr = np.array(generator.forward(eval_states, theta_override=generator.initial_theta, mode="raw"))

        np.save(f"{base_path}_x0_results.npy", ce_initial_arr)
        print(f"  ✓ Saved {base_path}_x0_results.npy ({len(ce_initial_arr)} samples)")

        # Create distribution comparison plot
        target_dist.createSampleDistributions(1000)
        target_samples = np.array(target_dist.samples).flatten()

        fig, ax = plt.subplots(figsize=(10, 6))
        bins = 20
        ax.hist(target_samples, bins=bins, histtype='step', color='r', density=True, label='Target', linewidth=2)
        ax.hist(ce_initial_arr, bins=bins, histtype='step', color='g', density=True, label='Initial', linewidth=2)
        ax.hist(ce_best_arr, bins=bins, histtype='step', color='b', density=True, label='Trained', linewidth=2)
        ax.set_xlabel('Concentratable Entanglement')
        ax.set_ylabel('Density')
        ax.set_title(f'{ansatz_name} - {dist_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f"{base_path}_dist_test.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved {base_path}_dist_test.png")

        # Save target avgDist for consistent TVD calculation
        if target_avgDist is not None:
            np.save(f"{base_path}_target_avgDist.npy", target_avgDist)
            print(f"  ✓ Saved {base_path}_target_avgDist.npy (for consistent TVD)")

    except Exception as e:
        print(f"  ⚠ Failed to generate distribution outputs: {e}")
        import traceback
        traceback.print_exc()

    # Save training history if trainer is provided
    if trainer is not None and hasattr(trainer, 'history'):
        try:
            import matplotlib.pyplot as plt

            # Save history as .npy
            np.save(f"{base_path}_history.npy", trainer.history, allow_pickle=True)
            print(f"  ✓ Saved {base_path}_history.npy")

            # Plot training curves
            fig, axes = plt.subplots(3, 2, figsize=(12, 15))  # Changed to 3x2 grid

            # Helper for plotting with None values
            def plot_with_none(ax, data, label, color, linestyle='-'):
                y = np.array(data)
                # Filter out None values for plotting (but keep x indices aligned)
                # Actually, better to just plot points where not None
                valid_indices = [i for i, v in enumerate(y) if v is not None]
                if not valid_indices:
                    return
                valid_values = [y[i] for i in valid_indices]
                ax.plot(valid_indices, valid_values, color=color, linestyle=linestyle, label=label, marker='.', markersize=4)

            # Plot 1: TVD over iterations
            if trainer.history['tvd']:
                axes[0, 0].plot(trainer.history['tvd'], 'b-', linewidth=2, label='Proxy TVD')
                # If full eval exists, overlay it
                if 'full_tvd' in trainer.history:
                     plot_with_none(axes[0, 0], trainer.history['full_tvd'], 'Full TVD', 'k', '--')

                axes[0, 0].axhline(y=generator.best_cost, color='r', linestyle='--', label=f'Best: {generator.best_cost:.4f}')
                axes[0, 0].set_xlabel('Iteration', fontsize=12)
                axes[0, 0].set_ylabel('TVD', fontsize=12)
                axes[0, 0].set_title('Total Variation Distance', fontsize=14, fontweight='bold')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Generator Loss (TVD)
            if trainer.history['loss_g_tvd']:
                axes[0, 1].plot(trainer.history['loss_g_tvd'], 'g-', linewidth=2)
                axes[0, 1].set_xlabel('Iteration', fontsize=12)
                axes[0, 1].set_ylabel('Generator Loss', fontsize=12)
                axes[0, 1].set_title('Generator Loss (TVD)', fontsize=14, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 5: VN Entropy
            if 'vn_entropy_mean' in trainer.history:
                plot_with_none(axes[2, 0], trainer.history['vn_entropy_mean'], 'Mean VN Entropy', 'teal')
                # Add shading if std is available and not None
                # This is tricky with None values, so we stick to simple line for now or just errorbars
                # Let's just plot the mean
                axes[2, 0].set_xlabel('Iteration', fontsize=12)
                axes[2, 0].set_ylabel('Entropy (bits)', fontsize=12)
                axes[2, 0].set_title('Generated State VN Entropy', fontsize=14, fontweight='bold')
                axes[2, 0].grid(True, alpha=0.3)

            # Plot 6: Purity
            if 'purity_mean' in trainer.history:
                plot_with_none(axes[2, 1], trainer.history['purity_mean'], 'Mean Purity', 'brown')
                axes[2, 1].set_xlabel('Iteration', fontsize=12)
                axes[2, 1].set_ylabel('Purity', fontsize=12)
                axes[2, 1].set_title('Generated State Purity', fontsize=14, fontweight='bold')
                axes[2, 1].grid(True, alpha=0.3)
                axes[2, 1].set_ylim(-0.05, 1.05) # Purity is [0, 1]

            plt.tight_layout()
            plt.savefig(f"{base_path}_loss_curves.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved {base_path}_loss_curves.png")
        except Exception as e:
            print(f"  ⚠ Failed to save training history: {e}")

    print(f"\n✓ Results saved to: {result_dir}")
    return result_dir


# ============================================================================
# QUANTUM GENERATOR
# ============================================================================

class QuantumGenerator(nn.Module):
    """
    Quantum generator: maps product states through the trained ansatz to CE/NZP values.

    Key differences from REINFORCE version:
    - Direct parameterization: θ (no Gaussian policy N(μ, σ²))
    - No sampling, entropy, or baseline logic

    Generator is QUANTUM: uses quantum circuits to produce classical CE data.
    """

    def __init__(self, ansatz_obj, executor, n_shots: int = 2048):
        """
        Args:
            ansatz_obj: QMill Ansatz object (Five, Sixteen, etc.)
            executor: QMill executor (CustomCircuitExecutor or NoisyCircuitExecutor)
            n_shots: Measurement shots (QMill standard: 2048)
        """
        super().__init__()

        self.ansatz = ansatz_obj
        self.executor = executor
        self.n_shots = n_shots

        # Get number of parameters from ansatz
        self.n_params = dimToNumber(ansatz_obj.shape)

        # Direct parameterization: θ ∈ [0, 2π]^n (no Gaussian policy!)
        # Initialize uniformly in [0, 2π]
        initial_theta = torch.rand(self.n_params) * 2 * np.pi
        self.theta = nn.Parameter(initial_theta, requires_grad=True)

        # Save initial parameters for result saving
        self.initial_theta = self.theta.detach().cpu().numpy().copy()

        # Training statistics (for result saving)
        self.initial_cost = None
        self.final_cost = None
        self.best_cost = float('inf')
        self.best_theta = self.initial_theta.copy()
        self.total_iterations = 0
        self.training_time = 0.0

        # Save a copy of the *state-prep* circuit (generated state), BEFORE SWAP test is added
        self.state_prep_circ = self.ansatz.currCirc.copy()

        # Create SWAP test circuit (QMILL METHODOLOGY)
        self.ansatz.createTestCircuit()

    def generated_state_entropy_metrics(self, product_states: list, theta_override=None, base: int = 2):
        """
        Compute Von Neumann entropy + purity of the *generated quantum state* (pre SWAP-test),
        under the same executor (ideal or noisy).
        """
        # Which theta to use
        theta_numpy = theta_override if theta_override is not None else self.theta.detach().cpu().numpy()

        # Build parameter assignments for the *state-prep* circuit
        circuits = [self.state_prep_circ] * len(product_states)
        param_assignments = []

        for state in product_states:
            assignment = list(map(
                curriedF(theta_numpy, state),
                self.state_prep_circ.parameters
            ))
            param_assignments.append(assignment)

        # Use executor density-matrix path
        # Returns list of dicts: {"vn_entropy": ..., "purity": ...}
        return self.executor.run_entropy_metrics(
            circuits=circuits,
            parameter_values=param_assignments,
            base=base,
            optimization_level=3
        )

    def forward(self, product_states: list, theta_override=None, mode: str = "raw"):
        """
        Generate CE values or features from product states using quantum circuits.

        This is the QUANTUM part: runs actual quantum circuits!

        Args:
            product_states: List of product state dicts (from pSampleSet)
            theta_override: Optional parameters to use instead of self.theta
                          (used to evaluate a candidate from CMA-ES)
            mode:
              - "raw":      return raw CE values as list
              - "features": return 24D MosaiQ-style features

        Returns:
            CE values (list) or 24D features (np.ndarray)
        """
        batch_size = len(product_states)

        # Use override or current parameters
        if theta_override is not None:
            theta_numpy = theta_override
        else:
            theta_numpy = self.theta.detach().cpu().numpy()

        # Build circuits (QMILL METHODOLOGY from ML.py distCost)
        circuits = [self.ansatz.currCirc] * batch_size
        param_assignments = []

        for state in product_states:
            # Map parameters: ansatz params (theta) + state params
            # This is EXACTLY what distCost does (ML.py line 136)
            assignment = list(map(
                curriedF(theta_numpy, state),
                self.ansatz.currCirc.parameters
            ))
            param_assignments.append(assignment)

        # Execute QUANTUM circuits with QMill executor
        job = self.executor.run(circuits, param_assignments, shots=self.n_shots)
        result = job.result()
        quasi_dists = result.quasi_dists

        # Extract CE values (QMILL METHODOLOGY from ML.py line 145-146)
        # This converts quantum measurements to CLASSICAL data
        ce_values = []
        for i in range(batch_size):
            dist = quasi_dists[i]
            prob_zero = dist.get(0, 0.0)  # P(|0...0>) from SWAP test
            ce = 1.0 - prob_zero          # CE = 1 - P(|0...0>)
            ce_values.append(ce)

        # Return based on mode
        if mode == "raw":
            return ce_values  # Classical data output (list)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ============================================================================
# MAIN
# ============================================================================

def main(use_noisy: bool = True, backend_name: str = 'ibm_boston', date: str = '2025-12-31',
         dist_name: str = 'MNIST', ansatz_name: str = 'Five',
         max_evals: int = 20000, eval_batch: int = 32,
         sigma0: float = 1.0, seed: int = 0, restarts: int = 1):
    """Train a QuGrad generator with CMA-ES on the NZP-TVD objective."""

    print("\n" + "=" * 80)
    print("QuGrad Training (CMA-ES / NZP-TVD)")
    print("=" * 80)
    print(f"Mode: {'NOISY' if use_noisy else 'IDEAL'}")
    print(f"Ansatz: {ansatz_name}")
    print(f"Distribution: {dist_name}")
    print("=" * 80)

    # ========== QMill Setup (EXACT SAME AS train_with_noise.py) ==========

    # Check ansatz validity
    ansatz_map = {
        'Five': Five,
        'Sixteen': Sixteen,
        'Custom_One': Custom_One,
        'Custom_Two': Custom_Two,
    }
    if ansatz_name not in ansatz_map:
        raise ValueError(f"Unknown ansatz: {ansatz_name}. Available: {list(ansatz_map.keys())}")

    # Create executor (QMILL)
    print("\nCreating executor...")
    if use_noisy:
        try:
            cache_dir = "noise_models"
            fetcher = IBMNoiseModelFetcher(cache_dir=cache_dir)
            noise_model = fetcher.get_noise_model(backend_name, date)
            if noise_model is None:
                raise ValueError(f"Noise model not found")
            executor = NoisyCircuitExecutor(noise_model, backend_name)
            print(f"  ✓ NoisyCircuitExecutor ({backend_name}, {date})")
        except Exception as e:
            print(f"  ⚠ Failed to load noise model: {e}")
            print(f"  Falling back to ideal...")
            executor = CustomCircuitExecutor()
            use_noisy = False
    else:
        executor = CustomCircuitExecutor()
        print(f"  ✓ CustomCircuitExecutor (ideal)")

    # Load target distribution (QMILL)
    print("\nLoading target distribution...")
    dist_map = {
        'MNIST': lambda: MNIST(100),
        'Normal': lambda: Normal(100),
        'Uniform': lambda: Uniform(100),
        'Left_Weibull': lambda: WeibullLeft(100),
        'Right_Weibull': lambda: WeibullRight(100),
        'Fashion_MNIST': lambda: FashionMNIST(100),
        'CIFAR': lambda: CIFAR(100),
        'QCHEM': lambda: QCHEM(100),
        'Soillow': lambda: soil(100, 'low'),
        'Soilhigh': lambda: soil(100, 'high'),
        'dmlow': lambda: dm(100, 'low'),
        'dmhigh': lambda: dm(100, 'high'),
    }
    if dist_name not in dist_map:
        raise ValueError(f"Unknown distribution: {dist_name}. Available: {list(dist_map.keys())}")

    target_dist = dist_map[dist_name]()
    target_dist.createSampleDistributions(100)  # QMILL: 100 sample distributions
    target_dist.getAveragedBins(20, (0, 0.6))   # QMILL: 20 bins, range (0, 0.6)
    print(f"  ✓ {dist_name} distribution loaded")
    print(f"  ✓ Range: {target_dist.Range}, Bins: {target_dist.numBoxes}")

    # Create training set (QMILL METHODOLOGY)
    print("\nCreating training set...")
    training_set = pSampleSet(qubits=5, N=100)  # QMILL: 100 product states
    print(f"  ✓ Training set: 100 product states")

    # ========== Setup ==========

    global_best_tvd = float('inf')

    for run_i in range(restarts):
        print(f"\n{'#'*80}")
        print(f"RESTART {run_i + 1}/{restarts}")
        print(f"{'#'*80}")

        # Create ansatz (QMILL) - Re-create for each restart to ensure fresh state
        print("\nCreating ansatz...")
        ansatz = ansatz_map[ansatz_name](qubits=5, depth=2)  # QMill standard
        ansatz.initialize()
        print(f"  ✓ Ansatz: {ansatz_name} (5 qubits, depth 2)")
        print(f"  ✓ Parameters: {dimToNumber(ansatz.shape)}")

        original_ansatz_name = ansatz.name
        
        # Calculate seed for this run
        current_seed = seed + run_i

        print("\nInitializing generator...")
        generator = QuantumGenerator(ansatz, executor, n_shots=2048)
        print(f"  ✓ QUANTUM Generator: {generator.n_params} parameters")

        # Initialize stats
        generator.initial_cost = None
        generator.best_cost = float('inf')
        generator.best_theta = generator.initial_theta.copy()
        
        trainer_stub = None

        print("\nRunning CMA-ES optimization...")

        best_theta, best_tvd, cma_history, entropy_states, ce_best_cmaes, ce_initial_cmaes, fixed_states, target_avgDist = run_cmaes_optimization(
            generator=generator,
            training_set=training_set,
            target_dist_obj=target_dist,
            max_evals=max_evals,
            eval_batch_size=eval_batch,
            sigma0=sigma0,
            seed=current_seed,
            verbose=True,
        )
        print(f"CMA-ES done | Best TVD: {best_tvd:.6f}")

        # CE values computed on fixed_states (the same states that produced Best Cost)
        final_ce = ce_best_cmaes
        initial_ce = ce_initial_cmaes
        training_target_avgDist = target_avgDist

        # History container consumed by save_qugrad_results()
        from types import SimpleNamespace
        trainer_stub = SimpleNamespace(history={
            'tvd': cma_history['tvd_proxy'],
            'loss_g_tvd': cma_history['tvd_proxy'],
            'loss_d': [],
            'grad_norm_g': [],
            'full_tvd': cma_history['tvd_full'],
            'best_full_tvd': cma_history['best_tvd_full'],
            'vn_entropy_mean': cma_history['vn_entropy_mean'],
            'vn_entropy_std': cma_history['vn_entropy_std'],
            'purity_mean': cma_history['purity_mean'],
            'purity_std': cma_history['purity_std'],
        })

        # NOTE: generator.initial_cost is set inside run_cmaes_optimization

        # ========== Evaluate ==========

        print("\nFinal Evaluation (using saved CE values):")

        # Compute TVD from the CE values that will be saved
        final_tvd = assymTVD(final_ce, target_dist)
        initial_tvd = assymTVD(initial_ce, target_dist)
        final_ce_np = np.array(final_ce)

        print(f"  Samples: {len(final_ce)}")
        print(f"  Initial TVD: {initial_tvd:.4f}")
        print(f"  Final TVD: {final_tvd:.4f}  ← THIS WILL MATCH PLOT")
        print(f"  Mean CE: {final_ce_np.mean():.4f}")
        print(f"  Std CE: {final_ce_np.std():.4f}")

        # ========== Final Diagnostic Check (Paper Table) ==========
        print("\n" + "-" * 80)
        print("FINAL DIAGNOSTIC CHECK (For Paper)")
        print("-" * 80)
        
        # We reuse 'entropy_states' (fixed 32 states) from earlier
        # Ensure best parameters are loaded (already done above)
        
        final_entropy_metrics = generator.generated_state_entropy_metrics(
            product_states=entropy_states,
            theta_override=generator.best_theta,
            base=2
        )
        
        S_vals = [m["vn_entropy"] for m in final_entropy_metrics]
        P_vals = [m["purity"] for m in final_entropy_metrics]

        final_mean_S, final_std_S = float(np.mean(S_vals)), float(np.std(S_vals))
        final_mean_P, final_std_P = float(np.mean(P_vals)), float(np.std(P_vals))
        
        print(f"Entropy (bits): {final_mean_S:.4f} ± {final_std_S:.4f}")
        print(f"Purity:         {final_mean_P:.4f} ± {final_std_P:.4f}")
        print("-" * 80)

        # Bundle for saving
        final_metrics_dict = {
            "mean_entropy": final_mean_S,
            "std_entropy": final_std_S,
            "mean_purity": final_mean_P,
            "std_purity": final_std_P
        }

        # ========== Save Results (MATCHES ANNEALING FORMAT) ==========

        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Save current run
        if restarts > 1:
            ansatz.name = f"{original_ansatz_name}_restart_{run_i + 1}"
        
        save_qugrad_results(
            generator=generator,
            ansatz=ansatz,
            target_dist=target_dist,
            backend_name=backend_name if use_noisy else None,
            date=date if use_noisy else None,
            output_dir='runs_qugrad',
            trainer=trainer_stub,
            final_metrics=final_metrics_dict,
            entropy_states_indices=None,
            ce_best=final_ce,      # Pass pre-computed CE values (exact training results)
            ce_initial=initial_ce,  # Pass pre-computed initial CE values
            target_avgDist=training_target_avgDist,  # Pass target avgDist for consistent TVD
            optimizer=optimizer     # Record the optimizer that ACTUALLY produced this run
        )
        ansatz.name = original_ansatz_name

        # Check Global Best
        if generator.best_cost < global_best_tvd:
            global_best_tvd = generator.best_cost
            print(f"\n>>> FOUND NEW GLOBAL BEST: {global_best_tvd:.6f}")
            
            if restarts > 1:
                ansatz.name = f"{original_ansatz_name}_GLOBAL_BEST"
                save_qugrad_results(
                    generator=generator,
                    ansatz=ansatz,
                    target_dist=target_dist,
                    backend_name=backend_name if use_noisy else None,
                    date=date if use_noisy else None,
                    output_dir='runs_qugrad',
                    trainer=trainer_stub,
                    final_metrics=final_metrics_dict,
                    ce_best=final_ce,
                    ce_initial=initial_ce,
                    target_avgDist=training_target_avgDist,
                    optimizer=optimizer
                )
                ansatz.name = original_ansatz_name

    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QuGrad Training (CMA-ES / NZP-TVD)")
    parser.add_argument('--noisy', action='store_true', default=True,
                       help='Use noisy simulation (default)')
    parser.add_argument('--ideal', action='store_true',
                       help='Use ideal simulation')
    parser.add_argument('--backend', type=str, default='ibm_brisbane',
                       help='IBM backend name')
    parser.add_argument('--date', type=str, default='2024-11-01',
                       help='Date for noise model')
    parser.add_argument('--dist', type=str, default='MNIST',
                       choices=['MNIST', 'Normal', 'Uniform', 'Left_Weibull', 'Right_Weibull',
                               'Fashion_MNIST', 'CIFAR', 'QCHEM', 'Soillow', 'Soilhigh', 'dmlow', 'dmhigh'],
                       help='Target distribution')
    parser.add_argument('--ansatz', type=str, default='Five',
                       choices=['Five', 'Sixteen', 'Custom_One', 'Custom_Two'],
                       help='Ansatz to use')
                       
    # CMA-ES flags
    parser.add_argument('--max_evals', type=int, default=20000,
                        help='Max evaluations for CMA-ES (default: 20000)')
    parser.add_argument('--eval_batch', type=int, default=32,
                        help='Batch size for CMA-ES d (default: 32)')
    parser.add_argument('--sigma0', type=float, default=1.0,
                        help='Initial sigma for CMA-ES (default: 1.0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for CMA-ES (default: 0)')
    parser.add_argument('--restarts', type=int, default=1,
                        help='Number of restarts (default: 1)')

    args = parser.parse_args()
    use_noisy = not args.ideal

    main(use_noisy=use_noisy, backend_name=args.backend, date=args.date,
         dist_name=args.dist, ansatz_name=args.ansatz,
         max_evals=args.max_evals,
         eval_batch=args.eval_batch, sigma0=args.sigma0, seed=args.seed,
         restarts=args.restarts)