import numpy as np
from circuits import Custom_One, Sixteen, Five, Custom_Two
from variety import getVarietyDists
from custom_executor import CustomCircuitExecutor
from dists import *

"""
NOTE: The output of this script was saved in swapres.txt manually; just change the print path to do this automatically
"""


def analyze_ansatz_dist_pair(ansatz_class, dist_class, params_file):
    """Analyze a single ansatz-distribution pair"""
    ansatz = ansatz_class(5, 1)  # All use 5 qubits, depth 1
    dist = dist_class(100)

    params = np.load(params_file)

    results, varieties = getVarietyDists(ansatz, params, dist, sample=1000)
    
    range_stats = []
    for swap_results, ce_range in varieties:
        if len(swap_results) > 0:
            range_stats.append({
                'CE_Range': f"{ce_range[0]:.2f}-{ce_range[1]:.2f}",
                'Mean_Similarity': np.mean(swap_results),
                'Std_Similarity': np.std(swap_results),
                'Num_Pairs': len(swap_results)
            })
    
    return range_stats


ansatzes = {
    'Custom_One': Custom_One,
    'Custom_Two': Custom_Two,
    'Sixteen': Sixteen,
    'Five': Five
}

distributions = {
    'Uniform': ('Uniform', Uniform),
    'Normal': ('Normal', Normal),
    'WeibullLeft': ('Left Weibull', WeibullLeft),
    'WeibullRight': ('Right Weibull', WeibullRight),
    'MNIST': ('MNIST', MNIST),
    'FashionMNIST': ('Fashion MNIST', FashionMNIST),
    'CIFAR': ('CIFAR', CIFAR),
    'QCHEM': ('QCHEM', QCHEM)
}

# Collect results
all_results = {}
for ansatz_name, ansatz_class in ansatzes.items():
    all_results[ansatz_name] = {}
    for code_name, (dir_name, dist_class) in distributions.items():
        try:
            params_file = f"./runs_qmill/{ansatz_name}/{dir_name}/5/1/1/{ansatz_name}_5_1.npy"
            stats = analyze_ansatz_dist_pair(ansatz_class, dist_class, params_file)
            all_results[ansatz_name][dir_name] = stats
            print(f"Processed {ansatz_name} - {dir_name}")
            print(f"Found {len(stats)} non-empty CE ranges")
        except Exception as e:
            print(f"Error processing {ansatz_name} - {dir_name}: {str(e)}")
            continue

# Print summary of results
for ansatz_name in all_results:
    print(f"\n{ansatz_name}:")
    for dist_name in all_results[ansatz_name]:
        stats = all_results[ansatz_name][dist_name]
        print(f"  {dist_name}: {len(stats)} ranges")
        if stats:  # Print details of first few ranges
            for range_stat in stats[:3]:
                print(f"    {range_stat['CE_Range']}: "
                      f"Mean={range_stat['Mean_Similarity']:.3f}, "
                      f"Std={range_stat['Std_Similarity']:.3f}, "
                      f"Pairs={range_stat['Num_Pairs']}")