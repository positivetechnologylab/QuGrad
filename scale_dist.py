import numpy as np
import matplotlib.pyplot as plt

def scale_distribution(input_file, output_file=None, target_min=0.0, target_max=1.4):
    """
    Scale the concentratable entanglement values from an input .npy file to a target range.
    
    Args:
        input_file (str): Path to input .npy file containing CE values
        output_file (str, optional): Path to save scaled values. If None, uses input_file_scaled.npy
        target_min (float): Minimum value of target range (default 0.0)
        target_max (float): Maximum value of target range (default 0.4)
    
    Returns:
        numpy.ndarray: Scaled values
    """
    # Load the data
    data = np.load(input_file)
    
    # Get current min and max
    current_min = np.min(data)
    current_max = np.max(data)
    
    # Scale the data to [0,1] first
    normalized = (data - current_min) / (current_max - current_min)
    
    scaled = normalized * (target_max - target_min) + target_min
    
    # Save if output file specified
    if output_file is None:
        # Create default output filename by inserting '_scaled' before .npy
        base = input_file.rsplit('.', 1)[0]
        output_file = f"{base}_scaled.npy"
    
    np.save(output_file, scaled)
    
    # Print statistics for verification
    print(f"Original distribution stats:")
    print(f"Min: {current_min:.6f}")
    print(f"Max: {current_max:.6f}")
    print(f"Mean: {np.mean(data):.6f}")
    print(f"Std: {np.std(data):.6f}\n")
    
    print(f"Scaled distribution stats:")
    print(f"Min: {np.min(scaled):.6f}")
    print(f"Max: {np.max(scaled):.6f}")
    print(f"Mean: {np.mean(scaled):.6f}")
    print(f"Std: {np.std(scaled):.6f}")
    
    return scaled

def plot_distributions(original_data, scaled_data, save_path=None):
    """
    Plot the original and scaled distributions side by side for comparison.
    
    Args:
        original_data (numpy.ndarray): Original CE values
        scaled_data (numpy.ndarray): Scaled CE values
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    plt.hist(original_data, bins=50, density=True, alpha=0.7, color='blue')
    plt.title('Original Distribution')
    # Then scale to target range
    plt.xlabel('Concentratable Entanglement')
    plt.ylabel('Density')
    
    # Scaled distribution
    plt.subplot(1, 2, 2)
    plt.hist(scaled_data, bins=50, density=True, alpha=0.7, color='green')
    plt.title('Scaled Distribution')
    plt.xlabel('Concentratable Entanglement')
    plt.ylabel('Density')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Scale the distribution
# scaled_data = scale_distribution('mnist_dist.npy', 'mnist_dist_scaled.npy')
# scaled_data = scale_distribution('fashionmnist_dist.npy', 'fashionmnist_dist_scaled.npy')
# scaled_data = scale_distribution('cifar_dist.npy', 'cifar_dist_scaled.npy')
scaled_data = scale_distribution('qchem_dist.npy', 'qchem_dist_scaled.npy')

# Load original data for comparison
original_data = np.load('qchem_dist.npy')

# Plot both distributions
plot_distributions(original_data, scaled_data, 'distribution_comparison.png')