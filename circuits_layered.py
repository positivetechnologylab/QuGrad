"""
Layered/Heterogeneous Ansatz Architectures

This module allows you to stack DIFFERENT ansatzes in layers,
similar to how CNNs stack Conv → Pool → Dense layers.

Example:
    Layer 1: Custom_One (exploits entanglement structure A)
    Layer 2: Sixteen (exploits entanglement structure B)
    Layer 3: Custom_Two (exploits entanglement structure C)

Each layer can have different connectivity patterns and gate types!
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from circuits import Five, Sixteen, Custom_One, Custom_Two
from classes import swap_test


class LayeredAnsatz:
    """
    Composable ansatz that stacks multiple different ansatzes sequentially.

    Think of this like:
        - Classical NN: Input → Dense(64) → ReLU → Dense(32) → ReLU → Output
        - Quantum: Input → Custom_One → Sixteen → Custom_Two → Measurement

    Each layer can be a DIFFERENT ansatz with different entanglement structure!
    """

    def __init__(self, qubits: int, layer_configs: list):
        """
        Args:
            qubits: Number of qubits
            layer_configs: List of (AnsatzClass, depth) tuples
                Example: [(Custom_One, 1), (Sixteen, 2), (Custom_Two, 1)]
        """
        self.qubits = qubits
        self.layer_configs = layer_configs
        self.layers = []
        self.name = "Layered_" + "_".join([cls.__name__ for cls, _ in layer_configs])

        # Create each layer
        for ansatz_class, depth in layer_configs:
            layer = ansatz_class(qubits=qubits, depth=depth)
            self.layers.append(layer)

        # Initialize will be called separately
        self.currCirc = None
        self.shape = None

    def initialize(self):
        """Initialize all layers and compose them into one circuit."""
        from qiskit.circuit import Parameter

        # Initialize each layer
        for layer in self.layers:
            layer.initialize()

        # Create a fresh circuit with all qubits
        self.currCirc = QuantumCircuit(self.qubits)

        # Track parameter offset for naming
        param_offset = 0
        all_params = []

        # Compose each layer with unique parameter names
        for layer_idx, layer in enumerate(self.layers):
            layer_circuit = layer.currCirc.copy()
            layer_params = sorted(layer_circuit.parameters, key=lambda p: str(p))
            n_layer_params = len(layer_params)

            # Create unique parameters for this layer (integers for QMill compatibility)
            unique_params = [
                Parameter(f"{param_offset + i}")
                for i in range(n_layer_params)
            ]
            all_params.extend(unique_params)

            # Create parameter mapping
            param_map = dict(zip(layer_params, unique_params))

            # Assign unique parameters and compose
            layer_circuit_bound = layer_circuit.assign_parameters(param_map)
            self.currCirc.compose(layer_circuit_bound, inplace=True)

            param_offset += n_layer_params

        # Calculate total parameter shape
        total_params = len(all_params)
        self.shape = (total_params,)  # Flatten to 1D
        self.depth = sum(layer.depth for layer in self.layers)
        
        # Create ref params for swap test (QMill standard)
        self.refpList = [(Parameter(f'theta_{i}'), Parameter(f'phi_{i}'), 
                          Parameter(f'lmbda_{i}')) for i in range(self.qubits)]

        print(f"Layered ansatz initialized:")
        for i, (layer, (ansatz_class, depth)) in enumerate(zip(self.layers, self.layer_configs)):
            print(f"  Layer {i+1}: {ansatz_class.__name__} (depth={depth}, params={np.prod(layer.shape)})")
        print(f"  Total parameters: {total_params}")

        return self

    def createTestCircuit(self):
        """Create SWAP test circuit (QMill interface)."""
        qubits = self.qubits
        
        # Store current ansatz as instruction
        ansatz_instr = self.currCirc.to_instruction()
        
        # Create new circuit for SWAP test
        self.currCirc = QuantumCircuit(3*qubits, qubits)
        
        # Reference unitaries
        URef = QuantumCircuit(self.qubits)
        for i in range(self.qubits):
            t, p, l = self.refpList[i]
            URef.u(t, p, l, i)
        
        # Build SWAP test
        self.currCirc.append(URef, range(qubits, 2*qubits))
        self.currCirc.append(URef, range(2*qubits, 3*qubits))
        
        # Append ansatz instructions - no need to specify qubits if they match instruction width
        # The ansatz_instr is already defined for 'qubits' width
        self.currCirc.append(ansatz_instr, range(qubits, 2*qubits))
        self.currCirc.append(ansatz_instr, range(2*qubits, 3*qubits))
        swap_test(self.currCirc)


class AlternatingAnsatz:
    """
    Ansatz that alternates between two different ansatz types.

    Example: [Custom_One, Sixteen, Custom_One, Sixteen, Custom_One]

    This creates a pattern where you alternate between different
    entanglement structures, potentially capturing richer correlations.
    """

    def __init__(self, qubits: int, ansatz_a, ansatz_b, num_layers: int):
        """
        Args:
            qubits: Number of qubits
            ansatz_a: First ansatz class (e.g., Custom_One)
            ansatz_b: Second ansatz class (e.g., Sixteen)
            num_layers: Total number of layers
        """
        self.qubits = qubits
        self.num_layers = num_layers
        self.name = f"Alternating_{ansatz_a.__name__}_{ansatz_b.__name__}_x{num_layers}"

        # Create alternating pattern
        layer_configs = []
        for i in range(num_layers):
            ansatz_class = ansatz_a if i % 2 == 0 else ansatz_b
            layer_configs.append((ansatz_class, 1))  # depth=1 per layer

        # Use LayeredAnsatz under the hood
        self.layered = LayeredAnsatz(qubits, layer_configs)

    def initialize(self):
        """Initialize the alternating ansatz."""
        self.layered.initialize()
        self.currCirc = self.layered.currCirc
        self.shape = self.layered.shape
        self.layers = self.layered.layers
        self.depth = self.layered.depth
        return self

    def createTestCircuit(self):
        """Delegate test circuit creation to underlying layered ansatz."""
        self.layered.createTestCircuit()
        self.currCirc = self.layered.currCirc


class PyramidAnsatz:
    """
    Pyramid-style ansatz where depth decreases per layer.

    Example with 3 layers:
        Layer 1: Sixteen (depth=3) - widest, most expressive
        Layer 2: Custom_One (depth=2) - narrower
        Layer 3: Five (depth=1) - narrowest, final refinement

    Inspired by ResNet/VGG pyramid architectures in classical CNNs.
    """

    def __init__(self, qubits: int, ansatz_classes: list, depths: list):
        """
        Args:
            qubits: Number of qubits
            ansatz_classes: List of ansatz classes [Wide, Medium, Narrow]
            depths: Corresponding depths [3, 2, 1]
        """
        self.qubits = qubits
        self.name = "Pyramid_" + "_".join([cls.__name__ for cls in ansatz_classes])

        layer_configs = list(zip(ansatz_classes, depths))
        self.layered = LayeredAnsatz(qubits, layer_configs)

    def initialize(self):
        """Initialize the pyramid ansatz."""
        self.layered.initialize()
        self.currCirc = self.layered.currCirc
        self.shape = self.layered.shape
        self.layers = self.layered.layers
        self.depth = self.layered.depth
        return self

    def createTestCircuit(self):
        """Delegate test circuit creation to underlying layered ansatz."""
        self.layered.createTestCircuit()
        self.currCirc = self.layered.currCirc


# ==============================================================================
# EXAMPLE ARCHITECTURES
# ==============================================================================

def create_deep_heterogeneous(qubits=5):
    """
    Deep heterogeneous architecture (like ResNet).

    5 layers with different ansatzes, total ~100-150 parameters.
    """
    layer_configs = [
        (Five, 2),   # Layer 1: 2 repetitions of Five
        (Sixteen, 1),      # Layer 2: 1 repetition of Sixteen
        (Custom_Two, 2),   # Layer 3: 2 repetitions of Custom_Two
        (Sixteen, 1),      # Layer 4: 1 repetition of Sixteen
        (Five, 1),   # Layer 5: 1 repetition of Five
    ]
    return LayeredAnsatz(qubits, layer_configs)


def create_alternating_shallow(qubits=5, num_layers=4):
    """
    Shallow alternating architecture (like AlexNet).

    Alternates between two ansatz types for exploration.
    """
    return AlternatingAnsatz(qubits, Custom_One, Sixteen, num_layers)


def create_pyramid(qubits=5):
    """
    Pyramid architecture (like VGG).

    Starts wide and deep, narrows down for refinement.
    """
    return PyramidAnsatz(
        qubits,
        ansatz_classes=[Sixteen, Custom_One, Five],
        depths=[3, 2, 1]
    )


def create_big_pyramid(qubits=5):
    """
    Big Pyramid architecture.
    Significantly deeper than standard pyramid.
    """
    return PyramidAnsatz(
        qubits,
        ansatz_classes=[Sixteen, Custom_Two, Custom_One, Five],
        depths=[4, 3, 2, 1]
    )


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("LAYERED/HETEROGENEOUS ANSATZ EXAMPLES")
    print("="*80)

    # Example 1: Simple 3-layer heterogeneous
    print("\n[Example 1] Simple 3-layer heterogeneous:")
    ansatz1 = LayeredAnsatz(
        qubits=5,
        layer_configs=[
            (Custom_One, 1),
            (Sixteen, 1),
            (Custom_Two, 1)
        ]
    )
    ansatz1.initialize()
    print(f"Circuit depth: {ansatz1.currCirc.depth()}")
    print(f"Total parameters: {np.prod(ansatz1.shape)}")

    # Example 2: Alternating pattern
    print("\n[Example 2] Alternating Custom_One ↔ Sixteen (4 layers):")
    ansatz2 = AlternatingAnsatz(
        qubits=5,
        ansatz_a=Custom_One,
        ansatz_b=Sixteen,
        num_layers=4
    )
    ansatz2.initialize()
    print(f"Circuit depth: {ansatz2.currCirc.depth()}")
    print(f"Total parameters: {np.prod(ansatz2.shape)}")

    # Example 3: Pyramid
    print("\n[Example 3] Pyramid (Sixteen→Custom_One→Five):")
    ansatz3 = create_pyramid(qubits=5)
    ansatz3.initialize()
    print(f"Circuit depth: {ansatz3.currCirc.depth()}")
    print(f"Total parameters: {np.prod(ansatz3.shape)}")

    # Example 4: Deep heterogeneous
    print("\n[Example 4] Deep heterogeneous (5 layers):")
    ansatz4 = create_deep_heterogeneous(qubits=5)
    ansatz4.initialize()
    print(f"Circuit depth: {ansatz4.currCirc.depth()}")
    print(f"Total parameters: {np.prod(ansatz4.shape)}")

    # Example 5: Big Pyramid
    print("\n[Example 5] Big Pyramid (Sixteen→Custom_Two→Custom_One→Five):")
    ansatz5 = create_big_pyramid(qubits=5)
    ansatz5.initialize()
    print(f"Circuit depth: {ansatz5.currCirc.depth()}")
    print(f"Total parameters: {np.prod(ansatz5.shape)}")

    print("\n" + "="*80)
    print("All architectures created successfully!")
    print("="*80)
