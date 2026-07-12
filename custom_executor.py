from qiskit_aer import AerSimulator
from qiskit.result import Result
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import DensityMatrix, entropy, purity as qiskit_purity
from typing import List, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# OPTIONAL METRIC HELPERS (NEW)
# ============================================================================

def von_neumann_entropy(rho: np.ndarray, base: int = 2) -> float:
    return float(entropy(DensityMatrix(rho), base=base))


def purity(rho, validate: bool = False) -> float:
    """
    Purity Tr(rho^2) using Qiskit's built-in purity().
    Accepts either:
      - np.ndarray (density matrix)
      - qiskit.quantum_info.DensityMatrix
    """
    if isinstance(rho, DensityMatrix):
        state = rho
    else:
        state = DensityMatrix(np.asarray(rho, dtype=np.complex128))
    return float(qiskit_purity(state, validate=validate))


def _bind_params_by_name(transpiled_circ: QuantumCircuit,
                         orig_param_names: List[str],
                         values: List[float]) -> Dict:
    """
    Robust parameter binding for transpiled circuits:
    maps values by param.name onto transpiled circuit params.
    """
    value_by_name = dict(zip(orig_param_names, values))
    binds = {}
    for p in list(transpiled_circ.parameters):
        if p.name not in value_by_name:
            raise ValueError(
                f"Missing value for parameter '{p.name}'. "
                f"Have: {sorted(value_by_name.keys())[:10]}..."
            )
        binds[p] = value_by_name[p.name]
    return binds


def _extract_rho(raw_results, i: int, label: str = "rho") -> np.ndarray:
    """
    Extract density matrix from Aer result robustly across versions.
    """
    data = raw_results.data(i)
    if label in data:
        return np.asarray(data[label])
    # common fallback keys
    if "density_matrix" in data:
        return np.asarray(data["density_matrix"])
    # last resort: show keys for debugging
    raise KeyError(f"Density matrix not found in result data. Keys: {list(data.keys())}")


# ============================================================================
# BASE EXECUTOR (UNCHANGED EXISTING FUNCTIONALITY + NEW DM FUNCTIONALITY)
# ============================================================================

class CustomCircuitExecutor:
    def __init__(self):
        # Existing backend for shot-based sampling
        self.backend = AerSimulator()

        # NEW: density-matrix backend (for VN entropy / purity / mixedness diagnostics)
        self.backend_dm = AerSimulator(method="density_matrix")

    def _fully_decompose(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Recursively decompose a circuit until it only contains basic gates.
        """
        decomposed = circuit.decompose()
        while any(op.name.startswith('circuit') for op, _, _ in decomposed.data):
            decomposed = decomposed.decompose()
        return decomposed

    def run(
        self,
        circuits: List[QuantumCircuit],
        parameter_values: List[List[float]],
        shots: int = 8192
    ) -> 'CircuitResults':
        """
        EXISTING FUNCTIONALITY (unchanged):
        - ensures measurements exist
        - binds parameters
        - runs shot-based simulator
        - returns quasi_dists via CircuitResults
        """
        template_circuit = self._fully_decompose(circuits[0])
        circuit_params = list(template_circuit.parameters)

        # Add measurements if not present
        if not template_circuit.cregs:
            template_circuit.add_register(ClassicalRegister(template_circuit.num_qubits, 'c'))
            template_circuit.measure_all()

        bound_circuits = []
        for params in parameter_values:
            if len(params) != len(circuit_params):
                raise ValueError(f"Parameter length mismatch: got {len(params)}, expected {len(circuit_params)}")
            param_dict = dict(zip(circuit_params, params))
            bound_circuits.append(template_circuit.assign_parameters(param_dict, inplace=False))

        job = self.backend.run(bound_circuits, shots=shots)
        raw_results = job.result()
        return CircuitResults(raw_results)

    # ----------------------------
    # NEW: Density-matrix execution
    # ----------------------------
    def run_density_matrices(
        self,
        circuits: List[QuantumCircuit],
        parameter_values: List[List[float]],
        optimization_level: int = 3
    ) -> List[np.ndarray]:
        """
        Execute measurement-free circuits and return density matrices ρ.

        - Removes FINAL measurements
        - Appends save_density_matrix('rho')
        - Transpiles once
        - Binds parameters ROBUSTLY by param.name on the transpiled circuit
        """
        template_circuit = self._fully_decompose(circuits[0])
        orig_params = list(template_circuit.parameters)
        orig_param_names = [p.name for p in orig_params]

        template_circuit = template_circuit.remove_final_measurements(inplace=False)
        template_circuit.save_density_matrix(label="rho")

        template_t = transpile(
            template_circuit,
            backend=self.backend_dm,
            optimization_level=optimization_level
        )

        bound_circuits = []
        for params in parameter_values:
            if len(params) != len(orig_params):
                raise ValueError(f"Parameter length mismatch: got {len(params)}, expected {len(orig_params)}")
            binds = _bind_params_by_name(template_t, orig_param_names, params)
            bound_circuits.append(template_t.assign_parameters(binds, inplace=False))

        raw_results = self.backend_dm.run(bound_circuits, shots=1).result()

        rhos: List[np.ndarray] = []
        for i in range(len(bound_circuits)):
            rhos.append(_extract_rho(raw_results, i, label="rho"))
        return rhos

    def run_entropy_metrics(
        self,
        circuits: List[QuantumCircuit],
        parameter_values: List[List[float]],
        base: int = 2,
        optimization_level: int = 3
    ) -> List[dict]:
        """
        Returns [{'vn_entropy': ..., 'purity': ...}, ...] per circuit instance.
        """
        rhos = self.run_density_matrices(
            circuits=circuits,
            parameter_values=parameter_values,
            optimization_level=optimization_level
        )
        return [{"vn_entropy": von_neumann_entropy(rho, base=base), "purity": purity(rho)} for rho in rhos]


# ============================================================================
# RESULTS WRAPPER (UNCHANGED)
# ============================================================================

class CircuitResults:
    def __init__(self, qiskit_result: Result):
        self.raw_result = qiskit_result
        self._process_results()

    def _process_results(self):
        self.quasi_dists = []
        for i in range(len(self.raw_result.results)):
            counts = self.raw_result.get_counts(i)
            total_shots = sum(counts.values())
            quasi_dist = {
                int(str(key).replace(' ', ''), 2): value / total_shots
                for key, value in counts.items()
            }
            self.quasi_dists.append(quasi_dist)

    def result(self) -> 'CircuitResults':
        return self


# ============================================================================
# NOISY EXECUTOR (UNCHANGED EXISTING FUNCTIONALITY + NEW DM FUNCTIONALITY)
# ============================================================================

class NoisyCircuitExecutor(CustomCircuitExecutor):
    def __init__(self, noise_model: NoiseModel, backend_name: str, coupling_map=None):
        self.noise_model = noise_model
        self.backend_name = backend_name
        self.coupling_map = coupling_map

        self.backend = AerSimulator(noise_model=noise_model, coupling_map=coupling_map)

        self.backend_dm = AerSimulator(
            method="density_matrix",
            noise_model=noise_model,
            coupling_map=coupling_map
        )

        # Cache for counts-mode template
        self._transpiled_template = None
        self._template_param_names = None

        # Cache for density-matrix template
        self._transpiled_template_dm = None
        self._template_param_names_dm = None

        logger.info(f"Created NoisyCircuitExecutor for {backend_name}")
        logger.info(f"Noise model has {len(noise_model.basis_gates)} basis gates")

    def run(
        self,
        circuits: List[QuantumCircuit],
        parameter_values: List[List[float]],
        shots: int = 8192
    ) -> 'CircuitResults':
        num_circuits = len(parameter_values)
        template_circuit = self._fully_decompose(circuits[0])
        orig_params = list(template_circuit.parameters)
        orig_param_names = [p.name for p in orig_params]

        if self._transpiled_template is None or self._template_param_names != orig_param_names:
            print(f"[NoisyExecutor] Transpiling template circuit (optimization_level=3)...")

            if not template_circuit.cregs:
                template_circuit.add_register(ClassicalRegister(template_circuit.num_qubits, 'c'))
                template_circuit.measure_all()

            self._transpiled_template = transpile(
                template_circuit,
                backend=self.backend,
                optimization_level=3,
                coupling_map=self.coupling_map
            )
            self._template_param_names = orig_param_names
            print(f"[NoisyExecutor] Template transpiled and cached")

        print(f"[NoisyExecutor] Binding {num_circuits} parameter sets to transpiled template...")
        bound_circuits = []
        for params in parameter_values:
            if len(params) != len(orig_params):
                raise ValueError(f"Parameter length mismatch: got {len(params)}, expected {len(orig_params)}")
            binds = _bind_params_by_name(self._transpiled_template, self._template_param_names, params)
            bound_circuits.append(self._transpiled_template.assign_parameters(binds, inplace=False))

        print(f"[NoisyExecutor] Running {num_circuits} circuits with {shots} shots...")
        raw_results = self.backend.run(bound_circuits, shots=shots).result()
        print(f"[NoisyExecutor] Execution completed")

        return NoisyCircuitResults(raw_results)

    def run_density_matrices(
        self,
        circuits: List[QuantumCircuit],
        parameter_values: List[List[float]],
        optimization_level: int = 3
    ) -> List[np.ndarray]:
        num_circuits = len(parameter_values)

        template_circuit = self._fully_decompose(circuits[0])
        orig_params = list(template_circuit.parameters)
        orig_param_names = [p.name for p in orig_params]

        if self._transpiled_template_dm is None or self._template_param_names_dm != orig_param_names:
            print(f"[NoisyExecutor] Transpiling DM template (optimization_level={optimization_level})...")

            template_circuit = template_circuit.remove_final_measurements(inplace=False)
            template_circuit.save_density_matrix(label="rho")

            self._transpiled_template_dm = transpile(
                template_circuit,
                backend=self.backend_dm,
                optimization_level=optimization_level,
                coupling_map=self.coupling_map
            )
            self._template_param_names_dm = orig_param_names
            print(f"[NoisyExecutor] DM template transpiled and cached")

        bound_circuits = []
        for params in parameter_values:
            if len(params) != len(orig_params):
                raise ValueError(f"Parameter length mismatch: got {len(params)}, expected {len(orig_params)}")
            binds = _bind_params_by_name(self._transpiled_template_dm, self._template_param_names_dm, params)
            bound_circuits.append(self._transpiled_template_dm.assign_parameters(binds, inplace=False))

        raw_results = self.backend_dm.run(bound_circuits, shots=1).result()

        rhos: List[np.ndarray] = []
        for i in range(num_circuits):
            rhos.append(_extract_rho(raw_results, i, label="rho"))
        return rhos

    def run_entropy_metrics(
        self,
        circuits: List[QuantumCircuit],
        parameter_values: List[List[float]],
        base: int = 2,
        optimization_level: int = 3
    ) -> List[dict]:
        rhos = self.run_density_matrices(
            circuits=circuits,
            parameter_values=parameter_values,
            optimization_level=optimization_level
        )
        return [{"vn_entropy": von_neumann_entropy(rho, base=base), "purity": purity(rho)} for rho in rhos]


# ============================================================================
# NOISY RESULTS WRAPPER
# ============================================================================

class NoisyCircuitResults(CircuitResults):
    def _process_results(self):
        self.quasi_dists = []
        for i in range(len(self.raw_result.results)):
            try:
                counts = self.raw_result.get_counts(i)
                total_shots = sum(counts.values())
                processed = {
                    int(str(key).replace(' ', ''), 2): value / total_shots
                    for key, value in counts.items()
                }
                self.quasi_dists.append(processed)
            except Exception as e:
                logger.warning(f"Could not get counts for experiment {i}: {e}")
                self.quasi_dists.append({0: 1.0})