import numpy as np

from ML import pSampleSet, curriedF
from custom_executor import CustomCircuitExecutor

"""
SWAP-test diversity analysis (paper Sec. V.C, Fig. 5).

Measures the DIVERSITY of the states a trained generator produces. A generator
can match the target CE histogram while repeatedly emitting highly similar
states; the SWAP test between pairs of generated states detects that collapse.

For a SWAP test between |a> and |b>, the ancilla measures 0 with probability
    P(0) = (1 + |<a|b>|^2) / 2
so P(0) ~ 0.5 means the two generated states are near-orthogonal (high
diversity), and P(0) ~ 1.0 means they have collapsed onto each other.

Because similarity is only meaningful between states of comparable entanglement,
states are first bucketed by their CE value and pairs are only drawn WITHIN a
bucket. Buckets are BIN_WIDTH = 0.02 wide across the target's CE range (note
this is a finer binning than the target distribution's own histogram, which uses
20 bins over (0, 0.6)). Within a bucket, states are shuffled and split into
disjoint pairs, giving k // 2 pairs for a bucket holding k states.

Both the state sampling and the within-bin pairing are random, so repeated runs
agree to within Monte-Carlo sampling noise rather than value-for-value; sparse
buckets (single-digit pair counts) are correspondingly the noisiest.

Consumed by swap_runs.py, whose output is recorded in swapres.txt and rendered
by swap_violin_plot.py.
"""

BIN_WIDTH = 0.02

circuit_executor = CustomCircuitExecutor()


def _ce_values(ansatz, params, states, shots):
    """CE = 1 - P(|0...0>) for each product state, via the SWAP-test circuit.

    Mirrors ML.py::distCost exactly.
    """
    ansatz.createTestCircuit()
    parameterList = ansatz.currCirc.parameters

    circuits = len(states) * [ansatz.currCirc]
    paramAssignments = [
        list(map(curriedF(params, state), parameterList)) for state in states
    ]

    job = circuit_executor.run(circuits, paramAssignments, shots=shots)
    dists = job.result().quasi_dists

    return [1 - dists[i][0] for i in range(len(states))]


def _pair_refs(stateA, stateB, qubits):
    """Merge two product states into one ref dict for the V-SWAP circuit.

    vCirc binds state A to (theta_i, phi_i, lmbda_i) and state B to the
    _2-suffixed parameters (classes.py::refpList2), so mapfn resolves both by name.
    """
    refs = dict(stateA)
    for i in range(qubits):
        refs[f'theta_{i}_2'] = stateB[f'theta_{i}']
        refs[f'phi_{i}_2'] = stateB[f'phi_{i}']
        refs[f'lmbda_{i}_2'] = stateB[f'lmbda_{i}']
    return refs


def _swap_similarities(ansatz, params, pairs, shots):
    """P(0) on the V-SWAP ancilla for each (stateA, stateB) pair."""
    if not pairs:
        return []

    ansatz.createVTestCircuit()
    parameterList = ansatz.vCirc.parameters
    qubits = ansatz.qubits

    circuits = len(pairs) * [ansatz.vCirc]
    paramAssignments = [
        list(map(curriedF(params, _pair_refs(a, b, qubits)), parameterList))
        for (a, b) in pairs
    ]

    job = circuit_executor.run(circuits, paramAssignments, shots=shots)
    dists = job.result().quasi_dists

    # vCirc measures the single ancilla into clbit 0.
    return [dists[i].get(0, 0.0) for i in range(len(pairs))]


def getVarietyDists(ansatz, params, dist, sample=1000, shots=2048, seed=None):
    """Bucket generated states by CE, then SWAP-test pairs within each bucket.

    Args:
        ansatz: an Ansatz (e.g. Sixteen(5, 1)), untrained -- params supply the angles
        params: trained parameter vector (np.load of runs_qmill/.../*.npy)
        dist:   target TestDist; only its .Range is used, for the CE binning extent
        sample: number of product states to push through the generator

    Returns:
        (results, varieties)
          results   -- list of CE values, one per sampled product state
          varieties -- list of (swap_results, (lo, hi)) per CE bin, as consumed by
                       swap_runs.py: np.mean/np.std/len over swap_results
    """
    rng = np.random.default_rng(seed)

    # Saved parameter files are shape (1, n) (see runs_qmill/.../*_5_1.npy), but
    # mapfn indexes thetas flat, so flatten here -- swap_runs.py passes them raw.
    params = np.asarray(params).flatten()

    states = pSampleSet(ansatz.qubits, sample)
    results = _ce_values(ansatz, params, states, shots)

    lo_all, hi_all = dist.Range
    n_bins = int(round((hi_all - lo_all) / BIN_WIDTH))

    varieties = []
    for b in range(n_bins):
        lo = lo_all + b * BIN_WIDTH
        hi = lo + BIN_WIDTH

        members = [s for s, ce in zip(states, results) if lo <= ce < hi]

        # PAIRING: the bucket is shuffled and split into DISJOINT pairs, so each
        # state is used at most once and len(swap_results) == len(members) // 2.
        pairs = []
        if len(members) >= 2:
            order = rng.permutation(len(members))
            for i in range(0, len(members) - 1, 2):
                pairs.append((members[order[i]], members[order[i + 1]]))

        swap_results = _swap_similarities(ansatz, params, pairs, shots)
        varieties.append((swap_results, (lo, hi)))

    return results, varieties
