from __future__ import annotations
import os, random
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

PHASE_BASE_FREE = 0.10
PHASE_JITTER    = 0.01
N_SAMPLES       = 250
N_QUBITS        = 3      # sensor qubits + measurement
N_ANCILLA       = 3      # one ancilla per sensor qubit
OUTPUT_DIR      = "ce_high_data"
SHOTS           = 2048 

# generate N_SAMPLES base soil‐phase values uniform on [0.5, 0.95]
# ps_bases = list(np.random.uniform(0.5, 0.95, size=N_SAMPLES))
ps_bases = np.random.uniform(0.05, 0.45, size=N_SAMPLES)  # low-moisture band

# ─── sensor circuit ───
def build_sensor_circuit(phi_soil: float, phi_free: float) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS)
    data_qubits = list(range(N_QUBITS - 1))
    meas_qubit  = N_QUBITS - 1

    qc.h(data_qubits[0])
    qc.x(data_qubits[1])
    for i in range(len(data_qubits)-1):
        qc.cx(data_qubits[i], data_qubits[i+1])
    for q in data_qubits[:1]:
        qc.p(phi_soil, q)
    for q in data_qubits[1:]:
        qc.p(phi_free, q)

    qc.barrier()
    qc.cx(data_qubits[0], meas_qubit)
    qc.barrier()
    for q in data_qubits:
        qc.h(q)
    qc.barrier()
    for q in data_qubits:
        qc.cx(q, meas_qubit)
    qc.barrier()
    return qc

# ─── SWAP‑test circuit ───
def build_swaptest_circuit(phi_s: float, phi_f: float) -> QuantumCircuit:
    total_qubits = N_ANCILLA + 2 * N_QUBITS
    qc = QuantumCircuit(total_qubits, N_ANCILLA)
    qc.compose(build_sensor_circuit(phi_s, phi_f),
               qubits=range(N_ANCILLA, N_ANCILLA + N_QUBITS), inplace=True)
    qc.compose(build_sensor_circuit(phi_s, phi_f),
               qubits=range(N_ANCILLA + N_QUBITS, N_ANCILLA + 2 * N_QUBITS), inplace=True)
    qc.barrier()
    for i in range(N_ANCILLA):
        qc.h(i)
        qc.cswap(i, N_ANCILLA + i, N_ANCILLA + N_QUBITS + i)
        qc.h(i)
    qc.barrier()
    qc.measure(range(N_ANCILLA), range(N_ANCILLA))
    return qc

service = QiskitRuntimeService()
backend = service.backend("ibm_sherbrooke")
pm      = generate_preset_pass_manager(backend=backend, optimization_level=3)
sampler = Sampler(mode=backend)
sampler.options.default_shots = SHOTS

os.makedirs(OUTPUT_DIR, exist_ok=True)
ps_list, pf_list, p0_list, ces = [], [], [], []

for idx in range(N_SAMPLES):
    # take the pre‑sampled soil phase and still add small jitter
    base_ps = ps_bases[idx]
    ds      = random.uniform(-PHASE_JITTER, PHASE_JITTER)
    df      = random.uniform(-PHASE_JITTER, PHASE_JITTER)
    ps, pf  = base_ps + ds, PHASE_BASE_FREE + df

    ps_list.append(ps)
    pf_list.append(pf)

    qc          = build_swaptest_circuit(ps, pf)
    isa_circuit = pm.run(qc)
    job         = sampler.run([isa_circuit])
    print(f"[{idx+1}/{N_SAMPLES}] Job ID: {job.job_id()}")

    res    = job.result()[0]
    counts = res.join_data().get_counts()
    p0     = counts.get("0" * N_ANCILLA, 0) / SHOTS
    ce     = 1.0 - p0

    p0_list.append(p0)
    ces.append(ce)
    print(f"    ps={ps:.4f}, pf={pf:.4f} → p0={p0:.4f}, CE={ce:.6f}")

ps_arr, pf_arr, p0_arr, ce_arr = map(np.array, (ps_list, pf_list, p0_list, ces))

np.savez(
    os.path.join(OUTPUT_DIR, "ce_swaptest_high_data.npz"),
    phi_soil=ps_arr, phi_free=pf_arr, p0=p0_arr, ce=ce_arr
)

import csv
with open(os.path.join(OUTPUT_DIR, "ce_swaptest_high_data.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["phi_soil", "phi_free", "p0", "ce"])
    for ps, pf, p0, ce in zip(ps_arr, pf_arr, p0_arr, ce_arr):
        writer.writerow([f"{ps:.6f}", f"{pf:.6f}", f"{p0:.6f}", f"{ce:.6f}"])

print("\nSaved results to:")
print("  -", os.path.join(OUTPUT_DIR, "ce_swaptest_high_data.npz"))
print("  -", os.path.join(OUTPUT_DIR, "ce_swaptest_high_data.csv"))
