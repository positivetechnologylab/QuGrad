#!/usr/bin/env python3
"""
Binary QML classifier with dual-annealing optimiser w/ 5-fold CV, Sherbrooke noise

"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator as AerEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import OptimizerResult

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from scipy.optimize import dual_annealing

df = pd.read_csv("qml_dataset copy.csv")
X = MinMaxScaler().fit_transform(df.drop("label", axis=1).values)
y = df["label"].map({0: -1, 1: 1}).astype(int).values
indices = np.arange(len(df))  # track original rows

nq   = 3
feat = ParameterVector("x", 3 * nq)

fm = QuantumCircuit(nq, name="Φ")
for q in range(nq):
    fm.ry(np.pi * feat[3*q],     q)
    fm.rz(np.pi * feat[3*q + 1], q)
    fm.rx(np.pi * feat[3*q + 2], q)

ansatz = RealAmplitudes(nq, reps=2, entanglement="full",
                        insert_barriers=False)

qc = QuantumCircuit(nq, name="QNN")
qc.compose(fm,     inplace=True)
qc.compose(ansatz, inplace=True)

obs = SparsePauliOp.from_list([("ZII", 1)])

here = Path(__file__).resolve().parent
conf  = json.load(open(here / "conf_sherbrooke.json"))
defs  = json.load(open(here / "defs_sherbrooke.json"))
props = json.load(open(here / "props_sherbrooke.json"))

try:
    noise_model = NoiseModel.from_backend_data(
        configuration=conf, defaults=defs, properties=props
    )
except AttributeError:  # fallback for older Aer
    from qiskit.providers.models import BackendProperties
    noise_model = NoiseModel.from_backend_properties(
        BackendProperties.from_dict(props)
    )

coupling_map = conf["coupling_map"]
basis_gates  = conf["basis_gates"]

backend = AerSimulator(
    method="density_matrix",
    noise_model=noise_model,
    coupling_map=coupling_map,
    basis_gates=basis_gates,
    seed_simulator=42,
)

estimator = AerEstimator(
    backend_options=dict(
        method="density_matrix",
        noise_model=noise_model,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        seed_simulator=42,
    ),
    run_options=dict(shots=1024, seed=42),
)

qnn = EstimatorQNN(
    circuit=qc,
    estimator=estimator,
    observables=[obs],
    input_params=feat,
    weight_params=ansatz.parameters,
)

class DualAnnealingWrapper:
    """SciPy dual-annealing wrapped for Qiskit ml"""

    def __init__(self, bounds, maxiter=2000, initial_temp=2e4,
                 seed=None, verbose=True):
        self.bounds = bounds
        self.maxiter = maxiter
        self.initial_temp = initial_temp
        self.seed = seed
        self.verbose = verbose

    def _wrap(self, fun):
        c = {"i": 0}
        def wrapped(x):
            val = fun(x)
            if self.verbose and c["i"] % 10 == 0:
                print(f"[eval {c['i']:05d}]  loss = {val:.6f}", flush=True)
            c["i"] += 1
            return val
        return wrapped

    def optimize(self, *args, **kw):
        fun = kw.pop("fun", None) or kw.pop("objective_function", None)
        if fun is None and len(args) >= 2:
            fun = args[1]
        if fun is None:
            raise TypeError("Objective function must be supplied.")

        x0 = kw.pop("x0", None)
        if x0 is None and "initial_point" in kw:
            x0 = kw.pop("initial_point")
        if x0 is None and len(args) >= 3:
            x0 = args[2]

        bounds = kw.pop("bounds", self.bounds)

        res = dual_annealing(
            func=self._wrap(fun),
            bounds=bounds,
            maxiter=self.maxiter,
            maxfun=500,
            initial_temp=self.initial_temp,
            seed=self.seed,
            x0=x0,
        )

        opt = OptimizerResult()
        opt.x, opt.fun, opt.nfev = res.x, res.fun, res.nfev
        opt.success, opt.message = res.success, res.message
        return opt

    minimize = optimize

bounds = [(0.0, 2*np.pi)] * len(ansatz.parameters)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_acc = []
all_preds = []

for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
    print(f"\n—— fold {fold}/5 —————————————————————")

    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    idx_te = indices[te]

    opt = DualAnnealingWrapper(bounds=bounds, maxiter=2000, seed=42)

    clf = NeuralNetworkClassifier(
        neural_network=qnn,
        loss="cross_entropy",
        optimizer=opt,
    )
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    fold_acc.append(acc)
    print(f"accuracy = {acc:.3f}")

    y_pred = clf.predict(X_te)
    t = (y_te   + 1)//2
    p = (y_pred + 1)//2

    print("confusion matrix:")
    print(confusion_matrix(t, p))
    print(classification_report(t, p, digits=3))

    all_preds.append(pd.DataFrame({
        "fold":       fold,
        "index":      idx_te.ravel(),
        "true_label": t.ravel(),
        "pred_label": p.ravel(),
    }))

print("\n=== summary ===")
print("fold accuracies:", np.round(fold_acc, 3))
print("mean ± std =", np.mean(fold_acc), "±", np.std(fold_acc))

pd.concat(all_preds).to_csv("dualann_noise_cv_results.csv", index=False)
print("\nPer-sample predictions saved to dualann_noise_cv_results.csv")
