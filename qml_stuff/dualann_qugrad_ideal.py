#!/usr/bin/env python3
"""Binary QML classifier with dual-annealing optimiser – 5-fold CV.
Using Ideal QuGrad data.
"""

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp

from qiskit.primitives import StatevectorEstimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

from qiskit_algorithms.optimizers import OptimizerResult

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from scipy.optimize import dual_annealing
import sys

# Output redirection to log file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("dualann_qugrad_ideal_new.log")

# Using generated ideal QuGrad dataset (explicitly)
df = pd.read_csv("qml_dataset_qugrad_ideal.csv")
X = MinMaxScaler().fit_transform(df.drop("label", axis=1).values)
y = df["label"].map({0: -1, 1: 1}).astype(int).values
indices = np.arange(len(df))  # track original rows

num_qubits = 3
feat = ParameterVector("x", 3 * num_qubits)

feature_map = QuantumCircuit(num_qubits, name="Φ")
for q in range(num_qubits):
    feature_map.ry(np.pi * feat[3 * q], q)
    feature_map.rz(np.pi * feat[3 * q + 1], q)
    feature_map.rx(np.pi * feat[3 * q + 2], q)

ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2,
                        entanglement="full", insert_barriers=False)

qc = QuantumCircuit(num_qubits, name="QNN")
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

obs = SparsePauliOp.from_list([("ZII", 1)])

estimator = StatevectorEstimator()

qnn = EstimatorQNN(
    circuit=qc,
    estimator=estimator,
    observables=[obs],
    input_params=feat,
    weight_params=ansatz.parameters,
)

class DualAnnealingWrapper:
    """SciPy dual-annealing wrapped for Qiskit ml"""

    def __init__(self, bounds, maxiter=10000, initial_temp=2e4,
                 seed=None, verbose=True):
        self.bounds = bounds
        self.maxiter = maxiter
        self.initial_temp = initial_temp
        self.seed = seed
        self.verbose = verbose

    def _wrap(self, fun):
        counter = {"i": 0}
        def wrapped(x):
            val = fun(x)
            if self.verbose and counter["i"] % 0 == 0:
                print(f"[eval {counter['i']:05d}]  loss = {val:.6f}", flush=True)
            counter["i"] += 1
            return val
        return wrapped

    def optimize(self, *args, **kw):
        # objective 
        fun = kw.pop("fun", None) or kw.pop("objective_function", None)
        if fun is None and len(args) >= 2:
            fun = args[1]
        if fun is None:
            raise TypeError("Objective function must be supplied.")

        x0 = kw.pop("x0", None)
        if x0 is None and "initial_point" in kw:
            x0 = kw.pop("initial_point")

        bounds = kw.pop("bounds", self.bounds)

        res = dual_annealing(
            func=self._wrap(fun),
            bounds=bounds,
            maxiter=self.maxiter,
            maxfun=10000,
            initial_temp=self.initial_temp,
            seed=self.seed,
            x0=x0,
        )

        opt_res = OptimizerResult()
        opt_res.x, opt_res.fun, opt_res.nfev = res.x, res.fun, res.nfev
        opt_res.success, opt_res.message = res.success, res.message
        return opt_res

    minimize = optimize

bounds = [(0.0, 2 * np.pi)] * len(ansatz.parameters)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_acc = []
all_preds = []

for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    print(f"\n—— fold {fold_id}/5 —————————————————————")

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    idx_te = indices[test_idx]

    optimizer = DualAnnealingWrapper(bounds=bounds, maxiter=10000, seed=42)

    clf = NeuralNetworkClassifier(
        neural_network=qnn,
        loss="cross_entropy",
        optimizer=optimizer,
    )
    clf.fit(X_tr, y_tr)

    acc = clf.score(X_te, y_te)
    fold_acc.append(acc)
    print(f"accuracy = {acc:.3f}")

    y_pred = clf.predict(X_te)
    t = (y_te + 1) // 2      # map to 0/1
    p = (y_pred + 1) // 2

    print("confusion matrix:")
    print(confusion_matrix(t, p))
    print(classification_report(t, p, digits=3))

    all_preds.append(pd.DataFrame({
        "fold": fold_id,
        "index": idx_te.ravel(),
        "true_label": t.ravel(),
        "pred_label": p.ravel(),
    }))

print("\n=== summary ===")
print("fold accuracies:", np.round(fold_acc, 3))
print("mean ± std =", np.mean(fold_acc), "±", np.std(fold_acc))

pd.concat(all_preds).to_csv("dualann_qugrad_ideal_cv_results.csv", index=False)
print("\nPer-sample predictions saved to dualann_qugrad_ideal_cv_results.csv")

