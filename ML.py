from circuits import *
from dists import *
import random
import matplotlib.pyplot as plt
import time
import scipy
from custom_executor import CustomCircuitExecutor
from qiskit.result import Result
from typing import List, Dict, Union
from classes import sinDist
import numpy as np

pi = np.pi
sinSampler = sinDist(a=0, b=np.pi)

circuit_executor = CustomCircuitExecutor()

def sin_sample():
    return np.arccos(1 - 2*np.random.random())

def getSampleSubset(N, sample):

    '''
    Given some list of product states (sample), return N product states from the
    list taken uniformly at random. 
    '''

    tot = len(sample)
    sub = []
    indices = random.sample(range(tot), N)
    mapfun = lambda x : sample[x]
    return list(map(mapfun, indices))

def curriedCost(Ansatz, trainingSet, xi, shots, cost, N=None):

    '''
    Partially apply cost function so that it only takes in thetas as argument.
    '''

    return lambda thetas : cost(thetas, Ansatz, trainingSet, xi, shots, N=N)



def sampleParamsDict(n):

    '''
    Given a number of qubits (n), generate a product state sampled
    uniformly from the set of product states on this number of qubits. 

    '''
    
    res = {}
    for i in range(n):
        phi = np.random.uniform(high=2*pi)
        lmbda = np.random.uniform(high=2*pi)
        theta = 2*(sinSampler.rvs())
        res[f'theta_{i}'] = theta
        res[f'lmbda_{i}'] = lmbda
        res[f'phi_{i}'] = phi
    return res



def pSampleSet(qubits, N):

    '''
    Create n product states of size <qubits>, sampled uniformly. 
    '''

    states = []
    for i in range(N):
        temp = sampleParamsDict(qubits)
        states.append(temp)
    return states



def mapfn(x, thetas, ref):

    '''
    Given a parameter x, list of gate angles thetas, and dictionary of reference
    state angles ref, return the element of either thetas or ref that x 
    corresponds to. 
    '''

    x = x.name
    if x.isdigit():
        return thetas[int(x)]
    else:
        return ref[x]



def curriedF(thetas, ref):

    '''
    Partially apply mapfn to only take in x as argument.
    '''

    return lambda x : mapfn(x, thetas, ref)



### COST FUNCTIONS ### ---------------------------------------------------------

def distCost(thetas, Ansatz, trainingSet, targetDist, shots, N = None):

    '''
    Objective function for distribution ML setup (new setup). Note that it 
    cannot be passed directly into the optimizer but must first be partially
    applied to only take thetas as argument.

    thetas: Parameters for the angles within the circuit. Must be in a 1D array.
    
    Ansatz: Ansatz object for which the objective function is being evaluated.
    
    trainingSet: Initial states to evaluate the circuit on.
    
    targetDist: distribution that we want the ansatz to match when applied to 
    uniformly sampled product states.

    shots: number of shots to run the circuits

    N: size of the subset of trainingSet to try. Defaults to len(trainingSet) if None.
    '''

    if N is None:
        N = len(trainingSet)
    sample = getSampleSubset(N, trainingSet)
    results = []
    parameterList = Ansatz.currCirc.parameters  

    circuits = N*[Ansatz.currCirc]
    paramAssignments = []  
    for i in range(N):
        currAssignments = list(map(curriedF(thetas, sample[i]), parameterList))
        paramAssignments.append(currAssignments)

    # Execute circuits with the configured number of shots (passed to curriedCost)
    job = circuit_executor.run(circuits, paramAssignments, shots=shots)
    result = job.result()
    dist = job.result().quasi_dists
    for i in range(N):
        res = dist[i]
        prob = res[0]
        result = 1 - prob
        results.append(result)

    final = assymTVD(results, targetDist)
    print(final)
    return final


Annealer = scipy.optimize.dual_annealing

def scipyGenerateData(Ansatz, sampleSize, target, cost,
                    minimizer = Annealer, executor = None):

    '''
    Function to setup ML procedure. Should be called once, and wrapped with
    another function to save results (only returns results, doesn't save them
    anywhere). Uses scipy annealer.

    Ansatz: Ansatz object to train parameters for.

    sampleSize: total number of input states to train on.

    target: Either the target distribution or target CE, see objective function.

    cost: objective function to use in ML procedure. Note: will be curried,
    does not need to partially applied yet.

    executor: Optional circuit executor to use instead of global circuit_executor.
    This enables noisy training by injecting a NoisyCircuitExecutor.
    '''

    # Handle custom executor for noisy training
    global circuit_executor
    original_executor = None
    if executor is not None:
        original_executor = circuit_executor
        circuit_executor = executor
        print(f"Using custom executor: {type(executor).__name__}")

    try:
        start = time.time()
        qubits = Ansatz.qubits

        TSet = pSampleSet(qubits, sampleSize)
        Ansatz.createTestCircuit()
        size = dimToNumber(Ansatz.shape)
        x0 = np.random.rand(size)
        x0 = x0*(2*pi)

        # Global counter for total function evaluations
        eval_counter = [0]

        def f_counted(x):
            eval_counter[0] += 1
            result = curriedCost(Ansatz, TSet, target, 2048, cost = cost)(x)
            print(f"[EVAL {eval_counter[0]}] Cost: {result}")
            return result

        f = f_counted

        count = 0
        currCost = f(x0)
        bestCost = currCost
        bestX = x0

        while count < 30:
            count += 1
            x0 = np.random.rand(size)*(2*pi)
            currCost = f(x0)
            if currCost < bestCost:
                bestCost = currCost
                bestX = x0

        firstCost = f(bestX)
        print(f"[INITIAL SEARCH COMPLETE] Total evaluations so far: {eval_counter[0]}")
        print(f'Using:{x0}')
        print(f'First cost: {firstCost}')

        end = time.time()
        overheadTime = end - start

        print(Ansatz.currCirc.parameters)

        start = time.time()
        print(start)

        result = Annealer(f, [(0, 2*pi)]*size, x0 = bestX, maxiter=50, maxfun=150)
        end = time.time()
        print(end)
        trainingTime = end - start

        print(f"[TRAINING COMPLETE] Total function evaluations: {eval_counter[0]}")
        print(result, overheadTime, trainingTime, firstCost)
        return (result, overheadTime, trainingTime, TSet, firstCost, x0)

    finally:
        # Restore original executor
        if original_executor is not None:
            circuit_executor = original_executor