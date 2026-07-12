from qiskit.circuit import QuantumCircuit, Parameter
import numpy as np
from scipy.stats import rv_continuous, gaussian_kde
from scipy.interpolate import interp1d

def dimToNumber(tuple):
    n = 1
    for i in tuple:
        n *= i
    return n 

def nIndexList(shape, k):
    return nIndexListHelper(0, shape, k)

def nIndexListHelper(n, shape, k):
    if len(shape) == 1:
        return [k(n + i) for i in range(shape[0])]
    else:
        m = shape[0]
        return [nIndexListHelper((n + i)*shape[1], shape[1:], k) for i in range(m)]

def sIndexList(shape):
    return sIndexListHelper('', shape)

def sIndexListHelper(prefix, shape):
    if len(shape) == 1:
        return [f'{prefix}.{i}' for i in range(shape[0])]
    else:
        n = shape[0]
        return [sIndexListHelper(f'{prefix}.{i}', shape[1:]) for i in range(n)]

#SWAP Test:
def swap_test(circuit):
    num = circuit.num_qubits
    circuit.barrier()
    for i in range(num//3):
        circuit.h(i)
    circuit.barrier()
    for i in range(num//3):
        circuit.cswap(i, i+(num//3), i+(2*(num//3)))
    circuit.barrier()
    for i in range(num//3):    
        circuit.h(i)
        circuit.measure(i, i)



#VSWAP Test:
def v_swap_test(circuit, qubits):
    num = circuit.num_qubits
    circuit.h(num - 1)
    for i in range(qubits):
        circuit.cswap(num-1, i, i+qubits)
    circuit.h(num - 1)
    circuit.measure(num - 1, 0)

def getBin(dist, min, max):
    total = 0
    for i in dist:
        if (min <= i) and (i <= max):
            total += 1
    return total

class Ansatz:
    def __init__(self, circuit, shape, qubits, depth, name='empty'):
        self.shape = shape
        self.circuit = circuit
        self.qubits = qubits
        self.depth = depth
        self.x0 = np.zeros(shape)
        self.name = name
        self.currCirc = None
        self.vCirc = None
        self.pList = [Parameter(f'{i}') for i in range(dimToNumber(shape))]
        self.refpList = [(Parameter(f'theta_{i}'), Parameter(f'phi_{i}'), 
                          Parameter(f'lmbda_{i}')) for i in range(qubits)]
        self.refpList2 = [(Parameter(f'theta_{i}_2'), Parameter(f'phi_{i}_2'), 
                          Parameter(f'lmbda_{i}_2')) for i in range(qubits)]

    def initialize(self): 
        shape = list(self.shape)
        params = nIndexList(shape, lambda x : self.pList[x])
        self.currCirc = QuantumCircuit(self.qubits)
        self.circuit(self.currCirc, params, self.depth)   

    def getEmptyCircuit(self):
        shape = list(self.shape)
        params = nIndexList(shape, lambda x : self.pList[x])
        qc = QuantumCircuit(self.qubits)
        self.circuit(qc, params, self.depth)   
        return qc

    def assignParams(self, params):
        assignments = {}
        for i in range(len(params)):
            assignments[self.pList[i]] = params[i]
        self.currCirc = self.currCirc.assign_parameters(assignments)
        del assignments

    def assignRefParams(self, refParams):
        assignments = {}
        for i in range(len(refParams)):
            t, p, l = refParams[i]
            first, second, third = self.refpList[i]
            assignments[first] = t
            assignments[second] = p
            assignments[third] = l
        self.currCirc = self.currCirc.assign_parameters(assignments)
        del assignments

    def createTestCircuit(self):
        qubits = self.qubits
        self.currCirc = QuantumCircuit(3*qubits, qubits)
        circ1 = QuantumCircuit(qubits)
        shape = list(self.shape)
        params = nIndexList(shape, lambda x : self.pList[x])
        self.circuit(circ1, params, self.depth)
        circ1.to_instruction()

        URef = QuantumCircuit(self.qubits)
        for i in range(self.qubits):
            t, p, l = self.refpList[i]
            URef.u(t, p, l, i)
        
        self.currCirc.append(URef, range(qubits, 2*qubits))
        self.currCirc.append(URef, range(2*qubits, 3*qubits))
        self.currCirc.append(circ1, range(qubits, 2*qubits))
        self.currCirc.append(circ1, range(2*qubits, 3*qubits))
        swap_test(self.currCirc)

    def createVTestCircuit(self):
        qubits = self.qubits
        self.vCirc = QuantumCircuit(2*qubits + 1, 1)
        circ1 = QuantumCircuit(qubits)
        shape = list(self.shape)
        params = nIndexList(shape, lambda x : self.pList[x])
        self.circuit(circ1, params, self.depth)
        circ1.to_instruction()

        U1 = QuantumCircuit(self.qubits)
        for i in range(self.qubits):
            t, p, l = self.refpList[i]
            U1.u(t, p, l, i)
        
        U2 = QuantumCircuit(self.qubits)
        for i in range(self.qubits):
            t, p, l = self.refpList2[i]
            U2.u(t, p, l, i)

        self.vCirc.append(U1, range(0, qubits))
        self.vCirc.append(U2, range(qubits, 2*qubits))
        self.vCirc.append(circ1, range(0, qubits))
        self.vCirc.append(circ1, range(qubits, 2*qubits))
        v_swap_test(self.vCirc, qubits)


    def getRefAssignments(self, refParams):
        assignments = {}
        for i in range(len(refParams)):
            t, p, l = refParams[i]
            first, second, third = self.refpList[i]
            assignments[first] = t
            assignments[second] = p
            assignments[third] = l
        return assignments
    
    def getAssignments(self, params):
        assignments = {}
        for i in range(len(params)):
            assignments[self.pList[i]] = params[i]
        return assignments

class Test:
    def __init__(self, ansatz, qubits, depth):
        self.qubits = qubits
        self.depth = depth
        self.ansatz = ansatz

class sinDist(rv_continuous):
    def _pdf(self, theta):
        return 0.5*np.sin(theta)

class TestDist:

    def __init__(self, fun, name, Range, numBoxes, params, size):
        self.fun = fun
        self.name = name
        self.samples = []
        self.avgDist = []
        self.Range = Range  
        self.params = params
        self.numBoxes = numBoxes
        self.size = size


    def createSampleDistributions(self, n, params=None, replace=True):
        """Override parent method to ignore params argument"""
        if replace:
            self.samples = []
            
        print("Creating sample distributions...")
        print("Sample size:", self.size)
        print("Number of samples:", n)
        
        for i in range(n):
            currDist = self.fun(*self.params)
            self.samples.append(currDist)
            
        return self.samples
    
    def sample(self, n):
        """
        Sample n random values from the distribution.
        
        Args:
            n: Number of values to sample
            
        Returns:
            List of n randomly sampled values from the distribution
        """
        import numpy as np
        # Flatten all samples into a single array
        all_values = np.concatenate(self.samples)
        # Randomly sample n values
        indices = np.random.choice(len(all_values), size=n, replace=True)
        return all_values[indices].tolist()
        
    @staticmethod
    def getBinsList(x, numBoxes, min, max):
        """
        Bins data into a histogram.
        
        Args:
            x: Data to bin
            numBoxes: Number of bins
            min: Minimum value for binning range
            max: Maximum value for binning range
        
        Returns:
            List of counts for each bin
        """
        delta = (max - min)/numBoxes  # Width of each bin
        bins = []
        
        for i in range(numBoxes):
            lower = min + (i*delta)     # Lower bound of current bin
            upper = lower + delta       # Upper bound of current bin
            num = getBin(x, lower, upper)  # Count elements in this bin
            bins.append(num)
        
        return bins


    def getAveragedBins(self, numBoxes, Range=None, mutate=True):
        """
        Calculates average bin counts across all samples.
        
        Args:
            numBoxes: Number of bins
            Range: Optional (min,max) tuple to override self.Range
            mutate: If True, update instance variables with results
        
        Returns:
            List of averaged bin counts
        """
        if Range == None:
            Range = self.Range
        (min, max) = Range
        
        # Create function to bin a single sample
        f = lambda x: self.getBinsList(x, numBoxes, min, max)
        
        # Map binning function across all samples
        result = list(map(f, self.samples))
        
        # Initialize total bins
        totalBins = [0 for i in range(numBoxes)]
        
        # Sum up counts across all samples
        for i in range(numBoxes):
            for sample in result:
                totalBins[i] += sample[i]
        
        # Calculate averages
        avgBins = list(map(lambda x: x/len(result), totalBins))
        
        if mutate:
            self.numBoxes = numBoxes
            self.Range = Range
            self.avgDist = avgBins
            
        return avgBins