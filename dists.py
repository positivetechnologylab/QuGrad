from classes import *
import matplotlib.pyplot as plt
import os


### Define TVD functions ###----------------------------------------------------

def TVD(dist1, dist2, numBoxes, min, max):
    if len(dist1) != len(dist2):
        return None
    total = len(dist1) + len(dist2)
    tvd = 0
    delta = (max-min)/numBoxes
    for i in range(numBoxes):
        lower = min + (i*delta)
        upper = lower + delta
        diff = abs(getBin(dist1, lower, upper) - getBin(dist2, lower, upper))
        tvd += diff
        print(lower, upper)
    result = tvd/total
    return result

def TVD2(boxes1, boxes2):
    """
    Calculate Total Variation Distance between two normalized distributions
    """
    if len(boxes1) != len(boxes2):
        return None
    
    # Check for zero sums
    if np.sum(boxes1) == 0:
        print("Warning: boxes1 sums to zero!")
        return 1.0  # Maximum distance if one distribution is empty
    if np.sum(boxes2) == 0:
        print("Warning: boxes2 sums to zero!")
        return 1.0
        
    boxes1 = np.array(boxes1, dtype=float)
    boxes2 = np.array(boxes2, dtype=float)
    boxes1 = boxes1 / np.sum(boxes1)
    boxes2 = boxes2 / np.sum(boxes2)
    
    # Calculate TVD
    tvd = 0.5 * np.sum(np.abs(boxes1 - boxes2))
    
    return tvd

def assymTVD(samples, dist):
    boxes2 = dist.avgDist
    (min, max) = dist.Range
    boxes1 = dist.getBinsList(samples, dist.numBoxes, min, max) 
    tvd = TVD2(boxes1, boxes2) 
    return tvd

def normalize(boxes):
    results = []
    total = sum(boxes)
    for box in boxes:
        results.append(box/total)
    return results


### Define Distributions ###----------------------------------------------------

def Uniform(n):
    dist = TestDist(np.random.uniform, 'Uniform', (0, 0.6), 20, (0, 0.4, n), n)
    return dist


def Normal(n):
    dist = TestDist(np.random.normal, 'Normal', (0, 0.6), 20, (0.2,0.05,n), n)
    return dist

def WeibullLeft(n):
    fun = lambda a, b : 0.05*np.random.weibull(a,b)
    dist = TestDist(fun, 'Left Weibull', (0, 0.6), 20, (1.2, n), n)
    return dist

def WeibullRight(n):
    fun = lambda a, b : 0.4*np.ones(b) - 0.05*np.random.weibull(a,b)
    dist = TestDist(fun, 'Right Weibull', (0, 0.6), 20, (1.2, n), n)
    return dist

def MNIST(n):
    """
    Creates a TestDist that samples from the scaled MNIST CE distribution.
    
    Args:
        n: size of samples to draw
    Returns:
        TestDist object configured to sample from mnist_dist_scaled.npy
    """
    # Look for file in current directory or parent directory
    filename = 'mnist_dist_scaled.npy'
    if not os.path.exists(filename) and os.path.exists(os.path.join('..', filename)):
        filename = os.path.join('..', filename)
        
    mnist_dist = np.load(filename)
    
    def mnist_sampler(size):
        return np.random.choice(mnist_dist, size=size)
        
    dist = TestDist(mnist_sampler, 'MNIST', (0, 0.6), 20, (n,), n)
    
    return dist


def FashionMNIST(n):
    # Look for file in current directory or parent directory
    filename = 'fashionmnist_dist_scaled.npy'
    if not os.path.exists(filename) and os.path.exists(os.path.join('..', filename)):
        filename = os.path.join('..', filename)
        
    fashionmnist_dist = np.load(filename)
    
    def fashionmnist_sampler(size):
        return np.random.choice(fashionmnist_dist, size=size)

    dist = TestDist(fashionmnist_sampler, 'Fashion MNIST', (0, 0.6), 20, (n,), n)
    
    return dist

def CIFAR(n):
    # Look for file in current directory or parent directory
    filename = 'cifar_dist_scaled.npy'
    if not os.path.exists(filename) and os.path.exists(os.path.join('..', filename)):
        filename = os.path.join('..', filename)
        
    cifar_dist = np.load(filename)
    
    def cifar_sampler(size):
        return np.random.choice(cifar_dist, size=size)
        
    dist = TestDist(cifar_sampler, 'CIFAR', (0, 0.6), 20, (n,), n)
    
    return dist

def QCHEM(n):
    # Look for file in current directory or parent directory
    filename = 'qchem_dist_scaled.npy'
    if not os.path.exists(filename) and os.path.exists(os.path.join('..', filename)):
        filename = os.path.join('..', filename)
        
    qchem_dist = np.load(filename)
    
    def qchem_sampler(size):
        return np.random.choice(qchem_dist, size=size)
        
    dist = TestDist(qchem_sampler, 'QCHEM', (0, 0.6), 20, (n,), n)
    
    return dist

def soil(n, size):
    filename = 'soil'+size+'_scaled.npy'
    if not os.path.exists(filename) and os.path.exists(os.path.join('..', filename)):
        filename = os.path.join('..', filename)
    soil_dist = np.load(filename)
    
    def soil_sampler(size):
        return np.random.choice(soil_dist, size=size)
        
    dist = TestDist(soil_sampler, 'Soil'+size, (0, 0.6), 20, (n,), n)
    
    return dist

def dm(n, size):
    filename = 'dm'+size+'_scaled.npy'
    if not os.path.exists(filename) and os.path.exists(os.path.join('..', filename)):
        filename = os.path.join('..', filename)
    dm_dist = np.load(filename)
    
    def dm_sampler(size):
        return np.random.choice(dm_dist, size=size)
        
    dist = TestDist(dm_sampler, 'dm'+size, (0, 0.6), 20, (n,), n)
    
    return dist
