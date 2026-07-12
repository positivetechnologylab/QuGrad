from ML import *
import os
from custom_executor import CustomCircuitExecutor

# Create a single instance of our executor to be used throughout
circuit_executor = CustomCircuitExecutor()

### Make Folders ### -----------------------------------------------------------

def makeFolders(categoriesList, currPath = ''):

    '''
    REQUIRES: categoriesList should be a list of lists of folder names.

    ENSURES: For the first list of folders in categories list, each folder within 
    the list will contain all of the folders in the next category, which will 
    each contain all of the folders in the next category, and so on. If this is 
    confusing, it may be helpful to just run it.

    This function really only exists to ensure that the proper folders exist to
    store the necessary files.     
    '''

    cwd = os.getcwd()
    category = categoriesList[0]
    for item in category:
        a = f'{item}'
        path = os.path.join(cwd, *currPath, a)
        print(path)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        if len(categoriesList) == 1:
            pass
        else:
            newPath = (*currPath, a)
            print(newPath)
            makeFolders(categoriesList[1:], newPath) 

def getFilePath(Ansatz, dist, num):

    '''
    Gets the filepath for a specific test, based on the current file setup.
    Use specifically for tests with distributions (new setup).
    '''

    name, qubits, depth = Ansatz.name, Ansatz.qubits, Ansatz.depth
    dname = dist.name
    filepath = f'{name}/{dname}/{qubits}/{depth}/{num}/{name}_{qubits}_{depth}'
   
    if num != 1:
        filepath = filepath + f'_no{num}'
    
    return filepath


### Distribution functions ### -------------------------------------------------

def CEntanglement(initialStates, params, Ansatz):
    """
    Given initialStates and params, runs the Ansatz with the assigned params on all
    of the initial states and returns their concentratable entanglement values.
    
    Returns:
        tuple: (results, initialStates) where results[i] corresponds to initialStates[i]
    """
    num = dimToNumber(params.shape)
    params = np.reshape(params, (num))
    parameterList = Ansatz.currCirc.parameters
    results = []
    sample = initialStates
    N = len(sample)

    circuits = N*[Ansatz.currCirc]
    paramAssignments = []  
    for i in range(N):
        currAssignments = list(map(curriedF(params, sample[i]), parameterList))
        paramAssignments.append(currAssignments)

    job = circuit_executor.run(circuits, paramAssignments, shots=2048)
    result = job.result()
    dist = result.quasi_dists
    for i in range(N):
        res = dist[i]
        prob = res[0]
        result = 1 - prob
        results.append(result)
     
    return (results, initialStates)


def makeDist(Ansatz, params, filePath, sample=1000, plot=False):
    """
    Given an ansatz and parameters, finds the CE for sample states.
    """
    qubits = Ansatz.qubits
    tset = pSampleSet(qubits, sample)
    
    results, extra = CEntanglement(tset, params, Ansatz)
    if plot:
        fig, ax = plt.subplots()
        ax.hist(results, bins=30)
        plt.savefig(f'{filePath}.png')

    return results


def checkAnsatz(Ansatz, dist, sample = 1000, basis = False, filepath = None, 
                num=1):
    

    if filepath == None:
        filepath = getFilePath(Ansatz, dist, num)

    if Ansatz.currCirc == None:
        Ansatz.createTestCircuit()
    
    data = np.load(f'{filepath}.npy')
    initial = np.load(f'{filepath}_x0.npy')
    if not basis:
        distFile = f'{filepath}_dist'
    else:
        distFile = f'{filepath}_dist_basis'
    res = makeDist(Ansatz, data, distFile, sample)
    res2 = makeDist(Ansatz, initial, distFile, sample)
    
    np.save(f'{filepath}_results', res)
    np.save(f'{filepath}_x0_results', res2)


    return (res, res2)


def fullDist(Ansatz,
             dist,
             filepath: str = None,
             num: int = 1,
             bins: int = None,
             resultFile: str = None,
             annealing: bool = True,
             getTVD: bool = False):
    # —————————————————————————————————————————————————————————————————————————
    # Build the resultFile path if not provided
    if resultFile is None:
        name, qubits, depth = Ansatz.name, Ansatz.qubits, Ansatz.depth
        resultpath = f"{name}_{qubits}_{depth}" + (f"_no{num}" if num != 1 else "")
        makeFolders([['Results'], ['Annealing'], ['Dists'], [dist.name]])
        resultFile = f"Results/Annealing/Dists/{dist.name}/{resultpath}"
    # —————————————————————————————————————————————————————————————————————————
    # Build the filepath prefix if not provided
    if filepath is None:
        filepath = getFilePath(Ansatz, dist, num)
        if annealing:
            filepath = f"runs_qmill/{filepath}"
    # —————————————————————————————————————————————————————————————————————————
    # Load your trained‐and‐initial output arrays
    results      = np.load(f"{filepath}_results.npy").flatten()
    init_results = np.load(f"{filepath}_x0_results.npy").flatten()
    # —————————————————————————————————————————————————————————————————————————
    # Regenerate the target sample and coarse bins
    dist.createSampleDistributions(1000)
    # If no bins override, use exactly the same number the optimizer saw
    if bins is None:
        bins = dist.numBoxes
    dist.getAveragedBins(bins)  # this also sets dist.avgDist under the hood
    sample = np.array(dist.samples).flatten()
    # —————————————————————————————————————————————————————————————————————————
    # If user only wants the numeric TVD, compute and return it
    if getTVD:
        # get the two normalized box‐lists
        boxes1 = normalize(dist.getBinsList(sample,    bins, *dist.Range))
        boxes2 = normalize(dist.getBinsList(results,    bins, *dist.Range))
        return TVD2(boxes1, boxes2)
    # —————————————————————————————————————————————————————————————————————————
    # Otherwise, draw the three overlaid histograms using the *same* bins
    fig, ax = plt.subplots()
    ax.hist(sample,     bins=bins, histtype='step', color='r', density=True, label='Target')
    ax.hist(init_results,bins=bins, histtype='step', color='g', density=True, label='Initial')
    ax.hist(results,    bins=bins, histtype='step', color='b', density=True, label='Trained')
    ax.set_xlabel('Concentratable Entanglement')
    ax.set_ylabel('Density')
    ax.legend()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # save into both locations
    plt.savefig(f"{filepath}_dist_test.png")
    plt.savefig(f"{resultFile}_result_dist.png")
    plt.close(fig)


def getBoxes(Ansatz,
             dist,
             filepath: str = None,
             num: int = 1,
             bins: int = None,
             resultFile: str = None,
             annealing: bool = True,
             getTVD: bool = False):
    """
    Exactly the same as fullDist, but returns the box‐lists (and optional TVD)
    rather than plotting.  Bins default to dist.numBoxes.
    """
    # —————————————————————————————————————————————————————————————————————————
    if resultFile is None:
        name, qubits, depth = Ansatz.name, Ansatz.qubits, Ansatz.depth
        resultpath = f"{name}_{qubits}_{depth}" + (f"_no{num}" if num != 1 else "")
        makeFolders([['Results'], ['Annealing'], ['Dists'], [dist.name]])
        resultFile = f"Results/Annealing/Dists/{dist.name}/{resultpath}"
    if filepath is None:
        filepath = getFilePath(Ansatz, dist, num)
        if annealing:
            filepath = f"runs_qmill/{filepath}"
    # —————————————————————————————————————————————————————————————————————————
    # Load outputs
    results      = np.load(f"{filepath}_results.npy").flatten()
    init_results = np.load(f"{filepath}_x0_results.npy").flatten()
    # —————————————————————————————————————————————————————————————————————————
    # Regenerate and bin
    dist.createSampleDistributions(1000)
    if bins is None:
        bins = dist.numBoxes
    dist.getAveragedBins(bins)
    sample = np.array(dist.samples).flatten()
    # —————————————————————————————————————————————————————————————————————————
    boxes1 = normalize(dist.getBinsList(sample,     bins, *dist.Range))
    boxes2 = normalize(dist.getBinsList(results,     bins, *dist.Range))
    if getTVD:
        return TVD2(boxes1, boxes2)
    # Otherwise return the raw box‐lists
    return boxes1, boxes2


def getLossCurve(Ansatz, dist, filepath = None, plot = True, num=1):
    if filepath == None:
        filepath = getFilePath(Ansatz, dist, num)

    data = np.load(f'{filepath}_loss.npy')
    if plot:
        fig, ax = plt.subplots()
        x = list(range(len(data)))
        F = data[x]
        ax.plot(x, F)
        plt.savefig(f'{filepath}_lossCurve.png')
    return data
