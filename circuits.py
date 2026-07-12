import qiskit
import qiskit.circuit.library as qcl
import numpy as np
from classes import *

def ansatz_FiveDeprecated(circuit, thetas, depth):
    num = circuit.num_qubits

    for d in range(depth):
        for i in range(num):
            circuit.rx(thetas[d][i][0], i)
            circuit.rz(thetas[d][i][1], i)
        for i in range(num-1, -1, -1):
            for j in range(num-1, -1, -1):
                if i == j:
                    pass
                elif i > j:
                    circuit.crz(thetas[d][i][j+2], i, j)
                else:
                    circuit.crz(thetas[d][i][j+1], i, j)
        circuit.barrier()
        for i in range(num):
            circuit.rx(thetas[d][i][num + 1], i)
            circuit.rz(thetas[d][i][num + 2], i)

def ansatz_Five(circuit, thetas, depth):
    num = circuit.num_qubits

    for d in range(depth):
        for i in range(num):
            circuit.rx(thetas[d*depth + 0][i], i)
            circuit.rz(thetas[d*depth + 1][i], i)
        for i in range(num-1, -1, -1):
            for j in range(num-1, -1, -1):
                if i == j:
                    pass
                elif i > j:
                    circuit.crz(thetas[d*depth + j + 2][i], i, j)
                else:
                    circuit.crz(thetas[d*depth + j + 1][i], i, j)
        circuit.barrier()
    for i in range(num):
        circuit.rx(thetas[depth*num][i], i)
        circuit.rz(thetas[depth*num + 1][i], i)

def Five(qubits, depth):
    if depth == 1:
        return Ansatz(ansatz_FiveDeprecated, (depth, qubits, qubits + 3), qubits, depth, name = 'Five')
    else:
        return Ansatz(ansatz_Five, ((depth*qubits+1)+2, qubits), qubits, depth, name = 'Five')

def ansatz_SixDeprecated(circuit, thetas, depth):
    num = circuit.num_qubits

    for d in range(depth):
        for i in range(num):
            circuit.rx(thetas[d][i][0], i)
            circuit.rz(thetas[d][i][1], i)
        for i in range(num-1, -1, -1):
            for j in range(num-1, -1, -1):
                if i == j:
                    pass
                elif i > j:
                    circuit.crx(thetas[d][i][j+2], i, j)
                else:
                    circuit.crx(thetas[d][i][j+1], i, j)
        circuit.barrier()
        for i in range(num):
            circuit.rx(thetas[d][i][num + 1], i)
            circuit.rz(thetas[d][i][num + 2], i)

def ansatz_Six(circuit, thetas, depth):
    num = circuit.num_qubits

    for d in range(depth):
        for i in range(num):
            circuit.rx(thetas[d*depth + 0][i], i)
            circuit.rz(thetas[d*depth + 1][i], i)
        for i in range(num-1, -1, -1):
            for j in range(num-1, -1, -1):
                if i == j:
                    pass
                elif i > j:
                    circuit.crx(thetas[d*depth + j + 2][i], i, j)
                else:
                    circuit.crx(thetas[d*depth + j + 1][i], i, j)
        circuit.barrier()
    for i in range(num):
        circuit.rx(thetas[depth*num][i], i)
        circuit.rz(thetas[depth*num + 1][i], i)


def Six(qubits, depth):
    if depth == 1:
        return Ansatz(ansatz_SixDeprecated, (depth, qubits, qubits + 3), qubits, depth, name = 'Six')
    else:
        return Ansatz(ansatz_Six, ((depth*qubits+1)+2, qubits), qubits, depth, name = 'Six')



def ansatz_Thirteen(circuit, thetas, depth):
    num = circuit.num_qubits

    for d in range(depth):
        for i in range(num):
            circuit.ry(thetas[d][i][0], i)
        circuit.barrier()
        for i in range(num-1, -1, -1):
            circuit.crz(thetas[d][i][1], i, (i+1)%num)
        circuit.barrier()
        for i in range(num):
            circuit.ry(thetas[d][i][2], i)
        circuit.barrier()
        for i in range(num):
            circuit.crz(thetas[d][i][3], (i - 1) % num, (i - 2) % num)

def Thirteen(qubits, depth):
    return Ansatz(ansatz_Thirteen, (depth, qubits, 4), qubits, depth, name = 'Thirteen')



def ansatz_Fourteen(circuit, thetas, depth):
    num = circuit.num_qubits

    for d in range(depth):
        for i in range(num):
            circuit.ry(thetas[d][i][0], i)
        for i in range(num-1, -1, -1):
            circuit.crx(thetas[d][i][1], i, (i+1)%num)
        for i in range(num):
            circuit.ry(thetas[d][i][2], i)
        for i in range(num):
            circuit.crx(thetas[d][i][3], (i - 1) % num, (i - 2) % num)

def Fourteen(qubits, depth):
    return Ansatz(ansatz_Fourteen, (depth, qubits, 4), qubits, depth, name = 'Fourteen')


def ansatz_Sixteen(circuit, thetas, depth):
    num = circuit.num_qubits

    for d in range(depth):
        i = 0
        for j in range(num):
            circuit.rx(thetas[d][i], j)
            i += 1
        for j in range(num):
            circuit.rz(thetas[d][i], j)
            i += 1
        for j in range(num - 1):
            if j % 2 == 1:
                pass
            else:
                circuit.crz(thetas[d][i], j+1, j)
                i += 1
        for j in range(num - 1):
            if j % 2 == 0:
                pass
            else:
                circuit.crz(thetas[d][i], j+1, j)
                i += 1

def Sixteen(qubits, depth):
    return Ansatz(ansatz_Sixteen, (depth, 3*qubits - 1), qubits, depth, name = 'Sixteen')
    



def ansatz_Seventeen(circuit, thetas, depth):
    num = circuit.num_qubits

    print(thetas)
    for d in range(depth):
        i = 0
        for j in range(num):
            circuit.rx(thetas[d][i], j)

            i += 1
        for j in range(num):
            circuit.rz(thetas[d][i], j)
            i += 1
            
        for j in range(num - 1):
            if j % 2 == 1:
                pass
            else:
                circuit.crx(thetas[d][i], j+1, j)
                i += 1
        for j in range(num - 1):
            if j % 2 == 0:
                pass
            else:
                circuit.crx(thetas[d][i], j+1, j)
                i += 1

def Seventeen(qubits, depth):
    return Ansatz(ansatz_Seventeen, (depth, 3*qubits - 1), qubits, depth, name = 'Seventeen')


def ansatz_Custom_One(circuit, thetas, depth):
    num = circuit.num_qubits
    
    for d in range(depth):
        i = 0
        for j in range(num):
            circuit.h(j)

        for j in range(num):
            circuit.ry(thetas[d][i], j)
            i += 1

        for j in range(num-1):
            circuit.crx(thetas[d][i], j, j+1)
            i += 1
            
        for j in range(num):
            circuit.rz(thetas[d][i], j)
            i += 1
            
        circuit.barrier()

def Custom_One(qubits, depth):
    return Ansatz(ansatz_Custom_One, (depth, 3*qubits-1), qubits, depth, name='Custom_One')

def ansatz_Custom_Two(circuit, thetas, depth):
    num = circuit.num_qubits
    
    for d in range(depth):
        i = 0
        for j in range(num):
            circuit.rx(thetas[d][i], j)
            i += 1
            
        for j in range(num-1):
            circuit.cx(j, j+1)
            circuit.ry(thetas[d][i], j+1)
            i += 1
            
        for j in range(num-1, 0, -1):
            circuit.crz(thetas[d][i], j, j-1)
            i += 1
            
        circuit.barrier()

def Custom_Two(qubits, depth):
    return Ansatz(ansatz_Custom_Two, (depth, 3*qubits-1), qubits, depth, name='Custom_Two')

#Takes in a bistring, and returns a gate that takes the all-zero state to the 
#basis state represented by the bitstring
def bitsToRef(bits):
    n = len(bits)
    temp = qiskit.QuantumCircuit(n)
    for i in range(n):
        if bits[i] == 1:
            temp.x(i)
    temp.to_gate()
    return temp



def ansatz_empty(circuit, thetas, depth):
    return circuit

def Empty(qubits, depth):
    return Ansatz(ansatz_empty, 0, qubits, depth)



def full_circuit(ansatz, params, qubits):
    circ1 = qiskit.QuantumCircuit(qubits)
    (U_r, thetas, depth) = params
    ansatz(circ1, thetas, depth)
    circ1.to_instruction()

    actualCirc = qiskit.QuantumCircuit(3*qubits, qubits)
    actualCirc.append(U_r, range(qubits, 2*qubits))
    actualCirc.append(U_r, range(2*qubits, 3*qubits))
    actualCirc.append(circ1, range(qubits, 2*qubits))
    actualCirc.append(circ1, range(2*qubits, 3*qubits))
    swap_test(actualCirc)

    return actualCirc