#!/usr/bin/env python

import numpy as np, os
import numpy.linalg as npl
import sys
import time

start_time = time.time()

np.set_printoptions(linewidth = 200)

molGeom = open('.\\Input_Files\\water.txt', 'r')
geomCont = molGeom.readlines()

hessRaw = open('.\\Input_Files\\waterHessian.txt', 'r')
hessCont = hessRaw.readlines()
molGeom.close()
hessRaw.close()

try:
	if(hessCont[0] != geomCont[0]):
		raise Exception("Number of Atoms in input and inputHessian do not match!")
except Exception as err:
	print("ERROR: " + str(err))
	sys.exit()
	
atomCount = int(geomCont[0].replace('\n', ''))
geomCont = geomCont[1:]
hessCont = hessCont[1:]

def pos(content):

	posMatrix = np.array([])

	for i in content:

		i = i.split('\t')
		i[-1] = i[-1].replace('\n','')
		i = i[1:]
		i = [float(x) for x in i]
		posMatrix = np.append(posMatrix, i)

	posMatrix = np.reshape(posMatrix, (len(content),3))

	return posMatrix

def atomicMass(content):

	an = np.array([])

	for i in content:

		i = i.split('\t')
		i = i[:1]
		i = [float(x) for x in i]
		an = np.append(an,i)

	for x in range(len(an)):
		if(an[x] == 6.):
			an[x] = 12.
		elif(an[x] == 8):
			an[x] = 16

	return an

def hessianDat(hess):
	
	hessMatrix = np.array([])

	for i in hess:
		i = i.split('\t')
		i[-1] = i[-1].replace('\n','')
		i = [float(x) for x in i]
		hessMatrix = np.append(hessMatrix, i)

	hessMatrix = np.reshape(hessMatrix, (atomCount*3,atomCount*3))

	return hessMatrix

def massWeight(H, masses):

	I = np.zeros(atomCount*3)

	for i in range(0, atomCount*3,3):

		I[i] = masses[int(i/3)]
		I[i+1] = masses[int(i/3)]
		I[i+2] = masses[int(i/3)]

	M = np.outer(I,I)
	M = np.sqrt(M)
	H = np.divide(H,M)

	return H

def hessFrequencies(H):

	evec, ev = npl.eigh(H)
	evec = np.sort(evec)
	evec = evec[::-1]

	evec = [np.sqrt(evec[i]*(1.0432*10**9))/(2*np.pi) for i in range(len(evec))]

	return evec


def v_(atom1, atom2, positions = pos(geomCont)):

	v_ij = (positions[atom2]-positions[atom1])
	return v_ij

def main(geometry, hessian):

	positions = pos(geometry)
	atomicMasses = atomicMass(geometry)
	hess = hessianDat(hessian)
	MH = massWeight(hess,atomicMasses)
	ev = hessFrequencies(MH)


	print("\nPositions:")
	print(positions)
	print("\nHessian:")
	print(hess)
	print("\nMass Weighted Hessian:")
	print(MH)
	print("\nHessian Eigenvalues:")

	print(np.around(ev, 6))
	

	return None

main(geomCont,hessCont)
print("--- %s seconds ---" % (time.time() - start_time))