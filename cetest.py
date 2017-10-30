import sympy as s
import numpy as np
x1 = s.Symbol('x1')
x2 = s.Symbol('x2')
x3 = s.Symbol('x3')
x4 = s.Symbol('x4')
O = s.Symbol('O')

X = np.array([x1, x2, x3, x4])


def make_qnu(LNU):
	nXh = np.sum(range(len(LNU)+1)) 
	Xh = np.full(nXh, O)

	idxh = 0
	for idx, xi in enumerate(LNU):
		for xj in LNU[idx:]:
			Xh[idxh] = xi*xj
			idxh += 1 

	return Xh

def make_cnu(LNU):
	nXh = np.sum(range(len(LNU)**2+1)) 
	Xh = np.full(nXh, O)

	idxh = 0
	for idi, xi in enumerate(LNU):
		for idj, xj in enumerate(LNU[idi:]):
			for idk, xk in enumerate(LNU[idj:]):
				Xh[idxh] = xi*xj*xk
				idxh += 1 

	return Xh


LNU = X
QNU = make_qnu(LNU)
CNU = make_cnu(LNU)


print LNU
print QNU
print CNU
