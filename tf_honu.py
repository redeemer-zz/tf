import numpy as np

def add_bias(X):
    bias_vec = np.ones([X.shape[0],1])
    return np.hstack((bias_vec, X))

def make_qnu_line(line):
    nXh = np.sum(range(len(line)+1)) 
    Xh = np.zeros(nXh)

    idxh = 0
    for idx, xi in enumerate(line):
        for xj in line[idx:]:
            Xh[idxh] = xi*xj
            idxh += 1 

    return Xh

def make_lnu(X):
    return add_bias(X)

def make_qnu(X, bias = False):
    nXh = np.sum(range(X.shape[1]+1)) 
    Xh = np.zeros([X.shape[0], nXh])

    for row_idx in range(X.shape[0]):
        Xh[row_idx,:] = make_qnu_line(X[row_idx,:])
    if bias: Xh = add_bias(Xh)
    return Xh

def honu_net(HONUs = []):
    net = HONUs[0]
    for idx in range(1, len(HONUs)):
        net = np.hstack((net, HONUs[idx]))
    return net

def xavier_init(n):
    x = np.sqrt(6./(n))
    return np.random.uniform(-x, x, n).astype(np.float32)
