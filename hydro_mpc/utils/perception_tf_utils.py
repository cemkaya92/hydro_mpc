import numpy as np

def vec(t): return np.array(t, dtype=np.float64).reshape(3,)
def mat(R): return np.array(R, dtype=np.float64).reshape(3,3)

def se3(R,t):
    T = np.eye(4)
    T[:3,:3] = R; T[:3,3] = t
    return T

def transform(T, p):
    p4 = np.ones(4); p4[:3] = p
    return (T @ p4)[:3]

def compose(A,B):
    return A @ B

def invert(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti
