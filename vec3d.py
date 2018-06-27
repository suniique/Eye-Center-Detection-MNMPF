import numpy as np


def midPoint(X1, W1, X2, W2):
    ''' calulate the mid-point of the Common Perpendicular of lines on different planes'''
    W1 = np.mat(W1).T / np.linalg.norm(W1)
    W2 = np.mat(W2).T / np.linalg.norm(W2)
    X1 = np.mat(X1).T
    X2 = np.mat(X2).T

    T1 = np.mat(np.diag([1, 1, 1])) - W1 * W1.T / (W1.T * W1)
    T2 = np.mat(np.diag([1, 1, 1])) - W2 * W2.T / (W2.T * W2)

    P = (T1 + T2).I * (T1 * X1 + T2 * X2)
    return P

def zPoint(X, W, h):
    t = (h - X[2]) / W[2]
    P = t * W + X
    return P

def zMidPoint(X1, W1, X2, W2, h):
    P1 = zPoint(X1, W1, h)
    P2 = zPoint(X2, W2, h)
    return (P1 + P2) / 2


def main():
    p = midPoint([0, 0, 0], [1, 0, 0], [1, 1, 1], [0, -1, 0])
    print(p)


if __name__ == '__main__':
    main()
