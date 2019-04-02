import numpy as np
import kmeans as km

# Calculando a distância do centro
def rbfGaus(x, c, s):
    return np.exp(-1 / (2 * s ** 2) * (km.distance(x,c)) ** 2)

def rbfMult(x,c,s):
    return (1 + (km.distance(x,c)) ** 2)**0.5

def rbfMultIn(x,c,s):
    return 1/(1+(km.distance(x,c)) ** 2)**0.5

class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbfGaus, inferStds=True):
        self.k = k  # grupos
        self.lr = lr
        self.epochs = epochs  # número de iterações
        self.rbf = rbf
        self.inferStds = inferStds  # se vai calcular o tamanho do cluster (std)

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y):
        if self.inferStds:
            # compute stds from data
            self.centers, self.stds = km.k_means(X, self.k)
        else:
            # use a fixed std
            self.centers, _ = km.k_means(X, self.k)
            dMax = max([np.abs(c1 - c2) for c1 in self.centers for c2 in self.centers])
            self.stds = np.repeat(dMax / np.sqrt(2 * self.k), self.k)

        # training
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
                # print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - F).flatten()

                # online update
                #chu = np.dot(a , error)
                self.w = self.w - self.lr *a*error
                self.b = self.b - self.lr * error

    def predict(self, X):
        y_pred = []
        error = 0
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)

        return np.array(y_pred)
