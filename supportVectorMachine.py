import numpy as np

## Linear SVM implementation based on platt's paper
## The code can be adjusted to fit non-linear datasets as well.

class svm:

    ## constructor with x as the n-dimensional input and y as the expected output. 
    def __init__(self, x, y, maxPasses = 100, bias = 0, C = 1):

        self.maxPasses = maxPasses
        self.x = x
        self.y = y
        self.m, self.n = self.x.shape
        self.alphas = np.zeros((self.m, 1))
        self.bias = bias
        self.passes = 0
        self.C = C
        self.weights = []
        self.intercept = []
        self.smo()

    # Selecting two random integers such that they are not equal to each other
    def randSelect(self, i):

        r = np.random.randint(0, self.m)
        if r != i:
            return r
        else:
            while (r == i):
                r = np.random.randint(0, self.m)
            return r
        
    # Calculates the error between the actual output and the measured output.
    def calcError(self, x, y, w, b):
        return np.sign(np.dot(w.T, x.T) + b).astype(int) - y

    def computeLH(self, ai, aj, C, yi, yj):
        if(yi != yj):
            return (max(0, aj-ai), min(C, C + aj - ai))
        else:
            return (max(0, ai + aj - C), min(C, ai + aj))

    def computeN(self, xi, xj):
        return -2*np.dot(xi, xj) + np.dot(xi, xi) + np.dot(xj, xj)

    # clipping alpha to limit to a CxC box
    def clipAlpha(self, a, L, H):

        ap = max(a, L)
        return min(ap, H)

    # This function finds the model parameters given the support vectors, alphas.
    def findBW(self, a, x, y):

        w = np.dot(x.T, np.multiply(a,y))
        b = y - np.dot(x, w)
        return [w, np.mean(b)]

    # SMO selects two alphas randomly and optimises their value keeping the others constant.
    # It is an extension of the coordinate ascent algorithm. It can be proved that most
    # alphas are zero. Only a few of them are non-zero and their indices
    # correspond to the support vectors in the input dataset.
    # Refer to platt's paper for full definitions.
    
    def smo(self):

        oldAlphas = np.zeros((self.m, 1))
        while self.passes <= self.maxPasses:
            oldAlphas = np.copy(self.alphas)
            for j in range(0, self.m):
                i = self.randSelect(j)
                eta = self.computeN(self.x[i], self.x[j])
                if eta == 0:
                    continue
                oAi = self.alphas[i]
                oAj = self.alphas[j]
                [L, H] = self.computeLH(oAi, oAj, self.C, self.y[i], self.y[j])
                [w, b] = self.findBW(self.alphas, self.x, self.y)
                Ei = self.calcError(self.x[i], self.y[i], w, b)   
                Ej = self.calcError(self.x[j], self.y[j], w, b)
                self.alphas[j] = oAj + (self.y[j]*(Ei - Ej))/eta
                self.alphas[j] = self.clipAlpha(self.alphas[j], L, H)
                self.alphas[i] = oAi + self.y[i]*self.y[j]*(oAj - self.alphas[j])
            
            self.passes += 1
        self.weights = w
        self.intercept = b

    # Predicting the class of a test vector, x.
    def predict(self, x):
        return np.sign(np.dot(x, self.weights) + self.intercept).astype(int)
        
def main():

    ## For demonstration, a dataset with two features (100 samples) has been created.
    ## 50 points fall under the region 2 to 3. The other 50 fall under 5 to 6.
    ## Output classes are -1 and 1.
    
    a = 2 + np.random.rand(50,2)
    b = 5 + np.random.rand(50,2)
    x = np.concatenate((a, b), axis=0)

    y1 = -1*np.ones((50, 1))
    y2 = np.ones((50, 1))
    y = np.concatenate((y1, y2), axis = 0)

    dataObject = svm(x, y)
    t = np.array([2, 2])
    print(dataObject.weights)
    print(dataObject.intercept)
    print(dataObject.predict(t))

if __name__ == '__main__':
    main()
