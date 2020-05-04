import numpy as np
from typing import List, Any, NoReturn


class Node:
    def __init__(self, X, Y, index: List[int], min_leaf_sample: int,
                 depth: int):
        '''
        A class for the Node of Regression Tree
        :param X: full 2-dimension training input X 
        :type X: numpy array,e.g.shape=(dim0,dim1)
        :param Y: full training input Y
        :type Y: numpy array,e.g.shape=(dim0,)
        :param index: index in this Node for splitting
        :param min_leaf_sample: the minimum sample number expected in the leaf node
        :param depth:  the depth of this node
        :type depth: none negtive int

        '''
        self.index = index
        self.x = X[self.index]
        self.y = Y[self.index]
        self.value = np.mean(self.y)
        self.min_leaf_sample = min_leaf_sample
        self.left = None
        self.right = None

        self.depth = depth

    def split(self) -> List:
        '''
        find the best split point with traversing method
        and return the nodes after splitting
        '''
        res = (np.Inf, None, None, None, None)
        for s in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                indexl, indexr = self.x[:, j] < self.x[
                    s, j], self.x[:, j] >= self.x[s, j]
                yl, yr = self.y[indexl], self.y[indexr]
                # check the one branch situation
                if yl.shape[0] and yr.shape[0]:
                    sse = np.sum((yl - np.mean(yl))**2) + np.sum(
                        (yr - np.mean(yr))**2)
                else:
                    continue
                if sse < res[0]:
                    res = (sse, s, j, [
                        self.index[i] for i, flag in enumerate(indexl) if flag
                    ], [
                        self.index[i] for i, flag in enumerate(indexr) if flag
                    ])
        callback = []
        if res[1] is None:
            return callback
        else:
            self.sse, self.s, self.j = res[0:3]
            self.threshold = self.x[self.s, self.j]
            if len(res[3]) > self.min_leaf_sample and len(
                    res[4]) > self.min_leaf_sample:
                self.left = Node(X, Y, res[3], self.min_leaf_sample,
                                 self.depth + 1)
                self.right = Node(X, Y, res[4], self.min_leaf_sample,
                                  self.depth + 1)
                callback += [self.left, self.right]
            return callback

    def predict(self, x) -> Any:
        '''
        return predicted value in a recursion way
        '''
        if self.left is None and self.right is None:
            return self.value
        else:
            if x[self.j] > self.threshold:
                return self.right.predict(x)
            else:
                return self.left.predict(x)

    def __repr__(self) -> str:
        return "Node: index:(%s) depth(%d)" % (str(self.index), self.depth)


class RegressionTree:
    def __init__(self, min_leaf_sample=1, max_depth=None):
        '''
        A class for Regression Tree build and prediction
        :param min_leaf_sample: minimum sample numer in a leaf node
        :param max_depth: maximum depth of the regression tree
        '''
        self.root = None
        self.min_leaf_sample = min_leaf_sample
        self.max_depth = max_depth

    def fit(self, X, Y) -> NoReturn:
        '''
        build the regression tree with breath-first search
        '''
        self.queue = []
        self.root = Node(X,
                         Y,
                         list(range(X.shape[0])),
                         self.min_leaf_sample,
                         depth=0)
        self.queue += self.root.split()
        while len(self.queue):
            head = self.queue.pop(0)
            if self.max_depth is not None and head.depth > self.max_depth:
                break
            self.queue += head.split()

    def predict(self, X):
        return np.array([self.root.predict(X[i]) for i in range(X.shape[0])])


if __name__ == "__main__":
    # simple test case
    X = np.random.random((100, 10))
    Y = np.random.randint(0, 2, (100, ))
    regression = RegressionTree(min_leaf_sample=3, max_depth=None)
    regression.fit(X, Y)
    sse = np.sum((regression.predict(X) - Y)**2)
    print("sse: %.4f" % sse)
