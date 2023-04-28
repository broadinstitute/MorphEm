import numpy as np
import faiss

########################################################
## KNN Classifier
########################################################

class FaissKNeighbors:
    def __init__(self, k=5, use_gpu: bool):
        self.index = None
        self.y = None
        self.k = k
        self.gpu = faiss.StandardGpuResources() if use_gpu else None

    def fit(self, X, y):
        if self.gpu:
            self.index = faiss.GpuIndexFlatL2(self.gpu, X.shape[1])
        else:
            self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions