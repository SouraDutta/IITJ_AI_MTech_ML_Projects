import numpy as np
import requests
import scipy
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn
from itertools import cycle, islice, permutations
import sys

from cust_kmeans import KMeans

class Spectral(object):
    
    def __init__(self):
        # self.arg = arg
        pass
        

    def squared_exponential(self, x, y, sig=0.8, sig2=1):
        x= x.T
        y= y.T
        norm = np.linalg.norm(x - y)
        dist = norm * norm
        return np.exp(- dist / (2 * sig * sig2))


    # affinity calculation

    def compute_affinity(self, X):
        N = X.shape[0]
        ans = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                #print(X[i], X[j])
                ans[i][j] = self.squared_exponential(X[i], X[j])
        return ans

    
    # finding normalized laplacian
    def laplacian(self, A):
        
        D = np.zeros(A.shape)
        w = np.sum(A, axis=0)
        D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
        return D.dot(A).dot(D)

    # clustering code

    def spectral_clustering(self, data_array, n_clusters):

        affinity = self.compute_affinity(data_array)
        L = self.laplacian(affinity)
        
        eig_val, eig_vect = scipy.sparse.linalg.eigs(L, n_clusters)
        X = eig_vect.real
        
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit(X)
        return labels

    # permute labels to get maximum accuracy

    def remap_labels(self, true_labels, pred_labels):
        
        pred_labels, true_labels = np.array(pred_labels), np.array(true_labels)
        assert pred_labels.ndim == 1 == true_labels.ndim
        assert len(pred_labels) == len(true_labels)
        cluster_names = np.unique(pred_labels)
        accuracy = 0

        perms = np.array(list(permutations(np.unique(true_labels))))

        remapped_labels = true_labels
        for perm in perms:
            flipped_labels = np.zeros(len(true_labels))
            for label_index, label in enumerate(cluster_names):
                flipped_labels[pred_labels == label] = perm[label_index]

            testAcc = np.sum(flipped_labels == true_labels) / len(true_labels)
            if testAcc > accuracy:
                accuracy = testAcc
                remapped_labels = flipped_labels

        return accuracy


# main function to run spectral only on dataset

def main(url):

    text_data=""
    data_set=[]
    class_labels = []
    plt.figure(figsize=(10, 10,))

    r = requests.get(url, allow_redirects=True)
    open('dataset.txt', 'wb').write(r.content)

    with open("dataset.txt","r") as f:
        text_data= f.readlines()
    
    # data preparation
    for each_row in text_data:
        row_data =(each_row.split("\t")[:2])
        row_data= [float(i) for i in row_data]
        data_set.append(row_data )
        class_labels.append(int(str((each_row.split("\t")[2:])[0]).replace("\\n","")))
    data_array= np.array(data_set)
    length = data_array.shape[0]
    K = len(set(class_labels))

    print("Spectral clustering on Jain dataset with 2 clusters --")

    label_encoder = LabelEncoder()

    class_labels_cat = label_encoder.fit_transform(class_labels)
    class_labels_list = class_labels_cat.tolist()

    spec = Spectral()
    Y = spec.spectral_clustering(data_array, K)
    
    overall_accuracy = spec.remap_labels(class_labels_list, Y.tolist())
    print("overall_accuracy", overall_accuracy)

    colors = np.array(list(islice(cycle(seaborn.color_palette()), int(max(Y) + 1))))
    plt.scatter(data_array[:, 0], data_array[:, 1], color=colors[Y], s=6, alpha=0.6)
    plt.savefig("spectral.png")


if __name__ == "__main__":

    url = sys.argv[1] if len(sys.argv) > 1 else None

    if(url is None):
        sys.exit('No input url!')

    main(url)