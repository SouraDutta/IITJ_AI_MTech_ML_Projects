import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn
from itertools import cycle, islice, permutations
import sys

class KMeans:

    def __init__(self, n_clusters=4):
        self.K = n_clusters

    #recursive function to fit kmeans
    def rec_fit(self, X, prev_label, centroids):

        self.labels = np.array([np.argmin(np.sqrt(np.sum((self.centroids - j) ** 2, axis=1))) for j in X])
        # print(self.labels)
        self.centroids = np.array([np.mean(X[self.labels == k], axis=0) for k in range(self.K)])

        if( not np.all(self.labels == prev_label)):
            return self.rec_fit(X, self.labels, self.centroids)
        else:
            return self.labels

    # calls the recursive function

    def fit(self, X):
        self.centroids = X[np.random.choice(len(X), self.K, replace=False)]
        return self.rec_fit(X, np.zeros(len(X)), self.centroids)


# permute labels to get maximum accuracy

def remap_labels(true_labels, pred_labels):
    
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

# main function to run kmeans only on dataset

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

    label_encoder = LabelEncoder()
    # print(class_labels)
    class_labels_cat = label_encoder.fit_transform(class_labels)
    class_labels_list = class_labels_cat.tolist()
    K = len(set(class_labels_list)) 

    print("KMeans clustering on Jain dataset with 2 clusters -- ")
    
    km = KMeans(K)
    Y = km.fit(data_array)

    overall_accuracy = remap_labels(class_labels_list,Y.tolist())
    print("overall_accuracy",overall_accuracy)

    colors = np.array(list(islice(cycle(seaborn.color_palette()), int(max(Y) + 1))))
    plt.scatter(data_array[:, 0], data_array[:, 1], color=colors[Y], s=6, alpha=0.6)
    plt.savefig('kmeans.png')

if __name__ == "__main__":

    url = sys.argv[1] if len(sys.argv) > 1 else None

    if(url is None):
        sys.exit('No input url!')

    main(url)