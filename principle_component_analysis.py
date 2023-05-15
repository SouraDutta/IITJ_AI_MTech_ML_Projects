import numpy as np

from sklearn.preprocessing import LabelEncoder



def PCA(X, num_components):
    # Step-1
    X_meaned = X - np.mean(X, axis=0)

    # Step-2
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # print(sorted_index, eigen_values)
    # Step-5
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced, X_meaned, np.dot(X_reduced, eigenvector_subset.transpose())

def recon_error(old, recon):
    
    return np.linalg.norm(np.array(old) - np.array(recon), ord='fro')




############################################################################
# from sklearn.preprocessing import StandardScaler
# X_scaled = StandardScaler().fit_transform(X)
#
#
# features = X_scaled.T
# cov_matrix = np.cov(features)
#
# values, vectors = np.linalg.eig(cov_matrix)
#
# explained_variances = []
# for i in range(len(values)):
#     explained_variances.append(values[i] / np.sum(values))
#
# print(np.sum(explained_variances), '\n', explained_variances)
#
# projected_1 = X_scaled.dot(vectors.T[0])
# projected_2 = X_scaled.dot(vectors.T[1])
# res = pd.DataFrame(projected_1, columns=['PC1'])
# res['PC2'] = projected_2
# res['Y'] = y
# res.head()
#
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# plt.figure(figsize=(20, 10))
# sns.scatterplot(res['PC1'], [0] * len(res), hue=res['Y'], s=200)
#
# plt.figure(figsize=(20, 10))
# sns.scatterplot(res['PC1'], [0] * len(res), hue=res['Y'], s=100)