"""

"""
#==============================================================================
# Loading the data
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

random_state = 32
dataset = fetch_olivetti_faces(shuffle=True, random_state=random_state)

X, y = dataset['data'], dataset['target']
n_x, n_y = dataset['images'][0].shape

X_data = X.reshape(-1, n_x, n_y).transpose(1, 2, 0)
#==============================================================================
# Get the connectivity
from sklearn.feature_extraction.image import grid_to_graph
from rena import weighted_connectivity_graph

connectivity_ward = grid_to_graph(n_x, n_y, 1)

mask = np.ones((n_x, n_y))
connectivity_rena = weighted_connectivity_graph(X_data, n_features=X.shape[1],
                                                mask=mask)
#==============================================================================
# Custering
from sklearn.cluster import AgglomerativeClustering
from rena import recursive_nearest_agglomeration

n_clusters = 250

ward = AgglomerativeClustering(n_clusters=n_clusters,
                               connectivity=connectivity_ward, linkage='ward')

labels_rena = recursive_nearest_agglomeration(X, connectivity_rena,
                                              n_clusters=n_clusters)

ward.fit(X.T)
labels_ward = ward.labels_
#==============================================================================
# Custering
from rena import reduce_data, approximate_data

X_red_rena = reduce_data(X, labels_rena)
X_red_ward = reduce_data(X, labels_ward)

X_approx_rena = approximate_data(X_red_rena, labels_rena)
X_approx_ward = approximate_data(X_red_ward, labels_ward)
#==============================================================================
# Visualize
import matplotlib.pyplot as plt
plt.close('all')

# X_approx_rena[].reshape(n_x, n_y)

# # fig, axx = plt.subplots(2, 4, **{'figsize': (10, 5)})
# # plt.gray()

# # for i in range(4):
# #     axx[0, i].imshow(X_approx_rena.re, i + 30])
# #     axx[0, i].set_axis_off()
# #     axx[0, i].set_title('Original')
# #     axx[1, i].imshow(imgs_compressed.get_data()[:, :, 0, i + 30])
# #     axx[1, i].set_axis_off()
# #     axx[1, i].set_title('Compressed')

# # fig.savefig('figures/faces.png', bbox_to_inches='tight')
# # plt.show()
