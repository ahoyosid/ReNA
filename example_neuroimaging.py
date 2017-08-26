"""
"""
from nilearn import datasets
dataset = datasets.fetch_haxby(n_subjects=1)

import numpy as np
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_strategy='epi', smoothing_fwhm=6, memory='cache')

X_masked = masker.fit_transform(dataset.func[0])
X_train = X_masked[:100, :]

X_data = masker.inverse_transform(X_train).get_data()

n_x, n_y, n_z, n_samples = X_data.shape
mask = masker.mask_img_.get_data()
#============================================================================
from sklearn.feature_extraction.image import grid_to_graph
from rena import weighted_connectivity_graph

connectivity_ward = grid_to_graph(n_x=n_x, n_y=n_y, n_z=n_z, mask=mask)

connectivity_rena = weighted_connectivity_graph(X_data,
                                                n_features=X_masked.shape[1],
                                                mask=mask)
#============================================================================
# Custering
from sklearn.cluster import AgglomerativeClustering
from rena import recursive_nearest_agglomeration

n_clusters = 2000

ward = AgglomerativeClustering(n_clusters=n_clusters,
                               connectivity=connectivity_ward, linkage='ward')

labels_rena = recursive_nearest_agglomeration(X_masked, connectivity_rena,
                                              n_clusters=n_clusters)

ward.fit(X_masked.T)
labels_ward = ward.labels_
#==============================================================================
# Custering
from rena import reduce_data, approximate_data

X_red_rena = reduce_data(X_masked, labels_rena)
X_red_ward = reduce_data(X_masked, labels_ward)

X_approx_rena = approximate_data(X_red_rena, labels_rena)
X_approx_ward = approximate_data(X_red_ward, labels_ward)
#============================================================================
# Loading different clustering algorithms

def visualize_labels(labels, masker):
    # Shuffle the labels (for better visualization):
    permutation = np.random.permutation(labels.shape[0])
    labels = permutation[labels]
    return masker.inverse_transform(labels)


cut_coords = (-34, -16)
n_image = 0


import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map, plot_epi
plt.close('all')

labels_rena_img = visualize_labels(labels_rena, masker)
labels_ward_img = visualize_labels(labels_ward, masker)


clusters_rena_fig = plot_stat_map(labels_rena_img, bg_img=dataset.anat[0],
                                  title='ReNA: clusters', display_mode='yz',
                                  cut_coords=cut_coords, colorbar=False)

clusters_ward_fig = plot_stat_map(labels_ward_img, bg_img=dataset.anat[0],
                                  title='Ward: clusters', display_mode='yz',
                                  cut_coords=cut_coords, colorbar=False)

compress_rena_fig = plot_epi(masker.inverse_transform(X_approx_rena[n_image]),
                             title='ReNA: approximated', display_mode='yz',
                             cut_coords=cut_coords)
compress_ward_fig = plot_epi(masker.inverse_transform(X_approx_ward[n_image]),
                             title='Ward: approximated', display_mode='yz',
                             cut_coords=cut_coords)

original_fig = plot_epi(masker.inverse_transform(X_masked[n_image]),
                        title='original', display_mode='yz',
                        cut_coords=cut_coords)

clusters_rena_fig.savefig('figures/clusters_rena.png')
clusters_ward_fig.savefig('figures/clusters_ward.png')
compress_rena_fig.savefig('figures/compress_rena.png')
compress_ward_fig.savefig('figures/compress_ward.png')
original_fig.savefig('figures/original.png')

plt.show()
