import numpy as np
import nibabel


# Auxiliary functions to use ReNA:
# ReNA is designed to use a NiftiMasker to define the connectivity graph
def _unmask(w, mask):
    if mask.sum() != len(w):
        raise ValueError("Expecting mask.sum() == len(w).")
    out = np.zeros(mask.shape, dtype=w.dtype)
    out[mask] = w
    return out


def to_niimgs(X, dim):
    p = np.prod(dim)
    mask = np.zeros(p).astype(np.bool)
    mask[:X.shape[-1]] = 1
    mask = mask.reshape(dim)
    X = np.rollaxis(np.array([_unmask(x, mask) for x in X]), 0, start=4)
    affine = np.eye(4)
    X_niimg = nibabel.Nifti1Image(X, affine)
    mask_niimg = nibabel.Nifti1Image(mask.astype(np.float), affine)
    return X_niimg, mask_niimg


from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils import check_random_state
rnd = check_random_state(23)
dataset = fetch_olivetti_faces(shuffle=True, random_state=rnd)

X, y = dataset['data'], dataset['target']

n_x, n_y = dataset['images'][0].shape

X_img, mask_img = to_niimgs(X, [n_x, n_y, 1])

from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_img, standardize=False).fit()

from rena import ReNA
cluster = ReNA(scaling=True, n_clusters=250, masker=masker)


X_clustered = cluster.fit_transform(X)
X_compressed = cluster.inverse_transform(X_clustered)

imgs_compressed = masker.inverse_transform(X_compressed)

import matplotlib.pyplot as plt
plt.close('all')


fig, axx = plt.subplots(2, 4, **{'figsize': (10, 5)})
plt.gray()

for i in range(4):
    axx[0, i].imshow(masker.inverse_transform(X).get_data()[:, :, 0, i + 30])
    axx[0, i].set_axis_off()
    axx[0, i].set_title('Original')
    axx[1, i].imshow(imgs_compressed.get_data()[:, :, 0, i + 30])
    axx[1, i].set_axis_off()
    axx[1, i].set_title('Compressed')

fig.savefig('figures/faces.png', bbox_to_inches='tight')
plt.show()
