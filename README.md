# Recursive nearest agglomeration (ReNA)



## Nilearn contains the latest version of ReNA:
https://nilearn.github.io/modules/generated/nilearn.regions.Parcellations.html#nilearn.regions.Parcellations


## XXX: This example is not currently working, please use the Nilearn implementation




ReNA is a fast clustering algorithm, mainly usef for the approximation of structured signals.


## Dependencies

* Python >= 2.6  
* scikit-learn >=0.15
* nilearn >= 0.2.4 
* nibabel >= 1.1.0 
* numpy >= 1.7.1
* scipy >= 0.15
* matplotlib (examples)


## Running the examples

### Clustering images of faces

Compression of face images from the Olivetti Faces Dataset using 250 clusters.

```python
python example_faces.py
```
![](figures/faces.png)

----

### Clustering brain images

Compression of a brain image (MRI) from the [Haxby2001](http://www.ncbi.nlm.nih.gov/pubmed/11577229) dataset using 2000 clusters.


```python
python examples_neuroimaging.py
```
![](figures/original.png)


![](figures/compress_ward.png)
![](figures/compress_rena.png)
![](figures/clusters_ward.png)
![](figures/clusters_rena.png)


