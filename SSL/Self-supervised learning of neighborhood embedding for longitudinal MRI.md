
# Methods

```
Definition
x_1: image acquired at 1st visit
x_2: image acquired at 2nd visit
Delta_t: the time difference of two visits
```
Taking $(x_1,x_2,\Delta t)$, and calculate 3 losses and generate embeddings $z_1,z_2$, with dimension=1000
## Losses
1. Reconstruction loss
	   x-> encoder -> z -> decoder -> x'
	   enforce: x~x'
2. ProtoNCE loss
   using k-means to cluster embeddings
   encourages z to be closer to its k-means center, and distant from other centers
3. Progression-consistent neighborhood loss
   $x$  with similar $z$ should also have similar $\Delta z$ ==Assumption==

## Sampling strategy for mini-batches
Sample pairs from each k-means center to approximate the dataset distribution


# Datasets

T1w, FA

Resampled to $64\times64\times64$
# Evaluation

- age prediction
- alcohol drinking classification
- AD, .. disease binary classification

# Limitations

ProtoNCE loss based on k-means clusters requires specifying number of clusters

