# gaussianfield

Python implementation of the Gaussian field classifier from Zhu 2003.

 Zhu, X; Lafferty, JK; Ghahramani, Z; (2003) Combining Active Learning and Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions. In: (Proceedings) 20th International Conference on Machine Learning workshop.

http://www.cs.cmu.edu/~zhuxj/pub/semisupervisedcode/active_learning/

## Brief usage snippet

Main inputs are a kernel matrix `K` for the full dataset, one-hot-encoded class labels for the observed examples, and an indicator vector for the observed examples. Unlike the Matlab implementation, observed and unobserved data don't need to be contiguous.

```python
import gaussianfield
from keras.utils.np_utils import to_categorical

# load data, labels, compute kernel matrix...
X, labels = load_dataset(...)
N, ndim = X.shape
K = compute_kernel_matrix(X)

# choose some random examples for initial observed set...
thresh = 0.05
observed = np.random.random(N) < thresh
N_observed = observed.sum()

field_solution, inverse_laplacian = gaussianfield.solve(K, labels[observed], observed)
class_predictions = np.argmax(field, axis=1)

# compute expected risk for active learning...
risk = gaussianfield.expected_risk(field_solution, inverse_laplacian)

# get the query index relative to the full dataset
_risk = 1000 * np.ones(labels.size)
_risk[~observed] = risk
query_idx = np.argmin(_risk)
```

