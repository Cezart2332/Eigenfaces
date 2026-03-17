# 👤 Eigenfaces – SVD vs EIG Comparison

A Python implementation of the **Eigenfaces algorithm** for facial recognition, comparing two decomposition methods: classical **EIG** (via covariance matrix) and **Truncated SVD** (via scikit-learn).

Built as part of the *Optimization Techniques* laboratory at university.

---

## Dataset

This project uses the **AT&T / ORL Faces Dataset** — 40 subjects, 10 grayscale images each (92×112 px), stored in folders `s1` to `s40`.

```
att_faces/
├── s1/
│   ├── 1.pgm
│   ├── 2.pgm
│   └── ...
├── s2/
│   └── ...
└── s40/
```

> Place the `att_faces/` folder in the same directory as `main.py`.

---

## Configuration

At the top of `main.py`, you can easily adjust the training/test split:

```python
NR_PERSOANE_TRAIN = 20   # number of subjects used for training (out of 40)
NR_POZE_TRAIN     = 8    # images per subject used for training (out of 10)
NR_POZE_TEST      = 2    # remaining images used for testing
```

---

## 🧠 How It Works

### 1. Data Loading
Images are flattened into 1D vectors of size **10304** (92×112) and arranged into a matrix of shape `(10304, n_train)`.

### 2. Preprocessing
The **mean face** is computed and subtracted from every image — this centers the data and removes what all faces have in common, leaving only what differentiates them.

### 3. Method 1 – EIG (Optimized with Matrix L)
Instead of computing the full covariance matrix `C = A·Aᵀ` of size `(10304×10304)`, we use the smaller matrix:

```
L = Aᵀ · A   →   shape (n_train × n_train)
```

Eigenfaces are then recovered from the eigenvectors of L. This is mathematically equivalent but **much faster**.

### 4. Method 2 – Truncated SVD
Uses `sklearn.decomposition.TruncatedSVD` with `n_components = k`, computing only the top-k components directly — no need to compute the full decomposition.

---

## Results

Both methods are compared for `k = 20, 40, 60, 80, 100` components:

| k   | Time EIG (s) | Time SVD (s) | Faster |
|-----|-------------|-------------|--------|
| 20  | ...         | ...         | SVD    |
| 40  | ...         | ...         | SVD    |
| 60  | ...         | ...         | SVD    |
| 80  | ...         | ...         | SVD    |
| 100 | ...         | ...         | SVD    |

### Key observations:
- **EIG** computes *all* eigenvectors regardless of k → roughly constant time
- **Truncated SVD** computes only k components → faster, especially for small k
- Both methods produce **visually similar eigenfaces** at the same k
- As k increases, eigenfaces become **sharper and more detailed**

---

##  Eigenfaces Visualization

Eigenfaces are saved as images for each method and each value of k (4×5 grid for k=20):

| EIG k=20 | SVD k=20 |
|----------|----------|
| ![eig](Eigenfaces%20k%20=20,%20method=eig.png) | ![svd](Eigenfaces%20k%20=20,%20method=svd.png) |

---

##  Requirements

```
numpy
Pillow
scikit-learn
matplotlib
scipy
```

Install with:

```bash
pip install numpy Pillow scikit-learn matplotlib scipy
```

---

##  Usage

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Compute eigenfaces using both EIG and Truncated SVD for k = 20, 40, 60, 80, 100
3. Save eigenface grid images to disk
4. Print a comparison table of execution times

---

##  References

- Turk, M. & Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience.
- [AT&T Faces Dataset](https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
- [scikit-learn TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)