from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
import scipy
import time
import numpy as np
import os
from PIL import Image

# ===== AJUSTEAZĂ AICI =====
NR_PERSOANE_TRAIN = 20
NR_POZE_TRAIN = 8      
NR_POZE_TEST = 2        
# ==========================

PATH = "att_faces"

X_train = []
X_test = []
y_train = []
y_test = []

for persoana in range(1, 41):  # s1 - s40
    folder = os.path.join(PATH, f"s{persoana}")
    
    for poza_nr in range(1, 11):  # 1 - 10
        cale_poza = os.path.join(folder, f"{poza_nr}.pgm")
        imagine = np.array(Image.open(cale_poza)).flatten()  # vector 1D
        
        if persoana <= NR_PERSOANE_TRAIN:
            if poza_nr <= NR_POZE_TRAIN:
                X_train.append(imagine)
                y_train.append(persoana)
            else:
                X_test.append(imagine)
                y_test.append(persoana)

X_train = np.array(X_train).T
X_test = np.array(X_test).T
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"Train: {X_train.shape}")  # ex: (160, 4096)
print(f"Test:  {X_test.shape}")   # ex: (40, 4096)


def preprocess(X_train):
    mean = np.mean(X_train,axis=1)
    X_centered = (X_train.T - mean).T
    return X_centered,mean

def compute_eigenfaces_eig(X_centered, k):
    t0 = time.perf_counter()
    l = np.dot(X_centered.T,X_centered)
    d,v = np.linalg.eig(l)
    idx = np.argsort(d)[::-1]
    d = d[idx]
    v = v[:, idx]
    HQPB = v[:, :k]
    eigenfaces = np.dot(X_centered, HQPB)
    t1 = time.perf_counter()
    time_to_compute = t1 - t0
    return eigenfaces,time_to_compute

def compute_eigenfaces_svd(X_centered,k):
    t0 = time.perf_counter()
    svd = TruncatedSVD(n_components=k)
    svd.fit(X_centered)
    eigenfaces = svd.components_

    t1 = time.perf_counter()
    time_to_compute = t1 - t0

    return eigenfaces,time_to_compute


def show_eigenfaces(eigenfaces,method, k):
    # eigenfaces are (10304, k) - o eigenface pe coloana
    rows = 4
    cols = k // rows  # 20//4 = 5

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for i, ax in enumerate(axes.flat):
        if i < k:
            eigenface = eigenfaces[:, i]          # coloana i → (10304,)
            eigenface = eigenface.reshape(112, 92) # înapoi la imaginea originală
            ax.imshow(eigenface.real, cmap='gray') # .real pt valori complexe
            ax.axis('off')
            ax.set_title(f"#{i+1}")
    
    plt.suptitle(f"Eigenfaces k ={k}, method={method}")
    plt.tight_layout()
    plt.savefig(f"Eigenfaces k ={k}, method={method}")

def compare_methods(X_centered):
    k_list = [20, 40, 60, 80, 100]

    print(f"\n{'k':<10} {'Timp EIG (s)':<20} {'Timp SVD (s)':<20} {'Mai rapid':<10}")
    print("=" * 60)

    for k in k_list:
        eigenfaces_eig, time_eig = compute_eigenfaces_eig(X_centered, k)
        eigenfaces_svd, time_svd = compute_eigenfaces_svd(X_centered.T, k)

        mai_rapid = "EIG" if time_eig < time_svd else "SVD"
        print(f"{k:<10} {time_eig:<20.4f} {time_svd:<20.4f} {mai_rapid:<10}")

        show_eigenfaces(eigenfaces_eig, "eig", k)
        show_eigenfaces(eigenfaces_svd.T, "svd", k)
        


X_centered,mean = preprocess(X_train)

compare_methods(X_centered)

    