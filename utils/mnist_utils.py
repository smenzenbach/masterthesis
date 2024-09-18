import os
import sys
import numpy as np
import matplotlib.pyplot as plt

file_dir = os.getcwd()
rrr_root = os.path.abspath(os.path.join(file_dir, "/workspace/home/old_repos/rrr"))

# Add the project root directory to sys.path
if rrr_root not in sys.path:
    sys.path.append(rrr_root)

import rrr.decoy_mnist as decoy_mnist


def save_mnist_images():
    # Xr,Xtr --> orginial MNIST; X,Xt --> decoy MNIST; E,Et --> explanations
    # y,yt --> same for decoy and orginial MNIST

    Xr, X, y, E, Xtr, Xt, yt, Et = decoy_mnist.generate_dataset(
        cachefile="../data/decoy-mnist.npz"
    )
    Xr = np.array([x.ravel() for x in Xr])
    Xtr = np.array([x.ravel() for x in Xtr])

    n_samples = 60000
    X = X.reshape((n_samples, 1, 28, 28))
    Xt = Xt.reshape((10000, 1, 28, 28))
    E = E.reshape(n_samples, 1, 28, 28)
    Xr = Xr.reshape((n_samples, 1, 28, 28))
    Xtr = Xtr.reshape((10000, 1, 28, 28))
    Et = Et.reshape(10000, 1, 28, 28)

    def save_images(X, output_dir, n_images=None):
        if n_images is None:
            n_images = X.shape[0]
        for i in range(n_images):
            print(f"Saving image {i}")
            plt.figure(figsize=(1, 1))
            plt.imshow(X[i, 0], cmap="gray")
            plt.axis("off")
            plt.savefig(
                os.path.join(output_dir, f"mnist_{i}.jpeg"),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close()

    output_dir_confounder = "/workspace/home/data/DecoyMNIST/confounder"
    os.makedirs(output_dir_confounder, exist_ok=True)

    output_dir_no_confounder = "/workspace/home/data/DecoyMNIST/no_confounder"
    os.makedirs(output_dir_no_confounder, exist_ok=True)

    save_images(X, output_dir_confounder, 1000)
    save_images(Xr, output_dir_no_confounder, 1000)


# save_mnist_images()
