"""
plot time series.

Usage:
    plot_tseries.py
"""

# import tomli
from tkinter import Tcl
import matplotlib.pyplot as plt

import h5py
import numpy as np


# TODO show parameters
def plot_and_stats(files, scalars_to_plot):
    """Plots time series from files and returns stats"""
    # with Path("par.toml").open("rb") as file_in:
    #     par = tomli.load(file_in)
    # # Parameters
    #     Lx, Lz = (par['physical']['Lx'], 1.)
    #     Ra = par['physical']['Rayleigh']
    # outputs
    # with Path("output.toml").open("rb") as file_in:
    #     out = tomli.load(file_in)
    #     Ta0 = out['Ta0']
    #     Ta1 = out['Ta1']
    #     rhoa0 = out['rhoa0']
    #     rhoa1 = out['rhoa1']
    #     mrhoa = out['mrhoa']
    #     RR = out['RR']

    # rho = np.array([])
    all_arrays = {}
    for scalar in scalars_to_plot:
        all_arrays[scalar] = np.array([])
    t = np.array([])
    for filename in Tcl().call("lsort", "-dict", files):
        with h5py.File(filename, mode="r") as file:
            print("Available scalars:", file["tasks"].keys(), end="")

            t = np.append(t, file["scales"]["sim_time"])
            for scalar in scalars_to_plot:
                print(file["tasks"][scalar].shape)
                all_arrays[scalar] = np.append(
                    all_arrays[scalar], file["tasks"][scalar][:, 0, 0]
                )

    # plots
    print(t.shape, all_arrays["mean_th"].shape)
    rowsnb = len(scalars_to_plot)
    if rowsnb == 1:
        rowsnb = 2
    tfig, tax = plt.subplots(rowsnb, 1, sharex=True, figsize=(5, 8))
    for i, scalar in enumerate(scalars_to_plot):
        tax[i].plot(t, all_arrays[scalar], label=scalar)
        tax[i].set_ylabel(scalar)
        tax[i].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    from pathlib import Path

    p = Path(".")

    ##################################
    files = list(p.glob("scalars/2111/*.h5"))
    scalars_to_plot = ["mean_th"]
    ##################################

    plot_and_stats(files, scalars_to_plot)
