"""
Plot 2D cartesian snapshots.

Usage:
    plot_rbds.py

"""

# Pour vidéo, besoin de faire
# conda install -c conda-forge ffmpeg
# conda install ffmpeg-python
# si ça marche toujours pas
# conda update ffmpeg

from dedalus.extras import plot_tools
import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")

# mpiexec -n 8 python3 plot_rbds.py
## !####################################################
run_name = "0612_Ra1e+05_Flot1e-01_X10_Y1_Le10_Pr1"
make_movie = True
clean_replot = True  # S'il y a deja des frames, repartir de zero.
all_tasks = {
    "meanx_flux_th": {
        "title": "Average thermal flux at interface",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
    },
    "meanx_flux_c": {
        "title": "Average salt flux at interface",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
    },
    "rms_u": {
        "title": "Velocity RMS",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
    },
}

## !####################################################


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings

    dpi = 150
    # Create multifigure

    # Plot writes
    # with h5py.File(filename, mode="r") as file:
    #     for n, (task, task_infos) in enumerate(all_tasks.items()):
    #         if task == "t":
    #             continue
    #         dset = file["tasks"][task]

    #         task_title = task_infos["title"]
    #         for index in range(start, start + count):
    #             # axes2d.title.set_text(task_title)
    #             if task_infos["axis"] == "t":
    #                 # t_patch = dset.dims[0][0][:].ravel()
    #                 # axs[n, 0].scatter(
    #                 #     [t_patch[index]], [dset[index, 0, 0]], color="black"
    #                 # )
    #                 axs[n, 0].set_title(task_title)
    #                 # axs[n,0].set_ylim(values_min, values_max, auto=True)

    #             else:
    #                 print("WHAT??")

    # Add time title
    # title = title_func(file["scales/sim_time"][index])
    # title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
    # fig.suptitle(f"{run_name} \n {title}", y=title_height, ha="center")
    # # Save figure
    # savename = savename_func(file["scales/write_number"][index])
    # savepath = output.joinpath(savename)
    # # print(savepath)

    # fig.clear()
    # plt.close(fig)


def init_tseries(task, files):
    from tkinter import Tcl

    t = np.array([])
    scalar = np.array([])
    for filename in Tcl().call("lsort", "-dict", files):
        with h5py.File(filename, mode="r") as file:
            t = np.append(t, file["scales"]["sim_time"])
            scalar = np.append(scalar, file["tasks"][task][:, 0, 0])
    return t, scalar


if __name__ == "__main__":
    import pathlib

    # from docopt import docopt
    import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync
    import glob

    logger = logging.getLogger(__name__)

    # args = docopt(__doc__)

    files = f"snapshots/{run_name}/*.h5"
    outputs = f"frames/{run_name}"
    args = {"--output": outputs, "<files>": files}
    globlist = glob.glob(args["<files>"])

    with h5py.File(globlist[0], mode="r") as file:
        logger.info(f"Available data: {list(file["tasks"].keys())}")

    nb_tseries = 0
    all_tasks["t"] = {"type": "1d"}
    for task, task_infos in all_tasks.items():
        if task_infos["type"] == "2d" and task_infos["axis"] == "t":
            nb_tseries += 1
            t, data = init_tseries(task, globlist)
            all_tasks["t"]["data"] = t
            all_tasks[task]["data"] = data

    # output_path = pathlib.Path(args["--output"]).absolute()
    # # Create output directory if needed

    # logger.info(f"Entering {output_path}")

    fft_folder = pathlib.Path(f"./fft").absolute()
    if not fft_folder.exists():
        fft_folder.mkdir()
    img_fft_path = str(fft_folder.joinpath(f"{run_name}.png"))
    logger.info(f"Rendering {img_fft_path}")

    fig, axs = plt.subplots(len(all_tasks) - 1, 3)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    fig.subplots_adjust(hspace=0.33)

    row = -1
    for _, (task, task_infos) in enumerate(all_tasks.items()):
        if task == "t":
            continue
        row += 1
        data = task_infos["data"]
        t = all_tasks["t"]["data"]
        tmin = 0.25
        mask = t > tmin
        axs[row, 0].plot(t, data)
        axs[row, 0].set_xlabel(r"$t$")
        axs[row, 0].set_ylabel(task)
        axs[row, 0].set_title(task_infos["title"])
        axs[row, 0].axvline(x=tmin, ls="--", color="black", lw=0.5)
        data = data[mask]
        t = t[mask]
        sp = np.fft.fft(data - np.mean(data))
        sp_abs = np.abs(sp)
        freq = np.fft.fftfreq(t.shape[-1])
        sortfreq = np.argsort(freq)
        sp_abs = sp_abs[sortfreq]
        freq = freq[sortfreq]

        axs[row, 1].plot(t, data)
        axs[row, 1].set_xlabel(r"$t$")
        axs[row, 1].set_title("Zoom")
        axs[row, 1].axvline(x=tmin, ls="--", color="black", lw=0.5)
        axs[row, 2].plot(freq, sp_abs)

        mask0 = freq > 0
        freq = freq[mask0]
        sp_abs_norm = np.max(sp_abs)
        sp_abs = sp_abs[mask0]
        imax = np.argmax(sp_abs)
        fftmax = sp_abs[imax]
        freqmax = freq[imax]
        axs[row, 2].scatter([freqmax], [fftmax], color="black", marker="o")
        axs[row, 2].text(x=0.05 + freqmax, y=0.05 + fftmax, s=f"{freqmax:.1e}")
        axs[row, 2].set_xlim(0, None)
        # axs[row, 2].set_ylim(0, 1.1 * np.max(sp_abs))
        axs[row, 2].set_title("|FFT| (normalized)")
        axs[row, 2].set_xlabel(r"$f$")
        dpi = 150
        fig.savefig(img_fft_path, dpi=dpi)
    # post.visit_writes(globlist, main, output=img_fft_path)

    # frames_pattern = f"{output_path}/*.png"
    # logger.info(f"{len(frames_pattern)} frames have been written.")
