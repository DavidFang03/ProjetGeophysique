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
run_name = "0612_Ra1e+04_Flot1e-01_X100_Y100_Le10_Pr1"
make_movie = True
clean_replot = True  # S'il y a deja des frames, repartir de zero.
all_tasks = {
    "th": {"title": "Temperature", "type": "3d", "clim": [0, 1]},
    "c": {"title": "Salinity", "type": "3d", "clim": "auto"},
    "vorticity": {"title": "Vorticity", "type": "3d", "clim": "auto"},
    "vz_times_theta": {"title": r"$v_z\times\theta$", "type": "3d", "clim": "auto"},
    "vz_times_c": {"title": r"$v_z\times c$", "type": "3d", "clim": "auto"},
    "meanx_th": {
        "title": "Temperature horizontal average",
        "type": "2d",
        "axis": "z",
        "range": [None, None],
    },
    "meanx_c": {
        "title": "Salinity horizontal average",
        "type": "2d",
        "axis": "z",
        "range": [None, None],
    },
    "flux_th": {
        "title": "Thermal flux at interface",
        "type": "2d",
        "axis": "x",
        "range": [None, None],
    },
    "flux_c": {
        "title": "Salt flux at interface",
        "type": "2d",
        "axis": "x",
        "range": [None, None],
    },
    "h": {
        "title": "Ice layer height",
        "type": "2d",
        "axis": "x",
        "range": [None, None],
    },
    "mean_th": {
        "title": "Average temperature",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
    },
    "mean_c": {
        "title": "Average salinity",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
    },
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
    "m_ice": {
        "title": "Ice mass",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
    },
}

## !####################################################


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings

    scale = 1
    dpi = 150

    def title_func(sim_time):
        return "t = {:.3f}".format(sim_time)

    def savename_func(write):
        return "write_{:06}.png".format(write)

    # Layout
    nrows, ncols = 4, 6  # à changer si jamais + de plots
    image = plot_tools.Box(4, 1)  # Lx=4, Lz=1
    pad = plot_tools.Frame(0.3, 0.3, 0, 0)
    margin = plot_tools.Frame(1, 0.2, 0, 0.5)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols + 1, image, pad, margin, scale)
    fig = mfig.figure

    # Plot writes
    with h5py.File(filename, mode="r") as file:
        for index in range(start, start + count):
            filled_rows = [
                0,
                0,
                0,
                0,
                0,
                0,
            ]  # Deux premières colonnes pour les plots 3d (colorbar) et les deux autres pour les 2d (profil spatial), les 2 dernières pour les profils temporels
            row3d = 0  # passera à 1 si la 1ère colonne pleine
            rowt = 4  # passera à 1 si l'avant dernière colonne pleine
            for n, (task, task_infos) in enumerate(all_tasks.items()):
                # Build subfigure axes

                if task == "t":
                    continue
                plot_type = task_infos["type"]

                if plot_type == "3d":
                    if filled_rows[row3d] == 3:
                        row3d += 1
                    j = row3d
                elif plot_type == "2d":
                    axis = task_infos["axis"]
                    if axis == "x":
                        j = 2
                    elif axis == "z":
                        j = 3
                    elif axis == "t":
                        if filled_rows[rowt] == 3:
                            rowt += 1
                        j = rowt
                filled_rows[j] += 1
                i = filled_rows[j] - 1

                # i, j = divmod(n, ncols)

                # Call 3D plotting helper, slicing in time
                dset = file["tasks"][task]
                task_title = task_infos["title"]
                if task_infos["type"] == "3d":
                    axes = mfig.add_axes(i, j, [j / ncols, 0, 1, 1])
                    if "clim" not in task_infos or task_infos["clim"] == "auto":
                        plot_tools.plot_bot_3d(
                            dset,
                            0,
                            index,
                            axes=axes,
                            title=task_title,
                            visible_axes=False,
                        )
                    else:
                        plot_tools.plot_bot_3d(
                            dset,
                            0,
                            index,
                            axes=axes,
                            title=task_title,
                            visible_axes=False,
                            clim=task_infos["clim"],
                        )
                elif task_infos["type"] == "2d":
                    values_min, values_max = task_infos["range"]
                    if task_infos["axis"] == "x":
                        x = dset.dims[1][0][:].ravel()
                        axes2d = mfig.add_axes(i, j, [j / ncols, 0, 1, 1])
                        data = dset[index, :, 0]
                        axes2d.plot(x, data)
                        axes2d.set_xlabel("x")
                        axes2d.set_ylabel(task_title)
                        axes2d.set_xticks([])
                        axes2d.set_ylim(values_min, values_max, auto=True)

                    elif task_infos["axis"] == "z":
                        z = dset.dims[2][0][:].ravel()
                        axes2d = mfig.add_axes(i, j, [j / ncols, 0, 1, 1])
                        data = dset[index, 0, :]
                        axes2d.plot(data, z)
                        axes2d.set_xlabel(task_title)
                        axes2d.set_ylabel("z")
                        axes2d.set_yticks([])
                        axes2d.set_xlim(values_min, values_max, auto=True)

                    # axes2d.title.set_text(task_title)
                    elif task_infos["axis"] == "t":
                        t_patch = dset.dims[0][0][:].ravel()
                        axes2d = mfig.add_axes(
                            i,
                            j,
                            [
                                (4 / ncols) + (0.0 * (j - 4) / ncols),
                                0,
                                0.6,
                                3 / (0.5 * nb_tseries),
                            ],
                        )
                        data = task_infos["data"]
                        t = all_tasks["t"]["data"]
                        axes2d.plot(t, data)
                        axes2d.scatter(
                            [t_patch[index]], [dset[index, 0, 0]], color="black"
                        )
                        axes2d.set_title(task_title)
                        if i == 2:  # last row
                            axes2d.set_xlabel("t")
                        else:
                            axes2d.set_xticklabels([])
                        axes2d.set_ylim(values_min, values_max, auto=True)

                else:
                    print("WHAT??")

            # Add time title
            title = title_func(file["scales/sim_time"][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(f"{run_name} \n {title}", y=title_height, ha="center")
            # Save figure
            savename = savename_func(file["scales/write_number"][index])
            savepath = output.joinpath(savename)
            # print(savepath)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


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

    output_path = pathlib.Path(args["--output"]).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
            else:
                logger.info(f"{output_path} already exists. Will be overwritten.")
                previousframes_pattern = f"{output_path}/*.png"
                previousframes = glob.glob(previousframes_pattern)

                if clean_replot and len(previousframes) > 0:
                    user_agree = (
                        input(
                            f"Will remove {len(previousframes)} frames: {previousframes_pattern} (y/n)"
                        )
                        == "y"
                    )
                    if user_agree:
                        import os

                        for f in previousframes:
                            os.remove(f)

    logger.info(f"Entering {output_path}")
    post.visit_writes(globlist, main, output=output_path)

    if make_movie:
        import ffmpeg

        with Sync() as sync:
            if sync.comm.rank == 0:
                pattern_png = str(output_path.joinpath("write_*.png"))

                # create ./videos directory if not there
                mp4_folder = pathlib.Path(f"./videos").absolute()
                if not mp4_folder.exists():
                    mp4_folder.mkdir()
                # folder_name = args["<files>"][0].split("/")[-2]
                filemp4 = str(mp4_folder.joinpath(f"{run_name}.mp4"))
                logger.info(f"Rendering {filemp4} from {pattern_png}")
                fps = 30
                ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).output(
                    filemp4,
                    vcodec="libx264",
                    crf=18,
                    preset="medium",
                    r=fps,
                    pix_fmt="yuv420p",
                    movflags="faststart",
                ).overwrite_output().run()

    # frames_pattern = f"{output_path}/*.png"
    # logger.info(f"{len(frames_pattern)} frames have been written.")
