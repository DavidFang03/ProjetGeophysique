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

import pathlib
import logging
from dedalus.tools import post
from dedalus.tools.parallel import Sync
import glob

matplotlib.use("Agg")

boundaries = {"c": [], "th": [], "buoyancy": []}

# mpiexec -n 8 python3 plot_rbds.py
## !####################################################
# run_name = "1512_Ra1e+04_Flot1e-01_X100_Y100_Le10_Pr1"
run_path = "/v2/2912_h0_1e-1_t2_-1_Y1e+00_Ra1e+04_Flot1e+02_X1e-01_Le10_Pr1"
run_name = run_path.split("/")[-1]
tmin = 0
make_movie = True
clean_replot = True  # S'il y ah deja des frames, repartir de zero.
color_th = "tab:brown"
color_c = "tab:gray"

c_clim = [0.5, 1.2]
th_clim = [0, 1]
buoyancy_clim = [66, 130]
all_tasks = [
    {
        "name": "h",
        "title": "Ice layer height",
        "type": "2d",
        "axis": "x",
        "range": [None, None],
        "row": 0,
        "col": 0,
        "color": "black",
    },
    {
        "name": "sr_first",
        "title": "Three terms of Stefan-Robin boundary conditions",
        "type": "2d",
        "axis": "x",
        "range": [None, None],
        "row": 2,
        "col": 0,
        "label": "Solid thermal flux",
        "color": "lightsteelblue",
        "height": 2,
        "chaperone": [
            {"name": "flux_th", "label": "Liquid thermal flux", "color": "aqua"},
            {"name": "sr_third", "label": "Latent heat term", "color": "tab:orange"},
        ],
    },
    {
        "name": "th",
        "title": "Temperature",
        "type": "3d",
        "clim": th_clim,
        "row": 3,
        "col": 0,
        "color": color_th,
    },
    {
        "name": "c",
        "title": "Salinity",
        "type": "3d",
        "clim": c_clim,
        "row": 4,
        "col": 0,
    },
    {
        "name": "vorticity",
        "title": "Vorticity",
        "type": "3d",
        "clim": "auto",
        "row": 4,
        "col": 1,
    },
    {
        "name": "vz_times_theta",
        "title": r"$v_z\times\theta$",
        "type": "3d",
        "clim": "auto",
        "row": 0,
        "col": 1,
    },
    {
        "name": "vz_times_c",
        "title": r"$v_z\times c$",
        "type": "3d",
        "clim": "auto",
        "row": 1,
        "col": 1,
    },
    {
        "name": "th_avgx",
        "title": "Temperature horizontal average",
        "type": "2d",
        "axis": "z",
        "range": [None, None],
        "row": 2,
        "col": 1,
        "color": color_th,
        "alpha": 0.1,
        "secondary_xaxis": {
            "name": "c_avgx",
            "title": "Salinity horizontal average",
            "color": color_c,
        },
    },
    {
        "name": "flux_th",
        "title": "Thermal flux at interface",
        "type": "2d",
        "axis": "x",
        "range": [None, None],
        "row": 3,
        "col": 1,
        "color": color_th,
        "secondary_yaxis": {
            "name": "flux_c",
            "title": "Salt flux at interface",
            "color": color_c,
        },
    },  #####################################################################################
    {
        "name": "th_avg",
        "title": "Average temperature",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
        "row": 0,
        "col": 2,
        "color": color_th,
        "secondary_yaxis": {
            "name": "c_avg",
            "title": "Average salinity",
            "color": color_c,
        },
    },
    {
        "name": "rms_u",
        "title": "Velocity RMS",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
        "row": 1,
        "col": 2,
        "color": "black",
    },
    {
        "name": "m_ice",
        "title": "Ice mass",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
        "row": 2,
        "col": 2,
        "color": "black",
    },
    {
        "name": "buoyancy",
        "title": "Buoyancy",
        "type": "3d",
        "clim": buoyancy_clim,
        "row": 3,
        "col": 2,
    },  ################################### 1st term = Fth + 3rd term
    {
        "name": "sr_first_avgx",
        "title": "Three terms of Stefan-Robin boundary conditions",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
        "row": 2,
        "col": 3,
        "height": 3,
        "label": "Solid thermal flux",
        "color": "lightsteelblue",
        "chaperone": [
            {
                "name": "flux_th",
                "label": "Liquid thermal flux",
                "color": "aqua",
            },
            {
                "name": "sr_third_avgx",
                "label": "Latent heat term",
                "color": "tab:orange",
            },
        ],
    },
    {
        "name": "flux_th_avgx",
        "title": "Average thermal flux at interface",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
        "row": 3,
        "col": 3,
        "color": color_th,
        "secondary_yaxis": {
            "name": "flux_c_avgx",
            "title": "Average salt flux at interface",
            "color": color_c,
        },
    },
    {
        "name": "c_avgx_interface",
        "title": "Average salinity at interface",
        "type": "2d",
        "axis": "t",
        "range": [None, None],
        "row": 4,
        "col": 3,
        "color": color_c,
        # "secondary_yaxis": {
        #     "name": "flux_c_avgx",
        #     "title": "Average salt flux at interface",
        #     "color": color_c,
        # },
    },
]

## !####################################################


def plot_3d(mfig, task_infos, dset, index):
    row, col, task_title = (
        task_infos["row"],
        task_infos["col"],
        task_infos["title"],
    )
    # full cell inside image box
    axes = mfig.add_axes(row, col, [0, 0, 1, 1])
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
    name = task_infos["name"]
    # print(name, name in boundaries)
    if name in boundaries:
        # print(data)
        boundaries[name].append(np.min(dset))
        boundaries[name].append(np.max(dset))


def plot_2d(mfig, task_infos, file, index, Time):
    row, col, task_title, plot_axis = (
        task_infos["row"],
        task_infos["col"],
        task_infos["title"],
        task_infos["axis"],
    )
    color = task_infos.get("color", "blue")
    values_min, values_max = task_infos["range"]  # TODO
    height = task_infos.get("height", 1.0)
    # rect = [left, bottom, width, height] in fractions of the image box
    ax = mfig.add_axes(row, col, [0, 0, 1, height])

    all_tasks = [task_infos]
    all_axes = [ax]
    if task_infos.get("secondary_yaxis"):
        ax2d = ax.twinx()
        ax2d.tick_params(axis="y", labelcolor=task_infos["secondary_yaxis"]["color"])
        ax.tick_params(axis="y", labelcolor=task_infos["color"])
        all_tasks.append(task_infos["secondary_yaxis"])
        all_axes.append(ax2d)
    elif task_infos.get("secondary_xaxis"):
        ax2d = ax.twiny()
        ax2d.tick_params(axis="x", labelcolor=task_infos["secondary_xaxis"]["color"])
        ax.tick_params(axis="x", labelcolor=task_infos["color"])
        all_tasks.append(task_infos["secondary_xaxis"])
        all_axes.append(ax2d)
    elif "chaperone" in task_infos:
        all_tasks += task_infos["chaperone"]
        for _ in task_infos["chaperone"]:
            all_axes.append(ax)

    for n, task in enumerate(all_tasks):
        color = task.get("color", "blue")
        alpha = task.get("alpha", 0.8)
        ax2d = all_axes[n]
        name = task["name"]
        # print(n, name)
        label = task.get("label", "")
        if "title" in task:
            task_title = task["title"]
        dset = file["tasks"][name]
        if plot_axis == "x":
            x = dset.dims[1][0][:].ravel()

            data = dset[index, :, 0]
            ax2d.plot(x, data, color=color, label=label)
            ax2d.set_xlabel(r"$x$")
            if "chaperone" in task_infos:
                ax2d.set_ylabel(task_title, color="black")
            else:
                ax2d.set_ylabel(task_title, color=color)
            # ax2d.set_xticks([])

        elif plot_axis == "z":
            z = dset.dims[2][0][:].ravel()
            data = dset[index, 0, :]
            ax2d.plot(data, z, color=color, label=label, alpha=alpha)
            ax2d.set_xlabel(task_title, color=color)
            ax2d.set_ylabel(r"$z$")
            if "range" in task:
                ax2d.set_xlim(*task["range"])
            # ax2d.set_yticks([])

        # ax2d.title.set_text(task_title)
        elif plot_axis == "t":
            t_patch = dset.dims[0][0][:].ravel()
            data = task["data"]
            ax2d.plot(Time, data, color=color, label=label)
            ax2d.set_xlabel(r"$t$")
            if t_patch[index] > tmin:
                ax2d.scatter([t_patch[index]], [dset[index, 0, 0]], color="black")
            ax2d.set_ylabel(task_title, color=color)

        else:
            raise NotImplementedError(plot_axis)

    if label != "":
        ax2d.legend()


def title_func(sim_time):
    return "t = {:.3f}".format(sim_time)


def savename_func(write):
    return "write_{:06}.png".format(write)


def main(filename, start, count, output, Time, all_tasks):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings

    scale = 1
    dpi = 150

    # Layout
    nrows, ncols = 6, 4  # à changer si jamais + de plots
    image = plot_tools.Box(4, 1)  # Lx=4, Lz=1
    # Increase horizontal padding and left outer margin so plots don't crowd
    margin = plot_tools.Frame(1, 0.1, 0.2, 0.8)
    pad = plot_tools.Frame(0.95, 0, 0.95, 0)

    # Plot writes
    with h5py.File(filename, mode="r") as file:
        for index in range(start, start + count):
            # create a fresh figure for each frame so axes don't accumulate
            mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
            fig = mfig.figure
            for task_infos in all_tasks:
                task = task_infos["name"]
                # Build subfigure axes

                plot_type = task_infos["type"]

                # Call 3D plotting helper, slicing in time
                dset = file["tasks"][task]
                task_title = task_infos["title"]
                if plot_type == "3d":
                    plot_3d(mfig, task_infos, dset, index)
                elif plot_type == "2d":
                    plot_2d(mfig, task_infos, file, index, Time)
                else:
                    print("WHAT??")

            # Add time title
            title = title_func(file["scales/sim_time"][index])
            # position suptitle using normalized margin (don't divide by figure inches)
            # title_height = 1.0 - 0.5 * mfig.margin.top
            fig.suptitle(f"{run_name} \n {title}", ha="center")
            # Save figure
            savename = savename_func(file["scales/write_number"][index])
            savepath = output.joinpath(savename)
            # print(savepath)
            fig.savefig(str(savepath), dpi=dpi)
            plt.close(fig)
            # all frames written


def init_tseries(task, files):

    t = np.array([])
    scalar = np.array([])
    # for filename in Tcl().call("lsort", "-dict", files):
    for filename in files:
        with h5py.File(filename, mode="r") as file:
            tfile = np.array(file["scales"]["sim_time"])
            mask_tmin = tfile > tmin
            tfile = tfile[mask_tmin]
            tdata = np.array(file["tasks"][task][:, 0, 0])
            t = np.append(t, tfile)
            scalar = np.append(scalar, tdata[mask_tmin])
    sort = np.argsort(t)
    return t[sort], scalar[sort]


def handle_directory(output_path):
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
            else:
                logger.info(f"{output_path} already exists. Will be overwritten.")
                previousframes_pattern = f"{output_path}/*.png"
                previousframes = glob.glob(previousframes_pattern)

                if clean_replot and len(previousframes) > 0:
                    # user_agree = (
                    #     input(
                    #         f"Will remove {len(previousframes)} frames: {previousframes_pattern} (y/n)"
                    #     )
                    #     == "y"
                    # )
                    user_agree = True
                    if user_agree:
                        import os

                        for f in previousframes:
                            os.remove(f)


def make_movie(output_path, fps):
    import ffmpeg

    with Sync() as sync:
        if sync.comm.rank == 0:
            pattern_png = str(output_path.joinpath("write_*.png"))

            movie_folder = pathlib.Path(f"outputs/{run_path}").absolute()
            if not movie_folder.exists():
                movie_folder.mkdir()
            # folder_name = args["<files>"][0].split("/")[-2]
            filemp4 = str(movie_folder.joinpath(f"{run_name}.mp4"))
            filegif = str(movie_folder.joinpath(f"{run_name}.gif"))
            logger.info(f"Rendering {filemp4} from {pattern_png} with fps={fps}")
            # fps = 30
            ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).output(
                filemp4,
                vcodec="libx264",
                crf=18,
                preset="medium",
                r=fps,
                pix_fmt="yuv420p",
                movflags="faststart",
            ).overwrite_output().run()
            print(filemp4)
            ffmpeg.input(filemp4).output(filegif).overwrite_output().run()
            print(filegif)


if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    files = f"outputs/{run_path}/snapshots/*.h5"
    outputs = f"outputs/{run_path}/frames"
    args = {"--output": outputs, "<files>": files}
    output_path = pathlib.Path(args["--output"]).absolute()
    globlist = glob.glob(args["<files>"])

    with h5py.File(globlist[0], mode="r") as file:
        logger.info(f"Available data: {list(file["tasks"].keys())}")

    for task_infos in all_tasks:
        if task_infos.get("type") == "2d" and task_infos.get("axis") == "t":
            Time, data = init_tseries(task_infos["name"], globlist)
            task_infos["data"] = data
            if "chaperone" in task_infos:
                for ii in range(len(task_infos["chaperone"])):
                    task_infos["chaperone"][ii]["data"] = init_tseries(
                        task_infos["chaperone"][ii]["name"], globlist
                    )[1]
            elif "secondary_yaxis" in task_infos:
                task_infos["secondary_yaxis"]["data"] = init_tseries(
                    task_infos["secondary_yaxis"]["name"], globlist
                )[1]

    # # ! Dirty trick to get the range of c(z) profile
    # _, c_avgx = init_tseries("c_avgx_interface", globlist)
    # c_avgx_min, c_avgx_max = np.min(c_avgx), np.max(c_avgx)
    # with h5py.File(globlist[0], mode="r") as file:
    #     dset = file["tasks"]["c_avgx"]
    #     data = dset[0, 0, :]
    #     c_avgx_min = np.min([c_avgx_min, np.min(data)])
    #     c_avgx_max = np.max([c_avgx_max, np.max(data)])

    # # print(c_avgx_min)
    # for task in all_tasks:
    #     if task["name"] == "th_avgx":
    #         task["secondary_xaxis"]["range"] = [
    #             c_avgx_min - 0.05 * np.abs(c_avgx_max - c_avgx_min),
    #             c_avgx_max + 0.05 * np.abs(c_avgx_max - c_avgx_min),
    #         ]

    tmax = np.max(Time)
    nb_outputs = len(Time)
    fps = nb_outputs / (3 * tmax)

    # Create output directory if needed
    handle_directory(output_path)

    logger.info(f"Entering {output_path}")
    post.visit_writes(
        globlist, main, output=output_path, Time=Time, all_tasks=all_tasks
    )

    if make_movie:
        make_movie(output_path, fps)

    with Sync() as sync:
        if sync.comm.rank == 0:
            showboundaries = {}
            for name in boundaries:
                print(boundaries[name])
                low = np.min(boundaries[name])
                top = np.max(boundaries[name])
                showboundaries[name] = f"[{low:.1e} {top:.1e}]"
            logger.info(f"I would recommand new clims: {showboundaries}")

    # frames_pattern = f"{output_path}/*.png"
    # logger.info(f"{len(frames_pattern)} frames have been written.")
