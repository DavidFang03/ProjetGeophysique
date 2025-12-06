"""
Plot 2D cartesian snapshots.

Usage:
    plot_rbds.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

# Pour vidéo, besoin de faire
# conda install -c conda-forge ffmpeg
# conda install ffmpeg-python
# si ça marche toujours pas
# conda update ffmpeg

import ffmpeg
from dedalus.extras import plot_tools
import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")

######################################################
all_tasks = {
    "th": {"title": "Temperature", "type": "3d", "clim": [0, 1]},
    "c": {"title": "Salinity", "type": "3d", "clim": "auto"},
    "vorticity": {"title": "Vorticity", "type": "3d", "clim": "auto"},
    "meanx_th": {"title": "Temperature horizontal average", "type": "2d", "axis": "z"},
    "meanx_c": {"title": "Salinity horizontal average", "type": "2d", "axis": "z"},
    "flux_th": {"title": "Thermal flux at interface", "type": "2d", "axis": "x"},
    "flux_c": {"title": "Salt flux at interface", "type": "2d", "axis": "x"},
}
######################################################


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings

    scale = 1.5
    dpi = 200

    def title_func(sim_time):
        return "t = {:.3f}".format(sim_time)

    def savename_func(write):
        return "write_{:06}.png".format(write)

    # Layout
    nrows, ncols = 3, 4  # à change si jamais + de plots
    image = plot_tools.Box(4, 1)
    pad = plot_tools.Frame(0.3, 0, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plot writes
    with h5py.File(filename, mode="r") as file:
        for index in range(start, start + count):
            filled_rows = [0, 0, 0]
            for n, (task, task_infos) in enumerate(all_tasks.items()):
                # Build subfigure axes
                plot_type = task_infos["type"]

                if plot_type == "3d":
                    filled_rows[0] += 1
                    i = filled_rows[0] - 1
                    j = 0
                elif plot_type == "2d":
                    axis = task_infos["axis"]
                    if axis == "x":
                        filled_rows[1] += 1
                        i = filled_rows[1] - 1
                        j = 1
                    elif axis == "z":
                        filled_rows[2] += 1
                        i = filled_rows[2] - 1
                        j = 2
                # i, j = divmod(n, ncols)

                # Call 3D plotting helper, slicing in time
                dset = file["tasks"][task]
                print(dset.shape)
                task_title = task_infos["title"]
                if task_infos["type"] == "3d":
                    axes = mfig.add_axes(i, j, [0, 0, 1, 1])
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

                    x = dset.dims[1][0][:].ravel()
                    z = dset.dims[2][0][:].ravel()
                    if task_infos["axis"] == "x":
                        axes2d = mfig.add_axes(i, j, [0.33, 0, 1, 1])
                        data = dset[index, :, 0]
                        axes2d.plot(x, data)
                        axes2d.set_xlabel("x")
                        axes2d.set_ylabel(task_title)
                        axes2d.set_xticks([])

                    elif task_infos["axis"] == "z":
                        axes2d = mfig.add_axes(i, j, [0.66, 0, 1, 1])
                        data = dset[index, 0, :]
                        axes2d.plot(data, z)
                        axes2d.set_xlabel(task_title)
                        axes2d.set_ylabel("z")
                        axes2d.set_yticks([])

                    # axes2d.title.set_text(task_title)

                else:
                    print("WHAT??")

            # Add time title
            title = title_func(file["scales/sim_time"][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.44, y=title_height, ha="left")
            # Save figure
            savename = savename_func(file["scales/write_number"][index])
            savepath = output.joinpath(savename)
            # print(savepath)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    logger = logging.getLogger(__name__)

    args = docopt(__doc__)

    output_path = pathlib.Path(args["--output"]).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
            else:
                logger.info(f"{output_path} already exists. Will be overwritten.")
    logger.info(f"Entering {output_path}")
    post.visit_writes(args["<files>"], main, output=output_path)

    with Sync() as sync:
        if sync.comm.rank == 0:
            pattern_png = str(output_path.joinpath("write_*.png"))

            # create ./videos directory if not there
            mp4_folder = pathlib.Path(f"./videos").absolute()
            if not mp4_folder.exists():
                mp4_folder.mkdir()
            folder_name = args["<files>"][0].split("/")[-2]
            filemp4 = str(mp4_folder.joinpath(f"{folder_name}.mp4"))
            logger.info(f"Rendering {filemp4} from {pattern_png}")
            fps = 60
            ffmpeg.input(pattern_png, pattern_type="glob", framerate=fps).output(
                filemp4,
                vcodec="libx264",
                crf=18,
                preset="medium",
                r=fps,
                pix_fmt="yuv420p",
                movflags="faststart",
            ).overwrite_output().run()


#  mpiexec -n 8 python3 plot_rbds.py snapshots/0512/*.h5 --output="./frames/0512"
