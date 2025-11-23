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
matplotlib.use('Agg')

# Pour les colorbars :
clims = {"u": "auto",
         "th": [-1, 1],
         "c": [-1, 1],
         "vorticity": "auto"}
tasks = ['th', 'c', 'vorticity']

tasks_titles = {"u": "Velocity",
                "th": "Temperature",
                "c": "Salinity",
                "vorticity": "Vorticity"}


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings

    scale = 1.5
    dpi = 200
    def title_func(sim_time): return 't = {:.3f}'.format(sim_time)
    def savename_func(write): return 'write_{:06}.png'.format(write)

    # Layout
    nrows, ncols = len(tasks), 1
    image = plot_tools.Box(4, 1)
    pad = plot_tools.Frame(0.3, 0, 0, 0)
    margin = plot_tools.Frame(0.2, 0.1, 0, 0)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                if task not in clims or clims[task] == "auto":
                    plot_tools.plot_bot_3d(
                        dset, 0, index, axes=axes, title=tasks_titles[task], even_scale=True, visible_axes=False)
                else:
                    plot_tools.plot_bot_3d(
                        dset, 0, index, axes=axes, title=tasks_titles[task], even_scale=True, visible_axes=False, clim=clims[task])

            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.44, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
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

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
            else:
                logger.info(
                    f'{output_path} already exists. Will be overwritten.')
    logger.info(f'Entering {output_path}')
    post.visit_writes(args['<files>'], main, output=output_path)

    with Sync() as sync:
        if sync.comm.rank == 0:
            pattern_png = str(output_path.joinpath('write_*.png'))

            # create ./videos directory if not there
            mp4_folder = pathlib.Path(f'./videos').absolute()
            if not mp4_folder.exists():
                mp4_folder.mkdir()
            folder_name = args['<files>'][0].split("/")[-2]
            filemp4 = str(mp4_folder.joinpath(f"{folder_name}.mp4"))
            logger.info(f"Rendering {filemp4} from {pattern_png}")
            fps = 60
            ffmpeg.input(pattern_png,
                         pattern_type='glob',
                         framerate=fps).output(filemp4,
                                               vcodec='libx264',
                                               crf=18,
                                               preset='medium',
                                               r=fps,
                                               pix_fmt='yuv420p',
                                               movflags='faststart').overwrite_output().run()


#  mpiexec -n 4 python3 plot_rbds.py snapshots/2111/*.h5 --output="./frames/2111"
