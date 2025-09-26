import pathlib
import subprocess
import h5py
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# # Clean up any old files
# import shutil
# shutil.rmtree('analysis', ignore_errors=True)



with h5py.File("snapshots/snapshots_s1.h5", mode='r') as file:
    # Load datasets
    # u = file['tasks']['u']
    print(file['tasks'].keys())
    # b = file['tasks']['buoyancy']
    b = file['tasks']['vorticity']
    print(b)
    t = b.dims[0]['sim_time']
    x = b.dims[1][0]
    y = b.dims[2][0]
    print(y)
    # print(b.dims[])
    # # Plot data
    # u_phase = np.arctan2(u[:].imag, u[:].real)
    # plt.figure(figsize=(6,7), dpi=100)
    X,Y = np.meshgrid(y,x)
    print(X.shape,Y.shape)
    print(b[0,:,:].shape)
    plt.pcolormesh(Y, X, b[49,:,:], shading='nearest', cmap='twilight_shifted')
    # plt.pcolormesh(x[:], t[:], u_phase, shading='nearest', cmap='twilight_shifted')
    # plt.colorbar(label='phase(u)')
    # plt.xlabel('x')
    # plt.ylabel('t')
    # plt.title('Hole-defect chaos in the CGLE')
    plt.show()