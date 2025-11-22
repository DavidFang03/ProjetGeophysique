"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D Cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. It should take about 5 cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ conda activate dedalus3
    $ mpiexec -n 8 python3 salt.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
import glob
import os
logger = logging.getLogger(__name__)


# ! Parameters

folder_path = "snapshots/2111"
restart = True

startfromprev = False
filerestart = ""

folder_name = folder_path.split("/")[-1]
restart_pattern = f"{folder_path}/{folder_name}_s*.h5"
logger.info(f"Writing in {restart_pattern}")

# Parameters
Lx, Lz = 4, 1
Nx, Nz = 256, 64

Rayleigh = 1e4  # Rayleigh de "température" sous-entendu
Flot = 0.1  # Rapport des Rayleigh
X = 100  # Relie flux thermique et chimique. X grand -> Flux chimique petit.
Le = 10    # = kappa/D. Normalement ~1000. Pour équations de diffusions.
Prandtl = 1    # = nu/kappa

stop_sim_time = 50
dt_init = 1e-9  # Simu : pas de temps initial
sim_dt = 1e-3  # Export : pas de temps
max_timestep = 0.125  # pas importance

# Ne pas toucher ?
dealias = 3/2
timestepper = d3.RK222
dtype = np.float64
nu = (Rayleigh / Prandtl)**(-1/2)
kappa = (Rayleigh * Prandtl)**(-1/2)

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis, zbasis))
th = dist.Field(name='th', bases=(xbasis, zbasis))
c = dist.Field(name='c', bases=(xbasis, zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
tau_p = dist.Field(name='tau_p')
tau_th1 = dist.Field(name='tau_th1', bases=xbasis)
tau_th2 = dist.Field(name='tau_th2', bases=xbasis)
tau_c1 = dist.Field(name='tau_c1', bases=xbasis)
tau_c2 = dist.Field(name='tau_c2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
def lift(A): return d3.Lift(A, lift_basis, -1)


grad_u = d3.grad(u) + ez*lift(tau_u1)  # First-order reduction
grad_th = d3.grad(th) + ez*lift(tau_th1)
grad_c = d3.grad(c) + ez*lift(tau_c1)
dzth = d3.Differentiate(th, coords['z'])
dzc = d3.Differentiate(c, coords['z'])

# Equations
problem = d3.IVP([p, th, c, u, tau_p, tau_th1, tau_th2, tau_c1,
                 tau_c2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")  # Conservation masse
# Equation advection-diffusion de température
problem.add_equation("dt(th) - div(grad_th) + lift(tau_th2) = - u@grad(th)")
# Equation advection-diffusion de salinité
problem.add_equation("dt(c) - (1/Le)*div(grad_c) + lift(tau_c2) = - u@grad(c)")
problem.add_equation(
    # Navier-Stokes
    "dt(u) - Prandtl*div(grad_u) + grad(p) - Prandtl*Rayleigh*(th - Flot*c)*ez + lift(tau_u2) = - u@grad(u)")

# Conditions aux limites
problem.add_equation("dzc(z=Lz)  = (1/X)*dzth(z=Lz)*c(z=Lz)")
problem.add_equation("th(z=Lz) = -1")
problem.add_equation("u(z=Lz) = 0")

problem.add_equation("th(z=0) = 0")  # Température au fond
problem.add_equation("dzc(z=0) = 0")  # Concentration au fond
problem.add_equation("u(z=0) = 0")

problem.add_equation("integ(p) = 0")  # ? Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if restart:
    list_of_files = glob.glob(restart_pattern)
    latest_file = max(list_of_files, key=os.path.getctime)
    write, initial_timestep = solver.load_state(latest_file)
    file_handler_mode = 'append'
    start_time = solver.sim_time
    logger.info('Start time = {}'.format(start_time))
elif startfromprev:
    list_of_files = glob.glob(filerestart)
    # logger.info('list = {}'.format(list_of_files))
    latest_file = max(list_of_files, key=os.path.getctime)
    logger.info('found file: {}'.format(latest_file))
    write, initial_timestep = solver.load_state(latest_file)
    initial_timestep = 1e-7
    file_handler_mode = 'overwrite'
    solver.sim_time = 0
    solver.iteration = 0
else:
    th.fill_random('g', seed=42, distribution='normal',
                   scale=1e-3)  # Random noise
    th['g'] *= z * (Lz - z)  # Damp noise at walls
    th['g'] += Lz - z  # Add linear background

    c.fill_random('g', seed=42, distribution='normal',
                  scale=1e-3)  # Random noise
    c['g'] *= z * (Lz - z)
    c['g'] += (Lz - z)/X
    c['g'] += 1
    file_handler_mode = 'overwrite'
    initial_timestep = dt_init


# ! Analysis
snapshots = solver.evaluator.add_file_handler(
    folder_path, sim_dt=sim_dt, max_writes=50, mode=file_handler_mode)
snapshots.add_task(th, name='th')
snapshots.add_task(c, name='c')
snapshots.add_task(u, name='u')
snapshots.add_task(p, name='p')
snapshots.add_task(tau_p, name='tau_p')
snapshots.add_task(tau_th1, name='tau_th1')
snapshots.add_task(tau_th2, name='tau_th2')
snapshots.add_task(tau_c1, name='tau_c1')
snapshots.add_task(tau_c2, name='tau_c2')
snapshots.add_task(tau_u1, name='tau_u1')
snapshots.add_task(tau_u2, name='tau_u2')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL (cadence = recalcul de dt)
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# ! Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')
flow.add_property(dzth(z=Lz), name='Fth')
flow.add_property(dzc(z=Lz), name='Fch')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            max_Fth = flow.max('Fth')
            max_Fch = flow.max('Fch')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f, max(Fth)=%f, max(Fch)=%f' % (
                solver.iteration, solver.sim_time, timestep, max_Re, max_Fth, max_Fch))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
