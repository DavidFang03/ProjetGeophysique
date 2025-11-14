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
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = 4, 1
Nx, Nz = 256, 64
Rayleigh = 2e6 # sous entendu rayleigh de "température"
Flot = 0
X = 0 # Nb sans dim dans la condition aux limites
D = 1e-4 # coeff diffusion
Prandtl = 1
dealias = 3/2
stop_sim_time = 50
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
th = dist.Field(name='th', bases=(xbasis,zbasis))
c = dist.Field(name='c', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_th1 = dist.Field(name='tau_th1', bases=xbasis)
tau_th2 = dist.Field(name='tau_th2', bases=xbasis)
tau_c1 = dist.Field(name='tau_c1', bases=xbasis)
tau_c2 = dist.Field(name='tau_c2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

# Substitutions
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_th = d3.grad(th) + ez*lift(tau_th1) # First-order reduction
grad_c = d3.grad(c) + ez*lift(tau_c1) # First-order reduction

# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, th, c, u, tau_p, tau_th1, tau_th2, tau_c1, tau_c2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")  # Conservation masse
problem.add_equation("dt(th) - kappa*div(grad_th) + lift(tau_th2) = - u@grad(th)") # Equation advection-diffusion de température
problem.add_equation("dt(c) - D*div(grad_c) + lift(tau_c2) = - u@grad(c)") # Equation advection-diffusion de salinité
problem.add_equation("dt(u) - Prandtl*div(grad_u) + grad(p) + Prandtl*Rayleigh*(th - Flot*c)*ez + lift(tau_u2) = - u@grad(u)") # Navier-Stokes

dzth = d3.Differentiate(th, coords['z'])
dzc = d3.Differentiate(c, coords['z'])
# Salinité entre rho, s,
# Transition de phase entre T,s
# CL
# problem.add_equation("th(z=0) = Lz") # ? Homogène à une longueur ? Dépend du diagramme de phase
# problem.add_equation("u(z=0) = 0") 
# problem.add_equation("c(z=0) = 0") # ? dépend de \dot s
# # ? IMposer les conditions aux limites des gradients ?
# problem.add_equation("th(z=Lz) = 0")  # ? Dépend du diagramme de phase
# problem.add_equation("u(z=Lz) = 0")
problem.add_equation("dzth(z=Lz) = X*dzc(z=Lz)/c(z=Lz)") # ? Stefan-Robin
# problem.add_equation("c(z=Lz) = -18.7*th(z=Lz) - 0.519*th(z=Lz)**2 - 0.00535*th(z=Lz)**3")
problem.add_equation("c(z=Lz) = 1")
problem.add_equation("u(z=Lz) = 0")
# problem.add_equation("c(z=0) = 0") # Pas besoin ?
# ? IMposer les conditions aux limites des gradients ?
problem.add_equation("th(z=0) = 1")  # Température au fond
problem.add_equation("c(z=0) = 1") # Concentration au fond
problem.add_equation("u(z=0) = 0") 


# problem.add_equation("c(z=0) = 1") # pas besoin ?

problem.add_equation("integ(p) = 0") # ? Pressure gauge


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions : On rajoute un bruit gaussien ?
th.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
th['g'] *= z * (Lz - z) # Damp noise at walls
th['g'] += Lz - z # Add linear background

# c['g'] = 0.1 # Add linear background
c.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
c+=1
# c['g'] *= z * (Lz - z) # Damp noise at walls
# c['g'] += Lz - z # Add linear background

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots/0711', sim_dt=0.25, max_writes=50)
snapshots.add_task(th, name='temperature')
snapshots.add_task(c, name='salinity')
snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u@u)/nu, name='Re')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
