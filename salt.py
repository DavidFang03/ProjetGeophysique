"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D Cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. It should take about 5 cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa_l = (Rayleigh * Prandtl)**(-1/2)
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
##!########################################################
run_date = "1812_h0_1e-1_t2_-1"
restart = False

startfromprev = False
filerestart = ""

## Parameters
# Résolution
Lx, Lz = 4, 1
Nx, Nz = 256, 64  # Augmenter ?
max_timestep = 1e-4  # Diminuer ?
dt_init = 1e-9  # Simu : pas de temps initial

# Physique
Rayleigh = 1e4  # Rayleigh de "température" sous-entendu
Flot = 1  # Rapport des Rayleigh
X = 10000  # (Stefan-Robin) Relie flux thermique et chimique. X grand -> Flux chimique petit.
Y = 0.1  # (Stefan-Robin) Deuxieme Nb sans dim dans la condition aux limites de Stefan-Robin
Le = 10  # = kappa_l/D. Normalement ~1000. Pour équations de diffusions.
Prandtl = 1  # = nu/kappa_l

# Run
stop_sim_time = 6
export_snapshots_dt = stop_sim_time / 100  # Export des *.h5 : pas de temps
# export_scalars_dt = 1e-3  # Export des scalaires : pas de temps
##!########################################################

run_name = f"{run_date}_Y{Y:.0e}_Ra{Rayleigh:.0e}_Flot{Flot:.0e}_X{X}_Le{Le}_Pr{Prandtl}"  # formater le reste si jamais
snapshots_folder = f"outputs/{run_name}/snapshots"
checkpoints_folder = f"outputs/{run_name}/checkpoints"
logger.info(f"Starting {run_name}")

# Ne pas toucher ?
dealias = 3 / 2
timestepper = d3.RK222
dtype = np.float64
nu = (Rayleigh / Prandtl) ** (-1 / 2)
kappa_l = (Rayleigh * Prandtl) ** (-1 / 2)

# Bases
coords = d3.CartesianCoordinates("x", "z")
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords["x"], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords["z"], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name="p", bases=(xbasis, zbasis))
th = dist.Field(name="th", bases=(xbasis, zbasis))
c = dist.Field(name="c", bases=(xbasis, zbasis))
u = dist.VectorField(coords, name="u", bases=(xbasis, zbasis))
tau_p = dist.Field(name="tau_p")
tau_th1 = dist.Field(name="tau_th1", bases=xbasis)
tau_th2 = dist.Field(name="tau_th2", bases=xbasis)
tau_c1 = dist.Field(name="tau_c1", bases=xbasis)
tau_c2 = dist.Field(name="tau_c2", bases=xbasis)
tau_u1 = dist.VectorField(coords, name="tau_u1", bases=xbasis)
tau_u2 = dist.VectorField(coords, name="tau_u2", bases=xbasis)

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)


def lift(A):
    return d3.Lift(A, lift_basis, -1)


grad_u = d3.grad(u) + ez * lift(tau_u1)  # First-order reduction
grad_th = d3.grad(th) + ez * lift(tau_th1)
grad_c = d3.grad(c) + ez * lift(tau_c1)
dzth = d3.Differentiate(th, coords["z"])
dzc = d3.Differentiate(c, coords["z"])
h = dist.Field(name="h", bases=(xbasis))

theta2 = -1  # = T2/T0 : doit être négatif
theta1 = 0  # interface
theta0 = 1  # au fond

## Equations
problem = d3.IVP(
    [p, th, c, u, tau_p, tau_th1, tau_th2, tau_c1, tau_c2, tau_u1, tau_u2, h],
    namespace=locals(),
)
problem.add_equation("trace(grad_u) + tau_p = 0")  # Conservation masse
# Equation advection-diffusion de température
problem.add_equation("dt(th) - div(grad_th) + lift(tau_th2) = - u@grad(th)")
# Equation advection-diffusion de salinité
problem.add_equation("dt(c) - (1/Le)*div(grad_c) + lift(tau_c2) = - u@grad(c)")
# Navier-Stokes
problem.add_equation(
    "dt(u) - Prandtl*div(grad_u) + grad(p) - Prandtl*Rayleigh*(th - Flot*c)*ez + lift(tau_u2) = - u@grad(u)"
)
problem.add_equation("dt(h)=(1/Le)*dzc(z=Lz)/c(z=Lz)")
# problem.add_equation("dt(h)=(1/(Le*X))*(dzth(z=Lz)-Y*theta2/h)")

## Conditions aux limites
# En z=Lz

problem.add_equation("dzc(z=Lz)  = (1/X)*dzth(z=Lz)*c(z=Lz) - (Y/X)*(theta2/h)*c(z=Lz)")
problem.add_equation("th(z=Lz) = 0")
problem.add_equation("u(z=Lz) = 0")
# En z=0
problem.add_equation("th(z=0) = 1")  # Température au fond
problem.add_equation("dzc(z=0) = 0")  # Concentration au fond
problem.add_equation("u(z=0) = 0")

problem.add_equation("integ(p) = 0")  # ? Pressure gauge


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if restart:
    restart_pattern = f"{checkpoints_folder}/checkpoints_s*.h5"
    list_of_files = glob.glob(restart_pattern)
    latest_file = max(list_of_files, key=os.path.getctime)
    write, initial_timestep = solver.load_state(latest_file)
    file_handler_mode = "append"
    start_time = solver.sim_time
    logger.info("[Restart] Start time = {}".format(start_time))
elif startfromprev:
    list_of_files = glob.glob(filerestart)
    # logger.info('list = {}'.format(list_of_files))
    latest_file = max(list_of_files, key=os.path.getctime)
    logger.info("found file: {}".format(latest_file))
    write, initial_timestep = solver.load_state(latest_file)
    initial_timestep = 1e-7
    file_handler_mode = "overwrite"
    solver.sim_time = 0
    solver.iteration = 0
else:
    th.fill_random("g", seed=42, distribution="normal", scale=1e-3)  # Random noise
    th["g"] *= z * (Lz - z)  # Damp noise at walls
    th["g"] += (Lz - z) / Lz  # Add linear background

    c.fill_random("g", seed=42, distribution="normal", scale=1e-3)  # Random noise
    c["g"] *= z * (Lz - z)  # Damp noise at walls
    dzth0 = -1
    h0 = 1e-1
    delta = 0.1
    c0 = 1

    St = -Y * theta2 / h0 + theta1 - theta0
    c1 = c0 / (1 - (1 - delta) * St)
    cuniform = c0
    clin = c1 * St * (z - (1 - delta)) + c0
    c["g"] += np.where(z > Lz - delta, clin, cuniform)

    h["g"] = h0

    file_handler_mode = "overwrite"
    initial_timestep = dt_init


# ! Analysis
# print(file_handler_mode)
# contenu = os.listdir(folder_path)
# print("avant", contenu)
# file_prefix = os.path.join(folder_path, folder_name)
# Ensure output folder exists and use a file-prefix (folder/name) so
# the FileHandler can find previous sets when using mode='append'.
# os.makedirs(folder_path, exist_ok=True)
# file_prefix = f"{folder_path}/{folder_name}"

checkpoints = snapshots = solver.evaluator.add_file_handler(
    checkpoints_folder,
    sim_dt=export_snapshots_dt,
    max_writes=100,
    mode=file_handler_mode,
)
checkpoints.add_tasks(solver.state)

snapshots = solver.evaluator.add_file_handler(
    snapshots_folder,
    sim_dt=export_snapshots_dt,
    max_writes=100,
    mode=file_handler_mode,
)

# snapshots.add_tasks(solver.state)
print("ok")
snapshots.add_task(th, name="th")
snapshots.add_task(c, name="c")
snapshots.add_task(u, name="u")
snapshots.add_task(p, name="p")
snapshots.add_task(tau_p, name="tau_p")
snapshots.add_task(tau_th1, name="tau_th1")
snapshots.add_task(tau_th2, name="tau_th2")
snapshots.add_task(tau_c1, name="tau_c1")
snapshots.add_task(tau_c2, name="tau_c2")
snapshots.add_task(tau_u1, name="tau_u1")
snapshots.add_task(tau_u2, name="tau_u2")
snapshots.add_task(-d3.div(d3.skew(u)), name="vorticity")
snapshots.add_task(-th + Flot * c, name="buoyancy")  # buoyancy

snapshots.add_task(d3.Average(th, coord="x"), name="th_avgx")
snapshots.add_task(d3.Average(c, coord="x"), name="c_avgx")
snapshots.add_task(-dzth(z=Lz), name="flux_th")
snapshots.add_task(-dzc(z=Lz), name="flux_c")
snapshots.add_task(h, name="h")  # dépend que de x
vz = d3.DotProduct(u, ez)
snapshots.add_task(vz * th, name="vz_times_theta")
snapshots.add_task(vz * c, name="vz_times_c")

# Stefan-Robin terms (only depend on x)
snapshots.add_task(-Y * theta2 / h, name="sr_first")  # solid thermal flux
# snapshots.add_task(dzth(z=Lz), name="sr_second") # liquid thermal flux already in output
snapshots.add_task(-X * dzc(z=Lz) / c(z=Lz), name="sr_third")
snapshots.add_task(d3.Average(-Y * theta2 / h, coord="x"), name="sr_first_avgx")
# snapshots.add_task(dzth(z=Lz), name="sr_second")
snapshots.add_task(
    d3.Average(-X * dzc(z=Lz) / c(z=Lz), coord="x"), name="sr_third_avgx"
)

# Temporal scalars
snapshots.add_task(d3.Average(th), name="th_avg")
snapshots.add_task(d3.Average(c), name="c_avg")
snapshots.add_task(d3.Average(-dzth(z=Lz)), name="flux_th_avgx")
snapshots.add_task(d3.Average(-dzc(z=Lz)), name="flux_c_avgx")
u2 = d3.DotProduct(u, u)
varu = d3.Average(u2) - d3.DotProduct(d3.Average(u), d3.Average(u))
snapshots.add_task(np.sqrt(varu), name="rms_u")
snapshots.add_task(d3.Integrate(h, "x"), name="m_ice")

# CFL (cadence = recalcul de dt)
CFL = d3.CFL(
    solver,
    initial_dt=initial_timestep,
    cadence=10,
    safety=0.5,
    threshold=0.05,
    max_change=1.5,
    min_change=0.5,
    max_dt=max_timestep,
)
CFL.add_velocity(u)

# ! Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u @ u) / nu, name="Re")
flow.add_property(-dzth(z=Lz), name="Fth")
flow.add_property(-dzc(z=Lz), name="Fch")
flow.add_property(h, name="h")

# Main loop
try:
    logger.info("Starting main loop")
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration - 1) % 10 == 0:
            max_Re = flow.max("Re")
            max_Fth = flow.max("Fth")
            max_Fch = flow.max("Fch")
            max_h = flow.max("h")
            logger.info(
                "Iteration=%i, Time=%e, dt=%e, max(Re)=%f, max(Fth)=%f, max(Fch)=%f"
                % (
                    solver.iteration,
                    solver.sim_time,
                    timestep,
                    max_Re,
                    max_Fth,
                    max_Fch,
                )
            )
except:
    logger.error("Exception raised, triggering end of main loop.")
    raise
finally:
    solver.log_stats()
    logger.info(f"run name is {run_name}")
