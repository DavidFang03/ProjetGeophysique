# Projet Géophysique M2 - Convection thermosolutale
- Lien du overleaf : https://www.overleaf.com/project/68f254d0e1582e502b564021
- Doc Dedalus : https://dedalus-project.readthedocs.io/en/latest/index.html
- Github Dedalus : https://github.com/DedalusProject/dedalus

## Usage
- Activer environnement conda
```bash
conda activate dedalus3
```
NB : Pour installer ffmpeg :
```bash
conda install -c conda-forge ffmpeg
conda install ffmpeg-python
# si ça marche pas :
conda update ffmpeg
```
### 1. Simulation :
`salt.py` pour run
```bash
mpiexec -n 8 python3 salt.py
```
Paramètres : $\text{Ra}, \text{Flot}, \text{X}, \text{Le}, \text{Pr}$

### 2. Analyse :
`plot_rbds` pour afficher la run : les images seront dans `--output` et la video dans `./videos`
```bash
mpiexec -n 4 python3 plot_rbds.py snapshots/2111/*.h5 --output="./frames/2111"
```
Paramètres : `tasks` pour sélectionner les grandeurs à afficher, `clims` pour les échelles des colorbars.
