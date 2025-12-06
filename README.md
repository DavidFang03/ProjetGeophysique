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
`salt.py` pour run. Il faut préciser une `run_name` (typiquement la date)
```bash
mpiexec -n 8 python3 salt.py
```
Paramètres : $\text{Ra}, \text{Flot}, \text{X}, \text{Le}, \text{Pr}$

### 2. Analyse :
`plot_rbds` pour afficher la run : les images seront dans `./frames` et la video dans `./videos`. Dans le fichier il suffit de préciser le nom de la run `run_name` (ligne 25).
```bash
mpiexec -n 4 python3 plot_rbds.py
```
Ligne 28 : `all_tasks` contient tous les paramètres des plots : titre et échelle.
