import autolens as al
from os import path

workspace_path = path.join(path.sep, "/Users", "samlange", "Code", "PAL")
lens = "0418"
filter = "f444w"
dataset_path = path.join(workspace_path, "cosma", "JWST", f"SPT-{lens}", filter)

if lens == "0418":
    positions = al.Grid2DIrregular.from_json(file_path="positions2.json")
    threshold = 0.5

print(positions)

plotter = al.plot.Grid2DPlotter(grid=positions)
plotter.figure_2d()

grid = positions
import matplotlib.pyplot as plt

plt.scatter(y=grid[:, 0], x=grid[:, 1])


import itertools

color = itertools.cycle(list("rgb"))
plt.scatter(y=grid[:, 0], x=grid[:, 1], c=next(color))
