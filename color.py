import pickle
import os

with open("cmap.pkl", "r+b") as f:
    cmap = pickle.loads(f.read())


print(cmap)


from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap(name="autolens", segmentdata=cmap._segmentdata)

print(cmap._segmentdata)
