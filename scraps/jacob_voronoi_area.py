import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import MultiPoint, Point, Polygon


def calculate_clipped_flux(voronoi, pix_vals):
    regions, vertices = voronoi_finite_polygons_2d(voronoi)
    pts = MultiPoint([Point(i) for i in voronoi.points])
    mask = pts.convex_hull
    new_vertices = []
    for region in regions:
        polygon = vertices[region]
        shape = list(polygon.shape)
        shape[0] += 1
        p = Polygon(np.append(polygon, polygon[0]).reshape(*shape)).intersection(mask)
        poly = np.array(
            list(zip(p.boundary.coords.xy[0][:-1], p.boundary.coords.xy[1][:-1]))
        )
        new_vertices.append(poly)
    clipped_areas = [PolyArea(polygon[:, 0], polygon[:, 1]) for polygon in new_vertices]
    # polygons = [new_vertices[new_regions[i]] for i in range(len(new_regions))]
    # areas = [PolyArea(polygon[:,0], polygon[:,1]) for polygon in polygons]
    fluxes = np.array([pix_vals[i] * clipped_areas[i] for i in range(len(pix_vals))])
    total_flux = np.sum(fluxes)
    return fluxes, clipped_areas


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
