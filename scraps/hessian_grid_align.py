grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05, sub_size=1)

print(grid.native[28, 47])
print(grid.native[67, 48])
print(grid.native[68, 48])

sis = MockEllipticalIsothermal(
    centre=(0.0, 0.0), elliptical_comps=(0.001, 0.001), einstein_radius=1.0
)

print(sis.magnification_from_grid(grid=grid).native[28, 47])
print(sis.magnification_from_grid(grid=grid).native[67, 48])
print(sis.magnification_from_grid(grid=grid).native[68, 48])

print()

buffer = 0.001

print(
    sis.magnification_via_hessian_from_grid(
        grid=ag.Grid2DIrregularGrouped(grid=[[(1.100039, -0.00742)]]), buffer=buffer
    )
)
print(
    sis.magnification_via_hessian_from_grid(
        grid=ag.Grid2DIrregularGrouped(grid=[[(-0.90039, -0.00585)]]), buffer=buffer
    )
)
print(
    sis.magnification_via_hessian_from_grid(
        grid=ag.Grid2DIrregularGrouped(grid=[[(-0.95039, -0.00585)]]), buffer=buffer
    )
)
print(
    sis.magnification_via_hessian_from_grid(
        grid=ag.Grid2DIrregularGrouped(grid=[[(-1.0, -0.00585)]]), buffer=buffer
    )
)
