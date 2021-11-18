import autolens as al

print(al.convert.angle_from(elliptical_comps=(1.0, 0.0)))
print(al.convert.angle_from(elliptical_comps=(0.0, 1.0)))
print(al.convert.angle_from(elliptical_comps=(-1.0, 0.0)))
print(al.convert.angle_from(elliptical_comps=(0.0, -1.0)))
print(al.convert.angle_from(elliptical_comps=(0.5, 0.5)))
print(al.convert.angle_from(elliptical_comps=(-0.5, 0.5)))
print(al.convert.angle_from(elliptical_comps=(0.5, -0.5)))
print(al.convert.angle_from(elliptical_comps=(-0.5, -0.5)))

print()

print(al.convert.shear_angle_from(elliptical_comps=(1.0, 0.0)))
print(al.convert.shear_angle_from(elliptical_comps=(0.0, 1.0)))
print(al.convert.shear_angle_from(elliptical_comps=(-1.0, 0.0)))
print(al.convert.shear_angle_from(elliptical_comps=(0.0, -1.0)))
print(al.convert.shear_angle_from(elliptical_comps=(0.5, 0.5)))
print(al.convert.shear_angle_from(elliptical_comps=(-0.5, 0.5)))
print(al.convert.shear_angle_from(elliptical_comps=(0.5, -0.5)))
print(al.convert.shear_angle_from(elliptical_comps=(-0.5, -0.5)))