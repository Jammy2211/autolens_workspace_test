"""
Strong lens clusters are made of 50+ galaxies, meaning the model API needs adaptation to make it easy to use.

The file `mass_profiles.csv` contains the parameters of 50 galaxies, it reflects a real galaxy cluster:

index,center_x(arcsec),center_y(arcsec),r_core(arcsec),r_cut(arcsec),kappa_scale,ellipticity,position_angle(deg)
0,-0.33174602292462385,-0.6373015906350705,14.550571956486197,432.8208212467319,834.691240427421,0,127.97425668889923
1,-0.09651805846120287,0.24981923786032878,0.18390512485114996,41.32264891183135,331.4867701209088,0,96.89669565127278
0,-31.692126743097194,17.538015180306594,0.05380077366778044,3.9039575609679513,104.32899485698229,0,0
1,-6.284379383680124,46.51382582630847,0.047758789496541665,3.4655317135217807,99.1798075965735,0,0
2,45.001629175143215,41.063707179766844,0.04487845637141711,3.256525457378097,96.59278640441109,0,0
3,21.436727689017204,-15.437163105603519,0.048222155206124585,3.4991550230412654,99.58752845618987,0,0
4,3.0866879294382086,43.41627637971287,0.04083451816941988,2.9630842660439143,92.79397156188364,0,0

...

This .csv file was made by a student modeling a cluster with their own code, we want to adapt this format to PyAutoFit.

__Task 1: .csv for normal lens model__

Lets make a .csv file for a clusters lens model for 3 lens galaxies with only mass profiles.
"""

import autofit as af
import autolens as al

lens_galaxies = {}

total_lens = 3

centre_list = [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]

for i in range(total_lens):

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre.centre_0 = centre_list[i]
    mass.centre.centre_1 = centre_list[i]
    mass.einstein_radius = af.UniformPrior(lower_limit=0.1, upper_limit=2.0)

    lens_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    lens_galaxies[f"lens_{i}"] = lens_galaxy

model = af.Collection(galaxies=af.Collection(**lens_galaxies))

print(model.info)

"""
Task 1 is to make it so a user can easily create a .csv file for the model above, with the 3 lines being:

index,profile,path,centre_0,centre_1,einstein_radius
0,IsothermalSph,galaxies.lens_0.mass,0.0,0.0,UniformPrior(0.1,2.0)
1,IsothermalSph,galaxies.lens_1.mass,1.0,1.0,UniformPrior(0.1,2.0)
2,IsothermalSph,galaxies.lens_2.mass,-1.0,-1.0,UniformPrior(0.1,2.0)

I'm not sure how to handle the `al.mp` import for the profile column.

__Task 2: .csv including light profiles__

Its common for each lens galaxy to have both light and mass profiles.
"""
lens_galaxies = {}

total_lens = 3

centre_list = [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]

for i in range(total_lens):

    light = af.Model(al.lp.Sersic)
    light.centre.centre_0 = centre_list[i]
    light.centre.centre_1 = centre_list[i]
    light.intensity = af.UniformPrior(lower_limit=0.1, upper_limit=10.0)
    light.effective_radius = 1.0
    light.sersic_index = 4.0

    mass = af.Model(al.mp.IsothermalSph)

    mass.centre.centre_0 = centre_list[i]
    mass.centre.centre_1 = centre_list[i]
    mass.einstein_radius = af.UniformPrior(lower_limit=0.1, upper_limit=2.0)

    lens_galaxy = af.Model(al.Galaxy, redshift=0.5, light=light, mass=mass)

    lens_galaxies[f"lens_{i}"] = lens_galaxy

model = af.Collection(galaxies=af.Collection(**lens_galaxies))

print(model.info)

"""
Task 2 is to make it so a user can easily create a .csv file for the model above, with the 3 lines being:

index,profile,path,centre_0,centre_1,intensity,effective_radius,sersic_index
0,Sersic,galaxies.lens_0.light,0.0,0.0,UniformPrior(0.1,10.0),1.0,4.0
1,Sersic,galaxies.lens_1.light,1.0,1.0,UniformPrior(0.1,10.0),1.0,4.0
2,Sersic,galaxies.lens_2.light,-1.0,-1.0,UniformPrior(0.1,10.0),1.0,4.0

Based on the .csv above identical code to the mass model .csv for Task 1 will work.

The important thing is to be sure that we can load both .csv files and create the correct model by combinig them
(e.g. the path of `galaxies.lens_0.light` and `galaxies.lens_0.mass` should be combined into one galaxy `lens_0` with both a light and mass profile).

If this is not easy in autofit, the alternative is to have one .csv file with both light and mass profiles, with empty columns for the profiles that do not exist.

index,profile,path,centre_0,centre_1,intensity,effective_radius,sersic_index,einstein_radius
0,Sersic,galaxies.lens_0.light,0.0,0.0,UniformPrior(0.1,10.0),1.0,4.0,
0,IsothermalSph,galaxies.lens_0.mass,0.0,0.0,,, ,UniformPrior(0.1,2.0)
1,Sersic,galaxies.lens_1.light,1.0,1.0,UniformPrior(0.1,10.0),1.0,4.0,
1,IsothermalSph,galaxies.lens_1.mass,1.0,1.0,,, ,UniformPrior(0.1,2.0)
2,Sersic,galaxies.lens_2.light,-1.0,-1.0,UniformPrior(0.1,10.0),1.0,4.0,
2,IsothermalSph,galaxies.lens_2.mass,-1.0,-1.0,,, ,UniformPrior(0.1,2.0)

__Task 3: Relational Models__

For cluster modeling it is common to link model parameters, for example light and mass.
"""
einstein_radius_m = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e11)
einstein_radius_c = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e11)
luminosity_star = 1e9

lens_galaxies = {}

for i in range(total_lens):

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre.centre_0 = centre_list[i]
    mass.centre.centre_1 = centre_list[i]
    mass.einstein_radius = (
        einstein_radius_m * (luminosity_star**0.5) + einstein_radius_c
    )

    lens_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    lens_galaxies[f"lens_{i}"] = lens_galaxy
