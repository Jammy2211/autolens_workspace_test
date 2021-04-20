import os
import time
import json
import random
import sys
import json

from autoconf import conf
import autofit as af
import numpy as np

# %%
"""Setup the path to the autolens_slacs, using a relative directory name."""

cosma_path = "/cosma7/data/dp004/dc-ethe1"

dataset_label = "slacs_mocks"

workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))


cosma_output_path = f"{cosma_path}/output/"

conf.instance.push(
    new_path=f"{workspace_path}/cosma/config", output_path=cosma_output_path
)

import autolens as al


cosma_array_id = int(sys.argv[1])


data_name = []
data_name.append("")  # Task number beings at 1, so keep index 0 blank

data_name.append("slacs0008-0004")  # Index 1
data_name.append("slacs0330-0020")  # Index 2
data_name.append("slacs0903+4116")  # Index 3
data_name.append("slacs0959+0410")  # Index 4
data_name.append("slacs1029+0420")  # Index 5
data_name.append("slacs1153+4612")  # Index 6
data_name.append("slacs1402+6321")  # Index 7
data_name.append("slacs1451-0239")  # Index 8
data_name.append("slacs2300+0022")  # Index 9
data_name.append("slacs0029-0055")  # Index 10
data_name.append("slacs0728+3835")  # Index 11
data_name.append("slacs0912+0029")  # Index 12
data_name.append("slacs0959+4416")  # Index 13
data_name.append("slacs1032+5322")  # Index 14
data_name.append("slacs1205+4910")  # Index 15
data_name.append("slacs1416+5136")  # Index 16
data_name.append("slacs1525+3327")  # Index 17
data_name.append("slacs2303+1422")  # Index 18
data_name.append("slacs0157-0056")  # Index 19
data_name.append("slacs0737+3216")  # Index 20
data_name.append("slacs0936+0913")  # Index 21
data_name.append("slacs1016+3859")  # Index 22
data_name.append("slacs1103+5322")  # Index 23
data_name.append("slacs1213+6708")  # Index 24
data_name.append("slacs1420+6019")  # Index 25
data_name.append("slacs1627-0053")  # Index 26
data_name.append("slacs0216-0813")  # Index 27
data_name.append("slacs0822+2652")  # Index 28
data_name.append("slacs0946+1006")  # Index 29
data_name.append("slacs1020+1122")  # Index 30
data_name.append("slacs1142+1001")  # Index 31
data_name.append("slacs1218+0830")  # Index 32
data_name.append("slacs1430+4105")  # Index 33
data_name.append("slacs1630+4520")  # Index 34
data_name.append("slacs0252+0039")  # Index 35
data_name.append("slacs0841+3824")  # Index 36
data_name.append("slacs0956+5100")  # Index 37
data_name.append("slacs1023+4230")  # Index 38
data_name.append("slacs1143-0144")  # Index 39
data_name.append("slacs1250+0523")  # Index 40
data_name.append("slacs1432+6317")  # Index 41
data_name.append("slacs2238-0754")  # Index 42
data_name.append("slacs2341+0000")  # Index 43


data_name = data_name[cosma_array_id]

aggregator_results_path = f"{cosma_output_path}/slacs_sub4_mocks_1/{data_name}"

agg_1 = af.Aggregator(directory=str(aggregator_results_path))

agg_lens = agg_1.filter(
    agg_1.directory.contains("phase[1]_mass[total]_source/settings"),
    agg_1.directory.contains("power_law"),
)

sersic_light_profile = [
    al.lp.EllipticalSersic(
        centre=(info["centre0"], info["centre1"]),
        intensity=info["intensity"],
        effective_radius=info["r_eff"],
        sersic_index=info["sersic_index"],
    )
    for info in agg_lens.values("info")
]

source_galaxy = al.Galaxy(redshift=1.0, light=sersic_light_profile)
masked_imaging_gen = al.agg.MaskedImaging(aggregator=agg_lens)
settings_masked_imaging = al.SettingsMaskedImaging(
    grid_class=al.Grid2DIterate, fractional_accuracy=0.9999
)
masked_imaging_gen_param = al.agg.MaskedImaging(
    aggregator=agg_lens, settings_masked_imaging=settings_masked_imaging
)
tracer_gen = al.agg.Tracer(aggregator=agg_lens)

settings_pixelization_1 = al.SettingsPixelization(is_stochastic=False, kmeans_seed=1)
settings_pixelization_2 = al.SettingsPixelization(is_stochastic=False, kmeans_seed=2)
settings_pixelization_3 = al.SettingsPixelization(is_stochastic=False, kmeans_seed=3)
settings_pixelization_rand = al.SettingsPixelization(is_stochastic=True)

filename = os.path.join(aggregator_results_path, "lh_surface.json")

### EXAMPLE OF HOW TO LOAD RESULTS ###

# with open(filename, "r") as f:
#     result_dict = json.load(f)
#
# print(result_dict["fit_1"])
#
# slopes = np.arange(1.5, 1.6, 0.05)

tracer = list(tracer_gen)[0]
masked_imaging = list(masked_imaging_gen)[0]
masked_imaging_param = list(masked_imaging_gen_param)[0]
lens_galaxy = tracer.galaxies[0]

fit_1_figures_of_merit = []
fit_2_figures_of_merit = []
fit_3_figures_of_merit = []
fit_rand_figures_of_merit = []
fit_param_figures_of_merit = []

slopes = np.arange(1.5, 1.6, 0.05)

result_dict = {"fit_1": [], "fit_2": [], "fit_3": [], "fit_rand": [], "fit_param": []}

output_counter = 0

for slope in slopes:

    output_counter += 1

    lens_galaxy.mass.slope = slope
    new_tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
    fit_1 = al.FitImaging(
        masked_imaging=masked_imaging,
        tracer=tracer,
        settings_pixelization=settings_pixelization_1,
    )
    fit_2 = al.FitImaging(
        masked_imaging=masked_imaging,
        tracer=tracer,
        settings_pixelization=settings_pixelization_2,
    )
    fit_3 = al.FitImaging(
        masked_imaging=masked_imaging,
        tracer=tracer,
        settings_pixelization=settings_pixelization_3,
    )
    fit_rand = al.FitImaging(
        masked_imaging=masked_imaging,
        tracer=tracer,
        settings_pixelization=settings_pixelization_rand,
    )
    fit_param = al.FitImaging(masked_imaging=masked_imaging_param, tracer=new_tracer)

    result_dict["fit_1"].append(fit_1.figure_of_merit)
    result_dict["fit_2"].append(fit_2.figure_of_merit)
    result_dict["fit_3"].append(fit_3.figure_of_merit)
    result_dict["fit_rand"].append(fit_rand.figure_of_merit)
    result_dict["fit_param"].append(fit_param.figure_of_merit)

    if output_counter == 1000:

        with open(filename, "w+") as f:
            json.dump(result_dict, f, indent=4)

        output_counter = 0
