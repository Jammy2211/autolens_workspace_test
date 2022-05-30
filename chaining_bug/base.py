from os import path
import sys
import json

import os

""" 
__AUTOLENS + DATA__
"""
cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "chaining_bug", "config"))

import autofit as af
import autolens as al

sys.path.insert(0, os.getcwd())
import slam

pixel_scales = 0.05

dataset_name = "slacs0728+3835"

dataset_path = path.join("dataset", dataset_name)

with open(path.join(dataset_path, "info.json")) as json_file:
    info = json.load(json_file)

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image_lens_light_scaled.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map_scaled.fits",
    pixel_scales=pixel_scales,
    name=dataset_name,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native,
    pixel_scales=pixel_scales,
    centre=(0.0, 0.0),
    radius=3.5,
)

positions = al.Grid2DIrregular.from_json(
    file_path=path.join(dataset_path, "positions.json")
)

imaging = imaging.apply_mask(mask=mask)

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, databse use, etc.
"""
sample_folder = "slacs_base_0"

settings_autofit = af.SettingsSearch(
    path_prefix=path.join(sample_folder),
    unique_tag=dataset_name,
    number_of_cores=1,
    session=None,
    info=info,
)


"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit.
"""
setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=True,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
    search_pixelized_dict={"nlive": 30, "sample": "rwalk"},
)

"""
__SOURCE PARAMETRIC PIPELINE (with lens light)__

The SOURCE PARAMETRIC PIPELINE (with lens light) uses three searches to initialize a robust model for the 
source galaxy's light, which in this example:

 - Uses a parametric `EllSersic` bulge and `EllExponential` disk with centres aligned for the lens
 galaxy's light.

 - Uses an `EllIsothermal` model for the lens's total mass distribution with an `ExternalShear`.

 __Settings__:

 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the MASS TOTAL PIPELINE).
"""

mass = af.Model(al.mp.EllIsothermal)

analysis = al.AnalysisImaging(
    dataset=imaging,
    positions=positions,
    settings_lens=al.SettingsLens(positions_threshold=0.7),
)

lens_bulge = af.Model(al.lp.EllSersicCore)
lens_bulge.radius_break = 0.05
lens_bulge.gamma = 0.0
lens_bulge.alpha = 2.0

lens_disk = af.Model(al.lp.EllExponentialCore)
lens_disk.radius_break = 0.05
lens_disk.gamma = 0.0
lens_disk.alpha = 2.0

lens_bulge.centre = lens_disk.centre

source_bulge = af.Model(al.lp.EllSersicCore)
source_bulge.radius_break = 0.05
source_bulge.gamma = 0.0
source_bulge.alpha = 2.0


source_parametric_results = slam.source_parametric.with_lens_light(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_hyper=setup_hyper,
    lens_bulge=lens_bulge,
    lens_disk=lens_disk,
    mass=mass,
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=source_bulge,
    mass_centre=(0.0, 0.0),
    redshift_lens=info["redshift_lens"],
    redshift_source=info["redshift_source"],
)

os._exit(1)

"""
Finish.
"""
