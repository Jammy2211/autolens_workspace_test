import json

from autoconf.dictable import Dictable
import autogalaxy as ag

json_file = "galaxy.json"

galaxy = ag.Galaxy(
        redshift=1.0, pixelization=ag.pix.VoronoiMagnification(), regularization=ag.reg.AdaptiveBrightness()
    )

with open(json_file, "w+") as f:
    json.dump(galaxy.dict(), f, indent=4)

with open(json_file, "r+") as f:
    galaxy_dict = json.load(f)

galaxy_from_dict = Dictable.from_dict(galaxy_dict)

print(galaxy_from_dict)

galaxy_from_dict = ag.Galaxy.from_dict(galaxy_dict)