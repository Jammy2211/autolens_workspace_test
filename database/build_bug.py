"""
__Imports__
"""
import os
from os import path
import warnings

warnings.filterwarnings("ignore")

import autofit as af
from autoconf import conf
import time

"""
__Paths__
"""
workspace_path = os.getcwd()

config_path = path.join(workspace_path, "config")
conf.instance.push(new_path=config_path)

"""
___Database__

The name of the database, which corresponds to the output results folder.
"""
database_name = "base"

"""
Remove database is making a new build (you could delete manually via your mouse). Building the database is slow, so 
only do this when you redownload results. Things are fast working from an already built database.
"""
try:
    os.remove(path.join("output", f"{database_name}.sqlite"))
except FileNotFoundError:
    pass

"""
Load the database. If the file `slacs.sqlite` does not exist, it will be made by the method below, so its fine if
you run the code below before the file exists.
"""
agg = af.Aggregator.from_database(
    filename=f"{database_name}.sqlite", completed_only=True
)

"""
Add all results in the directory "output/slacs" to the database, which we manipulate below via the agg.
Avoid rerunning this once the file `slacs.sqlite` has been built.
"""
start = time.time()
agg.add_directory(directory=path.join("output", database_name))
print(f"Time to add directory to database {time.time() - start}")