import pickle
import pandas as pd


results = pd.read_pickle(f"shear_and_orientations.pkl")
print(results.columns)
print(
    results[
        [
            "shear_mag",
            "bulge_PA_w_shear",
            "disk_PA_w_shear",
            "shear_angle",
            "PA_w_shear",
            "PA",
        ]
    ]
)
