import csv
import json
from os import path
import os

info_fieldnames = [
    "id_str",
]

result_fieldnames = [
    "einstein_radius_max_lh",
    "einstein_radius_median_pdf",
    "einstein_radius_lower_3_sigma",
    "einstein_radius_upper_3_sigma",
    "vis_total_lens_flux",
    "vis_total_lensed_source_flux",
    "vis_total_source_flux",
    "vis_magnification",
    "vis_max_lensed_source_signal_to_noise_ratio",
]

fieldnames = [""] + info_fieldnames + result_fieldnames

dataset_path = path.join("agg_png_csv", "dataset", "lens")

csv_file = path.join("agg_png_csv", "result.csv")

with open(csv_file, "w", newline="") as csvfile:

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i, dataset_name in enumerate(os.listdir(dataset_path)):

        if "py" in dataset_name:
            continue

        if "csv" in dataset_name:
            continue

        result_dict = {}

        dataset_main_path = path.join(dataset_path, dataset_name)

        info = {}

        try:
            with open(path.join(dataset_main_path, "info.json")) as f:
                info = json.load(f)
        except FileNotFoundError:
            info = {}

        result = {}

        try:
            with open(path.join(dataset_main_path, "result.json"), "r") as f:
                result = json.load(f)
        except FileNotFoundError:
            result = {}

        info_dict = {fieldname: info[fieldname] for fieldname in info_fieldnames}
        for fieldname in result_fieldnames:

            result_dict[fieldname] = result.get(fieldname, None)

        writer.writerow(
            {
                "": i,
                **info_dict,
                **result_dict
            }
        )