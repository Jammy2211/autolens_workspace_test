The ``slam`` folder contains the Source, Light and Mass (SLaM) pipeline, which are **advanced** template pipelines for
automated lens modeling.

The ``subhalo`` folder contains extensions to the SLaM pipelines which perform dark matter substructure detection and
sensiivity mapping.

Files (Advanced)
----------------

- ``detection.py``: A subhalo pipeline which performs dark matter subhalo detection via Bayesian model comparison.

- ``sensitivity_imaging.py``: Extends a SLaM pipeline with subhalo sensitivity mapping for imaging data.
- ``sensitivity_interferometer.py``: Extends a SLaM pipeline with subhalo sensitivity mapping for interferometer data.
