# Cost-Sensitive Thresholding vs. Cost-Sensitive Training for ICD Shocks

## Setup

**Requires Python 3.12+**

Run:
```bash
python setup.py
```

This creates a virtual environment, installs all dependencies, and registers a Jupyter kernel.

To run the end-to-end pipeline notebook, open `pipeline.ipynb` and select the `shock-thresholding` kernel. This notebook can be used to reproduce the results in the paper.

---

## Datasets

These datasets are automatically downloaded in the pipeline notebook `pipeline.ipynb`. If you'd like to download them manually:

**MIT-BIH Malignant Ventricular Ectopy Database** (~47.5 MB):
```commandline
wget -r -N -c -np -P data/vfdb https://physionet.org/files/vfdb/1.0.0/
```

**CU Ventricular Tachyarrhythmia Database** (~7.9 MB):
```commandline
wget -r -N -c -np -P data/cudb https://physionet.org/files/cudb/1.0.0/
```

These will download into a `data/` directory. The commands are idempotent.
