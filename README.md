# Classifying Thermodynamic Cloud Phase Using Machine Learning Models (Goldberger et al. 2025)

Code for "**Classifying Thermodynamic Cloud Phase Using Machine Learning Models**", by Lexie Goldberger*, Maxwell Levin*, Carlandra Harris, Andrew Geiss, Matthew D. Shupe, and Damao Zhang, 2025. ([preprint](https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1501/))

## Data

### Raw Data

The Thermodynamic Cloud Phase Value-Added-Product (VAP) dataset from the ARM program for the North Slope of Alaska site
is preprocessed and used as input for the models in this paper. The full dataset is too large to host on github (250+
GB), so a script is included to download the files from ARM.gov (`src/preprocessing/data/raw/download.py`).

The full dataset used as input is also [available here](https://adc.arm.gov/discovery/#/results/id::nsathermocldphaseC1.c1_cloud_phase_mplgr_macro_thermocloudphase_cloud?dataLevel=c1&showDetails=true) and can be cited as:

Atmospheric Radiation Measurement (ARM) user facility. 2018. Thermodynamic cloud phase (THERMOCLDPHASE). 2018-01-01 to 2022-12-31, North Slope Alaska (NSA) Central Facility, Barrow AK (C1). Compiled by D. Zhang and M. Levin. ARM Data Center. Data set accessed 2024-08-16 at <http://dx.doi.org/10.5439/1871014>.

### Output Data

The processed ML outputs on the test NSA dataset and the ANX dataset can be found here:

- <https://doi.org/10.5439/2568095>
- <https://adc.arm.gov/discovery/#/results/id::285_macro_thermocloudphase-ml_cloud_cloudphase>
- <https://iop.arm.gov/0pi-data/levin/thermocloudphase-ml/>
- <https://www.arm.gov/data/data-sources/pi-thermocloudphase-ml-285>

## Setup

The code was written with python 3.11 and requires python dependencies listed in the `environment.yml` file.

If you have conda installed, you may run

```shell
conda env create
conda activate cldphase
```

to create the python environment used in this codebase.

The project is structured in a `src/` layout, with `src/preprocessing` scripts intended to be run first, followed by
`src/processing` scripts, and finally `src/analysis`. The recommended order of operations is listed below.

```shell
cd src/preprocessing/data/raw
python download.py # <- add credentials first, then run for nsathermocldphaseC1.c1
python download.py # <- update for anxthermocldphaseM1.c1 and run again

cd ../.. # <- back to src/preprocessing
python cnn_inputs.py  # <- run with defaults (this is only for the training/validation data)

cd ../processing
python train_cnn.py # <- run with defaults
python train_cnn.py # <- after updating the code in main() to train the other model
python train_mlp_rf.py

python make_predictions.py  # <- run with defaults
python make_predictions.py  # <- after updating the file glob to use ANX *.nc files
python get_importances.py

cd ../analysis
python consolidate_predictions.py # <- run with defaults
python consolidate_predictions.py ../processing/data/predictions/anx*.nc  # <- ANX data

# Then open and run the notebooks in src/analysis and src/figure_code to generate the
# figures and tables in the paper/supplement
```
