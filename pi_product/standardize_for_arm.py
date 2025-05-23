import re
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm.auto import tqdm

OUTPUT_FOLDER = Path(__file__).parent / "data"
INPUT_FOLDER = Path(__file__).parent.parent / "src/processing/data/predictions"


def cleanup_file(file: Path) -> None:
    ds = xr.open_dataset(
        file,
        mask_and_scale=False,
        decode_times=False,
        decode_cf=False,
    )
    _cleanup_global_attrs(ds)
    _cleanup_coordinates(ds)
    regular_vars, confidence_vars, pred_vars, variable_order = _get_variable_order(ds)
    ds = ds[variable_order]
    _cleanup_confidence_vars(ds, confidence_vars)
    _cleanup_pred_vars(ds, pred_vars)
    _cleanup_encoding(ds, variable_order)
    save_dataset(ds, file)
    return None


def save_dataset(ds: xr.Dataset, input_file: Path) -> None:
    datastream = ds.attrs["datastream"]
    site = ds.attrs["site_id"]

    # get the new filename/path
    file_parts = input_file.name.split(".")
    new_first_part = datastream.split(".")[0]
    new_file_basename = ".".join([new_first_part, *file_parts[1:]])

    output_folder = OUTPUT_FOLDER / str(site) / str(datastream)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_filepath = output_folder / new_file_basename

    assert not output_filepath.exists(), f"{output_filepath} already exists!"
    ds.to_netcdf(output_filepath)
    return None


def _cleanup_global_attrs(ds: xr.Dataset) -> None:
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Update global attributes
    # - Change datastream from `thermocldphase.c1` to `cldphasemlamt.c1`
    # - Remove irrelevant global attributes from the input VAP: command_line,
    #   history, doi, dod_version, process_version, input_datastreams
    # - Remove threshold attrs
    # - Add description element explaining each model type
    site, fac = ds.attrs["site_id"], ds.attrs["facility_id"]
    ds.attrs["datastream"] = f"{site}cldphasemlamt{fac}.c1"
    ds.attrs["doi"] = "10.5439/2568095"  # reserved for this dataset
    ds.attrs["paper_doi"] = "10.5194/egusphere-2025-1501"
    ds.attrs["paper_title"] = (
        "Classifying Thermodynamic Cloud Phase Using Machine Learning Models"
    )

    del ds.attrs["command_line"]
    del ds.attrs["history"]
    del ds.attrs["dod_version"]
    del ds.attrs["process_version"]
    del ds.attrs["input_datastreams"]

    threshold_attrs = [
        att_name for att_name in list(ds.attrs) if "threshold" in att_name
    ]
    for att_name in threshold_attrs:
        del ds.attrs[att_name]

    del ds.attrs["radar_reflectivity_offset"]
    del ds.attrs["radar_reflectivity_offset_comment"]
    return None


def _cleanup_coordinates(ds: xr.Dataset) -> None:
    # Fix the phase coordinate (change from 0-7 to actual labels)
    ds["phase"] = xr.DataArray(
        data=[
            "clear_sky",
            "liquid",
            "ice",
            "mixed_phase",
            "drizzle",
            "liquid_drizzle",
            "rain",
            "snow",
        ],
        coords=dict(
            phase=[
                "clear_sky",
                "liquid",
                "ice",
                "mixed_phase",
                "drizzle",
                "liquid_drizzle",
                "rain",
                "snow",
            ]
        ),
        # coords=("phase",),
        dims="phase",
        attrs={
            "long_name": "Cloud Thermodynamic Phase",
            "units": "1",
        },
        name="phase",
    )
    return None


def _get_variable_order(
    ds: xr.Dataset,
) -> tuple[list[str], list[str], list[str], list[str]]:
    # Helper func to reorder data like so:
    # * all variables from the VAP
    # * pred variables
    #   - Confidence vars for: cnn, cnn_dropout, rf_balanced, mlp_balanced, rf_imbalanced, mlp_imbalanced
    #   - All cnn_ variables, starting with cnn, then the ablations (mpl*, rad*, mwr, sonde)
    #   - All cnn_dropout variables
    #   - All rf_balanced variables
    #   - All mlp_balanced variables
    #   - All rf_imbalanced variables
    #   - All mlp_imbalanced variables
    # * location/metadata variables (lat, lon, alt)

    def is_pred_var(v: str) -> bool:
        return any([v.startswith(n) for n in ["cnn", "rf", "mlp"]])

    vap_cloud_phase_vars = ["cloud_phase_mplgr", "qc_cloud_phase_mplgr"]
    regular_vars = [
        v
        for v in ds.data_vars
        if not (is_pred_var(v) or v in vap_cloud_phase_vars)
        and (v not in ["lat", "lon", "alt"])
    ]
    all_pred = sorted([v for v in ds.data_vars if is_pred_var(v)])
    confidence_vars = [v for v in all_pred if "confidence" in v]
    pred_vars = [v for v in all_pred if v not in confidence_vars]

    variable_order = [
        *regular_vars,
        *["cloud_phase_mplgr", "qc_cloud_phase_mplgr"],
        *confidence_vars,
        *pred_vars,
        *["lat", "lon", "alt"],
    ]
    return regular_vars, confidence_vars, pred_vars, variable_order


def _cleanup_confidence_vars(ds: xr.Dataset, confidence_vars: list[str]) -> None:
    pattern = r"^(?P<model>cnn_dropout|cnn|rf_balanced|rf_imbalanced|mlp_balanced|mlp_imbalanced)(?P<suffix>[a-z_]*)_confidence"

    # Now we fix some metadata for the confidence variables
    # - Update attributes: long_name, description
    # - Add comment attribute
    # - Remove flag_values and flag_meanings attrs (phase is an axis now, values are float)
    # - Remove ancillary_variables attr
    for var_name in confidence_vars:
        name_match = re.match(pattern, var_name).groupdict()  # type: ignore
        _model, *_ = (
            name_match.values()
        )  # no confidence recorded for ablation predictions
        model = _model.upper()

        ds[var_name].attrs["long_name"] = f"{model} Confidence Scores"
        ds[var_name].attrs["description"] = f"{model} confidence scores by cloud phase."
        ds[var_name].attrs["comment"] = (
            f"Confidence scores are the direct output of the model. Predictions are obtained using the argmax of confidence along the 'phase' axis, and are recorded by the '{_model}' variable in this dataset."
        )

        del ds[var_name].attrs["flag_values"]
        del ds[var_name].attrs["flag_meanings"]
        del ds[var_name].attrs["ancillary_variables"]

    return None


def _cleanup_pred_vars(ds: xr.Dataset, pred_vars: list[str]) -> None:
    pattern = r"^(?P<model>cnn_dropout|cnn|rf_balanced|rf_imbalanced|mlp_balanced|mlp_imbalanced)(?P<suffix>[a-z_]*)"

    # Each of the pred variables needs to have its attributes updated:
    # - Remove 'unknown' from flag_meanings
    # - Remove '8' from flag_values
    # - Remove ancillary_variables attr
    # - Update long_name
    # - Update description
    # We also want to change the data type from float64 to int8 since these are flag values
    # (integers 0 through 7), and there are no nan/missing values.
    for var_name in pred_vars:
        name_match = re.match(pattern, var_name).groupdict()  # type: ignore
        _model, suffix = name_match.values()

        model = _model.upper()
        long_name = {
            "": f"Predicted Thermodynamic Cloud Phase ({model})",
            "_mpl": f"Predicted Thermodynamic Cloud Phase ({model} no MPL)",
            "_mpl_b": f"Predicted Thermodynamic Cloud Phase ({model} no MPL B)",
            "_mpl_ldr": f"Predicted Thermodynamic Cloud Phase ({model} no MPL LDR)",
            "_mwr": f"Predicted Thermodynamic Cloud Phase ({model} no MWR)",
            "_rad": f"Predicted Thermodynamic Cloud Phase ({model} no Radar)",
            "_rad_ldr": f"Predicted Thermodynamic Cloud Phase ({model} no Radar LDR)",
            "_rad_mdv": f"Predicted Thermodynamic Cloud Phase ({model} no Radar MDV)",
            "_rad_ref": f"Predicted Thermodynamic Cloud Phase ({model} no Radar Z_e)",
            "_rad_spec": f"Predicted Thermodynamic Cloud Phase ({model} no Radar W)",
            "_sonde": f"Predicted Thermodynamic Cloud Phase ({model} no Sonde T)",
        }[suffix]
        description = {
            "": f"{model}-predicted thermodynamic cloud phase.",
            "_mpl": f"{model}-predicted thermodynamic cloud phase, with all MPL lidar data withheld during inference.",
            "_mpl_b": f"{model}-predicted thermodynamic cloud phase, with MPL backscatter data withheld during inference.",
            "_mpl_ldr": f"{model}-predicted thermodynamic cloud phase, with MPL linear depolarization ratio data withheld during inference.",
            "_mwr": f"{model}-predicted thermodynamic cloud phase, with MWR liquid water path data withheld during inference.",
            "_rad": f"{model}-predicted thermodynamic cloud phase, with all radar data withheld during inference.",
            "_rad_ldr": f"{model}-predicted thermodynamic cloud phase, with radar linear depolarization ratio data withheld during inference.",
            "_rad_mdv": f"{model}-predicted thermodynamic cloud phase, with radar mean doppler velocity data withheld during inference.",
            "_rad_ref": f"{model}-predicted thermodynamic cloud phase, with radar reflectivity data withheld during inference.",
            "_rad_spec": f"{model}-predicted thermodynamic cloud phase, with radar spectral width data withheld during inference.",
            "_sonde": f"{model}-predicted thermodynamic cloud phase, with radiosonde temperature data withheld during inference.",
        }[suffix]

        ds[var_name].attrs["long_name"] = long_name
        ds[var_name].attrs["description"] = description
        ds[var_name].attrs["flag_values"] = [0, 1, 2, 3, 4, 5, 6, 7]  # drop unknown
        ds[var_name].attrs["flag_meanings"] = (
            "clear_sky liquid ice mixed_phase drizzle liquid_drizzle rain snow"
        )

        del ds[var_name].attrs["ancillary_variables"]

        # Change dtype to int16. Some space savings here
        ds[var_name].data = ds[var_name].data.astype("int8")
        ds[var_name].encoding["dtype"] = ds[var_name].dtype

    return None


def _cleanup_encoding(ds: xr.Dataset, variable_order: list[str]) -> None:
    for var_name in variable_order:
        if "_FillValue" in ds[var_name].attrs:
            del ds[var_name].attrs["_FillValue"]
        if np.issubdtype(ds[var_name].data.dtype, np.number):
            ds[var_name].encoding = dict(
                complevel=3, dtype=ds[var_name].dtype, zlib=True, _FillValue=None
            )

    for var_name in list(ds.coords):
        ds[var_name].encoding["_FillValue"] = None
        if "_FillValue" in ds[var_name].attrs:
            del ds[var_name].attrs["_FillValue"]
    return None


if __name__ == "__main__":
    input_files = sorted(INPUT_FOLDER.glob("*.nc"))

    print(f"{len(input_files) = }")

    for filepath in tqdm(input_files):
        cleanup_file(filepath)
