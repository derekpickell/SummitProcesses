"""
Builds forcings for, and runs, the Community Firn Model (CFM) at the OGRE stations.

Produces the CFM runs read by the analysis scripts:
    physics       OGRE_CFM_DIR/<station>/<model>/output/output_firn_layers.csv
                  densification physics (testing_verticals.PHYSICS_MODELS; ogre_analysis.FIG3_MODEL at every Figure 3 station)
    conductivity  OGRE_CFM_DIR/<station>/MaxSP_<conductivity>/output/output_firn_layers.csv
                  thermal conductivity schemes (testing_verticals.CONDUCTIVITY_MODELS)
    density       OGRE_CFM_NO_OGRE_DIR/<station>/<model>_<density>/output/output_firn_layers.csv
                  surface densities for stacked_plot.py (stacked_plot.MODEL_NAME x stacked_plot.DENSITIES)
    sensitivity   OGRE_CFM_SENSITIVITY_DIR: surface density 300 vs 330 kg m⁻³ at Camp (8438), prints velocity bias

Forcing types (one set of forcing .csv files per station folder):
    ogre   MERRA-2 forcing at the station, accumulation replaced by OGRE GNSS-IR accumulation where available
    merra  MERRA-2 forcing at the station only
In both, MERRA-2 P-E is corrected for its bias against OGRE (STATION_PERCENT_DECIMAL) unless --no-correction is given.

SETUP
-----
Same environment variables as ogre_analysis.py (REFL_CODE, OGRE_STATIC_DIR, OGRE_CFM_DIR, OGRE_CFM_NO_OGRE_DIR), plus:
    OGRE_CFM_HOME            CommunityFirnModel checkout (contains CFM_main/main.py). Required unless --dry-run.
    OGRE_CFM_PYTHON          Python used to run CFM. Default: <OGRE_CFM_HOME>/env/bin/python if present, else this Python.
    OGRE_CFM_FORCING_DIR     MERRA-2 FLX .nc4 files, one time step (daily mean) per file
                             Default: ~/Desktop/Paper4_files/MERRA-2_flx
    OGRE_CFM_INIT_DENSITY    CFM initial firn profile (initfirnFile). Default: <OGRE_STATIC_DIR>/Summit_avg_density_profile.csv
    OGRE_CFM_SENSITIVITY_DIR output folder for the sensitivity test. Default: ~/Desktop/Paper4_files/cfm_sensitivity

USAGE
-----
    python generate_cfm.py physics                       # defaults reproduce the runs the analysis reads
    python generate_cfm.py conductivity
    python generate_cfm.py density
    python generate_cfm.py physics --stations 0009 --models GSFC2020 Crocus
    python generate_cfm.py density --densities 300 310 --skip-forcings
    python generate_cfm.py physics --dry-run             # write forcings + configs, don't run CFM
    python generate_cfm.py sensitivity
"""
import argparse
import copy
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, date

import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.nonparametric.smoothers_lowess import lowess

from ogre_analysis import (
    STATIC_DIR, CFM_DIR, CFM_NO_OGRE_DIR, SECONDS_PER_YEAR, FIG3_STATIONS, FIG3_MODEL, FIG3_AVERAGE_CFM_STATION, bbox,
    ppp_csv_path, gnssir_path, get_compaction,
)
from gnss_utils import read_ppp, read_gnssir_single, get_pole_column, POLE_LENGTH
from testing_verticals import CONDUCTIVITY_MODELS, PHYSICS_MODELS
from stacked_plot import STATIONS as STACKED_STATIONS, MODEL_NAME as STACKED_MODEL, DENSITIES as STACKED_DENSITIES, STATION_PERCENT_DECIMAL

########### PATHS ###########
CFM_HOME = os.environ.get('OGRE_CFM_HOME')
CFM_FORCING_DIR = os.environ.get('OGRE_CFM_FORCING_DIR', os.path.expanduser('~/Desktop/Paper4_files/MERRA-2_flx'))
CFM_INIT_DENSITY = os.environ.get('OGRE_CFM_INIT_DENSITY', os.path.join(STATIC_DIR, 'Summit_avg_density_profile.csv'))
CFM_SENSITIVITY_DIR = os.environ.get('OGRE_CFM_SENSITIVITY_DIR', os.path.expanduser('~/Desktop/Paper4_files/cfm_sensitivity'))

LAYERS_FILE = 'output_firn_layers.csv'  # per-layer compaction written by CFM (see README)
CONFIG_FILE = 'cfm_config.json'
FORCING_FILES = {  # CFM input name -> forcing .csv written by generate_forcings
    'temps': '1450-2024_Summit_temps.csv', 'bdot': '1450-2024_Summit_bdot.csv',
    'rain': '1450-2024_Summit_rain.csv', 'subl': '1450-2024_Summit_subl.csv',
}
MERRA_LON_RANGE = (-38.75, -37.5)  # MERRA-2 grid columns spanning the OGRE array; station longitudes are clamped to this
MERRA_LAT = 72.5

########### CFM CONFIG ###########
default_config = {
"InputFileFolder": "CFMinput_test",
"InputFileNameTemp": FORCING_FILES['temps'],
"InputFileNamebdot": FORCING_FILES['bdot'],
"InputFileNameIso": "none",
"InputFileNamerho": "none",
"InputFileNamemelt": "example_SMELT.csv",
"InputFileNameStrain": "none",
"InputFileNameSublim": FORCING_FILES['subl'],
"InputFileNameRain": FORCING_FILES['rain'],
"resultsFolder": "CFMoutput_test/csv",
"initfirnFile": "Summit_avg_density_profile.csv",
"initprofile": True,
"input_type": "csv",
"input_type_options": ["csv","dataframe"],
"DFresample": "none",
"DFfile": "none",
"physRho": "Arthern2010T",
"physRho_options":["HLdynamic","HLSigfus","Li2004","Li2011","Helsen2008","Arthern2010S","Arthern2010T","Li2015","Goujon2003","Barnola1991","Morris2014","KuipersMunneke2015","Crocus","Ligtenberg2011"],
"MELT": False,
"RAIN": True,
"ReehCorrectedT": False,
"FirnAir": False,
"AirConfigName": "none",
"TWriteInt": 1,
"TWriteStart": 2020.0,
"int_type": "nearest",
"int_type_options": ["nearest","linear"],
"SeasonalTcycle": False,
"SeasonalThemi": "north",
"coreless": False,
"TAmp":10.0,
"physGrain": True,
"calcGrainSize": False,
"GrGrowPhysics": "Arthern",
"GrGrowPhysics_options": ["Arthern", "Katsushima"],
"heatDiff": True,
"conductivity": "Calonne2019",
"conductivity_options": ["Schwander","Yen_fixed","Yen_var","Anderson","Yen_b","Sturm","VanDusen","Schwerdtfeger","Riche","Jiawen","mix","Calonne2011","Calonne2019"],
"variable_srho": False,
"srho_type": "userinput",
"srho_type_options": ["userinput","param","noise","Brils22"],
"rhos0": 300.0,
"r2s0": 1.0e-8,
"AutoSpinUpTime": False,
"yearSpin": 1,
"H": 3000,
"HbaseSpin": 2700.0,
"stpsPerYear": 12.0,
"D_surf": 1.0,
"bdot_type": "mean",
"bdot_type_options": ["instant","mean","stress"],
"grid_outputs": True,
"grid_output_res": 0.05,
"isoDiff": False,
"iso": ["d18O","dD"],
"isoOptions":["d18O","dD","NoDiffusion"],
"spacewriteint": 1,
"horizontal_divergence": False,
"strain_softening": False,
"tuning_bias_correction": False,
"residual_strain": 0.0002,
"outputs": ["density", "depth", "temperature", "age", "DIP","LWC","meltoutputs","climate", "compaction"],
"outputs_options": ["density", "depth", "temperature", "age", "Dcon", "bdot_mean", "climate", "compaction", "grainsize", "temp_Hx", "isotopes", "BCO", "DIPc", "DIP", "LWC","gasses", "PLWC_mem", "viscosity", "runoff", "refrozen"],
"resultsFileName": "CFMresults_Summit1.hdf5",
"spinFileName": "CFMspin_Summit1.hdf5",
"doublegrid": True,
"nodestocombine": 3,
"multnodestocombine": 12,
"grid1bottom": 5.0,
"grid2bottom": 15.0,
"spinup_climate_type": "mean",
"spinup_climate_type_options": ["mean", "initial"],
"manual_climate": False,
"deepT": 255.88,
"bdot_long": 0.49073,
"manual_iceout": False,
"iceout": 0.23,
"QMorris": 110.0e3,
"timesetup": "exact",
"timesetup_options": ["exact","interp","retmip"],
"liquid": "bucket",
"liquid_options": ["bucket","darcy","resingledomain","prefsnowpack"],
"merging": True,
"merge_min": 1e-9,
"LWCcorrect": False,
"manualT": False,
"no_densification": False,
"rad_pen": False,
"site_pressure": 1013.25,
"output_bits": "float32",
"spinUpdate": False,
"spinUpdateDate": 1980.0,
"DIPhorizon": 300.0,
"NewSpin": True,
"ColeouLesaffre": True,
"IrrVal": 0.02,
"RhoImp": 830,
"DownToIce": False,
"ThickImp": 0.1,
"Ponding": False,
"DirectRunoff": 0.0,
"RunoffZuoOerlemans": False,
"Slope": 0.1,
"SUBLIM": True,
"keep_firnthickness": False
}

# Settings shared by all OGRE runs (daily steps, 1 cm output grid, written from 2022)
OGRE_RUN_SETTINGS = {
    "MELT": False,
    "rhos0": 300.0,
    "grid_output_res": 0.01,
    "stpsPerYear": 365,
    "TWriteStart": 2022,
    "TWriteInt": 1,
    "int_type": "linear",
}


def _dir(path):
    """absolute folder path with trailing separator (CFM concatenates folder + file names)"""
    return os.path.join(os.path.abspath(path), '')


def generate_json(config_path, custom_params):
    """write a CFM config: default_config updated with custom_params (default_config itself is not modified)"""
    config = copy.deepcopy(default_config)
    config.update(custom_params)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as json_file:
        json.dump(config, json_file, indent=4)
    return config


def _cfm_python():
    python = os.environ.get('OGRE_CFM_PYTHON')
    if python:
        return python
    venv_python = os.path.join(CFM_HOME, 'env', 'bin', 'python')
    return venv_python if os.path.isfile(venv_python) else sys.executable


def run_cfm(config_path, result_folder):
    """
    Run CFM_main/main.py on config_path and put output_firn_layers.csv in result_folder.
    CFM runs in a scratch working directory; output_firn_layers.csv is picked up from there (or from result_folder).
    """
    if not CFM_HOME:
        raise EnvironmentError("Set OGRE_CFM_HOME to your CommunityFirnModel checkout (or use --dry-run).")
    main_py = os.path.join(CFM_HOME, 'CFM_main', 'main.py')
    if not os.path.isfile(main_py):
        raise FileNotFoundError(f"CFM main script not found at {main_py}")

    os.makedirs(result_folder, exist_ok=True)
    destination = os.path.join(result_folder, LAYERS_FILE)
    with tempfile.TemporaryDirectory() as workdir:
        result = subprocess.run([_cfm_python(), main_py, os.path.abspath(config_path)], cwd=workdir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"CFM failed for {config_path}:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
        produced = os.path.join(workdir, LAYERS_FILE)
        if os.path.isfile(produced):
            shutil.move(produced, destination)
    if not os.path.isfile(destination):
        raise FileNotFoundError(f"CFM finished but wrote no {LAYERS_FILE} for {config_path}. "
                                f"This output needs the CFM version described in the README.")
    print(f"-> wrote {destination}")
    return destination


########## FORCINGS ##########
def year_fraction(dt):
    """datetime -> decimal year. Includes the +0.0833 yr (1 month) offset used for the published runs."""
    if isinstance(dt, datetime):
        dt = dt.date()  # strip time
    start = date(dt.year, 1, 1).toordinal()
    year_length = date(dt.year+1, 1, 1).toordinal() - start
    return dt.year + float(dt.toordinal() - start) / year_length + .0833


def sumup_density_profile():
    """SUMup 2024 density profiles near Summit (ogre_analysis.bbox) -> LOWESS mean density [kg m⁻³] vs depth [m]"""
    df = pd.read_csv(os.path.join(STATIC_DIR, "SUMup_2024_density_greenland.csv"))
    cols = ["latitude", "longitude", "midpoint", "density"]
    df = df.dropna(subset=cols)
    df[cols] = df[cols].apply(pd.to_numeric)
    in_bbox = df[(df["latitude"] >= bbox[0][0]) & (df["latitude"] <= bbox[1][0]) &
                 (df["longitude"] >= bbox[0][1]) & (df["longitude"] <= bbox[1][1])]
    if in_bbox.empty:
        raise ValueError("No SUMup density profiles found within the bounding box.")

    sorted_idx = np.argsort(in_bbox["midpoint"].values)
    smoothed = lowess(endog=in_bbox["density"].values[sorted_idx], exog=in_bbox["midpoint"].values[sorted_idx], frac=0.2, return_sorted=True)
    return smoothed[:, 1], smoothed[:, 0]  # densities, depths


def get_corresponding_density(depths, densities, stake_depths):
    """using depth vs density curve, extract closest value for OGRE stake depth"""
    depths = np.array(depths); densities = np.array(densities)
    return np.array([densities[np.abs(depths - d).argmin()] for d in stake_depths])


def extend_data(list1, list2, start_year, repeat_year_start=1981.99836849, repeat_year_end=1989.99836849):
    """extend a forcing back to start_year by repeating the 1982-1990 segment"""
    list1 = np.array(list1); list2 = np.array(list2)

    # Find indices closest to the start and end of the repeat year range
    start_index = np.argmin(np.abs(list1 - repeat_year_start))
    end_index = np.argmin(np.abs(list1 - repeat_year_end))
    print(f"indices found: {list1[start_index]}, {list1[end_index]}")
    if start_index > 0 and end_index + 1 < len(list1):
        print(f"CHECK dates prior to start: {list1[start_index-1]}, {list1[end_index+1]}")

    # Calculate the time chunk and how many repetitions are needed
    time_chunk = repeat_year_end - repeat_year_start
    backward_chunks = int((repeat_year_start - start_year) / time_chunk)
    date_range = list1[start_index:end_index+1]
    data_range = list2[start_index:end_index+1]

    final_dates = list1.tolist(); final_data = list2.tolist()
    for x in range(1, backward_chunks + 1):
        date_segment = date_range - x * time_chunk
        mask = date_segment < final_dates[0]  # Ensure no overlap with existing data
        final_dates = list(date_segment[mask]) + final_dates
        final_data = list(data_range[mask]) + final_data

    return final_dates, final_data


def ogre_accumulation_rate(sn, step):
    """OGRE GNSS-IR accumulation as an ice-equivalent rate [m i.e. yr⁻¹] on a regular decimal-year grid (spacing `step`)"""
    densities, depths = sumup_density_profile()
    averages, datetimes, *_ = read_gnssir_single(sn, gnssir_path(sn), scale_averages=False)
    if sn == '8438':
        averages = averages[0:-2]; datetimes = datetimes[0:-2]

    net_acc = [0.0]; stake_depth = []
    for j in range(0, len(averages) - 1):
        stake_depth.append(POLE_LENGTH - averages[j+1])
        net_acc.append(averages[j] - averages[j + 1] + net_acc[j])
    stake_depth.append(POLE_LENGTH - averages[-1])

    # CONVERT NET ACCUMULATION to m. ice eq. using density @ stake depth
    stake_depth_densities = get_corresponding_density(depths, densities, stake_depth)
    masses = [0.0]
    for a in range(1, len(net_acc)):
        masses.append(stake_depth_densities[a] * (net_acc[a] - net_acc[a-1]) / 917 + masses[a-1])

    dates = np.array([year_fraction(x) for x in datetimes])
    ice_accumulation_rate = np.diff(masses) / np.diff(dates)  # m/year
    midpoint_dates = 0.5 * (dates[:-1] + dates[1:])
    gnss_times = np.arange(midpoint_dates[0], midpoint_dates[-1], step)
    return gnss_times, np.interp(gnss_times, midpoint_dates, ice_accumulation_rate)


def generate_forcings(ogre_location, output_folder, forcing='merra', station=None, correction=0.0, timestep='daily',
                      reanalysis_dir=None):
    """
    Write CFM forcing .csv files (FORCING_FILES) to output_folder, from 1450 to the end of the MERRA-2 record.
    ogre_location: [lat, lon] to interpolate MERRA-2 to
    forcing: 'merra' (MERRA-2 only) or 'ogre' (accumulation replaced by OGRE GNSS-IR accumulation; needs station)
    correction: MERRA-2 P-E bias as a fraction; P-E is divided by (1 + correction). E.g. -0.089 if MERRA-2 is 8.9% low.
    """
    reanalysis_dir = reanalysis_dir or CFM_FORCING_DIR
    lat, lon = ogre_location
    merra_files = sorted(glob.glob(os.path.join(reanalysis_dir, '*.nc4')))
    if not merra_files:
        raise FileNotFoundError(f"No MERRA-2 .nc4 files in {reanalysis_dir}. Set OGRE_CFM_FORCING_DIR.")
    os.makedirs(output_folder, exist_ok=True)

    # PRECSNO/EVAP/PRECTOT [kg m⁻² s⁻¹] -> m i.e. yr⁻¹ (* 31536000 s / 917 kg m⁻³), TSH [K]
    times = []; rains = []; bdot = []; surface_temp = []; evaps = []
    for file in merra_files:
        with xr.open_dataset(file) as ds:
            if ds.sizes.get('time', 1) != 1:
                raise ValueError(f"{file} has {ds.sizes['time']} time steps; forcing files must hold one daily mean each.")
            times.append(year_fraction(pd.Timestamp(ds['time'].values[0]).to_pydatetime()))

            point = dict(lat=lat, lon=lon, method='linear')
            point_snow = ds['PRECSNO'].interp(**point).values[0] * 31536000 / 917
            point_evap = ds['EVAP'].interp(**point).values[0] * 31536000 / 917
            point_precip = ds['PRECTOT'].interp(**point).values[0] * 31536000 / 917
            bdot.append((point_snow - point_evap) / (1 + correction))  # METERS ICE EQUIVALENT PER YEAR, corrected based on Pickell et al 2025
            evaps.append(point_evap)
            rains.append(point_precip - point_snow)
            surface_temp.append(float(ds['TSH'].interp(**point).values[0]))

    # EXTEND DATA TO 1450
    if timestep == "daily":
        step = 1/365.25
    elif timestep == "monthly":
        step = 1/12
    else:
        raise ValueError("timestep must be 'daily' or 'monthly'")
    daily_times = np.arange(1450.0, times[-1], step)

    def extend_and_interpolate(values):
        extended_times, extended_values = extend_data(times, values, daily_times[0])
        return np.interp(daily_times, extended_times, extended_values)

    final_temps = extend_and_interpolate(surface_temp)
    final_bdot = extend_and_interpolate(bdot)
    final_rains = extend_and_interpolate(rains)
    final_subl = extend_and_interpolate(evaps)

    if forcing == 'ogre':
        if station is None:
            raise ValueError("forcing='ogre' needs a station")
        print("-> accumulation replaced by OGRE GNSS-IR accumulation")
        gnss_times, gnss_rate = ogre_accumulation_rate(station, step)
        mask = (daily_times >= gnss_times[0]) & (daily_times <= gnss_times[-1])
        final_bdot[mask] = np.interp(daily_times[mask], gnss_times, gnss_rate)
    elif forcing == 'merra':
        print("-> MERRA-2 reanalysis forcing")
    else:
        raise ValueError("forcing must be 'ogre' or 'merra'")

    # one row of decimal-year times, one row of values
    for key, values in [('temps', final_temps), ('bdot', final_bdot), ('rain', final_rains), ('subl', final_subl)]:
        with open(os.path.join(output_folder, FORCING_FILES[key]), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(daily_times.tolist())
            writer.writerow(values.tolist())


def station_merra_position(sn):
    """[lat, lon] for the MERRA-2 interpolation at a station: OGRE longitude clamped to the MERRA-2 columns at 72.5 N"""
    _, _, _, longitudes = read_ppp(ppp_csv_path(sn))
    return [MERRA_LAT, float(np.clip(float(longitudes[0]), *MERRA_LON_RANGE))]


########## RUN SETS ##########
def plan_runs(run_set, stations=None, models=None, densities=None):
    """
    list of (station, folder name, CFM settings) for a run set. Defaults are taken from the scripts that read the runs,
    so the folder names match what they expect.
    """
    jobs = []
    if run_set == 'physics':
        if stations is None and models is None:
            # FIG3_MODEL at every Figure 3 station (plotVerticals), other models at the averaging station (compare_firn_models)
            plan = {FIG3_MODEL: FIG3_STATIONS}
            for m in PHYSICS_MODELS:
                plan.setdefault(m, [FIG3_AVERAGE_CFM_STATION])
        else:
            plan = {m: stations or [FIG3_AVERAGE_CFM_STATION] for m in (models or PHYSICS_MODELS)}
        for model, sns in plan.items():
            jobs += [(sn, model, {"physRho": model}) for sn in sns]

    elif run_set == 'conductivity':
        conductivities = models or [m.split('MaxSP_', 1)[1] for m in CONDUCTIVITY_MODELS]
        for sn in stations or [FIG3_AVERAGE_CFM_STATION]:
            jobs += [(sn, f"MaxSP_{c}", {"physRho": "MaxSP", "conductivity": c}) for c in conductivities]

    elif run_set == 'density':
        for sn in stations or STACKED_STATIONS:
            for model in models or [STACKED_MODEL]:
                jobs += [(sn, f"{model}_{d}", {"physRho": model, "rhos0": float(d)}) for d in densities or STACKED_DENSITIES]
    else:
        raise ValueError(f"unknown run set {run_set}")
    return jobs


def run_set(name, stations=None, models=None, densities=None, forcing=None, correct=True, skip_forcings=False, dry_run=False):
    """generate forcings (once per station) and run CFM for every job in the run set"""
    out_root = CFM_NO_OGRE_DIR if name == 'density' else CFM_DIR
    forcing = forcing or ('merra' if name == 'density' else 'ogre')
    jobs = plan_runs(name, stations, models, densities)
    print(f"{name}: {len(jobs)} runs, {forcing} forcing, MERRA-2 correction {'on' if correct else 'off'}, output {out_root}")

    if not os.path.isfile(CFM_INIT_DENSITY):
        raise FileNotFoundError(f"Initial firn profile not found at {CFM_INIT_DENSITY}. Set OGRE_CFM_INIT_DENSITY.")

    done_forcings = set()
    for sn, folder, settings in jobs:
        station_folder = os.path.join(out_root, sn)
        if sn not in done_forcings and not skip_forcings:
            print(f"Generating forcings for station {sn}")
            generate_forcings(station_merra_position(sn), station_folder, forcing=forcing, station=sn,
                              correction=STATION_PERCENT_DECIMAL[sn] if correct else 0.0)
        done_forcings.add(sn)

        run_folder = os.path.join(station_folder, folder)
        result_folder = os.path.join(run_folder, 'output')
        config = dict(OGRE_RUN_SETTINGS, **settings, InputFileFolder=_dir(station_folder), resultsFolder=_dir(result_folder),
                      initfirnFile=os.path.abspath(CFM_INIT_DENSITY))
        config_path = os.path.join(run_folder, CONFIG_FILE)
        generate_json(config_path, config)
        print(f"{sn} {folder}: config {config_path}")
        if not dry_run:
            run_cfm(config_path, result_folder)


######### SENSITIVITY TEST ############
def camp_firn_velocity(cfm_layers_file):
    """modeled firn velocity [m/yr] below the Camp (8438) pole base, with the June 2024 mast raise added back to the pole depth"""
    _, pole_dates, poles_depths = get_pole_column(gnssir_path('8438'), ppp_csv_path('8438'), '8438')
    pole_dates = np.array(pole_dates); poles_depths = np.array(poles_depths)
    poles_depths[pole_dates > datetime(2024, 6, 15)] += 0.9767500000002656  # mast raise

    bottom_depths = np.full(len(poles_depths), -1)  # -1 signifies bottom of the column -- assumes ice is at bottom of domain!!!
    firn_dates, firn_compactions, numeric_firn_dates = get_compaction(cfm_layers_file, poles_depths, pole_dates, bottom_depths)
    return np.gradient(np.cumsum(firn_compactions), numeric_firn_dates) * SECONDS_PER_YEAR, firn_dates


def sensitivity_test(models=('MaxSP',), densities=(300, 330), skip_forcings=False, dry_run=False):
    """Camp (8438) firn velocity for each model at each surface density; prints mean bias and std vs the first density"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    folder = CFM_SENSITIVITY_DIR
    if not skip_forcings:
        generate_forcings([MERRA_LAT, sum(MERRA_LON_RANGE) / 2], folder, forcing='merra')

    velocities = {}; dates = {}
    fig, ax = plt.subplots()
    for model in models:
        for density in densities:
            name = f"{model}_{density}"
            result_folder = os.path.join(folder, name, 'output')
            config_path = os.path.join(folder, name, CONFIG_FILE)
            generate_json(config_path, dict(physRho=model, rhos0=float(density), grid_output_res=0.01, stpsPerYear=365.0,
                                            InputFileFolder=_dir(folder), resultsFolder=_dir(result_folder),
                                            initfirnFile=os.path.abspath(CFM_INIT_DENSITY)))
            if dry_run:
                continue
            velocities[name], dates[name] = camp_firn_velocity(run_cfm(config_path, result_folder))
            ax.plot(dates[name], velocities[name], label=name)

    if dry_run:
        return
    print("\n--- MODEL PAIR BIAS & STD ---")
    for model in models:
        a = f"{model}_{densities[0]}"
        for density in densities[1:]:
            b = f"{model}_{density}"
            common = np.intersect1d(dates[a], dates[b])
            if len(common) == 0:
                print(f"{a} vs {b}: No overlapping dates"); continue
            va = np.interp(mdates.date2num(common), mdates.date2num(dates[a]), velocities[a])
            vb = np.interp(mdates.date2num(common), mdates.date2num(dates[b]), velocities[b])
            print(f"{b} - {a}: mean bias = {np.mean(vb - va):.4f}, std = {np.std(vb - va):.4f}")
    ax.set_ylabel("Firn velocity [m y⁻¹]")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split('SETUP')[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_set', choices=['physics', 'conductivity', 'density', 'sensitivity'])
    parser.add_argument('--stations', nargs='+', help="station IDs (default depends on run set)")
    parser.add_argument('--models', nargs='+', help="physRho models, or conductivity schemes for the conductivity set")
    parser.add_argument('--densities', nargs='+', type=float, help="surface densities [kg m⁻³] (density/sensitivity sets)")
    parser.add_argument('--forcing', choices=['ogre', 'merra'], help="default: ogre for physics/conductivity, merra for density")
    parser.add_argument('--no-correction', action='store_true', help="don't correct MERRA-2 P-E bias")
    parser.add_argument('--skip-forcings', action='store_true', help="reuse forcing .csv files already in the station folders")
    parser.add_argument('--dry-run', action='store_true', help="write forcings and configs, don't run CFM")
    args = parser.parse_args()

    if args.run_set == 'sensitivity':
        sensitivity_test(models=tuple(args.models or ['MaxSP']),
                         densities=tuple(int(d) if float(d).is_integer() else d for d in (args.densities or [300, 330])),
                         skip_forcings=args.skip_forcings, dry_run=args.dry_run)
    else:
        densities = [int(d) if float(d).is_integer() else d for d in args.densities] if args.densities else None
        run_set(args.run_set, args.stations, args.models, densities, args.forcing, not args.no_correction,
                args.skip_forcings, args.dry_run)
