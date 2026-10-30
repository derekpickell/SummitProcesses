"""
OGRE GNSS analysis: strain rates, SMB, vertical velocities, surface elevation.
Lives in <repo>/4JOG_Paper/. Velocity-product testing for Figure 2 is in testing_measures.py.

SETUP
-----
Required environment variable:
    REFL_CODE        gnssrefl processing directory (contains rinex/ and */results/)

Optional environment variables (defaults shown):
    OGRE_STATIC_DIR  <repo>/staticData
    OGRE_MERRA_DIR   <repo>/downloadScripts/download_merra        (MERRA-2 .nc4 used by plotSMB)
    OGRE_MEASURES_ANNUAL_DIR  ~/Desktop/MeasuresAnnualVelocities  (MEaSUREs annual vx/vy mosaics, v05)
    OGRE_MERRA2_DIR  ~/Desktop/MERRA_2                              (MERRA-2 .nc4 used by plot_merra_gnss)
    OGRE_CFM_DIR     ~/Desktop/Paper4_files/cfm_ogre_models         (Community Firn Model output, OGRE-forced)
    OGRE_CFM_NO_OGRE_DIR  ~/Desktop/Paper4_files/cfm_no_ogre_models_corrected_mass  (CFM output, MERRA-2 forced, by density; stacked_plot.py)
    OGRE_BEDMACHINE  ~/Desktop/BedMachineGreenland-v5_bed.tif

e.g.  export OGRE_CFM_DIR=/path/to/cfm_ogre_models
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so gnss_utils is found from 4JOG_Paper/
from gnss_utils import *  # read_ppp, read_ppp_ne, read_gnssir_single, get_elevation, get_pole_column, derek_colors

import glob
import calendar
import warnings
from functools import lru_cache
from datetime import datetime, timedelta
from collections import defaultdict
# from itertools import combinations  # needed for the all-combinations option in generate_station_triplets

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd
import statsmodels.api as sm
from pyproj import Transformer
from scipy.stats import linregress, t, pearsonr, theilslopes
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, uniform_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

########### PATHS ###########
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 4JOG_Paper/
REPO_DIR = os.path.dirname(BASE_DIR)                   # repo root (staticData/, downloadScripts/, gnss_utils.py)
STATIC_DIR = os.environ.get('OGRE_STATIC_DIR', os.path.join(REPO_DIR, 'staticData'))
MERRA_DAILY_DIR = os.environ.get('OGRE_MERRA_DIR', os.path.join(REPO_DIR, 'downloadScripts', 'download_merra'))
MEASURES_ANNUAL_DIR = os.environ.get('OGRE_MEASURES_ANNUAL_DIR', os.path.expanduser('~/Desktop/Paper4_files/MeasuresAnnualVelocities'))
MERRA2_DIR = os.environ.get('OGRE_MERRA2_DIR', os.path.expanduser('~/Desktop/Paper4_files/MERRA_2'))
CFM_DIR = os.environ.get('OGRE_CFM_DIR', os.path.expanduser('~/Desktop/Paper4_files/cfm_ogre_models'))
CFM_NO_OGRE_DIR = os.environ.get('OGRE_CFM_NO_OGRE_DIR', os.path.expanduser('~/Desktop/Paper4_files/cfm_no_ogre_models_corrected_mass'))
BEDMACHINE_TIF = os.environ.get('OGRE_BEDMACHINE', os.path.expanduser('~/Desktop/Paper4_files/BedMachineGreenland-v5_bed.tif'))

def _refl(*parts):
    """path under $REFL_CODE"""
    if 'REFL_CODE' not in os.environ:
        raise EnvironmentError("Set the REFL_CODE environment variable to your gnssrefl processing directory.")
    return os.path.join(os.environ['REFL_CODE'], *parts)

def ppp_sum_path(sn): return _refl('rinex', sn, '*', '*', '*.sum')  # horizontal position (northing, easting)
def ppp_csv_path(sn): return _refl('rinex', sn, '*', '*', '*.csv')  # vertical position
def gnssir_path(sn):  return _refl('*', 'results', sn, '*.txt')      # reflectometry

def cfm_no_ogre_file(sn, model_name, density):
    return os.path.join(CFM_NO_OGRE_DIR, sn, f"{model_name}_{density}", 'output', 'output_firn_layers.csv')

def cfm_file(sn, model_name):
    return os.path.join(CFM_DIR, sn, model_name, 'output', 'output_firn_layers.csv')

########### CONSTANTS ###########
cm = 1/2.54  # centimeters in inches
SECONDS_PER_YEAR = 31556952  # julian year

coordinates = [[72.64335806,-38.79178359,2.316], [72.64255883,-37.75334045,.635], [72.64344929,-38.53221852,1.85], [72.61697955,-38.54465462,1.86], [72.64270142,-38.01298833,1.001246401], [72.64297183,-38.27254235,1.489398398],[72.58555047,-38.55857554,1.935895867], [72.58582269,-38.29861095,1.7263743], [72.58515457,-38.03955263,1.096242753], [72.58510792,-38.81848999,2.464294297], [72.58512293,-37.77992538,0.626827623],[72.57794736,-38.43807224,1.823775282]]
stationNum = ["0015", "0005", "0012", "0009", "0008", "8438", "0013", "0011", "0014", "0010", "0017", "0007", "7313"]# ["879s", "879c", "879n", "8438"]#["0015", "0005", "0012", "0009", "0008", "8438", "0013", "0011", "0014", "0010", "0017", "0007"]#, "7313"]
stationNames = ['01', '02', '03', '04', '05', 'Camp', '07', '08', '09', '10', '11', '12','13'] # ["South", "Central", "North", "Bamboo"]#
# stationNum = ["0012", "0009", "0008", "8438", "879s", "879c", "879n"]# ["879s", "879c", "879n", "8438"]#["0015", "0005", "0012", "0009", "0008", "8438", "0013", "0011", "0014", "0010", "0017", "0007"]#, "7313"]
# stationNames = ['03', '04', '05', 'Camp', "South", "Central", "North"] # ["South", "Central", "North", "Bamboo"]#

stationColors = ['#8000ff', '#5247fc', '#2489f5', '#0ac0e8', '#3ae8d6', '#68fcc1', '#96fca7', '#c4e88a', '#f4c069', '#ff8947', '#ff4724', '#ff0000', '#ff004c', '#ff00ac' ]
bbox = [[72.5, -38.9], [72.7, -37.6]] # for merra: -38.9,72.5,-37.6,72.7

station_slopes = {"0005": 0.0016, "0015": 0.0016, "0012": 0.00136, "0008": 0.00136, "0009": 0.00136, "8438": 0.00125, 
                  "0013": 0.00115, "0011": 0.00115, "0010": 0.0007, "0014": 0.0007, "0007": 0.0007, "0017": 0.0007, 
                 "7313": 0.0005, "0000": 0.00} 

station_mean_yr_velocity = {"0005": 2.25, "0015": 2.4, "0012": 1.92, "0008": 1.8, "0009": 1.87, "8438": 1.7, 
                  "0013": 1.7, "0011": 1.4, "0010": 1.0, "0014": 1.0, "0007": 0.62, "0017": 0.53, 
                 "7313": 0.06, "0000": 0.0} # FOR CORRRECTION OF DOWNHILL ADVECTION

mpl.rcParams.update({
    'axes.labelsize': 12, 'axes.labelweight': 'light',
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'font.family': ['Avenir', 'sans-serif'],  # falls back if Avenir isn't installed
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.handlelength": 1.2,
    "legend.edgecolor": "0.8",
})

########### SHARED HELPERS ###########
def _style_ticks(axes, size):
    for ax in axes:
        ax.tick_params(axis='both', labelsize=size)
        ax.yaxis.get_offset_text().set_fontsize(size)

def _date_axis(ax, minticks, maxticks):
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

def _seconds_since_start(datetimes):
    return np.array([(dt - datetimes[0]).total_seconds() for dt in datetimes])

def _cumulative_from_zero(values):
    c = np.cumsum(values)
    return c - c[0]

def _net_accumulation(averages):
    """net_acc[k] = averages[0] - averages[k] (reflector height drops as snow accumulates)"""
    return averages[0] - np.asarray(averages)

@lru_cache(maxsize=None)
def _open_raster(path):
    return rioxarray.open_rasterio(path, masked=True).squeeze()  # drop band dim if present

def _fit_velocity(times, eastings, northings):
    """Two-pass least squares horizontal velocity with 3-sigma outlier rejection.
    Returns vel_e, vel_n, stderr_vel_e, stderr_vel_n (m/s), and horizontal RMS residual (m)."""
    # First-pass fit
    A = np.vstack([times, np.ones_like(times)]).T
    vel_e, intercept_e = np.linalg.lstsq(A, eastings, rcond=None)[0]
    vel_n, intercept_n = np.linalg.lstsq(A, northings, rcond=None)[0]

    # 3-sigma outlier rejection on combined horizontal residuals
    total_resid = np.sqrt((eastings - (vel_e * times + intercept_e))**2 + (northings - (vel_n * times + intercept_n))**2)
    mask = total_resid < (np.mean(total_resid) + 3 * np.std(total_resid))
    times_clean, eastings_clean, northings_clean, A_clean = times[mask], eastings[mask], northings[mask], A[mask]

    # Refit with filtered data (keep the refit intercepts)
    vel_e, intercept_e = np.linalg.lstsq(A_clean, eastings_clean, rcond=None)[0]
    vel_n, intercept_n = np.linalg.lstsq(A_clean, northings_clean, rcond=None)[0]

    # Residuals and standard errors
    resid_e = eastings_clean - (vel_e * times_clean + intercept_e)
    resid_n = northings_clean - (vel_n * times_clean + intercept_n)
    AtA_inv = np.linalg.inv(A_clean.T @ A_clean)
    dof = len(times_clean) - 2
    se_e = np.sqrt(np.sum(resid_e**2) / dof * AtA_inv[0, 0])
    se_n = np.sqrt(np.sum(resid_n**2) / dof * AtA_inv[0, 0])
    rms_h = np.sqrt(np.mean(resid_e**2 + resid_n**2))
    return vel_e, vel_n, se_e, se_n, rms_h

def weighted_least_squares(x, values, errors):
    """WLS fit of values vs x with weights 1/errors². Returns slope p-value, and line (x_vals, y_vals)."""
    x = np.asarray(x, dtype=float)
    results = sm.WLS(np.asarray(values), sm.add_constant(x), weights=1 / np.asarray(errors)**2).fit()
    p_value = 2 * t.sf(np.abs(results.params[1] / results.bse[1]), len(values) - 2)
    x_vals = np.linspace(x.min(), x.max(), 100)
    return p_value, x_vals, results.params[0] + results.params[1] * x_vals

def load_reflector_heights(sn):
    """GNSS-IR daily reflector heights [m] (8438 trimmed after the mast raise)"""
    averages, datetimes, *_ = read_gnssir_single(sn, gnssir_path(sn), scale_averages=False)
    if sn == '8438':
        averages = averages[0:-52]; datetimes = datetimes[0:-52]
    return datetimes, averages

def load_surface_elevation(sn, print_statement=False):
    """surface elevation [m] (PPP antenna height - reflector height), advection corrected, zeroed at the start"""
    elevations, datetimes, *_ = get_elevation(gnssir_path(sn), ppp_csv_path(sn), sn)
    elevations = remove_advection(sn, datetimes, elevations, print_statement=print_statement)
    if sn == '8438':
        elevations = elevations[:-49]; datetimes = datetimes[:-49]
    return datetimes, [x - elevations[0] for x in elevations]

def _correct_8438_mast_raise(datetimes, vertical_positions):
    """Remove the offset from raising the 8438 mast in June 2024, then drop the last 52 epochs."""
    cutoff_before = datetime(2024, 6, 13)
    cutoff_after = datetime(2024, 6, 14)
    datetimes = np.array(datetimes); vertical_positions = np.array(vertical_positions)

    # Split the data: segment A (before shutdown), segment B (after mast raised)
    mask_before = datetimes < cutoff_before
    mask_after = datetimes > cutoff_after

    # Use the last few days before and first few days after
    window_days = 14
    mask_near_before = (datetimes > (cutoff_before - timedelta(days=window_days))) & mask_before
    mask_near_after = (datetimes < (cutoff_after + timedelta(days=window_days))) & mask_after
    offset = np.median(vertical_positions[mask_near_after]) - np.median(vertical_positions[mask_near_before])
    print("-> stake raise offset: ", offset)

    datetimes = np.concatenate([datetimes[mask_before], datetimes[mask_after]])
    vertical_positions = np.concatenate([vertical_positions[mask_before], vertical_positions[mask_after] - offset])
    return datetimes[:-52], vertical_positions[:-52]

def _merra_point_pe(files, lat, lon):
    """Daily MERRA-2 P-E [mm w.e. d⁻¹] at a point, sorted by date. Unreadable files are skipped."""
    pe = []; dates = []
    for filename in files:
        try:
            ds = xr.open_dataset(filename, engine='netcdf4')
        except Exception:
            # print(f"Failed to open {filename}. Skipping")
            continue

        # extract location-dependent variables
        point_precip = ds['PRECTOT'][0].interp(lat=lat, lon=lon)
        point_evap = ds['EVAP'][0].interp(lat=lat, lon=lon)
        pe.append(float(np.mean(point_precip) * 86400 - np.mean(point_evap) * 86400))
        dates.append(pd.to_datetime(point_precip['time'].values).to_pydatetime())
        # dates.append(datetime.strptime(filename[-19:-11], '%Y%m%d'))

    dates, pe = zip(*sorted(zip(dates, pe)))
    return pd.to_datetime(dates), np.array(pe)

########### Strain Rate ###########
def remove_outliers(x, y):
    # Calculate the IQR for both x and y
    Q1_x, Q3_x = np.percentile(x, [25, 75])
    Q1_y, Q3_y = np.percentile(y, [25, 75])
    IQR_x = Q3_x - Q1_x
    IQR_y = Q3_y - Q1_y
    
    # Create a mask that filters out outliers (1.5 IQR bounds)
    mask = (x >= Q1_x - 1.5 * IQR_x) & (x <= Q3_x + 1.5 * IQR_x) & (y >= Q1_y - 1.5 * IQR_y) & (y <= Q3_y + 1.5 * IQR_y)
    return x[mask], y[mask]

def generate_station_triplets(station_list, line1_ids, line2_ids):
    """
    Generate valid triangle triplets from a list of GPS station IDs,
    filtering out those with more than 2 from either linear group,
    and return them sorted west to east by centroid easting.

    Args:
        station_list (list of str): All station IDs.
        line1_ids (set of str): Stations in horizontal 'line 1' that shall not form a triangle
        line2_ids (set of str): Stations in horizontal line 2 that shall not form a triangle

    Returns:
        list of list: Valid triplets sorted west to east.
    """
    station_coords = {}
    for station in station_list:
        _, northing, easting = read_ppp_ne(ppp_sum_path(station))
        station_coords[station] = (easting[0], northing[0])

    # Get all 3-station combos
    # all_triplets = list(combinations(station_list, 3))
    all_triplets = list([("0015", "0005", "0008"), ("0015", "0008", "0012"), ("0012", "0008", "0011"), ("0012", "0011", "0013"), ("0013", "0011", "0010"), ("0013", "0010", "0014"), ("0014", "0010", "0007"), ("0014", "0007", "0017"), ("0015", "0005", "0009"), ("0009", "0013", "0011"), ("0015", "0005", "0012"), ("0008", "0005", "0012"), ("0008", "0013", "0012"), ("0008", "0013", "0011"), 
                         ("0017", "0014", "0010"), ("0017", "0010", "0007"), ("0014", "0013", "0011"), ("0014", "0011", "0010"), ("8438", "0008", "0011"), ("8438", "0008", "0015"), ("8438", "0005", "0008"), ("8438", "0009", "0012"), ("8438", "0011", "0013"), ("8438", "0011", "0014"), ("7313", "0007", "0017"),
                         
                         ])
                        #  ("8438", "0010", "0014"), ("8438", "0011", "0010")])
    excluded = [{"0008", "0009", "0012"}, {"0009", "0011", "0015"}, {"0005", "0009", "0013"},
                {"0010", "0009", "0015"}, {"8438", "0007", "0013"}, {"0009", "0010", "0007"}] #or "0013" in set(triplet)

    valid_triplets = []; centroids = []; eastings_all = []
    for triplet in all_triplets:
        if set(triplet) in excluded:
            continue

        line1_count = sum(1 for s in triplet if s in line1_ids)
        line2_count = sum(1 for s in triplet if s in line2_ids)
        if line1_count <= 2 and line2_count <= 2:
            valid_triplets.append(list(triplet))
            centroid = (np.mean([station_coords[s][0] for s in triplet]), np.mean([station_coords[s][1] for s in triplet]))
            centroids.append(centroid)
            eastings_all.append(centroid[0])

    # Sort triplets by centroid easting (west to east)
    sorted_triplets = [t for _, t in sorted(zip(centroids, valid_triplets), key=lambda x: x[0][0])]

    return sorted_triplets, sorted(eastings_all)

def get_ogre_velocities(sn):
    datetimes, northings, eastings = read_ppp_ne(ppp_sum_path(sn))
    eastings = np.array(eastings); northings = np.array(northings)
    vel_e, vel_n, se_e, se_n, rms_h = _fit_velocity(_seconds_since_start(datetimes), eastings, northings)
    return eastings[0], northings[0], vel_e, vel_n, se_e, se_n, rms_h

def get_ogre_station_velocities(station_list=None):
    """{station: {'pos': (easting, northing), 'vel': (v_e, v_n), 'vel_err': (se_e, se_n), 'rms_h': rms}} in UTM 24N, m and m/s"""
    stations = {}
    for station in station_list or stationNum:
        e, n, v_e, v_n, se_e, se_n, rms_h = get_ogre_velocities(station)
        stations[station] = {'pos': (e, n), 'vel': (v_e, v_n), 'vel_err': (se_e, se_n), 'rms_h': rms_h}
    return stations

def get_biases_from_stations(magnitude, stations):
    """
    Given a raster dataset `magnitude` in EPSG:3413 (Arctic Polar Stereographic) and a dictionary of stations
    in EPSG:32624 (UTM Zone 24N), compute the bias between the station's velocity and the velocity
    from the raster at the transformed station's position.
    """
    bias_percentages = []; biases = []

    # transformer from UTM 24 N to Arctic Stereo
    transformer = Transformer.from_crs("EPSG:32624", "EPSG:3413", always_xy=True)

    for station, data in stations.items():
        easting, northing = data['pos']
        station_velocity_e, station_velocity_n = data['vel']

        # Transform the station's UTM coordinates to EPSG:3413 and extract raster velocity there
        x_3413, y_3413 = transformer.transform(easting, northing)
        sat_velocity = magnitude.sel(x=x_3413, y=y_3413, method="nearest").values.item()
        
        station_velocity_magnitude = np.sqrt(station_velocity_e**2 + station_velocity_n**2) * SECONDS_PER_YEAR
        bias = sat_velocity - station_velocity_magnitude
        # print(f"-> _percent (m/y) {bias}")
        bias_percentages.append(round(bias / station_velocity_magnitude * 100, 1))
        biases.append(bias)

    return bias_percentages, biases

def get_satellite_vectors(ds, stations):
    """
    Return satellite velocity vector (vx, vy) in UTM Zone 24 coordinates.

    Parameters:
        ds (xarray.Dataset): Satellite dataset in EPSG:3413, m/yr.
        stations (dict): station -> {'pos': (easting, northing)} in EPSG:32624.

    Returns:
        tuple list: [(vx_utm24, vy_utm24)] in m/s
    """
    to_3413 = Transformer.from_crs("EPSG:32624", "EPSG:3413", always_xy=True)
    to_utm24 = Transformer.from_crs("EPSG:3413", "EPSG:32624", always_xy=True)
    vx_ds = ds.vx.squeeze(); vy_ds = ds.vy.squeeze()

    satellite_vel_vectors = []
    for station, data in stations.items():
        # Transform station point to EPSG:3413
        x_3413, y_3413 = to_3413.transform(*data['pos'])

        # Get local velocity in EPSG:3413
        vx = vx_ds.sel(x=x_3413, y=y_3413, method='nearest').item() / SECONDS_PER_YEAR
        vy = vy_ds.sel(x=x_3413, y=y_3413, method='nearest').item() / SECONDS_PER_YEAR

        # Use finite difference to get the local direction of (1, 0) and (0, 1) in UTM
        delta = 1.0  # meter step
        x_dx, y_dx = to_utm24.transform(x_3413 + delta, y_3413)
        x_dy, y_dy = to_utm24.transform(x_3413, y_3413 + delta)
        x0, y0 = to_utm24.transform(x_3413, y_3413)

        # Basis vectors in UTM24 space
        ex = np.array([x_dx - x0, y_dx - y0])  # EPSG:3413 x-direction mapped to UTM
        ey = np.array([x_dy - x0, y_dy - y0])  # EPSG:3413 y-direction mapped to UTM

        satellite_vel_vectors.append(tuple(vx * ex + vy * ey))

    return tuple(satellite_vel_vectors)

def azimuth_difference(vel, vel_sat):
    """signed angle [degrees] from vel to vel_sat (2D vectors), wrapped to ±180°"""
    return (np.degrees(np.arctan2(vel_sat[1], vel_sat[0]) - np.arctan2(vel[1], vel[0])) + 180) % 360 - 180

def get_dem_utm():
    """Load and crop PGC Arctic DEM to UTM24N extent and transform to UTM coordinates."""
    
    # Define target UTM extent (you want this range in the plot)
    utm_minx, utm_maxx = 495000, 545000
    utm_miny, utm_maxy = 8050000, 8064000

    # Convert UTM bounding box to EPSG:3413 to match DEM CRS
    to_3413 = Transformer.from_crs("EPSG:32624", "EPSG:3413", always_xy=True)
    minx_3413, miny_3413 = to_3413.transform(utm_minx, utm_miny)
    maxx_3413, maxy_3413 = to_3413.transform(utm_maxx, utm_maxy)

    # Load and crop DEM
    dem = rioxarray.open_rasterio(os.path.join(STATIC_DIR, 'MEASURES', '22_43_1_1_2m_v4.1', '22_43_1_1_2m_v4.1_dem.tif'))
    dem = dem.rio.clip_box(minx=minx_3413, maxx=maxx_3413, miny=miny_3413, maxy=maxy_3413)
    dem = dem.to_dataset('band').rename({1: 'elevation'})

    # Extract coordinate grid and convert to UTM
    xx, yy = np.meshgrid(dem.x.values, dem.y.values)
    xx_utm, yy_utm = Transformer.from_crs("EPSG:3413", "EPSG:32624", always_xy=True).transform(xx, yy)
    return xx_utm, yy_utm, dem.elevation.values

def get_dem_from_geopackage():
    """arctic data center -> QGIS -> extract contour -> extract by expression $length > 350
    -> export as UTM 24 geopackage, geometry linestring"""
    gdf = gpd.read_file(os.path.join(STATIC_DIR, 'MEASURES', '1m_contours_utm24.gpkg'))

    x_list = []; y_list = []; elev_list = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        # Support both LineString and MultiLineString
        lines = [geom] if geom.geom_type == 'LineString' else (geom.geoms if geom.geom_type == 'MultiLineString' else [])
        for line in lines:
            xs, ys = line.xy
            x_list.extend(xs); y_list.extend(ys)
            elev_list.extend([row['ELEV']] * len(xs))

    return np.array(x_list), np.array(y_list), np.array(elev_list)

# Velocity products
GRIP_ID = "7313"  # far from the divide; excluded when tuning smoothing and averaging angular errors
VELOCITY_PRODUCTS = {
    # name: plot title. Annual MEaSUREs files are v05, Dec-Nov years (14 = Dec 2014 - Nov 2015)
    'promice':              'PROMICE Multi-year, 2016-2022',
    'measures_multiyear':   'MEaSUREs Multi-year, 1995-2016',
    'measures_annual_mean': 'MEaSUREs Annual Velocity Mosaic Average, 2014-2023',
    'measures_2020_2021':   'MEaSUREs Annual Velocity Mosaic, 2020-2021',
}

def _measures_annual_files(years):
    """(vx, vy) file pairs for MEaSUREs annual mosaics starting December of each 2-digit year."""
    name = os.path.join(MEASURES_ANNUAL_DIR, 'GL_vel_mosaic_Annual_01Dec{:02d}_30Nov{:02d}_{}_v05.0.tif')
    return [(name.format(y, y + 1, 'vx'), name.format(y, y + 1, 'vy')) for y in years]

def load_velocity_product(product, bbox_latlon=[[70, -42], [75, -34]]):
    """
    Load a satellite velocity product as an xarray Dataset (vx, vy in m/yr, EPSG:3413),
    coarsely cropped to bbox_latlon [[lat, lon], [lat, lon]].

    Datasets:
        MEaSUREs GRIMP .tif in annual or multiyear: https://nsidc.org/grimp (Arctic Stereo)
        PROMICE multiyear in .nc format (Arctic Stereo)
    """
    print(f"--> Using {VELOCITY_PRODUCTS[product]}")
    to_3413 = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    bbox_espg = [to_3413.transform(*coord[::-1]) for coord in bbox_latlon]  # lon,lat order
    crop = dict(x=slice(bbox_espg[0][0], bbox_espg[1][0]), y=slice(bbox_espg[1][1], bbox_espg[0][1]))

    if product == 'promice':
        ds = xr.open_dataset(os.path.join(STATIC_DIR, 'PROMICE', 'IV_20160128_20221222.nc')).rename({
            "land_ice_surface_easting_velocity": "vx", "land_ice_surface_northing_velocity": "vy"})
        vx = ds.vx.squeeze().sel(**crop) * 365.25  # PROMICE is in m/day
        vy = ds.vy.squeeze().sel(**crop) * 365.25
    elif product == 'measures_multiyear':
        vx = rioxarray.open_rasterio(os.path.join(STATIC_DIR, 'MEASURES', 'greenland_vel_mosaic250_vx_v1.tif')).squeeze().sel(**crop)
        vy = rioxarray.open_rasterio(os.path.join(STATIC_DIR, 'MEASURES', 'greenland_vel_mosaic250_vy_v1.tif')).squeeze().sel(**crop)
    else:
        # annual mosaics: average all years (ignoring NaNs), or the single 2020-2021 year
        years = range(14, 23) if product == 'measures_annual_mean' else [20]
        pairs = _measures_annual_files(years)
        vx = xr.concat([rioxarray.open_rasterio(f).squeeze().sel(**crop) for f, _ in pairs], dim='stack').mean(dim='stack', skipna=True)
        vy = xr.concat([rioxarray.open_rasterio(f).squeeze().sel(**crop) for _, f in pairs], dim='stack').mean(dim='stack', skipna=True)

    return xr.Dataset({'vx': (('y', 'x'), vx.values), 'vy': (('y', 'x'), vy.values)}, coords={'x': vx.x.values, 'y': vx.y.values})

def smooth_velocity(ds, pixels):
    """Uniform (boxcar) smoothing of vx, vy over `pixels` grid cells. pixels=None returns ds unchanged."""
    if not pixels:
        return ds
    # gaussian_filter(ds[c].values, sigma=pixels) # alternative smoothing kernel
    return xr.Dataset({c: (('y', 'x'), uniform_filter(ds[c].values, size=pixels, mode='nearest')) for c in ['vx', 'vy']},
                      coords={'x': ds.x.values, 'y': ds.y.values})

def velocity_magnitude(ds):
    return np.sqrt(ds.vx**2 + ds.vy**2)

# Strain
def _velocity_gradient(positions, velocities, weights=None):
    """
    Fit u = u0 + du/dx dx + du/dy dy (and likewise v) about the triangle centroid.
    With 3 stations the system is exactly determined, so weights only affect the covariance.

    Returns:
        (x0, y0), strain rate tensor [s⁻¹], parameter covariance for (u0, v0, exx, exy, eyx, eyy)
    """
    x0, y0 = np.mean(positions, axis=0)

    # Build design matrix and observation vector
    G = []; d = []
    for (x, y), (u, v) in zip(positions, velocities):
        dx = x - x0; dy = y - y0
        G.append([1, 0, dx, dy, 0, 0]); d.append(u)  # u equation
        G.append([0, 1, 0, 0, dx, dy]); d.append(v)  # v equation
    G = np.array(G); d = np.array(d)
    W = np.diag(weights) if weights is not None else np.eye(len(d))

    # Weighted least squares: m = (Gᵗ W G)⁻¹ Gᵗ W d
    # m, _, _, _ = np.linalg.lstsq(G, d, rcond=None)
    GTWG_inv = np.linalg.inv(G.T @ W @ G)
    _, _, exx, exy, eyx, eyy = GTWG_inv @ (G.T @ W @ d)

    # STRAIN RATE TENSOR
    strain_tensor = np.array([
        [exx, 0.5 * (exy + eyx)],
        [0.5 * (exy + eyx), eyy]
    ])
    return (x0, y0), strain_tensor, GTWG_inv

def compute_gnss_strain(N=3, use_bedmachine=True):
    """
    Strain rates and vertical ice velocity from GNSS station triangles (weighted least squares).
    N: Glen's flow law exponent. WARNING: ASSUMPTION HERE!!!
    use_bedmachine: ice thickness from BedMachine at each centroid, else the precomputed ice_thicknesses list

    Returns:
        stationTriplets, eastings_all (centroids, west to east), station_cache, results (dict of lists, one entry per triplet)
    """
    ice_thicknesses = [
    3088.1498615885416, 3120.227736979167,3137.3410167968755,3103.721830859375,3134.4795212239583,3092.798092057292,3114.7359582031254,3054.109449479167,3043.8567035156248,
    3097.6585938802077,3078.718253515625,3028.6251838541666,3050.314387369792,3073.07073359375,3004.7449020833333,2997.103259244792,2981.2069600260415,3040.1545733072912,2990.705424869792,
    3058.6393953124993,3019.8298570312495,3052.577865625,3128.27132734375,3126.859565625,3141.9331244791665
    ]
    
    line1 = {"0005", "0008", "0010", "0007", "0011"}
    line2 = {"0015", "0012", "0013", "0014", "0017", "8438"}
    stationTriplets, eastings_all = generate_station_triplets(stationNum, line1, line2)

    station_cache = {}
    results = defaultdict(list)
    for n, triplet in enumerate(stationTriplets):
        
        # ITERATE THROUGH EACH TRIPLET STATION (read + fit each station once)
        for sn in triplet:
            if sn not in station_cache:
                datetimes, northings, eastings = read_ppp_ne(ppp_sum_path(sn))
                elevations, _, _, _ = read_ppp(ppp_csv_path(sn))
                eastings = np.array(eastings); northings = np.array(northings)
                vel_e, vel_n, stderr_vel_e, stderr_vel_n, _ = _fit_velocity(_seconds_since_start(datetimes), eastings, northings)

                # STORE initial position and velocity
                station_cache[sn] = {
                    'pos': (eastings[0], northings[0]),
                    'vel': (vel_e, vel_n),
                    'vel_err': (stderr_vel_e, stderr_vel_n),
                    'ele': elevations[0]  # Just the first profile?
                }
        stations = [station_cache[sn] for sn in triplet]

        # SOLVE FOR VELOCITY GRADIENT (weights = 1/σ² for u and v of each station)
        weights = [w for s in stations for w in (1 / s['vel_err'][0]**2, 1 / s['vel_err'][1]**2)]
        (x0, y0), strain_tensor, cov_m = _velocity_gradient([s['pos'] for s in stations], [s['vel'] for s in stations], weights)
        ele0 = np.mean([s['ele'] for s in stations])

        # STRAIN uncertainty
        std_errors = np.sqrt(np.diag(cov_m))  # For u0, v0, exx, exy, eyx, eyy
        strain_std = std_errors[2:6] * SECONDS_PER_YEAR  # exx, exy, eyx, eyy
        shear_std = 0.5 * np.sqrt(strain_std[1]**2 + strain_std[2]**2)
        eig_std = np.sqrt(strain_std[0]**2 + strain_std[3]**2 + shear_std**2) #Assumes uncorrelated errors → max principal ~ conservative estimate

        # CONVERT to strain rate per year (y⁻¹)
        strain_tensor_per_year = strain_tensor * SECONDS_PER_YEAR 
        # eigvals, eigvecs = np.linalg.eig(strain_tensor_per_year)
        eigvals, eigvecs = np.linalg.eigh(strain_tensor_per_year)  # symmetric tensor; ascending order
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]          # [max, min]
        du_dx = strain_tensor_per_year[0, 0]  # e_xx
        dv_dy = strain_tensor_per_year[1, 1]  # e_yy

        # VERTICAL velocity
        thickness = bedmachine_corrected_elevation(BEDMACHINE_TIF, x0, y0, ele0) if use_bedmachine else ice_thicknesses[n]
        w = (du_dx+dv_dy) * (N+1)/(N+2)*thickness
        # w = (eigvals[0]+eigvals[1]) * (N+1)/(N+2)*thickness

        # SAVE
        results['centroid'].append((x0, y0))
        results['eigvals'].append(eigvals); results['eigvecs'].append(eigvecs)
        results['max_principal'].append(eigvals[0]); results['min_principal'].append(eigvals[1])
        results['principal_err'].append(eig_std)
        results['angle'].append(np.degrees(np.arctan2(eigvecs[:,0][1], eigvecs[:,0][0])))
        results['du_dx'].append(du_dx); results['dv_dy'].append(dv_dy); results['shear'].append(strain_tensor_per_year[0, 1])
        results['w'].append(-w)
        results['w_err'].append(eig_std * np.sqrt(2) * (N + 1) / (N + 2) * thickness)  # Uncertainty propagation to w

    print(f"-> N set to {N}. Using scaling n+1/(n+2) * (du/dx + dv/dy) * thickness assumed constant or calculated from BedMachine.")
    w_arr = np.array(results['w']); w_err_arr = np.array(results['w_err'])
    print(f"-> Vertical velocity: {w_arr.mean():.3f} ± {w_arr.std(ddof=1):.3f} m/yr "
          f"(mean ± 1σ spread across {len(w_arr)} triads); mean formal error {w_err_arr.mean():.3f} m/yr")
    print(f"-> Strain median uncertainty of principal strain rate: {np.mean(results['principal_err'])}")
    # print("Mean x strain: ", np.mean(results['du_dx']))
    # print("Mean y strain: ", np.mean(results['dv_dy']))
    # print("Mean shear: ", np.mean(results['shear']))

    return stationTriplets, eastings_all, station_cache, results

def compute_satellite_strain(ds, stationTriplets, station_cache, N=3, thickness=None):
    """
    Strain rates and vertical ice velocity from a satellite velocity field (vx, vy in m/yr, EPSG:3413),
    sampled at the GNSS station positions and solved on the same triangles.
    Triplets without satellite data are NaN, so results stay aligned with the GNSS triplets.
    thickness: constant ice thickness [m]; None uses BedMachine at each centroid.

    Returns:
        max_principals, min_principals, ws (arrays, one value per triplet)
    """
    to_3413 = Transformer.from_crs("EPSG:32624", "EPSG:3413", always_xy=True)
    max_principals = []; min_principals = []; ws = []
    for triplet in stationTriplets:
        positions = [station_cache[sn]['pos'] for sn in triplet]

        # Satellite velocity at each station, m/yr -> m/s
        velocities = []
        for x, y in positions:
            x_3413, y_3413 = to_3413.transform(x, y)
            velocities.append((ds.vx.sel(x=x_3413, y=y_3413, method="nearest").item() / SECONDS_PER_YEAR,
                               ds.vy.sel(x=x_3413, y=y_3413, method="nearest").item() / SECONDS_PER_YEAR))

        if not np.all(np.isfinite(velocities)):
            print(f"-> Skipping triplet {triplet}: no satellite velocity at one or more stations")
            max_principals.append(np.nan); min_principals.append(np.nan); ws.append(np.nan)
            continue

        (x0, y0), strain_tensor, _ = _velocity_gradient(positions, velocities)
        strain_tensor_per_year = strain_tensor * SECONDS_PER_YEAR
        # eigvals, _ = np.linalg.eig(strain_tensor_per_year)
        eigvals = np.sort(np.linalg.eigvalsh(strain_tensor_per_year))[::-1]

        # Estimate vertical velocity
        if thickness is None:
            ele0 = np.mean([station_cache[sn]['ele'] for sn in triplet])
            H = bedmachine_corrected_elevation(BEDMACHINE_TIF, x0, y0, ele0)
        else:
            H = thickness
        w = -np.trace(strain_tensor_per_year) * (N + 1) / (N + 2) * H

        max_principals.append(eigvals[0]); min_principals.append(eigvals[1]); ws.append(w)

    return np.array(max_principals), np.array(min_principals), np.array(ws)

########### FIGURE 2 ###########
FIG2_PRODUCT = 'measures_2020_2021'  # best agreement with OGRE, see testing_measures.compare_products()
FIG2_SMOOTHING_PX = 44               # uniform filter width minimizing spread of station biases (testing_measures.compare_products())
FIG2_STRAIN_SMOOTHING_PX = None      # satellite strain from the unsmoothed product; set to FIG2_SMOOTHING_PX to smooth

def get_strain(ax, ax1, ax2, ax3, ds_satellite=None):
    """map of principal strain axes (ax), max/min principal strain (ax1, ax2) and vertical velocity (ax3) vs easting.
    If ds_satellite is given, satellite-derived estimates on the same triangles are added to ax1-ax3."""
    stationTriplets, eastings_all, station_cache, gnss = compute_gnss_strain()

    # PLOTTING triangles, stations and principal strain axes
    scale = 5e7  # principal strain arrow scale
    for triplet, (x0, y0), eigvals, eigvecs in zip(stationTriplets, gnss['centroid'], gnss['eigvals'], gnss['eigvecs']):
        x_vals = [station_cache[sn]['pos'][0] for sn in triplet]
        y_vals = [station_cache[sn]['pos'][1] for sn in triplet]
        ax.plot(x_vals + x_vals[:1], y_vals + y_vals[:1], color='#787878', linewidth=.5)
        ax.scatter(x_vals, y_vals, color='black', s=8)

        # arrows point out for extension, in for compression
        for i, color in zip([0, 1], [derek_colors['red'], derek_colors['blue']]):
            magnitude = eigvals[i]
            dx, dy = eigvecs[:, i] * scale/2 * abs(magnitude)
            for tip in [(x0 + dx, y0 + dy), (x0 - dx, y0 - dy)]:
                posA, posB = ((x0, y0), tip) if magnitude >= 0 else (tip, (x0, y0))
                ax.add_patch(FancyArrowPatch(posA, posB, arrowstyle='-|>', color=color, linewidth=2, mutation_scale=5))

    # PLOT SCALE BAR
    strain_example = 6e-5
    length = scale/2 * strain_example
    ax.add_patch(FancyArrowPatch((506000+length, 8053000), (506000+length+length, 8053000), arrowstyle='-|>', color=derek_colors['red'], linewidth=2, mutation_scale=5))
    ax.add_patch(FancyArrowPatch((506000+length, 8053000), (506000, 8053000), arrowstyle='-|>', color=derek_colors['red'], linewidth=2, mutation_scale=5))
    ax.text(506000 + length + length + 200, 8053000, f"\u03F5\u0307 = 6e-5 y⁻¹", ha='left', va='center', fontsize=6)

    # GNSS estimates vs easting
    for a, key, err, color in [(ax1, 'max_principal', 'principal_err', 'red'),
                               (ax2, 'min_principal', 'principal_err', 'blue'),
                               (ax3, 'w', 'w_err', 'black')]:
        a.scatter(eastings_all, gnss[key], color=derek_colors[color])
        a.errorbar(eastings_all, gnss[key], yerr=gnss[err], fmt='o', color=derek_colors[color], markersize=4, capsize=2)
    mean_ws = np.mean(gnss['w'])
    ax3.axhline(y = mean_ws, linestyle='dotted', color='k', label=rf'$\overline{{z_{{\mathrm{{ice}}}}}}$ = {mean_ws:.2f} m yr$^{{-1}}$')

    def add_regression_line(ax, x, y):
        x, y = remove_outliers(np.array(x), np.array(y))
        slope, intercept, _, p_value, _ = linregress(x, y)
        print("-> P value: ", p_value)
        if p_value < 0.05:  # Statistically significant slope
            ax.plot(x, slope * x + intercept, color=derek_colors['purple'], label=f'Linear fit (p={p_value:.1e}, m={slope:.1e})')

    for a, key in [(ax1, 'max_principal'), (ax2, 'min_principal'), (ax3, 'w')]:
        add_regression_line(a, eastings_all, gnss[key])

    # SATELLITE estimates on the same triangles
    if ds_satellite is not None:
        sat_max, sat_min, sat_ws = compute_satellite_strain(ds_satellite, stationTriplets, station_cache)
        ax1.scatter(eastings_all, sat_max, color='black', alpha=0.3, label='Satellite-derived estimates')
        ax2.scatter(eastings_all, sat_min, color='black', alpha=0.3)
        ax3.scatter(eastings_all, sat_ws, color='black', alpha=0.3)
        differences = np.array(gnss['w']) - sat_ws
        print(f"-> Difference in vertical velocities (OGRE - satellite): {np.nanmean(differences):.3f} +/- {np.nanstd(differences):.3f} m/yr")

    for a in [ax1, ax2, ax3]:
        if a.get_legend_handles_labels()[0]:
            a.legend(loc='upper right', fontsize=6)

def get_measures_comparison(ax1, ds, smoothing_px=FIG2_SMOOTHING_PX):
    """
    Map of satellite velocity magnitude (ds: vx, vy in m/yr, EPSG:3413) with OGRE and satellite
    velocity vectors and percent bias at each station. ALL data ultimately plotted in UTM Zone 24N.
    """
    # GET OGRE VELOCITIES
    stations = get_ogre_station_velocities()
    for station, data in stations.items():
        (v_e, v_n), (se_e, se_n), rms_h = data['vel'], data['vel_err'], data['rms_h']
        speed = np.hypot(v_e, v_n)
        speed_se = np.sqrt((v_e * se_e)**2 + (v_n * se_n)**2) / speed
        print(f"Station {station} Velocity: {speed*SECONDS_PER_YEAR:.3f} ± {speed_se*SECONDS_PER_YEAR:.3f} m/yr, "
              f"horizontal RMS residual: {100*rms_h:.2f} cm")

    # BIASES before and after smoothing
    _, biases = get_biases_from_stations(velocity_magnitude(ds), stations)
    print(f"-> Mean bias across all stations, without smoothing: {np.mean(biases)}, {np.std(biases)}")
    ds_smoothed = smooth_velocity(ds, smoothing_px)
    magnitude = velocity_magnitude(ds_smoothed)
    bias_percentages, biases = get_biases_from_stations(magnitude, stations)
    pixel_size = abs(ds.x.values[1] - ds.x.values[0])
    print(f"-> Smoothing = {smoothing_px} pixels, pixel size = {pixel_size}, wavelength [m] = {(smoothing_px or 0)*pixel_size}")
    print(f"-> Mean bias across all stations, with smoothing: {np.mean(biases)}, {np.std(biases)}")

    # Now transform velocity map coordinates to UTM Zone 24N
    xx, yy = np.meshgrid(ds_smoothed.x.values, ds_smoothed.y.values)
    xx_utm, yy_utm = Transformer.from_crs("EPSG:3413", "EPSG:32624", always_xy=True).transform(xx, yy)

    # CROP MAP DATA (bounding box indices of mask, applied to all three arrays)
    xmin, xmax = 495000, 555000
    ymin, ymax = 8050000, 8064000
    rows, cols = np.where((xx_utm >= xmin) & (xx_utm <= xmax) & (yy_utm >= ymin) & (yy_utm <= ymax))
    rs = slice(rows.min(), rows.max()+1); cs = slice(cols.min(), cols.max()+1)

    # PLOT color mesh of SATELLITE magnitude
    im = ax1.pcolormesh(xx_utm[rs, cs], yy_utm[rs, cs], magnitude.values[rs, cs], cmap='Blues', shading='auto')
    cax = make_axes_locatable(ax1).append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('velocity [m y⁻¹]', fontsize=5)  

    # PLOT contours
    demx, demy, _ = get_dem_from_geopackage()
    ax1.scatter(demx, demy, c='gray', s=.001, alpha=.1)

    # PLOT velocity vectors and bias at each station
    vels = get_satellite_vectors(ds_smoothed, stations)
    vel_scale = 1e11  # adjust as needed for visualization
    angle_errors = []
    for i, (station, data) in enumerate(stations.items()):
        x, y = data['pos']
        # OGRE velocity
        dx, dy = data['vel'][0] * vel_scale, data['vel'][1] * vel_scale
        ax1.add_patch(FancyArrowPatch((x, y), (x + dx, y + dy), arrowstyle='->', color='black', linewidth=1, mutation_scale=6))

        # Satellite velocity
        dx_sat, dy_sat = vels[i][0] * vel_scale, vels[i][1] * vel_scale
        ax1.add_patch(FancyArrowPatch((x, y), (x + dx_sat, y + dy_sat),arrowstyle='->', color='black', linewidth=.5, linestyle='dotted', mutation_scale=6))

        ### CALCULATE AZIMUTH ANGULAR DIFFERENCE
        angle_diff = azimuth_difference(data['vel'], vels[i])
        print(f"-> {station} angular error (degrees): {angle_diff}")
        if station != GRIP_ID:
            angle_errors.append(angle_diff)

        # Stations
        ax1.scatter(x, y, color='black', s=8)
        label_x = x - (x * 1.005 - x) if i == len(stations)-1 else x * 1.00008
        t3 = ax1.annotate(f"{bias_percentages[i]}%", (label_x, y * 1.00008), fontsize=9)
        t3.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

    print(f"-> Total angular bias (excluding GRIP): {np.mean(angle_errors)} +/- {np.std(angle_errors)}")

    # Legend arrows
    dx = 1 / (3.154e7) * vel_scale  # 1 m/yr in m/s, scaled
    ax1.add_patch(FancyArrowPatch((537460 - 1000, 8051000 - 100), (537460 - 1000 + dx, 8051000 - 100),arrowstyle='->', color='black',linewidth=1, mutation_scale=6))
    ax1.add_patch(FancyArrowPatch((537460 - 1000, 8051000 - 100 + 1500),(537460 - 1000 + dx, 8051000 - 100 + 1500),arrowstyle='->',color='black',linewidth=.5,linestyle='dotted',mutation_scale=6))
    t0 = ax1.text(537460-1000, 8051000+500,  f"OGRE Velocity      1 m y⁻¹", ha='left', va='center', fontsize=6)
    t1 = ax1.text(537460-1000, 8051000+2000, f"Measures Velocity  1 m y⁻¹", ha='left', va='center', fontsize=6)
    for txt in [t0, t1]:
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])

def plotStrain(product=FIG2_PRODUCT, smoothing_px=FIG2_SMOOTHING_PX, strain_smoothing_px=FIG2_STRAIN_SMOOTHING_PX):
    ### PLOT
    # fig = plt.figure(figsize=(17.9*cm, 8.56*cm))
    fig = plt.figure(figsize=(11.5, 5.5))

    # Left Column
    gs1 = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs1[0])
    ax2 = fig.add_subplot(gs1[1])

    # Right Column
    gs2 = gridspec.GridSpec(3, 1)
    ax3 = fig.add_subplot(gs2[0])
    ax4 = fig.add_subplot(gs2[1])
    ax5 = fig.add_subplot(gs2[2])

    ds = load_velocity_product(product)
    get_measures_comparison(ax1, ds, smoothing_px)
    get_strain(ax2, ax3, ax4, ax5, ds_satellite=smooth_velocity(ds, strain_smoothing_px))

    # Axis settings
    ax1.set_xlim(495000, 551000)
    ax1.set_ylim(8050000, 8064000)
    ax2.set_xlim(505000, 551000)
    ax2.set_ylim(8052000, 8063000)
    ax3.set_ylim(.00005, .000075)
    ax4.set_ylim(-.00001, .00005)
    ax2.set_xlabel("Easting [m]", fontsize=6, fontweight='bold')
    ax1.set_ylabel("Northing [m]", fontsize=6, fontweight='bold')
    ax2.set_ylabel("Northing [m]", fontsize=6, fontweight='bold')
    ax1.set_aspect('equal')  
    ax2.set_aspect('equal')
    ax1.set_title(f"OGRE Horizontal Velocity vs {VELOCITY_PRODUCTS[product]}", fontsize=7)
    ax2.set_title("Principal Strain Rate Axes from Station Triangles", fontsize=7)

    ax3.set_ylabel("Maximum Principal \u03F5\u0307 [y⁻¹]", fontsize=6, fontweight='bold')
    ax4.set_ylabel("Minimum Principal \u03F5\u0307 [y⁻¹]", fontsize=6, fontweight='bold')
    ax5.set_ylabel("Vertical Velocity [m yr⁻¹]", fontsize=6, fontweight='bold')
    # ax5.set_ylabel("Principal \u03F5\u0307 Angle [°]", fontsize=6, fontweight='bold')
    ax5.set_xlabel("Easting [m]", fontsize=6, fontweight='bold')
    ax5.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax5.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))  # scientific notation
    ax5.xaxis.get_offset_text().set_fontsize(6)  # Match your tick label font size

    ax3.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
    ax4.tick_params(axis='x', bottom=False, top=False, labelbottom=False)

    _style_ticks([ax1, ax2, ax3, ax4, ax5], 7)
    fig.align_ylabels([ax1, ax2])
    fig.align_ylabels([ax3, ax4, ax5])

    for ax in [ax3, ax4, ax5]:
        ax.grid(True, alpha=.3)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gs1.tight_layout(fig, rect=[0.01, 0, 0.7, 1], h_pad=0.5)
        gs2.tight_layout(fig, rect=[0.72, 0, 1, 1], h_pad=0.5, w_pad=0.5)
        plt.show()

def bedmachine_corrected_elevation(tif_path, easting, northing, elevation):
    """
    Given a BedMachine GeoTIFF in EPSG:3413 and a UTM 24N (easting, northing, elevation),
    return ice thickness: elevation - bed value at that point (nearest pixel).
    """
    bed = _open_raster(tif_path)  # cached, so repeated calls don't re-open the file

    # Transformer: UTM zone 24N → EPSG:3413
    x3413, y3413 = Transformer.from_crs("EPSG:32624", bed.rio.crs, always_xy=True).transform(easting, northing)
    bed_val = bed.sel(x=x3413, y=y3413, method='nearest').item()

    # Handle nodata
    if np.isnan(bed_val):
        raise ValueError("No bed value found at this location (possibly outside raster bounds).")

    return elevation - bed_val
    
########### SMB ###########
def get_density_depth(ax):
    """take SUMUP dataset, find data near Summit (global bbox), and construct a mean density vs depth 
    curve for all data"""
    df = pd.read_csv(os.path.join(STATIC_DIR, "SUMup_2024_density_greenland.csv"))

    # Drop rows with missing critical data and convert to proper data types
    cols = ["latitude", "longitude", "midpoint", "density"]
    df = df.dropna(subset=cols)
    df[cols] = df[cols].apply(pd.to_numeric)

    # Filter by bounding box
    in_bbox = df[
        (df["latitude"] >= bbox[0][0]) & (df["latitude"] <= bbox[1][0]) &
        (df["longitude"] >= bbox[0][1]) & (df["longitude"] <= bbox[1][1])]

    if in_bbox.empty:
        print("No profiles found within bounding box.")
    else:
        # Plot profiles
        for profile_key, group in in_bbox.groupby("profile_key"):
            group_sorted = group.sort_values("midpoint")  # Ensure depth increases
            ax.plot(group_sorted["density"], group_sorted["midpoint"], color='black', alpha=.1, zorder=1)

        # Fit LOWESS smoothing to all data, sorted by depth
        sorted_idx = np.argsort(in_bbox["midpoint"].values)
        smoothed = lowess(endog=in_bbox["density"].values[sorted_idx], exog=in_bbox["midpoint"].values[sorted_idx], frac=0.2, return_sorted=True)

        # Plot smoothed LOESS curve
        ax.plot(smoothed[:, 1], smoothed[:, 0], color=derek_colors['red'], linewidth=2.5, label='Best Fit', zorder=3)
        ax.set_xlim(100, 550)
        ax.set_ylim(0, 2.5)
        ax.invert_yaxis() 

    return smoothed[:,1], smoothed[:,0]

def get_corresponding_density(depths, densities, stake_depths):
    """using depth vs density curve, extract closest value for OGRE stake depth"""
    depths = np.array(depths)
    densities = np.array(densities)
    return np.array([densities[np.abs(depths - d).argmin()] for d in stake_depths])

def calculate_interval_smb(plot_dates, plot_heights, days=365):
    rolling_annual_acc = []; rolling_dates = []
    for i in range(len(plot_dates)):
        start_date = plot_dates[i]
        end_date = start_date + pd.Timedelta(days=days)

        # Find index of end_date
        if end_date in plot_dates.values:
            j = plot_dates.get_loc(end_date)
            rolling_annual_acc.append((plot_heights[j] - plot_heights[i])*365/days)
            rolling_dates.append(start_date)

    return rolling_dates, rolling_annual_acc

def plotSMB():
    """plot accumulation for all stations using GNSS IR interpolated to daily estimates starting from 2022-08-20"""
    stationNum = ["0015", "0005", "0012", "0009", "0008", "8438", "0013", "0011", "0014", "0010", "0017", "0007", "7313"]

    ### PLOT
    fig = plt.figure(figsize=(10, 7))

    # Left Column: 2 rows, 2 columns (top row is split into 2 subplots sharing y-axis)
    gs1 = gridspec.GridSpec(2, 2, width_ratios=[4, 2.3]) 
    ax1a = fig.add_subplot(gs1[0, 0])
    ax1b = fig.add_subplot(gs1[0, 1], sharey=ax1a)
    ax2 = fig.add_subplot(gs1[1, :])  # Bottom row: single subplot spanning both columns

    # Right Column
    gs2 = gridspec.GridSpec(1, 1)
    ax3 = fig.add_subplot(gs2[0])
    ax_inset = inset_axes(ax3, width="40%", height="20%", loc='upper right',bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax3.transAxes, borderpad=2)

    stake_length = 3.048  # meters
    plot_start_date = pd.to_datetime("2022-08-30")
    days = 90  # interval for rolling SMB
    densities, depths = get_density_depth(ax1b)

    # iterate through each station
    all_masses = []; all_dates = []
    for i in range(len(stationNum)):
        print("Processing station: ", stationNum[i])
        datetimes, averages = load_reflector_heights(stationNum[i])

        averages = np.asarray(averages)
        net_acc = _net_accumulation(averages)
        stake_depth = stake_length - np.append(averages[1:], averages[-1])  # each epoch uses the next reflector height

        # INTERPOLATE TO DAILY
        start_date = datetimes[0]; end_date = datetimes[-1]
        full_date_range = [start_date + timedelta(days=d) for d in range((end_date - start_date).days + 1)]
        date_nums = [(date - start_date).days for date in datetimes]
        full_date_nums = [(date - start_date).days for date in full_date_range]
        interpolated_heights = np.interp(full_date_nums, date_nums, net_acc)
        interpolated_stake_depths = np.interp(full_date_nums, date_nums, stake_depth)

        # NOW START ON SAME DAY and MASK
        full_date_range = pd.to_datetime(full_date_range)
        mask = full_date_range >= plot_start_date
        plot_dates = full_date_range[mask]
        plot_heights = interpolated_heights[mask]
        interpolated_stake_depths = interpolated_stake_depths[mask]
        plot_heights = plot_heights - plot_heights[0]  # SET NET ACCUMULATION TO START AT ZERO
        
        # CONVERT NET ACCUMULATION to MMWE using density @ stake depth
        # e.g., .05 m * 300 kg/m^3 / 1000 kg/m^3 * 1000 mm / 1 m= 15 mm
        stake_depth_densities = get_corresponding_density(depths, densities, interpolated_stake_depths)
        masses = np.concatenate([[0.0], np.cumsum(stake_depth_densities[1:] * np.diff(plot_heights))])

        # GET INTERVAL SMB AVERAGE
        rolling_dates, rolling_masses = calculate_interval_smb(plot_dates, masses, days=days)

        # Plotting
        ax1a.plot(plot_dates, interpolated_stake_depths, zorder=3, label="Station " + stationNames[i], color=stationColors[i])
        ax3.plot(plot_dates, masses, label="Station " + stationNames[i], color=stationColors[i])
        ax_inset.plot(rolling_dates, rolling_masses, color=stationColors[i])

        # evaluate merra bias and drift (rmse)
        evaluate_merra(plot_dates, masses, stationNum[i])

        all_masses.append(masses); all_dates.append(plot_dates)

    # ADD DATA FROM from MERRA-2
    get_merra(ax3, ax_inset, ax2, plot_start_date, days=days)

    # ADD SLOPE ANALYSIS
    test_east_west_gradient(all_masses, all_dates, stationNum, ax2) 
    
    _date_axis(ax1a, 2, 4)
    _date_axis(ax_inset, 3, 8)
    _date_axis(ax3, 3, 8)
    ax_inset.tick_params(axis='both', labelsize=7)
    _style_ticks([ax1a, ax1b, ax2, ax3], 8)

    ax_inset.set_ylabel('90 Day SMB [mm w.e. y$^{-1}$]', fontsize=7, fontweight='heavy')
    ax1b.legend(loc='lower left', fontsize=8).set_zorder(10)
    ax3.legend(loc='upper left', fontsize=8, bbox_to_anchor=(0.055, .98), borderaxespad=0)
    ax3.set_ylim(-10, 700)
    ax1b.set_xlim(150, 420)
    ax2.set_ylim(240, 290)

    for ax in fig.get_axes():
        ax.grid(True, zorder=0, alpha=0.5)

    label_kw = dict(fontsize=8, fontweight='heavy', fontfamily='sans-serif')
    ax1a.set_ylabel("Pole Depth below Surface [m]", **label_kw)
    ax1b.tick_params(axis='y', labelleft=False)  # Hide labels, keep ticks
    ax1b.set_xlabel("Density [kg m$^{-3}$]", **label_kw)
    ax3.set_ylabel("Cumulative SMB [mm w.e.]", **label_kw)
    ax2.set_xlabel('Easting [m]', **label_kw)
    ax2.set_ylabel(r'Accumulation Rates [mm w.e. y$^{-1}$]', **label_kw)
    ax2.legend(loc='upper right', fontsize=8)

    panel_kw = dict(fontsize=12, fontweight='heavy', va='top', fontfamily='sans-serif')
    ax1a.text(0.02, 0.95, 'A', transform=ax1a.transAxes, **panel_kw)
    ax1b.text(0.02, 0.95, 'B', transform=ax1b.transAxes, **panel_kw)
    ax2.text(0.02, 0.95, 'D', transform=ax2.transAxes, **panel_kw)
    ax3.text(0.02, 0.979, 'C', transform=ax3.transAxes, **panel_kw)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gs1.tight_layout(fig, rect=[0.01, 0, 0.4, 1], h_pad=0.5)
        gs2.tight_layout(fig, rect=[0.4, 0, 1, 1], h_pad=0.5, w_pad=0.5)
        plt.show()
 
def get_merra(ax, ax_inset, ax2, plot_start_date = pd.to_datetime("2022-08-30"), days = 365):
    """cross sectional slope: is there a difference in precip/evap?"""
    files = sorted(glob.glob(os.path.join(MERRA_DAILY_DIR, '*.nc4')))
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32624", always_xy=True)

    # GRIP TO GISP -38.464245
    positions = [[72.58511415888755, -37.63], [72.58511152254049, -38.46] ]
    for i, position in enumerate(positions):
        print(f"Processing Position: {position}")
        dates, merra2 = _merra_point_pe(files, *position)

        # mask and get cumulative
        mask = (dates >= plot_start_date) & (dates <= pd.to_datetime("2024-06-15"))
        dates = dates[mask]
        cumulative_precip = _cumulative_from_zero(merra2[mask])

        # plot
        color = 'gray' if i == 0 else 'k'
        ax.plot(dates, cumulative_precip, label=f'Merra-2 at Longitude {position[1]}', lw=2, color=color, linestyle='dashed')

        # GET ANNUAL RATE AND PLOT IN FIGURE 2
        slope, _, _, _, _ = linregress(mdates.date2num(dates), cumulative_precip)  # days since epoch
        slope_annual = slope * 365.25
        print(f"MERRA-2 mean rate: {slope_annual:.3f} mm/year")
        x_utm, y_utm = to_utm.transform(position[1], position[0])
        ax2.scatter(x_utm, slope_annual, color=color, marker='x')

        # GET ROLLING RATES
        rolling_dates, rolling_masses = calculate_interval_smb(dates, cumulative_precip, days=days)
        ax_inset.plot(rolling_dates, rolling_masses, color=color, linestyle='dashed')

def evaluate_merra(ogre_dates, masses, stationNum):
    """Mean bias and rate bias of MERRA-2 P-E vs OGRE cumulative SMB at a station."""
    files = sorted(glob.glob(os.path.join(MERRA_DAILY_DIR, '*.nc4'))) # merra2_data_smb/*.nc4

    _, _, latitude, longitude = read_ppp(ppp_csv_path(stationNum))
    dates, merra2 = _merra_point_pe(files, float(latitude[0]), float(longitude[0]))

    # mask and get cumulative
    mask = (dates >= ogre_dates[0]) & (dates <= ogre_dates[-1])
    dates = dates[mask]
    cumulative_precip = _cumulative_from_zero(merra2[mask])

    def align_data(datetimes_1, datetimes_2, data_1, data_2, max_diff):
        ogre_dates = []; ogre_data = []; comp_data = []
        for i in range(len(datetimes_1)):
            index = min(range(len(datetimes_2)), key=lambda x: abs(datetimes_2[x] - datetimes_1[i]).total_seconds())
            if abs((datetimes_1[i] - datetimes_2[index]).total_seconds()) / 3600 <= max_diff:
                ogre_dates.append(datetimes_1[i])
                ogre_data.append(data_1[i])
                comp_data.append(data_2[index])
        return ogre_dates, ogre_data, comp_data
    
    ogre_date, ogre_mass, comp_data = align_data(ogre_dates, dates, masses, cumulative_precip, 24)
    comp_data = np.array(comp_data); ogre_mass = np.array(ogre_mass)

    # MEAN BIAS ASSESSMENT: model - observation
    residuals = comp_data - ogre_mass
    bias = np.mean(residuals)
    percent_bias = (bias / np.mean(ogre_mass)) * 100
    slope, _, _, p_value, _ = linregress(mdates.date2num(ogre_date), residuals)
    print(f"-> {stationNum} Mean bias (MERRA2 - OGRE): {bias:.2f}, {percent_bias:.04f}% [mm], slope: {slope:.04f} pvalue: {p_value:.04f}")

    # RATE ASSESSMENT 
    ogre_rate = np.diff(ogre_mass)
    comp_rate = np.diff(comp_data)
    rate_bias = np.mean(comp_rate - ogre_rate)
    percent_rate_bias = (rate_bias / np.mean(ogre_rate)) * 100
    print(f"-> {stationNum} RATE bias (MERRA2 - OGRE): {rate_bias:.2f}, {percent_rate_bias:.04f}% [mm/yr]")

def test_east_west_gradient(all_masses, all_dates, stations, ax):
    """
    Test for an east-west gradient in mass accumulation slope using weighted least squares.
    """
    slopes = []; stderrs = []; eastings = []
    for i, station in enumerate(stations):
        x = mdates.date2num(all_dates[i])  # days
        res = sm.OLS(np.array(all_masses[i]), sm.add_constant(x)).fit(cov_type='HAC', cov_kwds={'maxlags': 60})
        slope, stderr = res.params[1], res.bse[1]
        slopes.append(slope); stderrs.append(stderr)
        print(f"-> {station} slope [mm y⁻¹] {slope*365.25}, period (yrs): {(x[-1] - x[0])/365}")
        
        _, _, easting = read_ppp_ne(ppp_sum_path(station))
        eastings.append(easting[-1])
        print(f"-> {station} accumulation: {slope*365.25:.1f} ± {stderr*365.25:.1f} mm w.e./yr")
    eastings = np.array(eastings); slopes = np.array(slopes); stderrs = np.array(stderrs)

    # Weighted least squares with intercept
    results = sm.WLS(slopes, sm.add_constant(eastings), weights=1 / (stderrs ** 2)).fit()

    # Extract slope coefficient and stats
    slope_coef = results.params[1]
    slope_se = results.bse[1]
    t_stat = slope_coef / slope_se
    p_value = 2 * t.sf(np.abs(t_stat), len(slopes) - 2)

    print(f"Slope coefficient: {slope_coef:.4e}")
    print(f"Std err: {slope_se:.4e}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    # Multiply regression slope and intercept by 365.25 to convert model line to per year
    x_vals = np.linspace(eastings.min(), eastings.max(), 100)
    slope_year = results.params[1] * 365.25
    y_vals_year = results.params[0] * 365.25 + slope_year * x_vals

    # Plot with error bars and regression line
    for i in range(len(stations)): 
        ax.errorbar(eastings[i], slopes[i] * 365.25, yerr=stderrs[i] * 365.25, fmt='o', capsize=3, color=stationColors[i])
    
    ax.plot(x_vals, y_vals_year, 'k-', label=f'slope = {slope_year*1000:.2e} mm w.e. per km, p: {p_value:.3f} ')

########### FIGURE 3 ###########
FIG3_STATIONS = ["0005", "0012", "0009", "0008", "8438", "0011", "0010", "0017", "0007", "7313"]
FIG3_NAMES = ["02", "03", "04", "05", "Camp", "08", "10", "11", "12", "13"]
FIG3_COLORS = ['#5247fc', '#2489f5', '#0ac0e8', '#3ae8d6', '#68fcc1', '#c4e88a', '#ff8947', '#ff4724', '#ff0000', "#ff004c"]
FIG3_MODEL = 'GSFC2020'            # CFM firn model, see testing_verticals.compare_firn_models() #'Arthern2010T' #'KuipersMunneke2015' #'Ligtenberg2011'
FIG3_AVERAGE_CFM_STATION = "0009"  # CFM run (forcing) used for the site-wide average

def _velocity_settings(sn, window):
    """(window [days], position uncertainty [m]) for the Monte Carlo velocities. GRIP is noisier."""
    return (80, .03) if sn == GRIP_ID else (window, .01)

def load_vertical_positions(sn, year_conv=SECONDS_PER_YEAR, print_statement=True):
    """PPP vertical positions [m] corrected for downhill advection (and the 8438 mast raise)"""
    vertical_positions, datetimes, _, _ = read_ppp(ppp_csv_path(sn))
    vertical_positions = remove_advection(sn, datetimes, vertical_positions, year_conv=year_conv, print_statement=print_statement)
    if sn == '8438':
        datetimes, vertical_positions = _correct_8438_mast_raise(datetimes, vertical_positions)
    return datetimes, vertical_positions

def load_pole_depths(sn):
    """dates and pole base depth below the surface [m], from GNSS-IR"""
    _, pole_dates, poles_depths = buried_column(sn)
    if sn == '8438':
        pole_dates = np.array(pole_dates[:-52]); poles_depths = np.array(poles_depths[:-52])
    return pole_dates, poles_depths

def load_site_data(stations, year_conv=SECONDS_PER_YEAR):
    """zeroed vertical positions and pole depths for each station, for site-wide averaging"""
    station_dates = []; station_positions = []; pole_dates_all = []; pole_depths_all = []
    for sn in stations:
        print("Processing station: ", sn)
        datetimes, vertical_positions = load_vertical_positions(sn, year_conv, print_statement=False)
        station_dates.append(datetimes); station_positions.append([x - vertical_positions[0] for x in vertical_positions])
        pole_dates, poles_depths = load_pole_depths(sn)
        pole_dates_all.append(pole_dates); pole_depths_all.append(poles_depths)
    return station_dates, station_positions, pole_dates_all, pole_depths_all

def firn_velocity_below_pole(cfm_station, pole_dates, poles_depths, window, model_name=FIG3_MODEL, year_conv=SECONDS_PER_YEAR, num_sim=1000, position_uncertainty=.01):
    """Modeled (CFM) firn compaction velocity [m/yr] between the pole base and the bottom of the model domain"""
    bottom_depths = np.full(len(poles_depths), -1) # -1 signifies bottom of the column -- assumes ice is at bottom of domain!!!
    firn_dates, firn_compactions, _ = get_compaction(cfm_file(cfm_station, model_name), poles_depths, pole_dates, bottom_depths)
    firn_dates, firn_velocity, _, _ = monte_carlo_velocity(np.cumsum(firn_compactions), firn_dates, window, year_conv=year_conv, num_simulations=num_sim, position_uncertainty=position_uncertainty)
    return firn_dates, np.array(firn_velocity)

def _agreement_stats(observed, modeled):
    """R² (Pearson), NRMSE [%] of the demeaned series, and amplitude ratio (model/observed)"""
    observed = np.asarray(observed); modeled = np.asarray(modeled)
    r, _ = pearsonr(modeled, observed)
    observed = observed - np.mean(observed); modeled = modeled - np.mean(modeled)
    nrmse = 100 * np.sqrt(np.mean((modeled - observed)**2)) / np.std(observed)
    amp_ratio = np.std(modeled) / np.std(observed)
    print(f"-> Amplitude ratio (model/observed): {amp_ratio:.2f}")
    return r**2, nrmse, amp_ratio

def get_average(station_dates, station_positions, pole_dates_all, pole_depths_all, window=90, model_name=FIG3_MODEL,
                year_conv=SECONDS_PER_YEAR, num_sim=1000, ogre_velocity=None):
    """
    Site-wide average: interpolate and average station positions and pole depths, then compare the
    averaged OGRE vertical velocity with modeled firn velocity below the averaged pole depth.
    ogre_velocity: optional precomputed (velocity_dates, velocities) to skip the OGRE Monte Carlo.

    Returns dict: dates, residuals (OGRE - firn), firn_dates, firn_velocity, velocity_dates, velocities, r_squared, nrmse, amp_ratio
    """
    # FLATTEN AND AVERAGE TOGETHER
    date_grid, mean_vertical, _ = interpolate_and_average_timeseries(station_dates, station_positions)
    date_pole, mean_depths, _ = interpolate_and_average_timeseries(pole_dates_all, pole_depths_all)

    # GET VERTICAL VELOCITY w/ WINDOWING, OGRE and firn model
    if ogre_velocity is None:
        velocity_dates, velocities, _, _ = monte_carlo_velocity(mean_vertical, date_grid, window, year_conv=year_conv, num_simulations=num_sim, position_uncertainty=0.01)
    else:
        velocity_dates, velocities = ogre_velocity
    firn_dates, firn_velocity = firn_velocity_below_pole(FIG3_AVERAGE_CFM_STATION, date_pole, mean_depths, window, model_name, year_conv, num_sim)

    # MATCH DATES (first velocity match per firn date)
    velocity_index = {}
    for j, d in enumerate(d.date() for d in velocity_dates):
        velocity_index.setdefault(d, j)
    common_dates = []; common_velocities = []; common_firn_velocities = []
    for i, d in enumerate(d.date() for d in firn_dates):
        if d in velocity_index:
            common_dates.append(d)
            common_velocities.append(velocities[velocity_index[d]]); common_firn_velocities.append(firn_velocity[i])
    common_velocities = np.array(common_velocities); common_firn_velocities = np.array(common_firn_velocities)

    r_squared, nrmse, amp_ratio = _agreement_stats(common_velocities, common_firn_velocities)
    return dict(dates=np.array(common_dates), residuals=common_velocities - common_firn_velocities,
                firn_dates=firn_dates, firn_velocity=firn_velocity, velocity_dates=velocity_dates, velocities=np.array(velocities),
                r_squared=r_squared, nrmse=nrmse, amp_ratio=amp_ratio)

def plotVerticals(window = 90, year_conv = SECONDS_PER_YEAR, num_sim = 1000, model_name = FIG3_MODEL):
    """
    Figure 3. Left: vertical positions (A), OGRE total (ice + firn) and modeled firn velocity (C),
    residuals with variance explained inset (E). Right, vs easting: mean OGRE vertical velocity (B),
    mean modeled firn velocity (D), and vertical ice velocity = mean OGRE - firn residual (F).
    window: how many days to average to derive velocity
    """
    # PLOT
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex='col', gridspec_kw={'width_ratios': [2, 1]})
    ax1, ax4 = axes[0]; ax2, ax5 = axes[1]; ax3, ax6 = axes[2]
    ax_inset = inset_axes(ax3, width="55%", height="30%", loc='lower left',bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax3.transAxes, borderpad=.5)

    station_dates = []; station_positions = []; pole_dates_all = []; pole_depths_all = []
    eastings = []; mean_velocities = []; mean_velocities_std_err = []; mean_firn_velocities = []
    ice_velocities = []; ice_velocities_std = []; std_anomalies = []
    for sn, name, color in zip(FIG3_STATIONS, FIG3_NAMES, FIG3_COLORS):
        print("Processing station: ", sn) # LOAD DATA
        datetimes, vertical_positions = load_vertical_positions(sn, year_conv)
        eastings.append(np.mean(read_ppp_ne(ppp_sum_path(sn))[2]))
        window_t, position_uncertainty = _velocity_settings(sn, window)

        # Zero out first elevation
        mod_positions = [x - vertical_positions[0] for x in vertical_positions] 
        station_dates.append(datetimes); station_positions.append(mod_positions)
        ax1.plot(datetimes, mod_positions, label="Station " + name, color=color)

        # GET VERTICAL VELOCITY w/ WINDOWING
        velocity_dates, velocities, lower_bound, upper_bound = monte_carlo_velocity(vertical_positions, datetimes, window_t, year_conv=year_conv, num_simulations=num_sim, position_uncertainty=position_uncertainty)  
        ax2.plot(velocity_dates, velocities, color=color)
        ax2.fill_between(velocity_dates, lower_bound, upper_bound, alpha=0.3, color=color)

        # GET MEAN VELOCITY (linear fit over the whole record)
        mean_velocity, _, _, _, mean_velocity_std_error = linregress(_seconds_since_start(datetimes) / SECONDS_PER_YEAR, vertical_positions)
        print(f"-> {name} mean vertical velocity: {mean_velocity:.3f} ± {mean_velocity_std_error:.3f} m/yr")
        mean_velocities.append(mean_velocity); mean_velocities_std_err.append(mean_velocity_std_error)
        ax4.scatter(eastings[-1], mean_velocity, s=30, color=color)
        ax4.errorbar(eastings[-1], mean_velocity, yerr=mean_velocity_std_error, color=color, capsize=4, elinewidth=1.5) 

        # GET VERTICAL VELOCITY FROM CFM FOR EACH STATION
        pole_dates, poles_depths = load_pole_depths(sn)
        pole_dates_all.append(pole_dates); pole_depths_all.append(poles_depths)
        firn_dates, firn_velocity = firn_velocity_below_pole(sn, pole_dates, poles_depths, window_t, model_name, year_conv, num_sim, position_uncertainty)
        if sn == '8438':
            firn_velocity = firn_velocity[:-30]; firn_dates = firn_dates[:-30]
        ax2.plot(firn_dates, firn_velocity, color=color, linestyle='dotted')

        # BONUS: Calculate compaction rate between surface and pole bottom
        firn_dates2, firn_compactions2, _ = get_compaction(cfm_file(sn, model_name), np.full(len(poles_depths), 0.0), pole_dates, poles_depths)
        slope = linregress([(d - datetimes[0]).total_seconds() / SECONDS_PER_YEAR for d in firn_dates2], np.cumsum(firn_compactions2))[0]
        print(f"-> Mean compaction rate between surface and pole bottom over study period: {slope}")
        
        # PLOT AVERAGE FIRN
        mean_firn_velocity = np.mean(firn_velocity)
        mean_firn_velocities.append(mean_firn_velocity)
        ax5.scatter(eastings[-1], mean_firn_velocity, s=30, color=color)
        ax5.errorbar(eastings[-1], mean_firn_velocity, yerr=np.std(firn_velocity)/np.sqrt(len(firn_velocity)), color=color, capsize=4, elinewidth=1.5) 

        # ICE VELOCITY = mean residual between OGRE and firn velocities
        residuals, residual_dates, r_squared, nrmse = get_residuals(firn_dates, firn_velocity, velocity_dates, velocities)
        ice_velocity = np.mean(residuals)
        span_days = (residual_dates[-1] - residual_dates[0]).days
        n_indep = max(span_days / window_t, 1)
        ice_velocity_std = np.std(residuals, ddof=1) / np.sqrt(n_indep)  # standard error, ~independent windows
        print(f"-> mean residual bias = {ice_velocity:.4f} ± {ice_velocity_std:.4f}")
        print(f"-> R-squared: {r_squared}, NRMSE: {nrmse}")

        detrended_residuals = np.array(residuals) - ice_velocity
        ax3.plot(residual_dates, detrended_residuals, color=color)
        ax_inset.bar(name, r_squared*100, color=color)
        std_anomalies.extend(detrended_residuals)

        ax6.scatter(eastings[-1], ice_velocity, s=30, color=color)
        ax6.errorbar(eastings[-1], ice_velocity, yerr=ice_velocity_std, color=color, capsize=4, elinewidth=1.5)  
        ice_velocities.append(ice_velocity); ice_velocities_std.append(ice_velocity_std)

    # GET SITE-WIDE AVERAGE (excluding GRIP)
    keep = [i for i, sn in enumerate(FIG3_STATIONS) if sn != GRIP_ID]
    avg = get_average([station_dates[i] for i in keep], [station_positions[i] for i in keep],
                      [pole_dates_all[i] for i in keep], [pole_depths_all[i] for i in keep], window, model_name, year_conv, num_sim)
    ax2.plot(avg['velocity_dates'], avg['velocities'], color='k', lw=3, label='OGRE-Averaged Velocity')
    ax2.plot(avg['firn_dates'], avg['firn_velocity'], color='k', linestyle='dotted')
    ax3.plot(avg['dates'], avg['residuals'] - np.mean(avg['residuals']), color='k')
    ax_inset.bar('Mean', avg['r_squared']*100, color='k')
    print(f"**Average of All OGRE Stations -> R-squared: {avg['r_squared']}, NRMSE: {avg['nrmse']}")

    # CHECK EAST-WEST GRADIENT
    # average total vertical velocity
    p_value, x, y = weighted_least_squares(eastings, mean_velocities, mean_velocities_std_err)
    print(f"-> Total vertical velocity gradient p = {p_value:.3f}")
    if p_value < 0.05:  
        ax4.plot(x, y)
    # average firn velocity (excluding GRIP), line always shown
    p_value, x_firn, y_firn = weighted_least_squares([eastings[i] for i in keep], [mean_firn_velocities[i] for i in keep], np.ones(len(keep)))
    print(f"-> Firn vertical velocity gradient p = {p_value:.3f}")
    ax5.plot(x_firn, y_firn, c='black', linestyle='dashed')
    # average underlying ice velocity
    p_value, x, y = weighted_least_squares(eastings, ice_velocities, np.ones(len(ice_velocities)))
    print(f"-> Ice vertical velocity gradient p = {p_value:.3f}")
    if p_value < 0.05:  
        ax6.plot(x, y)

    # plot average ice velocity
    mean_ice_velocity = np.mean(ice_velocities)
    ice_se = np.std(ice_velocities, ddof=1) / np.sqrt(len(ice_velocities))
    print(f"-> Network-mean dynamic thinning: {mean_ice_velocity:.3f} ± {ice_se:.3f} m/yr")
    ax6.axhline(y=mean_ice_velocity, c='black', linestyle='dashed')
    ax6.text(x = ax6.get_xlim()[0], y=mean_ice_velocity, s=rf'$\overline{{z_{{ice}}}}$ = {mean_ice_velocity:.2f} m $\mathrm{{yr}}^{{-1}}$', va='bottom', ha='left', fontsize=6, fontfamily='sans-serif')

    ax2.plot([], [], linestyle='dotted', color='k', label=f'{model_name} Modeled Firn Velocities')
    print(f"Standard deviation of residuals of all stations (VV-Firn): {np.std(std_anomalies):.04f}")
    print(f"Mean residual vertical velocity: {mean_ice_velocity:.04f}")

    # FORMATTING
    _style_ticks([ax1, ax2, ax3, ax4, ax5, ax6], 7)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.grid(True, zorder=0, alpha=0.5)
    _style_ticks([ax_inset], 5)
    ax_inset.grid(False)
    for ax in [ax4, ax5, ax6]:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

    for label, ax in zip(['A', 'C', 'E', 'B', 'D', 'F'], [ax1, ax2, ax3, ax4, ax5, ax6]):
        ax.text(0.01, 0.98, label, transform=ax.transAxes, fontsize=9, fontweight='bold', va='top', ha='left', fontfamily='sans-serif')

    ax2.set_ylim(-.8, -.3)
    ax3.set_ylim(-.3, .2)
    _date_axis(ax3, 3, 7)
    title_kw = dict(fontsize=7, fontweight='bold', fontfamily='sans-serif')
    label_kw = dict(fontsize=6, fontweight='bold', fontfamily='sans-serif')
    ax1.set_title("OGRE Vertical Position Time Series", **title_kw)
    ax2.set_title(f"OGRE Total Velocity (Ice + Firn) & Modeled Firn, {window} Day Smoothing", **title_kw)
    ax3.set_title("Residuals between OGRE Velocities and Modeled Firn Velocity", **title_kw)
    ax4.set_title("Mean OGRE Vertical Velocities", **title_kw)
    ax5.set_title("Modeled Firn Mean Velocity", **title_kw)
    ax6.set_title("Vertical Ice Velocity", **title_kw)
    ax1.set_ylabel("[m]", **label_kw)
    for ax in [ax2, ax3, ax4, ax5, ax6]:
        ax.set_ylabel("[m y⁻¹]", **label_kw)
    ax6.set_xlabel("Easting [m]", **label_kw)
    ax_inset.set_ylabel("Percent Variance [%]", fontsize=5, fontweight='bold', fontfamily='sans-serif')
    ax1.legend(loc='lower left', fontsize=6, ncol=2) 
    ax2.legend(loc='upper right', fontsize=6, ncol=3)

    fig.align_ylabels()
    plt.tight_layout()
    plt.show()   

def interpolate_and_average_timeseries(list_of_dates, list_of_values):
    """
    Interpolate multiple time series onto a daily grid and average at each timestep.

    Parameters:
        list_of_dates: list of lists of datetime objects (one list per time series)
        list_of_values: list of lists of floats (same shape as list_of_dates)

    Returns:
        date_grid: list of datetime objects (daily)
        mean_values: numpy array of mean values per day (ignoring NaNs)
        std_values: numpy array of std dev per day (ignoring NaNs)
    """
    # Build daily date grid over global time range
    all_dates = [dt for dates in list_of_dates for dt in dates]
    start_date = min(all_dates)
    date_grid = [start_date + timedelta(days=d) for d in range((max(all_dates) - start_date).days + 1)]

    # Convert dates to float days since fixed epoch
    def days_since_epoch(dates):
        epoch = datetime(2000, 1, 1)
        return np.array([(d - epoch).total_seconds() / 86400.0 for d in dates])

    grid_days = days_since_epoch(date_grid)

    # Interpolate each time series onto the grid
    interpolated_arrays = []
    for dates, values in zip(list_of_dates, list_of_values):
        days = days_since_epoch(dates)
        if len(days) == 0:
            interpolated_arrays.append(np.full_like(grid_days, np.nan))
        else:
            interpolated_arrays.append(np.interp(grid_days, days, values, left=np.nan, right=np.nan))

    # Stack and average
    interpolated_stack = np.vstack(interpolated_arrays)
    return date_grid, np.nanmean(interpolated_stack, axis=0), np.nanstd(interpolated_stack, axis=0)

def compareAverage(window = 90, year_conv = SECONDS_PER_YEAR, num_sim = 1000, model_name = FIG3_MODEL):
    """AVERAGE ALL STATION POSITIONS TOGETHER (excluding GRIP)
    then calculate vertical velocity and compare to firn model
    # """
    stations = [sn for sn in FIG3_STATIONS if sn != GRIP_ID]
    # stations = ["0015", "0005", "0012", "0009", "0008", "8438", "0013", "0011", "0014", "0010", "0017", "0007"]
    avg = get_average(*load_site_data(stations, year_conv), window, model_name, year_conv, num_sim)

    plt.plot(avg['velocity_dates'], avg['velocities'], label='OGRE-averaged velocity')
    plt.plot(avg['firn_dates'], avg['firn_velocity'], label=f'{model_name} firn velocity')
    plt.plot(avg['dates'], avg['residuals'], marker='o', label='Residual (vel - firn)')
    plt.legend()
    plt.show()

    residuals = avg['residuals']
    print(f"-> Mean residual (ice velocity): {np.mean(residuals):.4f} ± {np.std(residuals):.4f}")
    print(f"-> percent error: {np.std(residuals)/np.std(avg['velocities'])*100}, {np.std(avg['velocities'])}")
    print(f"-> R-squared: {avg['r_squared']:.3f}, NRMSE: {avg['nrmse']:.1f}%, amplitude ratio: {avg['amp_ratio']:.2f}")
    return avg

def percent_variance_explained(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 100 * (1 - ss_res / ss_tot)

### HELPER FUNCTIONS FOR PLOTTING VERTICALS
def remove_advection(station, datetimes, vertical_positions, year_conv = SECONDS_PER_YEAR, print_statement=False):
    """uses local slope and mean overall velocity to correct the vertical positions"""
    if station not in station_slopes:
        print(f"-> Station {station} does not have a corresponding slope. No advection correction")
        return vertical_positions
    horizontal_dist_traveled = _seconds_since_start(datetimes) * (station_mean_yr_velocity[station]/year_conv)
    verticals_corrected = list(np.asarray(vertical_positions) + np.tan(station_slopes[station]) * horizontal_dist_traveled)

    if print_statement:
        x = vertical_positions[0] - vertical_positions[-1]
        y = verticals_corrected[0] - verticals_corrected[-1]
        print(f"-> Advection corrected total displacement from {x:.05f} to {y:05f}, delta: {x-y:.05f}m")
    return verticals_corrected  

def calculate_yearly_velocity(stationNum, datetimes, vertical_positions, year_conv):
    """calculates yearly velocity average for OGRE, and removes advection"""
    corrected_z = remove_advection(stationNum, datetimes, vertical_positions, print_statement=True)
    yearly_datetime_0 = datetimes[0]; yearly_ele= corrected_z[0]
    yearly_velocities = []; yearly_dates = []
    for k in range(len(datetimes)-1):
        if datetimes[k + 1].year > yearly_datetime_0.year or datetimes[k + 1] == datetimes[-1]:
            delta_t = (datetimes[k + 1] - yearly_datetime_0).total_seconds()  # This does account for leap years
            velocity_year = (corrected_z[k+1] - yearly_ele) / delta_t * year_conv  # Convert to m/yr
            yearly_velocities.extend([velocity_year, velocity_year])
            yearly_dates.extend([yearly_datetime_0, datetimes[k+1]])
            yearly_datetime_0 = datetimes[k+1]; yearly_ele = corrected_z[k+1]# reset starter

    # get overall velocity (average)
    end = -3 if stationNum == '8438' else -1
    overall_velocity = ((corrected_z[end] - corrected_z[0]) / (datetimes[end] - datetimes[0]).total_seconds())*year_conv
    return yearly_velocities, yearly_dates, overall_velocity

def getWindowedVelocity(window, vertical_positions, datetimes, year_conv = SECONDS_PER_YEAR, interp = True, plot=False):
    """vertical velocity [m/yr] from position differences `window` days apart, averaged over all window offsets.
    Positions should already be corrected for advection (see load_vertical_positions)."""
    if interp:
        datetimes, vertical_positions, _ = interpolate_daily(datetimes, vertical_positions)
        end_step = window
    else:
        end_step = 1

    if plot:
        _, ax = plt.subplots()

    velocity_data = defaultdict(list)
    for k in range(0, end_step): 
        # every `window`-th sample, starting at offset k, zeroed at start
        datetimes_shortened = datetimes[k::window]
        vertical_positions_shortened = np.asarray(vertical_positions[k::window], dtype=float)
        if len(vertical_positions_shortened) < 2:
            continue  # Skip if too few points to compute gradient
        vertical_positions_shortened = vertical_positions_shortened - vertical_positions_shortened[0]

        # calculate velocities, then reinterpolate to daily velocities
        velocity_z = np.gradient(vertical_positions_shortened, _seconds_since_start(datetimes_shortened))*year_conv
        d_d, z_d, _ = interpolate_daily(datetimes_shortened, velocity_z)
        for date, velocity in zip(d_d, z_d):
            velocity_data[date].append(velocity)
        
        if plot:
            ax.plot(datetimes_shortened, velocity_z)

    # Compute averages, sorted by date
    unique_dates = sorted(velocity_data)
    average_velocities = [np.mean(velocity_data[d]) for d in unique_dates]

    if interp:
        crop = int(window/2)
        unique_dates = unique_dates[crop:-crop]
        average_velocities = average_velocities[crop:-crop]
    
    if plot:
        ax.plot(unique_dates, average_velocities, c='r')
        ax.legend()
        plt.show()

    return unique_dates, average_velocities

def interpolate_daily(datetimes, dependent_variable):
    start_date = datetimes[0]
    daily_datetimes = [start_date + timedelta(days=i) for i in range((datetimes[-1] - start_date).days + 1)]

    original_times = np.array([(dt - start_date).total_seconds() / (24 * 3600) for dt in datetimes])
    daily_times = np.array([(dt - start_date).total_seconds() / (24 * 3600)  for dt in daily_datetimes])

    return daily_datetimes, np.interp(daily_times, original_times, dependent_variable), daily_times

def monte_carlo_velocity(vertical_positions, datetimes, window, year_conv=SECONDS_PER_YEAR, num_simulations=1000, position_uncertainty=0.01, interp=True):
    """
    Perform Monte Carlo simulations to estimate uncertainty bounds for velocities.

    Args:
        vertical_positions (list): Measured vertical positions.
        datetimes (list): Corresponding datetime objects.
        window (int): Window size for velocity calculation (in days).
        year_conv (float): Conversion factor for velocity (seconds to years).
        num_simulations (int): Number of Monte Carlo iterations.
        position_uncertainty (float): Standard deviation of random noise to add to positions (in meters).
        interp (bool): Whether to interpolate positions before calculating velocities.

    Returns:
        tuple: (dates, mean_velocities, lower_bounds, upper_bounds)
    """
    velocity_simulations = []
    for _ in range(num_simulations):
        # Add Gaussian noise to the vertical positions, then compute windowed velocities
        noisy_positions = vertical_positions + np.random.normal(0, position_uncertainty, len(vertical_positions))
        unique_dates, simulated_velocities = getWindowedVelocity(window, noisy_positions, datetimes, year_conv=year_conv, interp=interp)
        velocity_simulations.append(simulated_velocities)

    # Calculate mean and percentiles (5th and 95th for 90% confidence interval)
    velocity_simulations = np.array(velocity_simulations)
    return (unique_dates, np.mean(velocity_simulations, axis=0),
            np.percentile(velocity_simulations, 5, axis=0), np.percentile(velocity_simulations, 95, axis=0))

def buried_column(stationNum):
    return get_pole_column(gnssir_path(stationNum), ppp_csv_path(stationNum), stationNum)  # elevations, datetimes, poles

def get_compaction(cfm_file, pole_depths, pole_times, bottom_depths):
    """
    Function: For every model timestep, we have a list of compaction values
    that correspond to some depth-layer. We interpolate OGRE pole base data to this 
    dataset, and use that pole base depth to grab the TOTAL compaction from the 
    base of the pole to the bottom of the model domain, which should be the ice.

    compaction_velocity_2 in format: grid_resolution, numeric date, #, #, #, #, 
    Where #1 is the topmost firn copaction
    """
    data = np.genfromtxt(cfm_file, delimiter=',')
    grid_resolution = data[0,0]
    numeric_dates = data[:, 1]
    data = data[:,2:]

    # CONVERT DECIMAL-YEAR DATES TO DATETIMES
    dates = []
    for date in numeric_dates:
        year = int(date)
        days_in_year = 366 if calendar.isleap(year) else 365
        dates.append(datetime(year, 1, 1) + timedelta(days=(date - year) * days_in_year))

    # INTERPOLATE POLE_DEPTHS to FIRN DATES (seconds since epoch)
    pole_times_num = np.array([t.timestamp() for t in pole_times])
    dates_num = np.array([t.timestamp() for t in dates])
    interpolated_depths = interp1d(pole_times_num, pole_depths, kind='linear', bounds_error=False, fill_value=np.nan)(dates_num)
    interpolated_bottom_depths = interp1d(pole_times_num, bottom_depths, kind='linear', bounds_error=False, fill_value=np.nan)(dates_num)
    
    # Crop nan values from depths and pole bottoms, and corresponding firn compaction times
    valid_mask = ~np.isnan(interpolated_depths)
    valid_depths = interpolated_depths[valid_mask]  
    valid_bottom_depths = interpolated_bottom_depths[valid_mask]
    valid_dates = np.array(dates)[valid_mask]
    valid_numeric_dates = dates_num[valid_mask]
    valid_data = data[valid_mask]

    # NOW EXTRACT TOTAL COMPACTION FOR THE LAYER 
    # DEFINED AS FROM POLE DEPTH to BOTTOM DEPTH
    # WARNING: indexing depth_index = int(depth/grid_resolution) 
    # is essentially interpolation, source of error! 
    firn_compactions_at_depth = []
    for i, depth in enumerate(valid_depths):
        depth_index = int(depth/grid_resolution)
        bottom_depth = -1 if valid_bottom_depths[i] == -1 else int(valid_bottom_depths[i]/grid_resolution)
        firn_compactions_at_depth.append(np.sum(valid_data[i][depth_index:bottom_depth])*grid_resolution)

    return valid_dates, firn_compactions_at_depth, valid_numeric_dates

def get_residuals(firn_dates, firn_velocities, ogre_dates, ogre_velocities):
    """
    Residuals between OGRE velocities and firn velocities interpolated to the OGRE dates.
    
    Returns:
        residuals (OGRE - firn), valid OGRE dates, R² (Pearson), NRMSE [%]
    """
    firn_times = np.array([date.timestamp() for date in firn_dates])
    ogre_times = np.array([date.timestamp() for date in ogre_dates])

    # Mask OGRE times within firn range
    valid_mask = (ogre_times >= firn_times.min()) & (ogre_times <= firn_times.max())
    valid_ogre_velocities = np.array(ogre_velocities)[valid_mask]
    valid_ogre_dates = np.array(ogre_dates)[valid_mask]

    # Interpolate firn velocities at the valid OGRE times
    interpolated_firn = np.interp(ogre_times[valid_mask], firn_times, firn_velocities)
    residuals = valid_ogre_velocities - interpolated_firn

    # r_squared = percent_variance_explained(valid_ogre_velocities, interpolated_firn) / 100  # alternative metric
    r_squared, nrmse, _ = _agreement_stats(valid_ogre_velocities, interpolated_firn)

    return residuals.tolist(), valid_ogre_dates.tolist(), r_squared, nrmse

### Elevations
def plotSurfaceElevation():
    """LEGACY: superseded by plotSurfaceElevation_2. Will not run as-is because
    test_east_west_gradient_elevation now requires eastings and returns a dict."""
    _, ax = plt.subplots(1, figsize=(17.9*cm, 8.95*cm))
    ax_inset = inset_axes(ax, width="25%", height="35%", loc='upper left', bbox_to_anchor=(0.08, 0.05, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=.5)

    # stationNum = ["0005", "0012", "0009", "0008", "8438", "0011", "0010", "0007", "0017"]#, "7313"] 
    # stationNames = ["02", "03", "04", "05", "Camp", "08", "10", "11", "East", "GRIP"]
    # stationColors = ['#5247fc', '#2489f5', '#0ac0e8', '#3ae8d6', '#68fcc1', '#c4e88a', '#ff8947', '#ff4724', '#ff0000', '#ff004c']

    stationNum = ["0005", "0012", "0009", "0008", "8438", "0011", "0010", "0017", "0007", "7313"] 
    stationNames = ["02", "03", "04", "05", "Camp", "08", "10", "11", "12", "13"]
    stationColors = ['#5247fc', '#2489f5', '#0ac0e8', '#3ae8d6', '#68fcc1', '#c4e88a', '#ff8947', '#ff4724', '#ff0000', "#ff004c"]

    # stationNum = ["0015", "0005", "0012", "0009", "0008", "8438", "0011", "0014", "0010", "0017", "0007", "7313"]# ["879s", "879c", "879n", "8438"]#["0015", "0005", "0012", "0009", "0008", "8438", "0013", "0011", "0014", "0010", "0017", "0007"]#, "7313"]
    # stationNames = ['West', '02', '03', '04', '05', 'Camp', '08', '09', '10', '11', 'East','GRIP'] # ["South", "Central", "North", "Bamboo"]#
    # stationColors = ['#8000ff', '#5247fc', '#2489f5', '#0ac0e8', '#3ae8d6', '#68fcc1', '#c4e88a', '#f4c069', '#ff8947', '#ff4724', '#ff0000', '#ff004c', '#ff00ac' ]

    elevations_all = []; dates_all = []
    for i in range(len(stationNum)): 
        print("Processing station: ", stationNum[i])

        ##### GET REFLECTOMETRY & XYZ #####
        elevations, datetimes, *_ = get_elevation(gnssir_path(stationNum[i]), ppp_csv_path(stationNum[i]), stationNum[i])
        elevations = remove_advection(stationNum[i], datetimes, elevations, print_statement=True)
        _, northing, easting = read_ppp_ne(ppp_sum_path(stationNum[i]))
        mean_easting = np.mean(easting)

        if stationNum[i] == '8438':
            elevations = elevations[:-49]; datetimes = datetimes[:-49]
        # if stationNum[i] == '0012':
        #     elevation2, datetimes2, *_ = get_elevation(gnssir_path('879s'), ppp_csv_path('879s'), '879s')
        #     elevations.extend(elevation2); datetimes.extend(datetimes2)

        # if stationNum[i] == '0009':
        #     elevation2, datetimes2, *_ = get_elevation(gnssir_path('879c'), ppp_csv_path('879c'), '879c')
        #     elevations.extend(elevation2); datetimes.extend(datetimes2)

        # if stationNum[i] == '0008':
        #     elevation2, datetimes2, *_ = get_elevation(gnssir_path('879n'), ppp_csv_path('879n'), '879n')
        #     elevations.extend(elevation2); datetimes.extend(datetimes2)

        # ---- EXPORT RAW DATA TO CSV ----
        # df = pd.DataFrame({
        #     'datetime': datetimes,
        #     'elevation': elevations,   # raw, before subtracting first value
        #     'northing': northing[:len(datetimes)] ,
        #     'easting': easting[:len(datetimes)]   # in case lengths match
        # })

        # # Optional: create output folder
        # out_folder = os.path.join(os.path.expanduser('~/Desktop'), "gnss_elevations_advection_corrected")
        # os.makedirs(out_folder, exist_ok=True)

        # csv_path = os.path.join(out_folder, f"{stationNum[i]}_raw.csv")
        # df.to_csv(csv_path, index=False)
        # print(f"Saved raw data for station {stationNum[i]} to {csv_path}")

        elevations = [x-elevations[0] for x in elevations]
        ax.plot(datetimes, elevations, label="Station " + stationNames[i], color=stationColors[i])
        elevations_all.append(elevations); dates_all.append(datetimes)

        # Convert datetime to numeric (in years)
        dates_num = _seconds_since_start(datetimes) / SECONDS_PER_YEAR
        slope, _, _, _, std_err = linregress(dates_num, np.array(elevations))
        # print("slope: ", slope)

        ax_inset.errorbar(mean_easting, slope, yerr=std_err*3, color=stationColors[i], capsize=4, elinewidth=1.5, zorder=1)  
        ax_inset.scatter(mean_easting, slope, color=stationColors[i], zorder=2)  

    p_value, xs, ys, rate = test_east_west_gradient_elevation(elevations_all, dates_all, stationNum)
    if p_value < 0.05: 
        ax_inset.plot(xs, ys, linestyle='dashed', lw=1, color='k', label=f'Slope: {rate:.1e}, P: {p_value:.02f}', zorder=3)

    _date_axis(ax, 3, 7)
    _style_ticks([ax], 7)
    ax.xaxis.get_offset_text().set_fontsize(7)
    ax.set_ylim(0,.5)
    ax_inset.legend(fontsize=5)
    ax_inset.set_ylabel(r"Mean Rate [m y$^{-1}$]", fontsize=6)
    ax_inset.tick_params(axis='x', labelbottom=False, bottom=False)
    ax_inset.tick_params(axis='y', labelsize=6)
    ax.legend(loc='upper right', ncol=4, fontsize=6)
    ax.grid(True, zorder=0, alpha=0.5)
    ax.set_title("Surface Elevation", fontsize=6, fontweight='heavy', fontfamily='sans-serif')
    ax.set_ylabel("[m]", fontsize=6, fontweight='heavy', fontfamily='sans-serif')
    plt.tight_layout()
    plt.show()

### MERRA Analysis
def get_density(reanalysis, ground):
    from sklearn.linear_model import LinearRegression
    reanalysis = np.array(reanalysis); ground = np.array(ground)

    # Mask out NaNs
    mask = ~np.isnan(reanalysis) & ~np.isnan(ground)

    # Fit linear regression through origin; scaling factor is the inverse coefficient
    model = LinearRegression(fit_intercept=False)
    model.fit(reanalysis[mask].reshape(-1, 1), ground[mask])
    return 1/model.coef_[0] 

def get_merra_2(stationPosition):
    files = sorted(glob.glob(os.path.join(MERRA2_DIR, "*.nc4")))

    precips = []; evaps = []; dates = []
    for filename in files:
        try:
            ds = xr.open_dataset(filename, engine='netcdf4')
        except:
            continue

        precip = ds['PRECTOT'].interp(lat=stationPosition[0], lon=stationPosition[1], method='linear').values
        evap = ds['EVAP'].interp(lat=stationPosition[0], lon=stationPosition[1], method='linear').values
        precips.append(np.mean(precip)); evaps.append(np.mean(evap))
        dates.append(datetime.strptime(filename[-12:-4], '%Y%m%d'))

    dates, precips, evaps = zip(*sorted(zip(dates, precips, evaps)))

    # get pe
    return dates, np.array(precips)*86400 - np.array(evaps)*86400

def plot_merra_gnss(upper_firn_compaction=0):
    """compare MERRA P-E with ground data"""
    stationNum = ["0005", "0012", "0009", "0008", "8438", "0013", "0011", "0014", "0010", "0017", "0007", "7313"]# ["879s", "879c", "879n", "8438"]#["0015", "0005", "0012", "0009", "0008", "8438", "0013", "0011", "0014", "0010", "0017", "0007"]#, "7313"]

    for station in stationNum:
        print("Station: ", station)
        if station == '7313':
            continue
        ogre_averages, match_dates, *_ = read_gnssir_single(station, gnssir_path(station))
        _, _, latitude, longitude = read_ppp(ppp_csv_path(station))
        station_position = [float(latitude[0]), float(longitude[0])]

        trim = -52 if station == '8438' else -5
        ogre_averages = ogre_averages[:trim]; match_dates = match_dates[:trim]
        ogre_averages = [x - ogre_averages[0] for x in ogre_averages]

        # scale by densification of firn layer between surface and base of pole...
        years_since_ref = np.array([(d - match_dates[0]).days / 365.25 for d in match_dates])
        ogre_averages = np.array(ogre_averages) + upper_firn_compaction * years_since_ref  # 10 cm in meters

        merra_dates, pe = get_merra_2(station_position)
        merra_dates = [pd.to_datetime(d).to_pydatetime() for d in merra_dates]
        match_dates = [pd.to_datetime(x.date()).to_pydatetime() for x in match_dates] # might cause issues since this is date interpolation

        pe_sums = [0]; cumulative_pe = [0]
        for i in range(1, len(match_dates)):
            try:
                index1 = merra_dates.index(match_dates[i-1])
                index2 = merra_dates.index(match_dates[i])
            except ValueError as e:
                print(f"Date not found in merra_dates: {e}")
                pe_sums.append(np.nan)
                cumulative_pe.append(np.nan)
                continue

            pe_sum = np.sum(pe[index1:index2])
            pe_sums.append(pe_sum)
            cumulative_pe.append(pe_sum + cumulative_pe[-1])

        # estimate density and correct
        density = get_density(cumulative_pe, ogre_averages)
        cumulative_pe = cumulative_pe/density
        print(f"->Estimated (corrected for 10cm) Density: {density:.03f}")

        fig, ax = plt.subplots()
        ax.plot(match_dates, ogre_averages, color=derek_colors['blue'], label=f'{station}')
        ax.plot(match_dates, cumulative_pe, color=derek_colors['red'], label='MERRA-2 P-E')
        ax.set_title(f"MERRA-2 Precip-Evap vs {station} dH")
        ax.legend()
        ax.set_ylabel("meters")
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        # plt.show()

### Horizontal velocity analysis - see if temporally stable
def smooth_horizontal_velocity():
    window = 300; year_conv = SECONDS_PER_YEAR; num_sim = 10
    stationNum = ["0005", "0012", "0009", "0008", "8438", "0011", "0010", "0007", "0017", "7313"] 

    fig, ax = plt.subplots()
    for i, station in enumerate(stationNum[:-1]):
        print(f"Processing {station}")
        datetimes, northings, eastings = read_ppp_ne(ppp_sum_path(station))
        northings = np.array(northings); eastings = np.array(eastings)
        # Convert datetimes to float (e.g., decimal years)
        times = np.array([dt.year + dt.timetuple().tm_yday / 365.25 for dt in datetimes])
        
        # Calculate velocity over time (difference in northing or magnitude)
        # Option 1: velocity magnitude
        displacements = np.sqrt((northings - northings[0])**2 + (eastings - eastings[0])**2)
        velocities = np.gradient(displacements, times)

        # Option 2: just use northing OR easting if interested in N-S motion
        # velocities = np.gradient(eastings, times)

        # Run linear regression on velocity vs. time
        slope, intercept, r_value, p_value, std_err = linregress(times, velocities)

        # plt.plot(times, velocities)
        # plt.show()

        # # Interpretation
        # trend = "increasing" if slope > 0 and p_value < 0.05 else \
        #         "decreasing" if slope < 0 and p_value < 0.05 else \
        #         "no significant trend"
        
        # print(f"Station {station} velocity trend: {trend} (slope={slope:.4f}, p={p_value:.4f})")

        velocity_dates, eastings, _, _ = monte_carlo_velocity(eastings, datetimes, window, year_conv=year_conv, num_simulations=num_sim, position_uncertainty=0.01)
        _, northings, _, _ = monte_carlo_velocity(northings, datetimes, window, year_conv=year_conv, num_simulations=num_sim, position_uncertainty=0.01)

        velocities = np.sqrt(np.array(northings)**2 + np.array(eastings)**2)
        ax.plot(velocity_dates, velocities, color=stationColors[i], label=stationNames[i])

        # CHECK SLOPE
        times = np.array([dt.year + dt.timetuple().tm_yday / 365.25 for dt in velocity_dates])
        slope, intercept, r_value, p_value, std_err = linregress(times, velocities)

        trend = "increasing" if slope > 0 and p_value < 0.05 else \
                "decreasing" if slope < 0 and p_value < 0.05 else \
                "no significant trend"
        
        print(f"Station {station} velocity trend: {trend} (slope={slope:.4f}, p={p_value:.4f})")

    ax.legend()
    plt.show()

### elevation and accumulation correlation
def correlate_elevation_accumulation():
    """correlate surface elevation, using GNSS IR pole height, with accumulation. 
    just to see what happens
    """
    stationNum = ["0005", "0012", "0009", "0008", "8438", "0011", "0010", "0007", "0017"] 
    stationNames = ["02", "03", "04", "05", "Camp", "08", "10", "11", "East"]
    stationColors = ['#5247fc', '#2489f5', '#0ac0e8', '#3ae8d6', '#68fcc1', '#c4e88a', '#ff8947', '#ff4724', '#ff0000']

    results = []
    for i in range(len(stationNum)): 
        print("Processing station: ", stationNum[i])

        ##### GET REFLECTOMETRY & XYZ #####
        averages, datetimes, *_ = read_gnssir_single(stationNum[i], gnssir_path(stationNum[i]), scale_averages=False)
        net_acc = _net_accumulation(averages)
        elevations, datetimes, *_ = get_elevation(gnssir_path(stationNum[i]), ppp_csv_path(stationNum[i]), stationNum[i])

        if stationNum[i] == '8438':
            elevations = elevations[:-49]; datetimes = datetimes[:-49]
            net_acc = net_acc[:-49]

        try:
            result = pearsonr(elevations, net_acc)
            print(result)
            results.append(result)
        except:
            print("reflectometry and elevations not aligned")
    print(results)

def _robust_rate(dates, elevations, alpha=0.95):
    """Theil-Sen rate in m/yr with an approximate 1-sigma error.
    The error is the 95% CI half-width divided by 1.96. Daily data is
    autocorrelated, so treat it as optimistic."""
    dates = np.asarray(dates)
    t_yr = _seconds_since_start(dates) / SECONDS_PER_YEAR
    med, _, lo, hi = theilslopes(np.asarray(elevations), t_yr, alpha=alpha)
    return med, (hi - lo) / (2 * 1.96), (t_yr[-1] - t_yr[0])

def test_east_west_gradient_elevation(all_elevations, all_dates, stations, eastings, names=None, min_overlap_years=0.5):
    """
    Robust (Theil-Sen) elevation rates per station over (a) each station's whole
    record and (b) the overlap window shared by all stations. Then a weighted
    least squares fit of overlap rates vs easting.

    Returns dict with: p_value, x_vals, y_vals (m/yr line), slope_per_km (m/yr/km),
    eastings, rates, errs (overlap period, m/yr), overlap_start, overlap_end.
    """
    names = names or stations
    eastings = np.asarray(eastings, dtype=float)

    all_start = max(min(d) for d in all_dates)
    all_end = min(max(d) for d in all_dates)
    overlap_yrs = (all_end - all_start).days / 365.25
    if overlap_yrs < min_overlap_years:
        raise ValueError(f"Not enough temporal overlap ({overlap_yrs:.2f} yr) among stations.")

    print(f"\nOverlap window: {all_start:%Y-%m-%d} to {all_end:%Y-%m-%d} ({overlap_yrs:.2f} yr)")
    print(f"{'Stn':>6} {'Easting':>9} | {'Full rate':>10} {'±σ':>7} {'yrs':>5} | {'Overlap rate':>12} {'±σ':>7}")

    full_rates, full_errs, ov_rates, ov_errs, ov_east, ov_idx = [], [], [], [], [], []
    for i, name in enumerate(names):
        dates = np.array(all_dates[i]); elev = np.array(all_elevations[i])

        # whole period
        r_full, e_full, span = _robust_rate(dates, elev)
        full_rates.append(r_full); full_errs.append(e_full)

        # overlap period
        mask = (dates >= all_start) & (dates <= all_end)
        if mask.sum() < 3:
            print(f"{name:>6} {eastings[i]:9.0f} | {r_full:10.4f} {e_full:7.4f} {span:5.2f} | {'skipped (<3 pts)':>20}")
            continue
        r_ov, e_ov, _ = _robust_rate(dates[mask], elev[mask])
        ov_rates.append(r_ov); ov_errs.append(e_ov)
        ov_east.append(eastings[i]); ov_idx.append(i)
        print(f"{name:>6} {eastings[i]:9.0f} | {r_full:10.4f} {e_full:7.4f} {span:5.2f} | {r_ov:12.4f} {e_ov:7.4f}")

    if len(ov_rates) < 3:
        raise ValueError("Not enough stations with overlapping data to perform regression.")

    ov_rates = np.array(ov_rates); ov_errs = np.array(ov_errs); ov_east = np.array(ov_east)

    # WLS of overlap rates vs easting (rates already in m/yr)
    res = sm.WLS(ov_rates, sm.add_constant(ov_east), weights=1 / ov_errs**2).fit()
    slope_coef, slope_se = res.params[1], res.bse[1]
    p_value = 2 * t.sf(abs(slope_coef / slope_se), len(ov_rates) - 2)

    print(f"\nOverlap-period gradient: {slope_coef*1000:.4e} ± {slope_se*1000:.4e} m/yr per km, p = {p_value:.3g}")

    x_vals = np.linspace(ov_east.min(), ov_east.max(), 100)
    y_vals = res.params[0] + slope_coef * x_vals

    return dict(p_value=p_value, x_vals=x_vals, y_vals=y_vals,
                slope_per_km=slope_coef * 1000, eastings=ov_east,
                rates=ov_rates, errs=ov_errs, idx=ov_idx,
                overlap_start=all_start, overlap_end=all_end)

def plotSurfaceElevation_2():
    """plot surface elevation, using GNSS IR pole height"""
    _, ax = plt.subplots(1, figsize=(17.9*cm, 8.95*cm))
    ax_inset = inset_axes(ax, width="25%", height="35%", loc='upper left',
                          bbox_to_anchor=(0.08, 0.05, 0.9, 0.9), bbox_transform=ax.transAxes, borderpad=.5)

    stationNum = ["0005", "0012", "0009", "0008", "8438", "0011", "0010", "0017", "0007", "7313"]
    stationNames = ["02", "03", "04", "05", "Camp", "08", "10", "11", "12", "13"]
    stationColors = ['#5247fc', '#2489f5', '#0ac0e8', '#3ae8d6', '#68fcc1', '#c4e88a', '#ff8947', '#ff4724', '#ff0000', "#ff004c"]

    elevations_all, dates_all, mean_eastings = [], [], []
    for i in range(len(stationNum)):
        print("Processing station: ", stationNum[i])

        datetimes, elevations = load_surface_elevation(stationNum[i], print_statement=True)
        mean_eastings.append(np.mean(read_ppp_ne(ppp_sum_path(stationNum[i]))[2]))
        ax.plot(datetimes, elevations, label="Station " + stationNames[i], color=stationColors[i])
        elevations_all.append(elevations); dates_all.append(datetimes)

    # robust rates: full period + overlap period (printed); overlap used for inset
    out = test_east_west_gradient_elevation(elevations_all, dates_all, stationNum,
                                            mean_eastings, names=stationNames)

    for e, r, err, i in zip(out['eastings'], out['rates'], out['errs'], out['idx']):
        ax_inset.errorbar(e, r, yerr=err, color=stationColors[i], capsize=4, elinewidth=1.5, zorder=1)
        ax_inset.scatter(e, r, color=stationColors[i], zorder=2)

    if out['p_value'] < 0.05:
        ax_inset.plot(out['x_vals'], out['y_vals'], linestyle='dashed', lw=1, color='k', zorder=3,
                      label=f"Slope: {out['slope_per_km']:.1e} m/yr/km, P: {out['p_value']:.02f}")

    # show the overlap window on the main axis
    ax.axvspan(out['overlap_start'], out['overlap_end'], color='gray', alpha=0.08, zorder=0)

    _date_axis(ax, 3, 7)
    _style_ticks([ax], 7)
    ax.xaxis.get_offset_text().set_fontsize(7)
    ax.set_ylim(0, .5)
    ax_inset.legend(fontsize=5)
    ax_inset.set_ylabel(r"Overlap-period Rate [m y$^{-1}$]", fontsize=6)
    ax_inset.tick_params(axis='x', labelbottom=False, bottom=False)
    ax_inset.tick_params(axis='y', labelsize=6)
    ax.legend(loc='upper right', ncol=4, fontsize=6)
    ax.grid(True, zorder=0, alpha=0.5)
    ax.set_title("Surface Elevation", fontsize=6, fontweight='heavy', fontfamily='sans-serif')
    ax.set_ylabel("[m]", fontsize=6, fontweight='heavy', fontfamily='sans-serif')
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    plotStrain()           # figure 2
    plotVerticals()        # figure 3
    plotSMB()              # figure 4
    plotSurfaceElevation_2()   # figure 5
    compareAverage()         # table X?

    plot_merra_gnss(upper_firn_compaction=0.023006636590525216)
    smooth_horizontal_velocity()
    correlate_elevation_accumulation()

    # densities - no correction
    # densities = [323.631, 309.029, 321.123, 320.561, 320.479, 315.357, 328.278, 326.242]
    # upper firn compaction rates
    upper_compactions = [-0.023454151963225554, -0.02393916998364272, -0.024044632829127697, -0.02917360523389139, -0.02158751582260213, -0.021144359029020367, -0.018694101260544625, -0.022015556602147247]
    # densities - assuming adjustment above^ - mean = 309.1978888888889
    densities2 = [315.546, 301.599, 312.904, 284.333, 312.220, 312.235, 307.194, 319.442, 317.308]
    # compareAverage()
