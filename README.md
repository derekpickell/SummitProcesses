# OGRE GNSS analysis: strain, vertical velocity, SMB and firn compaction at Summit, Greenland

This code accompanies JOG manuscript "A single instrument approach to ice flow, surface mass balance, and firn compaction validation in central Greenland" by Pickell, Hawley, Stevens.

### Python

Python 3.9 or newer is required, with the following packages:

```
numpy scipy pandas matplotlib statsmodels scikit-learn xarray netCDF4 rioxarray geopandas pyproj
```

`ogre_analysis.py` imports all of them at module level. Every other script imports `ogre_analysis`, so every script needs the full set.  

### Community Firn Model (only for `generate_cfm.py`)

`generate_cfm.py` calls `CFM_main/main.py` from a CommunityFirnModel checkout. < 

### Environment variables

| Variable | Required | Default | Contents |
|---|---|---|---|
| `REFL_CODE` | yes | – | gnssrefl processing directory with GNSS-IR results and PPP solutions |
| `OGRE_STATIC_DIR` | no | `<repo>/staticData` | static datasets |
| `OGRE_MERRA_DIR` | no | `<repo>/downloadScripts/download_merra` | daily MERRA-2 files (SMB, surface elevation budget) |
| `OGRE_MERRA2_DIR` | no | `~/Workspace/files/MERRA_2` | daily MERRA-2 files (`plot_merra_gnss`) |
| `OGRE_MEASURES_ANNUAL_DIR` | no | `~/Workspace/files/MeasuresAnnualVelocities` | MEaSUREs annual velocity mosaics |
| `OGRE_BEDMACHINE` | no | `~/Workspace/files/BedMachineGreenland-v5_bed.tif` | BedMachine v5 bed elevation |
| `OGRE_CFM_DIR` | no | `~/Workspace/files/cfm_ogre_models` | CFM runs, OGRE-forced |
| `OGRE_CFM_NO_OGRE_DIR` | no | `~/Workspace/files/cfm_no_ogre_models_corrected_mass` | CFM runs, MERRA-2-forced, by surface density |
| `OGRE_CFM_HOME` | for CFM runs | – | CommunityFirnModel checkout |
| `OGRE_CFM_PYTHON` | no | `$OGRE_CFM_HOME/env/bin/python` if present, else the current Python | Python used to run CFM |
| `OGRE_CFM_FORCING_DIR` | for CFM runs | `~/Workspace/files/MERRA-2_flx` | MERRA-2 files used to build CFM forcings |
| `OGRE_CFM_INIT_DENSITY` | for CFM runs | `$OGRE_STATIC_DIR/Summit_avg_density_profile.csv` | initial firn density profile for CFM |
| `OGRE_CFM_SENSITIVITY_DIR` | no | `~/Workspace/files/cfm_sensitivity` | output of `generate_cfm.py sensitivity` |

For example:

```bash
export REFL_CODE=/path/to/refl_code
export OGRE_CFM_DIR=/path/to/cfm_ogre_models
```

## Input data


### GNSS data (under `$REFL_CODE`)

The analysis uses 13 OGRE stations: `0015 0005 0012 0009 0008 8438 0013 0011 0014 0010 0017 0007 7313`. Their station names are 01–05, Camp, 07–13, in that order; 7313 is GRIP.

| Data | Path pattern | Format |
|---|---|---|
| GNSS-IR reflector heights | `$REFL_CODE/<year>/results/<station>/<doy>.txt` | Output of the gnssrefl `gnssir` command, one file per day. The readers skip 6 header lines and use these whitespace-delimited columns: 0 year, 1 day of year, 2 reflector height [m], 3 satellite, 4 UTC time [h], 5 azimuth [°], 10 frequency. |
| PPP time series | `$REFL_CODE/rinex/<station>/<dir>/<dir>/*.csv` | One file per RINEX file. Comma-delimited with one header row. The columns used are 0 latitude [°], 1 longitude [°], 2 ellipsoidal height [m], 3 decimal hour (UTC), 4 day of year, 5 year. The position is taken from the last row, and the time is the midpoint of the first and last rows. |
| PPP summary | `$REFL_CODE/rinex/<station>/<dir>/<dir>/*.sum` | One file per RINEX file. The lines used are `BEG YYYY-mm-dd HH:MM:SS.f`, `END YYYY-mm-dd HH:MM:SS.f`, and `PRJ  UTM <zone> <easting> <northing> ...`, which gives horizontal position in UTM zone 24N. |

The two `<dir>` levels under each station (for example `<year>/<doy>`) can have any names. Only the depth matters.

### Static datasets (under `$OGRE_STATIC_DIR`)

| File | Format | Source | Used by |
|---|---|---|---|
| `SUMup_2024_density_greenland.csv` | CSV with columns `latitude`, `longitude`, `midpoint` (depth, m), `density` (kg m⁻³), and `profile_key` | SUMup 2024 density release | `plotSMB` (snow density for SMB), and `generate_cfm.py` (converts OGRE accumulation to ice equivalent for `ogre` forcing) |
| `MEASURES/1m_contours_utm24.gpkg` | GeoPackage of LineStrings in UTM 24N, with an `ELEV` attribute | 1 m contours from ArcticDEM tile `22_43_1_1_2m_v4.1`, filtered to length > 350 m | Strain maps (`plotStrain`), and the animation in `testing_measures.py` |
| `MEASURES/22_43_1_1_2m_v4.1/22_43_1_1_2m_v4.1_dem.tif` | GeoTIFF, EPSG:3413 | ArcticDEM 2 m mosaic tile | `get_dem_utm` only (not called by any figure) |
| `MEASURES/greenland_vel_mosaic250_vx_v1.tif`, `..._vy_v1.tif` | GeoTIFF, EPSG:3413, m yr⁻¹ | MEaSUREs multi-year Greenland velocity mosaic, 250 m | Velocity product `measures_multiyear` |
| `PROMICE/IV_20160128_20221222.nc` | NetCDF with `land_ice_surface_easting_velocity` and `land_ice_surface_northing_velocity` in m day⁻¹, EPSG:3413 | PROMICE multi-year ice velocity | Velocity product `promice` |
| `Summit_avg_density_profile.csv` | CSV in the format CFM expects for `initfirnFile` | <!-- TODO: provenance --> | `generate_cfm.py` (initial firn profile) |

### External datasets (paths set by environment variables)

| Variable | Contents | Format |
|---|---|---|
| `OGRE_MEASURES_ANNUAL_DIR` | MEaSUREs annual Greenland velocity mosaics, version 5, for the December–November years 2014–2015 through 2022–2023 | GeoTIFF files named `GL_vel_mosaic_Annual_01DecYY_30NovYY_{vx,vy}_v05.0.tif`, EPSG:3413, m yr⁻¹. The 2020–2021 pair is enough for the Figure 2 default (`FIG2_PRODUCT = 'measures_2020_2021'`). |
| `OGRE_BEDMACHINE` | BedMachine Greenland v5 bed elevation | Single-band GeoTIFF, EPSG:3413, m. Used for ice thickness in the strain-derived vertical velocity. |
| `OGRE_MERRA_DIR` | Daily MERRA-2 surface flux files covering the OGRE array | `.nc4` files containing `PRECTOT` and `EVAP` (kg m⁻² s⁻¹) on `lat`/`lon`. Only the first time step of each file is used, so each file should hold one daily mean. |
| `OGRE_MERRA2_DIR` | The same kind of MERRA-2 files, for `plot_merra_gnss` | As above. The date is read from the file name, which must end in `YYYYMMDD.nc4` (standard MERRA-2 naming, e.g. `MERRA2_400.tavg1_2d_flx_Nx.20220101.nc4`). This can be the same directory as `OGRE_MERRA_DIR`. |
| `OGRE_CFM_FORCING_DIR` | MERRA-2 FLX collection (`tavg1_2d_flx_Nx`) for CFM forcing, from 1982 to the end of the study period | `.nc4` files with exactly one time step (a daily mean) each, containing `PRECSNO`, `EVAP`, `PRECTOT` (kg m⁻² s⁻¹) and `TSH` (K). They must cover 72.5° N and 38.75–37.5° W. The 1982–1990 period is repeated back to 1450 for spin-up. |

### CFM output (under `$OGRE_CFM_DIR` and `$OGRE_CFM_NO_OGRE_DIR`)

Each run's output goes to `<station>/<run>/output/output_firn_layers.csv`. Use `generate_cfm.py` to produce these files, or download the archived runs.

`output_firn_layers.csv` has no header and one row per CFM output step. Column 0 is the output grid resolution [m], column 1 is the decimal year, and the remaining columns give the compaction of each grid layer from the surface downward. Compaction between two depths is the sum of those layers multiplied by the grid resolution. This file is not part of standard CFM output, which is HDF5. 

| Run folder | Used by |
|---|---|
| `$OGRE_CFM_DIR/<station>/GSFC2020` at all Figure 3 stations | Figure 3 (`plotVerticals`, `FIG3_MODEL`), `compareAverage`, `plotSurfaceElevation_2` |
| `$OGRE_CFM_DIR/0009/{GSFC2020, KuipersMunneke2015, Ligtenberg2011, Crocus, HLdynamic}` | `testing_verticals.py` (`PHYSICS_MODELS`) |
| `$OGRE_CFM_DIR/0009/MaxSP_<conductivity>` for 13 conductivity schemes | `testing_verticals.py` (`CONDUCTIVITY_MODELS`) |
| `$OGRE_CFM_NO_OGRE_DIR/<station>/GSFC2020_<density>` for densities 280–340 at the Figure 3 stations except GRIP | `stacked_plot.py` |

## File functionality

### `gnss_utils.py`

This file contains readers for the GNSS data described above. None of its functions plot anything.

- `read_gnssir_single` returns daily reflector heights from the GNSS-IR results. It merges solutions from a RINEX day that spans two calendar days (`clump_by_date`).
- `read_ppp` returns daily vertical positions, latitudes and longitudes from the PPP `.csv` files.
- `read_ppp_ne` returns daily UTM northings and eastings from the PPP `.sum` files.
- `get_elevation` returns surface elevation, computed as the PPP antenna height minus the reflector height.
- `get_pole_column` returns the depth of the pole base below the surface, computed as the 3.048 m pole length minus the reflector height.

### `Scripts/ogre_analysis.py`

| Function | Output | Inputs |
|---|---|---|
| `plotStrain` | Figure 2: strain rates and vertical ice velocity from OGRE compared with satellite velocity | PPP `.csv` and `.sum`, velocity products, contour `.gpkg`, BedMachine |
| `plotVerticals` | Figure 3: OGRE vertical velocity compared with modeled firn velocity | GNSS-IR, PPP, CFM runs (`OGRE_CFM_DIR`) |
| `plotSMB` | Figure 4: surface mass balance from GNSS-IR compared with MERRA-2 | GNSS-IR, PPP, SUMup, MERRA-2 (`OGRE_MERRA_DIR`) |
| `plotSurfaceElevation_2` | Figure 5: surface elevation change | all of the above |
| `compareAverage` | site-averaged OGRE and firn velocity statistics | GNSS-IR, PPP, CFM runs |
| `plot_merra_gnss` | MERRA-2 P−E compared with OGRE surface height change | GNSS-IR, PPP, MERRA-2 (`OGRE_MERRA2_DIR`) |
| `smooth_horizontal_velocity`, `correlate_elevation_accumulation` | supporting diagnostics | PPP, GNSS-IR |

Three choices that the figures depend on are set at the top of the file, from the results of the test scripts: `FIG2_PRODUCT` and `FIG2_SMOOTHING_PX` (from `testing_measures.py`), and `FIG3_MODEL` (from `testing_verticals.py`).

### `Scripts/testing_measures.py`

This script tests the satellite velocity products for Figure 2. It compares each product in `VELOCITY_PRODUCTS` with the OGRE stations, looking at speed bias, flow direction and strain-derived vertical velocity, and finds the smoothing width that best matches the stations. That determines `FIG2_PRODUCT` and `FIG2_SMOOTHING_PX`. It also has functions for strain maps, synthetic strain triangles, and a velocity animation; the animation is saved as `Scripts/velocity_animation.gif`.

Inputs: PPP `.sum` files, all velocity products, the contour `.gpkg` and BedMachine.

### `Scripts/testing_verticals.py`

This script tests the firn model runs for Figure 3. It compares the site-averaged OGRE vertical velocity (all Figure 3 stations except GRIP) with the modeled firn velocity from each CFM run. For each run it prints R², NRMSE, amplitude ratio and the implied ice velocity, which determines `FIG3_MODEL`.

Inputs: GNSS-IR, PPP `.csv`, and the CFM runs in `$OGRE_CFM_DIR/0009/`.

### `Scripts/stacked_plot.py`

This script computes the surface elevation budget at each station:

surface elevation change = accumulation (GNSS-IR) + firn compaction (CFM) + vertical ice velocity (strain)

It searches over Glen's n, surface snow density and firn model to find the combination with the smallest residual drift.

Inputs: GNSS-IR, PPP, MERRA-2 (`OGRE_MERRA_DIR`), BedMachine, and the CFM runs in `$OGRE_CFM_NO_OGRE_DIR`.

### `Scripts/generate_cfm.py`

This script builds the CFM forcings and runs CFM to produce the run folders listed above. For each station it writes four forcing `.csv` files (temperature, accumulation, rain and sublimation), each with a row of decimal-year times and a row of values. It then writes a `cfm_config.json` into each run folder and runs CFM.

```bash
python generate_cfm.py physics         # density physics runs (OGRE_CFM_DIR)
python generate_cfm.py conductivity    # MaxSP conductivity runs (OGRE_CFM_DIR)
python generate_cfm.py density         # surface density runs (OGRE_CFM_NO_OGRE_DIR)
python generate_cfm.py sensitivity     # 300 vs 330 kg m⁻³ surface density at Camp (8438)

# options
--stations 0009 0012      # default: the stations the analysis reads for that run set
--models GSFC2020 Crocus  # physRho models, or conductivity schemes for the conductivity set
--densities 300 310       # density and sensitivity sets
--forcing ogre|merra      # default: ogre for physics/conductivity, merra for density
--no-correction           # don't correct the MERRA-2 P−E bias
--skip-forcings           # reuse the forcing files already in the station folders
--dry-run                 # write forcings and configs without running CFM
```

There are two forcing types. `merra` interpolates MERRA-2 to the station. `ogre` does the same, then replaces the accumulation with OGRE GNSS-IR accumulation wherever OGRE data exists. In both, MERRA-2 P−E is divided by (1 + bias), using the per-station bias against OGRE in `stacked_plot.STATION_PERCENT_DECIMAL`.

The default stations, models and densities come from `ogre_analysis`, `testing_verticals` and `stacked_plot`. That way, the folder names it writes match what those scripts read.

Inputs: PPP `.csv` (station longitude), GNSS-IR (`ogre` forcing), SUMup, MERRA-2 (`OGRE_CFM_FORCING_DIR`), the initial density profile, and CFM.


## Disclosure
The provided code and documentation have been modified by Anthropic Sonnet v5.5. This LLM was used for the following purposes: 
- Scan all original human-written code and refactor (with human verification) of the following: 
    - replace static file paths (e.g., /my/personal/computer/path/file.abc)
    - transcribe functions from external scripts otherwise not associated with this manuscript
    - refactor code for readability/flow
- Cleanup of doc strings & function comments

Any changed content, outputs, and functionality to these scripts were verified by the author.