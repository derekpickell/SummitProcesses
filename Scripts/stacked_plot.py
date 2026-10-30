"""
Surface elevation budget ("stacked" components) at each OGRE station:

    surface elevation change = accumulation (GNSS-IR) + firn compaction (CFM) + vertical ice velocity (strain)

Searches Glen's n, snow/firn density and firn model for the combination with the smallest residual drift.
All station loading, strain and firn code is shared with ogre_analysis.py (same setup / environment variables).
CFM runs forced without OGRE are read from OGRE_CFM_NO_OGRE_DIR/<station>/<model>_<density>/output/.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

from ogre_analysis import (
    MERRA_DAILY_DIR, GRIP_ID, FIG3_STATIONS, FIG3_NAMES, derek_colors, read_ppp,
    ppp_csv_path, cfm_no_ogre_file, _cumulative_from_zero, _merra_point_pe, _velocity_settings,
    compute_gnss_strain, load_reflector_heights, load_surface_elevation, load_pole_depths,
    get_compaction, firn_velocity_below_pole,
)
plt.rcParams["mathtext.fontset"] = "dejavusans"

STATIONS = [sn for sn in FIG3_STATIONS if sn != GRIP_ID]
STATION_NAMES = [name for sn, name in zip(FIG3_STATIONS, FIG3_NAMES) if sn != GRIP_ID]
MODEL_NAME = 'GSFC2020'
DENSITIES = [280, 290, 295, 300, 305, 310, 315, 320, 330, 340] #[290, 295, 300, 305, 310, 315]

# MERRA-2 P-E bias per station (fraction), applied as P-E / (1 + bias)
STATION_PERCENT_DECIMAL = {"0015": -0.113288, "0005": -0.105171, "0012": -0.098341,"0009": -0.126334,"0008": -0.088673,
    "8438": -0.156717,"0013": 0.035526,"0011": -0.056248, "0014": -0.028636,"0010": -0.091310, "0017": -0.087818,"0007": -0.067686, "7313": 0.057330
}
# STATION_PERCENT_DECIMAL = {sn: 0.0 for sn in STATION_PERCENT_DECIMAL}  # no MERRA-2 correction

# Mean vertical ice velocity [m/yr] for each Glen's n (from glen_n_spectrum(N_SPECTRUM), hardcoded since it is slow)
WS_MODELED_SPECTRUM = [-0.209852757136161, -0.21115619041029873, -0.21240415843872834,
                       -0.2136001277993068, -0.21474728208394323, -0.2158485501971942,
                       -0.21690663132561183, -0.21792401702601338, -0.21890301081319222,
                       -0.21984574557121636, -0.22075419906531227, -0.22163020779176193,
                       -0.2224754793699151, -0.22329160365226985, -0.2240800627047143,
                       -0.22484223978874396, -0.22557942746018247, -0.22629283488415525,
                       -0.2269835944533986, -0.22765276778610327, -0.2283013511701093,
                       -0.2289302805121757, -0.22954043584403117]
N_SPECTRUM = np.linspace(2.5, 4.7, len(WS_MODELED_SPECTRUM))

def _to_ns(dates):
    """dates -> int64 nanoseconds since epoch, for np.interp"""
    return np.array(dates, dtype='datetime64[ns]').astype('int64')

def _years_since_start(dates):
    dates_np = np.array(dates, dtype='datetime64[ns]')
    return (dates_np - dates_np[0]).astype('timedelta64[s]').astype(float) / (365.25*24*3600)

#### ACCUMULATION (MERRA-2) ####
def merra_cumulative_pe(ogre_dates, sn, correction=0):
    """
    Cumulative MERRA-2 P-E [mm w.e. = kg/m²] at the station, divided by (1 + correction),
    zeroed at the first OGRE date and interpolated to ogre_dates. Divide by density for snow thickness [m].
    """
    files = sorted(glob.glob(os.path.join(MERRA_DAILY_DIR, '*.nc4')))
    _, _, latitude, longitude = read_ppp(ppp_csv_path(sn))
    dates, pe = _merra_point_pe(files, float(latitude[0]), float(longitude[0]))

    # mask to ogre_dates range, cumulative sum
    mask = (dates >= ogre_dates[0]) & (dates <= ogre_dates[-1])
    cumulative_pe = _cumulative_from_zero(pe[mask] / (1 + correction))
    return np.interp(_to_ns(ogre_dates), _to_ns(dates[mask]), cumulative_pe)

#### VERTICAL VELOCITY FROM STRAIN ####
def get_strain(n_glen, use_bedmachine=False):
    """mean vertical ice velocity [m/yr] over all station triangles, and per-triangle uncertainties"""
    _, _, _, gnss = compute_gnss_strain(N=n_glen, use_bedmachine=use_bedmachine)
    return np.mean(gnss['w']), gnss['w_err']

def get_strain_at_station(n_glen, station):
    """mean vertical ice velocity [m/yr] of the triangles that include `station` (BedMachine thickness)"""
    stationTriplets, _, _, gnss = compute_gnss_strain(N=n_glen, use_bedmachine=True)
    return np.mean([w for triplet, w in zip(stationTriplets, gnss['w']) if station in triplet])

def glen_n_spectrum(n_values, station=None):
    """
    Vertical ice velocity for each Glen's n. w scales with (n+1)/(n+2) and the strain rates
    do not depend on n, so the strain is computed once (n=3) and rescaled.
    """
    w3 = get_strain(3)[0] if station is None else get_strain_at_station(3, station)
    return [w3 * ((n + 1) / (n + 2)) / (4 / 5) for n in n_values]

#### FIRN COMPACTION (CFM) ####
def firn_velocity_ogre(model_name=MODEL_NAME, sn="0005", window=90, year_conv=31556952, num_sim=1000):
    """modeled firn velocity [m/yr] below the pole base, from the OGRE-forced CFM run (as in Figure 3)"""
    window_t, position_uncertainty = _velocity_settings(sn, window)
    pole_dates, poles_depths = load_pole_depths(sn)
    firn_dates, firn_velocity = firn_velocity_below_pole(sn, pole_dates, poles_depths, window_t, model_name, year_conv, num_sim, position_uncertainty)
    if sn == '8438':
        firn_velocity = firn_velocity[:-30]; firn_dates = firn_dates[:-30]
    return firn_dates, firn_velocity

def modeled_firn_displacement(sn, density, dates_num, model_name=MODEL_NAME, poles_depths=None, pole_dates=None):
    """
    Cumulative firn compaction [m] from the MERRA-2 forced CFM run with surface density `density`,
    between the pole base (or the surface, if no pole depths given) and the bottom of the model domain,
    interpolated to dates_num (ns since epoch) and zeroed at the first of those dates.
    """
    if poles_depths is None:
        _, pole_dates, _, _ = read_ppp(ppp_csv_path(sn))
        poles_depths = np.zeros(len(pole_dates))
    bottom_depths = np.full(len(poles_depths), -1) # -1 signifies bottom of the column -- assumes ice is at bottom of domain!!!
    firn_dates, firn_compactions, _ = get_compaction(cfm_no_ogre_file(sn, model_name, density), poles_depths, pole_dates, bottom_depths)
    firn_positions = np.interp(dates_num, _to_ns(firn_dates), _cumulative_from_zero(firn_compactions))
    return firn_positions - firn_positions[0]

#### MODEL SEARCHES ####
def find_model_n(stationNum="0012", possible_n=np.arange(2, 5.0, 0.1),
                 possible_models=['GSFC2020', 'Ligtenberg2011', "Crocus", "Veldhuijsen2023", "HLdynamic", 'KuipersMunneke2015']):
    """residual spread of (elevation - accumulation - firn - ice) for each Glen's n and firn model at one station"""
    # Preload datasets that don't change with n or model_name
    dates_elevations, surface_elevations = load_surface_elevation(stationNum)
    master_dates_all = pd.to_datetime(dates_elevations); surface_elevations = np.array(surface_elevations)
    dates_acc, accumulations = load_reflector_heights(stationNum)
    accumulations = np.interp(_to_ns(master_dates_all), _to_ns(dates_acc), accumulations)  # match elevation dates

    # Firn displacement for each model, from daily velocities
    firn_cache = {}
    for model_name in possible_models:
        dates_firn, modeled_firn = firn_velocity_ogre(model_name=model_name, sn=stationNum)
        firn_cache[model_name] = (pd.to_datetime(dates_firn), np.nancumsum(np.array(modeled_firn) / 365))

    # Ice velocity for each n
    ws_values = glen_n_spectrum(possible_n)

    results = []
    for n, ws in zip(possible_n, ws_values):
        for model_name in possible_models:
            dates_firn, firn_displacements = firn_cache[model_name]

            # Clip to common date range
            mask = (master_dates_all >= dates_firn[0]) & (master_dates_all <= dates_firn[-1])
            master_dates = master_dates_all[mask]
            firn_displacements_interp = np.interp(_to_ns(master_dates), _to_ns(dates_firn), firn_displacements)

            # Normalize
            surface_elevations_norm = surface_elevations[mask] - surface_elevations[mask][0]
            accumulations_norm = -(accumulations[mask] - accumulations[mask][0])
            firn_displacements_norm = firn_displacements_interp - firn_displacements_interp[0]

            # Dynamic thinning term
            ws_scaled = np.concatenate([[0], np.cumsum(np.diff(master_dates.values).astype('timedelta64[D]').astype(float) / 365.25 * ws)])

            # Sum and residuals
            residuals = surface_elevations_norm - (firn_displacements_norm + ws_scaled + accumulations_norm)
            results.append({"n": n, "model_name": model_name, "mean_residual": np.mean(residuals), "std_residual": np.std(residuals)})
            print(model_name, np.mean(residuals), n)

    # Find best combination
    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results['std_residual'].idxmin()]
    print(f"Best n={best_row['n']}, model={best_row['model_name']}")
    print(f"Residual mean={best_row['mean_residual']:.4f}, std={best_row['std_residual']:.4f}")

    # Plot results
    fig, ax = plt.subplots()
    for model_name in possible_models:
        subset = df_results[df_results['model_name'] == model_name]
        ax.plot(subset['n'], subset['std_residual'], marker='o', label=model_name)
    ax.set_xlabel("n")
    ax.set_ylabel("Residual Std Dev (m)")
    ax.set_title(f"Residual spread vs. n for {stationNum}")
    ax.legend()
    plt.show()

def optimized_forward_model(ax, stationNum, dates_elevations, surface_elevations, ws_modeled_spectrum=WS_MODELED_SPECTRUM, n_spectrum=N_SPECTRUM,
                            densities=DENSITIES, station_percent_decimal=STATION_PERCENT_DECIMAL, MODEL_NAME=MODEL_NAME, penalty_weight=1e-3):
    """
    Forward model: MERRA-2 accumulation / rho_precip + surface-to-bottom firn compaction (rho_firn) + ice velocity (n).
    Picks the combination with the smallest residual drift, penalizing rho_precip != rho_firn.
    """
    # Observed surface elevation
    surface_elevations_translated = np.array(surface_elevations) - surface_elevations[0]
    dates_num = _to_ns(dates_elevations)
    years = _years_since_start(dates_elevations)
    dt_years = np.diff(years)

    # Precompute firn and MERRA cumulative sums for each density (MERRA read once, then scaled)
    firn_dict = {density: modeled_firn_displacement(stationNum, density, dates_num, model_name=MODEL_NAME) for density in densities}
    merra_cumulative_pe_mm = merra_cumulative_pe(dates_elevations, stationNum, correction=station_percent_decimal[stationNum])
    merra_dict = {density: merra_cumulative_pe_mm / density for density in densities}
    
    # Nested search over n and densities
    best_metric = np.inf
    all_modeled = []
    for n_val, ws_val in zip(n_spectrum, ws_modeled_spectrum):
        cum_modeled_ws = np.concatenate([[0], np.cumsum(dt_years * ws_val)])
        for rho_precip in densities:
            for rho_firn in densities:
                modeled_sum = merra_dict[rho_precip] + firn_dict[rho_firn] + cum_modeled_ws
                slope, _ = np.polyfit(years, surface_elevations_translated - modeled_sum, 1)

                # Penalty for mismatch between cum_precip and firn densities
                metric = np.abs(slope) + penalty_weight * np.abs(rho_precip - rho_firn)
                if metric < best_metric:
                    best_metric = metric; best_slope = np.abs(slope)
                    best_n = n_val; best_rho_precip = rho_precip; best_rho_firn = rho_firn
                    best_model_sum = modeled_sum
                all_modeled.append(modeled_sum)

    # Plot shaded envelope of all candidates
    all_modeled = np.vstack(all_modeled)
    ax.fill_between(dates_elevations, np.min(all_modeled, axis=0), np.max(all_modeled, axis=0), color=derek_colors['red'], alpha=0.2, zorder=1)

    print(f"-> BEST combination for station {stationNum}: n={best_n:.2f}, "
            f"rho_precip={best_rho_precip}, rho_firn={best_rho_firn}, "
            f"residual sol = {best_metric:.4f} residual = {best_slope:.03f} m/yr")
    ax.plot(dates_elevations, best_model_sum, color=derek_colors['red'], lw=2, zorder=9)

    # Plot observed and annotate best parameters
    ax.plot(dates_elevations, surface_elevations_translated, c='k', lw=3)
    textstr = f"Residuals drift: {best_slope:.3f} m/yr\n$n$: {best_n:.2f}\n$\\rho_f$: {best_rho_firn}, $\\rho_p$: {best_rho_precip}"
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=6, va='top', ha='left', color=derek_colors['red'], zorder=10)

def surface_elevation_comparisons(stations=STATIONS, names=STATION_NAMES, densities=DENSITIES, show_misfit=True):
    """USES concept of Surface Elevation = Z - GNSS_IR, such that accumulation is fixed
    and we are estimating our choice of firn model, surface snow density, and Glen's N"""
    fig, axes = plt.subplots(nrows=3, ncols=3)
    for ax, stationNum, name in zip(axes.flatten(), stations, names):
        print("Processing station: ", stationNum)

        # Observed surface elevation
        dates_elevations, surface_elevations = load_surface_elevation(stationNum)
        surface_elevations_translated = np.array(surface_elevations) - surface_elevations[0]
        dates_num = _to_ns(dates_elevations)
        years = _years_since_start(dates_elevations)
        dt_years = np.diff(years)

        # Get accumulation (reflector height decrease), interpolated to the elevation dates
        dates_accumulations, accumulations = load_reflector_heights(stationNum)
        acc = -np.array(accumulations)
        acc_interp = np.interp(dates_num, _to_ns(dates_accumulations), acc - acc[0])
        
        # Firn compaction below the pole base for each density
        pole_dates, poles_depths = load_pole_depths(stationNum)
        firn_dict = {density: modeled_firn_displacement(stationNum, density, dates_num, MODEL_NAME, poles_depths, pole_dates) for density in densities}
            
        # Nested search over n and density
        best_metric = np.inf
        misfit_matrix = np.zeros((len(densities), len(N_SPECTRUM)))
        all_solutions = []
        for n_idx, (n_val, ws_val) in enumerate(zip(N_SPECTRUM, WS_MODELED_SPECTRUM)):
            # ws_val = get_strain_at_station(n_val, stationNum)
            cum_modeled_ws = np.concatenate([[0], np.cumsum(dt_years * ws_val)])

            for d_idx, density in enumerate(densities):
                modeled_sum = acc_interp + firn_dict[density] + cum_modeled_ws
                slope, _ = np.polyfit(years, surface_elevations_translated - modeled_sum, 1)
                misfit_matrix[d_idx, n_idx] = np.abs(slope)
                # print(f"drift: {np.abs(slope):.05f}, n: {n_val}, density: {density}")

                if np.abs(slope) < best_metric:
                    best_metric = np.abs(slope); best_n = n_val; best_density = density
                    best_model_sum = modeled_sum
                all_solutions.append(modeled_sum)

        # Plot shaded envelope of all candidates, best solution and observed
        all_modeled = np.vstack(all_solutions)
        ax.fill_between(dates_elevations, np.min(all_modeled, axis=0), np.max(all_modeled, axis=0), color=derek_colors['blue'], alpha=0.4, zorder=1)
        print(f"-> BEST combination for station {stationNum}: n={best_n:.2f}, density={best_density}, residual drift={best_metric:.4f} m/yr")
        ax.plot(dates_elevations, best_model_sum, color=derek_colors['blue'], lw=2, zorder=9)
        ax.plot(dates_elevations, surface_elevations_translated, c='k', lw=3, zorder=10)
        ax.set_title(name, fontsize=8)
        ax.set_ylim(0, 0.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.tick_params(axis='x', rotation=45, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

        # Annotate best parameters
        textstr = f"Residuals drift: {best_metric:.3f} m/yr\nn: {best_n:.2f}\n$\\rho$: {best_density}"
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=6, va='top', ha='right', color='#001330', zorder=10)
        
        # FORWARD MODEL (MERRA-2 accumulation)
        optimized_forward_model(ax, stationNum, dates_elevations, surface_elevations, densities=densities)

        # SOLUTION SPACE
        if show_misfit:
            fig2, ax2 = plt.subplots()
            im = ax2.imshow(misfit_matrix, aspect='auto', extent=[N_SPECTRUM[0], N_SPECTRUM[-1], densities[0], densities[-1]],
                            origin='lower', cmap='viridis')
            fig2.colorbar(im, ax=ax2, label="Residual drift [m/yr]")
            ax2.scatter(best_n, best_density, color='red', s=50, edgecolor='k', zorder=5, label='Best fit')
            ax2.set_xlabel("Glen's n")
            ax2.set_ylabel("Density [kg/m³]")
            ax2.set_title(f"Solution space at station {name}")
            plt.show()

    custom_lines = [Line2D([0], [0], color='k', linewidth=3), Line2D([0], [0], color=derek_colors['blue']), Line2D([0], [0], color=derek_colors['red'])]
    fig.legend(custom_lines, ['Surface Elevation', 'Sum of OGRE-Model Components', 'Forward Model'], loc='upper center', ncol=3, fontsize=10) # , bbox_to_anchor=(0.5, 1.05)
    fig.supylabel('[m]', fontsize=8)
    # fig.suptitle("OGRE Ele = Acc - Vertical - Thinning, tuning firn and verticals")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(wspace=0.2, hspace=0.5) 
    plt.show()


if __name__=="__main__":
    surface_elevation_comparisons()
    # find_model_n()
    # print(glen_n_spectrum(N_SPECTRUM))  # regenerate WS_MODELED_SPECTRUM
