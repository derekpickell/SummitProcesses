"""
Firn model tests for Figure 3.

Compares the site-averaged OGRE vertical velocity with modeled firn velocities from
different Community Firn Model (CFM) runs (densification physics, conductivity, ...).
The results set FIG3_MODEL in ogre_analysis.py. All loading, velocity and firn code is
shared with ogre_analysis.

Same setup (environment variables) as ogre_analysis.py. CFM runs are read from
OGRE_CFM_DIR/<station>/<model>/output/output_firn_layers.csv (see generate_cfm.py).
"""
import numpy as np
import matplotlib.pyplot as plt

from ogre_analysis import (
    SECONDS_PER_YEAR, GRIP_ID, FIG3_STATIONS, cm,
    load_site_data, get_average, interpolate_and_average_timeseries, monte_carlo_velocity, plotVerticals,
)

# CFM runs to compare (folder names under OGRE_CFM_DIR/<station>/)
CONDUCTIVITY_MODELS = ["MaxSP_Schwander", "MaxSP_Yen_fixed", "MaxSP_Yen_var", "MaxSP_Anderson", "MaxSP_Yen_b", "MaxSP_Sturm",
                       "MaxSP_VanDusen", "MaxSP_Schwerdtfeger", "MaxSP_Riche", "MaxSP_Jiawen", "MaxSP_mix", "MaxSP_Calonne2011", "MaxSP_Calonne2019"]
PHYSICS_MODELS = ['GSFC2020', 'KuipersMunneke2015', 'Ligtenberg2011', "Crocus", "HLdynamic"] # Veldhuijsen2023
# DENSITY_MODELS = ["MaxSP_300", "MaxSP_301", "MaxSP_302", "MaxSP_303", "MaxSP_304", "MaxSP_305", "MaxSP_306", "GSFC2020"]

########### FIRN MODEL COMPARISON ###########
def compare_firn_models(model_names=CONDUCTIVITY_MODELS, window=90, year_conv=SECONDS_PER_YEAR, num_sim=1000):
    """
    Site-averaged OGRE vertical velocity (excluding GRIP) vs modeled firn velocity for each CFM run.
    Prints R², NRMSE, amplitude ratio and mean residual (ice velocity) per model, and plots the velocities.
    """
    # Load and average position/depth data, OGRE velocity computed once for all models
    site_data = load_site_data([sn for sn in FIG3_STATIONS if sn != GRIP_ID], year_conv)
    date_grid, mean_vertical, _ = interpolate_and_average_timeseries(site_data[0], site_data[1])
    ogre_velocity = monte_carlo_velocity(mean_vertical, date_grid, window, year_conv=year_conv, num_simulations=num_sim, position_uncertainty=0.01)[:2]

    fig, ax = plt.subplots(figsize=(17.9*cm, 9*cm))
    ax.plot(*ogre_velocity, color='k', lw=2, label='OGRE-Averaged Velocity', linestyle='dotted')
    cmap = plt.cm.tab20  # distinct color per model
    summary = []
    for idx, model in enumerate(model_names):
        print("Processing model: ", model)
        avg = get_average(*site_data, window, model, year_conv, num_sim, ogre_velocity=ogre_velocity)
        summary.append((model, avg['r_squared'], avg['nrmse'], avg['amp_ratio'], np.mean(avg['residuals']), np.std(avg['residuals'])))
        ax.plot(avg['firn_dates'], avg['firn_velocity'], color=cmap(idx % cmap.N), label=f"{model} Modeled Firn Velocity", lw=2)

    # PRINT summary
    print(f"\n{'Model':<22} {'R²':>6} {'NRMSE [%]':>10} {'amp ratio':>10} | {'ice velocity [m/yr]':>20}")
    for model, r2, nrmse, amp, ice, ice_std in sorted(summary, key=lambda r: -r[1]):
        print(f"{model:<22} {r2:>6.3f} {nrmse:>10.1f} {amp:>10.2f} | {ice:>12.3f} ± {ice_std:.3f}")

    ax.legend(fontsize=6, ncol=2)
    ax.set_ylabel("[m y⁻¹]", fontsize=8, fontweight='bold')
    ax.set_title(f"OGRE vs. Firn Model Velocities, {window} Day Smoothing", fontsize=8, fontweight='bold')
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    plt.show()

    return summary


if __name__=="__main__":
    ## FIRN MODEL COMPARISON -> sets FIG3_MODEL in ogre_analysis.py
    compare_firn_models(CONDUCTIVITY_MODELS)
    # compare_firn_models(PHYSICS_MODELS)

    ## FIGURE 3 LAYOUT FOR ANY MODEL
    # plotVerticals(model_name='KuipersMunneke2015')
