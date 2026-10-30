"""
Satellite velocity product tests for Figure 2.

Compares each product in ogre_analysis.VELOCITY_PRODUCTS to the OGRE stations
(speed bias, flow direction, strain-derived vertical velocity) and finds the smoothing
width that best matches the stations. The results set FIG2_PRODUCT and FIG2_SMOOTHING_PX
in ogre_analysis.py. All loading, smoothing and strain code is shared with ogre_analysis.

Same setup (environment variables) as ogre_analysis.py.
"""
import os
import numpy as np
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import sobel
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ogre_analysis import (
    BASE_DIR, SECONDS_PER_YEAR, GRIP_ID, VELOCITY_PRODUCTS, cm, derek_colors,
    load_velocity_product, smooth_velocity, velocity_magnitude, azimuth_difference,
    get_ogre_station_velocities, get_biases_from_stations, get_satellite_vectors,
    compute_gnss_strain, compute_satellite_strain, get_dem_from_geopackage,
    plotStrain, FIG2_STRAIN_SMOOTHING_PX,
)

########### PRODUCT COMPARISON ###########
def smoothing_search(ds, stations, pixel_widths=range(1, 100)):
    """
    Spread (std) of satellite - OGRE speed biases for each uniform smoothing width [pixels].
    GRIP is excluded since it sits off the divide.

    Returns:
        best pixel width (smallest spread), spreads (array, one per width)
    """
    tuning_stations = {sn: d for sn, d in stations.items() if sn != GRIP_ID}  # USE ALL BUT GRIP STATION
    spreads = np.array([np.std(get_biases_from_stations(velocity_magnitude(smooth_velocity(ds, px)), tuning_stations)[1])
                        for px in pixel_widths])
    return list(pixel_widths)[int(np.argmin(spreads))], spreads

def compare_products(products=list(VELOCITY_PRODUCTS), pixel_widths=range(1, 100)):
    """
    For each velocity product: station speed bias (raw and best-smoothed), flow direction
    error, and difference between OGRE and satellite strain-derived vertical velocity.
    Prints a summary table and plots bias spread vs smoothing wavelength.
    """
    stations = get_ogre_station_velocities()
    stationTriplets, _, station_cache, gnss = compute_gnss_strain()

    fig, ax = plt.subplots(figsize=(17.9*cm, 8*cm))
    summary = []
    for product in products:
        ds = load_velocity_product(product)
        pixel_size = abs(ds.x.values[1] - ds.x.values[0])

        # SPEED BIAS, before and after smoothing
        _, raw_biases = get_biases_from_stations(velocity_magnitude(ds), stations)
        best_px, spreads = smoothing_search(ds, stations, pixel_widths)
        ds_smoothed = smooth_velocity(ds, best_px)
        _, biases = get_biases_from_stations(velocity_magnitude(ds_smoothed), stations)

        # FLOW DIRECTION (smoothed product, excluding GRIP)
        sat_vels = get_satellite_vectors(ds_smoothed, stations)
        angles = [azimuth_difference(d['vel'], v) for (sn, d), v in zip(stations.items(), sat_vels) if sn != GRIP_ID]

        # VERTICAL VELOCITY from strain on the same triangles
        _, _, sat_ws = compute_satellite_strain(smooth_velocity(ds, FIG2_STRAIN_SMOOTHING_PX), stationTriplets, station_cache)
        w_diff = np.array(gnss['w']) - sat_ws

        summary.append(dict(product=product, best_px=best_px, wavelength=best_px * pixel_size,
                            raw_bias=np.mean(raw_biases), raw_spread=np.std(raw_biases),
                            bias=np.mean(biases), spread=np.std(biases),
                            angle=np.mean(angles), angle_std=np.std(angles),
                            w_diff=np.nanmean(w_diff), w_diff_std=np.nanstd(w_diff)))

        # PLOT spread vs smoothing wavelength
        wavelengths = np.array(list(pixel_widths)) * pixel_size
        ax.plot(wavelengths, spreads, label=VELOCITY_PRODUCTS[product])
        ax.scatter(best_px * pixel_size, spreads.min(), color='k', s=10, zorder=3)

    # PRINT summary (speeds in m/yr, angles in degrees)
    print(f"\n{'Product':<22} {'px':>4} {'λ [m]':>7} | {'raw bias':>9} {'±':>6} | {'bias':>7} {'±':>6} | {'angle':>6} {'±':>5} | {'ΔW (OGRE-sat)':>13} {'±':>6}")
    for r in summary:
        print(f"{r['product']:<22} {r['best_px']:>4} {r['wavelength']:>7.0f} | {r['raw_bias']:>9.3f} {r['raw_spread']:>6.3f} | "
              f"{r['bias']:>7.3f} {r['spread']:>6.3f} | {r['angle']:>6.1f} {r['angle_std']:>5.1f} | {r['w_diff']:>13.3f} {r['w_diff_std']:>6.3f}")

    ax.set_xlabel("Smoothing wavelength [m]", fontsize=8, fontweight='bold')
    ax.set_ylabel("Spread of station speed biases [m y⁻¹]", fontsize=8, fontweight='bold')
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=.3)
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()

    return summary

########### STRAIN MAPS ###########
def plot_strain_maps(product='measures_annual_mean', smoothing=2, fields=('magnitude', 'vertical_velocity'), N=3, thickness=3200):
    """
    Strain-rate fields from Sobel gradients of a smoothed satellite product, with OGRE stations (EPSG:3413).
    fields: any of 'magnitude', 'strain_rate', 'max_principal', 'vertical_velocity', 'along_flow'
    thickness: constant ice thickness [m] for vertical velocity (BedMachine not used here)
    """
    to_3413 = Transformer.from_crs("EPSG:32624", "EPSG:3413", always_xy=True)
    station_xy = np.array([to_3413.transform(*d['pos']) for d in get_ogre_station_velocities().values()])

    ds = smooth_velocity(load_velocity_product(product, bbox_latlon=[[72, -39], [74, -35]]), smoothing)
    # bbox_latlon=[[70, -45], [75, -32]] or [[66, -45], [85, -30]] for larger areas
    vx, vy = ds.vx.values, ds.vy.values

    # GRADIENTS with Sobel filter. Signed spacing so d/dy is correct for north-up rasters (y decreasing with row)
    # NOTE: sobel returns ~8x the centered difference, so scale = 100 is an empirical normalization
    scale = 100
    x_spacing = ds.x.values[1] - ds.x.values[0]
    y_spacing = ds.y.values[1] - ds.y.values[0]
    dvx_dx = sobel(vx, axis=1) / (scale * x_spacing)
    dvx_dy = sobel(vx, axis=0) / (scale * y_spacing)
    dvy_dx = sobel(vy, axis=1) / (scale * x_spacing)
    dvy_dy = sobel(vy, axis=0) / (scale * y_spacing)

    # STRAIN RATE TENSOR components and flow direction
    exx = dvx_dx; eyy = dvy_dy; exy = 0.5 * (dvx_dy + dvy_dx)
    speed = np.sqrt(vx**2 + vy**2)
    ux = np.nan_to_num(vx / speed); uy = np.nan_to_num(vy / speed)  # avoid division by zero

    # field: (values, cmap, vmin, vmax, label)
    maps = {
        'magnitude':         (speed, 'plasma', None, None, "Smoothed Velocity Magnitude [m/yr]"),
        'strain_rate':       (np.sqrt((dvx_dx - dvy_dy)**2 + (dvy_dx + dvx_dy)**2), 'viridis', .0001, .002, "Strain rate [1/yr]"),
        'max_principal':     (0.5 * (exx + eyy) + np.sqrt((0.5 * (exx - eyy))**2 + exy**2), 'magma', .00002, .00015, "Max principal strain rate [1/yr]"),
        'vertical_velocity': (-(exx + eyy) * (N + 1) / (N + 2) * thickness, 'magma', -.4, 0, "vertical velocity estimate [m/yr]"),
        'along_flow':        (exx * ux**2 + 2 * exy * ux * uy + eyy * uy**2, 'coolwarm', -.001, .001, "Along-flow Strain Rate [1/yr]"),
    }

    X, Y = np.meshgrid(ds.x.values, ds.y.values)
    for field in fields:
        values, cmap, vmin, vmax, label = maps[field]
        fig, ax = plt.subplots()
        im = ax.pcolormesh(X, Y, values, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=label)
        ax.scatter(station_xy[:, 0], station_xy[:, 1], c='cyan', edgecolors='black', s=40, label='GNSS Stations')
        ax.set_title(f"{label.split(' [')[0]} ({VELOCITY_PRODUCTS[product]})", fontsize=8)
        ax.set_xlabel("X (EPSG:3413)")
        ax.set_ylabel("Y (EPSG:3413)")
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

########### SYNTHETIC TRIANGLES ###########
def scale_velocity_magnitude(ds, correction):
    """Subtract a constant speed bias [m/yr] while keeping flow direction."""
    v_mag = velocity_magnitude(ds)
    return ds * xr.where(v_mag > 0, (v_mag - correction) / v_mag, 0.0).clip(min=0)

def synthetic_triangle_vv(product='measures_multiyear', smoothing=2, satellite_correction=0, thickness=3200):
    """
    Satellite strain-derived vertical velocity on a zigzag of synthetic stations (1.5 km spacing,
    3 km apart in northing) across the ice sheet, to see the regional pattern beyond the OGRE array.
    """
    n = 279  # 279 from 400000 m; earlier versions: 70 from 500000 m, 219 from 400000 m
    eastings = 400000 + 1500 * np.arange(n)
    northings = np.where(np.arange(n) % 2 == 0, 8056405, 8059405)
    station_cache = {k: {'pos': (e, nn), 'ele': 0.0} for k, (e, nn) in enumerate(zip(eastings, northings))}
    stationTriplets = [[j, j + 1, j + 2] for j in range(0, n - 2, 2)]

    ds = load_velocity_product(product, bbox_latlon=[[65, -50], [80, -25]])
    ds = scale_velocity_magnitude(smooth_velocity(ds, smoothing), satellite_correction)
    _, _, ws = compute_satellite_strain(ds, stationTriplets, station_cache, thickness=thickness)

    x_centroids = [np.mean([station_cache[sn]['pos'][0] for sn in t]) for t in stationTriplets]
    fig, ax = plt.subplots()
    ax.scatter(x_centroids, ws, color='black', alpha=0.3)
    ax.set_xlabel("Easting [m]"); ax.set_ylabel("Vertical Velocity [m yr⁻¹]")
    ax.set_title(f"Synthetic triangles, {VELOCITY_PRODUCTS[product]}, H = {thickness} m", fontsize=8)
    ax.grid(True, alpha=.3)
    plt.show()

########### ANIMATION ###########
def animate_velocities(output_file=os.path.join(BASE_DIR, 'velocity_animation.gif')):
    """particles advected from each OGRE station at its measured velocity, over the DEM"""
    fig, ax = plt.subplots(figsize=(12,5))
    demx, demy, demelev = get_dem_from_geopackage()
    step = 50  # plot every 50th point
    xmin, xmax = 495000, 551000
    ymin, ymax = 8050000, 8064000
    mask = (demx >= xmin) & (demx <= xmax) & (demy >= ymin) & (demy <= ymax)
    im = ax.scatter(demx[mask][::step], demy[mask][::step], c=demelev[mask][::step], cmap='gray', s=0.5, alpha=.8)
    ax.set_aspect('equal')

    # Station positions [m] and velocities [m/yr]
    station_list = [{'pos': np.array(d['pos']), 'vel': np.array(d['vel']) * SECONDS_PER_YEAR} for d in get_ogre_station_velocities().values()]
    for s in station_list:
        s['speed'] = np.linalg.norm(s['vel'])
        ax.scatter(*s['pos'], c=derek_colors['red'], s=30, zorder=3)

    # Parameters
    dt = 100  # years per frame
    max_distance = 12000  # max distance in meters from origin
    desired_spacing = 1000  # desired spatial spacing between particles in meters

    # Spawn intervals (in frames) for each station, avoid division by zero
    spawn_intervals = [max(1, int(round(desired_spacing / (s['speed'] * dt)))) for s in station_list]
    last_spawn_frame = [-1000000] * len(station_list)  # initialize far in past

    # Particle data
    particle_positions = []; particle_velocities = []; particle_origin_idx = []
    particle_scatter = ax.scatter([], [], c=derek_colors['blue'], s=5)

    def update(frame):
        nonlocal particle_positions, particle_velocities, particle_origin_idx

        # Spawn new particles per station according to their spawn interval
        for idx, s in enumerate(station_list):
            if (frame - last_spawn_frame[idx]) >= spawn_intervals[idx]:
                particle_positions.append(s['pos'].copy())
                particle_velocities.append(s['vel'].copy())
                particle_origin_idx.append(idx)
                last_spawn_frame[idx] = frame

        # Move all particles forward, remove particles too far from origin
        particle_positions = [p + v * dt for p, v in zip(particle_positions, particle_velocities)]
        keep = [i for i, p in enumerate(particle_positions) if np.linalg.norm(p - station_list[particle_origin_idx[i]]['pos']) <= max_distance]
        particle_positions = [particle_positions[i] for i in keep]
        particle_velocities = [particle_velocities[i] for i in keep]
        particle_origin_idx = [particle_origin_idx[i] for i in keep]

        particle_scatter.set_offsets(np.vstack(particle_positions) if particle_positions else np.empty((0, 2)))
        return particle_scatter,

    # plot Summit
    ax.scatter([518030], [8053781], c=derek_colors['purple'], s=18, marker="^")
    ax.annotate("Summit Station/GISP2", xy=(518030, 8051950), ha='right', fontsize=9, font='Avenir', path_effects=[path_effects.Stroke(linewidth=1.5, foreground='white'),
                                  path_effects.Normal()])
    # ax.scatter([518030], [8043781], c=derek_colors['purple'], s=5, marker="^")
    # ax.annotate("GRIP", xy=(518030, 8043781), ha='right', fontsize=9, font='Avenir')

    # plot latitude and longitude lines
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32624", always_xy=True)
    for lon in [-38, -38.6]:
        lat_range = np.linspace(72.1, 72.8, 100)
        ax.plot(*to_utm.transform(np.full_like(lat_range, lon), lat_range), color='k', linestyle='--', linewidth=1.5, alpha=0.8)
    for lat in [72.59, 72.65]:
        lon_range = np.linspace(-37, -40, 100)
        ax.plot(*to_utm.transform(lon_range, np.full_like(lon_range, lat)), color='k', linestyle='--', linewidth=1, alpha=0.8)

    # PLOTTING JUNK
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cb_ax = inset_axes(ax, width="2%", height="20%", loc='lower left', bbox_to_anchor=(0.01, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label("Elevation [m]", font='Avenir', fontsize=9, path_effects=[path_effects.Stroke(linewidth=1.5, foreground='white'),path_effects.Normal()])
    cbar.ax.tick_params(labelsize=6)
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])  # white border

    for spine in ax.spines.values():
        spine.set_linestyle('dotted')

    # Major ticks every 10 km, minor every 1 km
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(mticker.MultipleLocator(10_000))
        axis.set_minor_locator(mticker.MultipleLocator(1_000))
        axis.set_ticks_position('both')
    ax.tick_params(axis='both', which='both', direction='in', labelbottom=False, labelleft=False)

    # 5 km bar
    ax.plot([551000-7000, 551000-2000], [8050000+1500, 8050000+1500], c='k', lw=5)
    ax.annotate("5 km", xy=(546500, 8051500+700), ha='center', fontsize=7, font='Avenir')
    ani = animation.FuncAnimation(fig, update, frames=200, interval=400, blit=False)
    ani.save(output_file, writer='pillow', fps=30, dpi=200)

    plt.show()


if __name__=="__main__":
    ## PRODUCT COMPARISON -> sets FIG2_PRODUCT and FIG2_SMOOTHING_PX in ogre_analysis.py
    compare_products()

    ## FIGURE 2 LAYOUT FOR ANY PRODUCT (e.g. with the smoothing width from compare_products)
    # plotStrain(product='measures_multiyear', smoothing_px=12)

    ## STRAIN MAPS
    # plot_strain_maps(product='measures_annual_mean', fields=('magnitude', 'vertical_velocity'))

    ## SYNTHETIC TRIANGLES ACROSS THE ICE SHEET
    # synthetic_triangle_vv()

    ## ANIMATION
    # animate_velocities()
