"""
Input formats:
GNSS-IR results (gnssrefl `gnssir` output), one .txt per day
PPP time series (.csv), one per RINEX file
PPP summary (.sum), one per RINEX file
"""
import glob
import warnings
from datetime import datetime, timedelta

import numpy as np

derek_colors = {"blue": '#3867B1', "black": '#292930', "red": '#BE3445', "purple": '#4a0e82',
                "light_blue": '#1f8db5', "yellow": '#ffe138', "orange": "#B55E1F"}

POLE_LENGTH = 3.048  # [m] OGRE antenna pole length (10 ft)
MATCH_TOLERANCE_S = 43200  # [s] max time offset when pairing a GNSS-IR day with a PPP solution


def _find_files(path):
    """sorted file names matching a glob pattern; raises if nothing matches"""
    file_names = sorted(glob.glob(path))
    if not file_names:
        raise FileNotFoundError(f"No files found matching {path}. Check REFL_CODE and the directory layout.")
    return file_names


########### GNSS-IR ###########
def clump_by_date(datetimes, heights, azimuths, sat_num, frequency, utc):
    """When a 24HR RINEX file spans 2 calendar days, this produces 2 SNR file solutions. However, we still want to consider 24 hour data,
    So this code clumps adjacent days into a singular day and averages the data.
    For non-24 hour RINEX files, so long as they are > 14 hours apart, they will not be clumped
    """
    clumped_heights = []; clumped_datetimes = []; clumped_azimuths = []
    clumped_sat_num = []; clumped_frequency = []; clumped_utc = []

    i = 0
    while i < len(datetimes)-1:
        if (datetimes[i+1] - datetimes[i]).total_seconds()//3600 < 14:
            clumped_heights.append(np.concatenate((heights[i], heights[i+1])))
            clumped_azimuths.append(np.concatenate((azimuths[i], azimuths[i+1])))
            clumped_sat_num.append(np.concatenate((sat_num[i], sat_num[i+1])))
            clumped_frequency.append(np.concatenate((frequency[i], frequency[i+1])))
            clumped_utc.append(np.concatenate((utc[i], utc[i+1])))
            start_hour = int(min(utc[i])); end_hour = int(max(utc[i+1]))
            average_date = datetimes[i].replace(hour=start_hour) + (datetimes[i+1].replace(hour=end_hour) - datetimes[i].replace(hour=start_hour)) / 2
            clumped_datetimes.append(average_date)
            i += 2
        else:
            clumped_heights.append(heights[i])
            clumped_azimuths.append(azimuths[i])
            clumped_datetimes.append(datetimes[i])
            clumped_sat_num.append(sat_num[i])
            clumped_frequency.append(frequency[i])
            clumped_utc.append(utc[i])
            i += 1
        if i+1 == len(datetimes):
            clumped_heights.append(heights[i])
            clumped_azimuths.append(azimuths[i])
            clumped_datetimes.append(datetimes[i])
            clumped_sat_num.append(sat_num[i])
            clumped_frequency.append(frequency[i])
            clumped_utc.append(utc[i])

    return clumped_datetimes, clumped_heights, clumped_azimuths, clumped_sat_num, clumped_frequency, clumped_utc


def read_gnssir_single(station, path, scale_averages=True):
    """GNSS-IR reflector heights from gnssrefl result .txt files (glob pattern `path`) for a SINGLE station.
    NOTE: this code clumps data into 24hr groups that may span 2 calendar days.

    Returns: averages, datetimes, stdev, heights, azimuths, sat_num, frequency, utc (one entry per clumped day)
    scale_averages: if True, averages are flipped and zeroed so they increase from 0 as snow accumulates
    """
    heights = []; datetimes = []
    azimuths = []; sat_num = []; frequency = []; utc = []

    for file_name in _find_files(path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = np.genfromtxt(file_name, skip_header=6, dtype=float, usecols=[0, 1, 2, 3, 4, 5, 10])

        if len(data) <= 2:  # day too short
            continue

        if data.ndim != 2:
            print("not 2D:", data.shape, "in", file_name)
            continue

        date = str(int(data[-1][0])) + " " + str(int(data[-1][1])).zfill(3)
        data = data.T
        heights.append(data[2]); azimuths.append(data[5])
        sat_num.append(data[3]); frequency.append(data[6]); utc.append(data[4])
        first_hr = int(min(data[4])); last_hr = int(max(data[4]))
        datetimes.append(datetime.strptime(f'{date} {int((first_hr + last_hr) / 2)}', '%Y %j %H'))

    datetimes_c, heights_c, azimuths_c, sat_num_c, frequency_c, utc_c = clump_by_date(datetimes, heights, azimuths, sat_num, frequency, utc)

    stdev = [np.std(x) for x in heights_c]
    averages = [np.mean(x) for x in heights_c]

    if scale_averages:
        max_average = max(averages)
        averages = [-(x - max_average) for x in averages]  # start at 0

    return averages, datetimes_c, stdev, heights_c, azimuths_c, sat_num_c, frequency_c, utc_c


########### PPP ###########
def parse_fractional_datetime(hour_str, day_of_year, year):
    """datetime from fractional UTC hour, day of year and year (used by read_ppp)"""
    hour_float = float(hour_str)
    whole_hour = int(hour_float)
    total_seconds = (hour_float - whole_hour) * 3600
    dt = datetime.strptime(f"{whole_hour} {day_of_year} {year}", '%H %j %Y')
    return dt + timedelta(minutes=int(total_seconds // 60), seconds=int(total_seconds % 60))


def read_ppp(path):
    """PPP .csv files (glob pattern `path`), one solution per file.
    Returns vertical_positions [m] (final epoch), midpoint datetimes, latitudes, longitudes (as strings), sorted by date.
    """
    vertical_positions = []; datetimes = []
    latitude = []; longitude = []

    for file_name in _find_files(path):
        data = np.genfromtxt(file_name, skip_header=1, dtype=str, usecols=[0, 1, 2, 3, 4, 5], delimiter=',')
        if data.ndim == 1:  # a single epoch
            data = data[np.newaxis, :]
        if data.size == 0:
            print("Issue reading this file:", file_name)
            continue
        first = data[0]; last = data[-1]

        first_datetime = parse_fractional_datetime(first[3], first[4], first[5])
        second_datetime = parse_fractional_datetime(last[3], last[4], last[5])
        datetimes.append(first_datetime + (second_datetime - first_datetime) / 2)
        vertical_positions.append(last[2]); longitude.append(last[1]); latitude.append(last[0])

    combined = sorted(zip(datetimes, vertical_positions, longitude, latitude))
    datetimes, vertical_positions, longitude, latitude = (list(x) for x in zip(*combined))
    vertical_positions = [float(x) for x in vertical_positions]

    return vertical_positions, datetimes, latitude, longitude


def read_ppp_ne(path):
    """PPP .sum files (glob pattern `path`), one solution per file.
    Returns midpoint datetimes, UTM northings [m], UTM eastings [m], sorted by date.
    """
    datetimes = []; northings = []; eastings = []

    for file_name in _find_files(path):
        beg_time = end_time = None
        with open(file_name, 'r') as f:
            for line in f:
                if line.startswith('PRJ'):
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[1] == 'UTM':
                        eastings.append(float(parts[3])); northings.append(float(parts[4]))
                elif line.startswith("BEG"):
                    beg_time = datetime.strptime(line.strip().split(maxsplit=1)[1], "%Y-%m-%d %H:%M:%S.%f")
                elif line.startswith("END"):
                    end_time = datetime.strptime(line.strip().split(maxsplit=1)[1], "%Y-%m-%d %H:%M:%S.%f")

        if beg_time is None or end_time is None:
            raise ValueError(f"No BEG/END lines in {file_name}")
        datetimes.append(beg_time + (end_time - beg_time) / 2)

    combined = sorted(zip(datetimes, northings, eastings))
    datetimes, northings, eastings = (list(x) for x in zip(*combined))

    return datetimes, northings, eastings


########### COMBINED GNSS-IR + PPP ###########
def _match_ppp(datetimes_ir, datetimes_ppp):
    """for each GNSS-IR date, index of the first PPP solution within MATCH_TOLERANCE_S (or None)"""
    matches = []
    for dt1 in datetimes_ir:
        matches.append(next((idx for idx, dt2 in enumerate(datetimes_ppp)
                             if abs((dt2 - dt1).total_seconds()) <= MATCH_TOLERANCE_S), None))
    return matches


def get_elevation(path1, path2, stationNum):
    """INPUTS: path1 - GNSS-IR results glob, path2 - PPP .csv glob
    OUTPUTS: surface elevations (PPP height - reflector height), datetimes, reflector height SDs, latitudes, longitudes"""
    averages, datetimes, stdevs, *_ = read_gnssir_single(stationNum, path1, scale_averages=False)
    vertical_positions, datetimes2, latitude, longitude = read_ppp(path2)

    datetimes_final = []; elevations = []; stdevs_final = []; lats = []; lons = []
    for idy, idx in enumerate(_match_ppp(datetimes, datetimes2)):
        if idx is None:
            print(f"Warning: Could not find a PPP vertical for every reflectometry file. Check your data. {datetimes[idy]}")
            continue
        elevations.append(np.array(vertical_positions[idx], dtype=float) - np.array(averages[idy], dtype=float))
        datetimes_final.append(datetimes[idy])
        stdevs_final.append(stdevs[idy])
        lats.append(latitude[idx]); lons.append(longitude[idx])

    return elevations, datetimes_final, stdevs_final, lats, lons


def get_pole_column(path1, path2, stationNum):
    """INPUTS: path1 - GNSS-IR results glob, path2 - PPP .csv glob
    OUTPUTS: surface elevations, datetimes of matching dates, and pole length below surface [m] (not geodetic)"""
    averages, datetimes, *_ = read_gnssir_single(stationNum, path1, scale_averages=False)
    vertical_positions, datetimes2, _, _ = read_ppp(path2)

    if len(averages) != len(vertical_positions):
        print("Warning: Could not find a PPP vertical for every reflectometry file. Check your data.")

    datetimes_final = []; elevations = []; poles = []
    for idy, idx in enumerate(_match_ppp(datetimes, datetimes2)):
        if idx is None:
            continue
        elevations.append(np.array(vertical_positions[idx], dtype=float) - np.array(averages[idy], dtype=float))
        datetimes_final.append(datetimes[idy])
        poles.append(POLE_LENGTH - np.array(averages[idy], dtype=float))

    return elevations, datetimes_final, poles
