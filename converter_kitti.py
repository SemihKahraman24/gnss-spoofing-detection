"""
converter_kitti.py - KITTI Raw Data -> Pipeline Format Converter
================================================================
KITTI oxts klasoründen gps_*.csv ve imu_*.csv üretir.
100 Hz'e upsample ederek Istanbul IMU verileriyle uyumlu hale getirir.

Kullanim:
  python converter_kitti.py 2011_09_26_drive_0009_sync --out-prefix kitti_09
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timezone

GRAVITY = 9.80665
RAD2DEG = 180.0 / np.pi


def navstat_to_fixtype(navstat):
    if navstat >= 3:
        return 3
    elif navstat == 2:
        return 2
    else:
        return 0


def parse_kitti_timestamp(ts_str):
    ts_str = ts_str.strip()
    if '.' in ts_str:
        base, frac = ts_str.split('.')
        frac = frac[:6]
        ts_str = base + '.' + frac
    dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() * 1000.0


def find_sequence_dir(seq_dir):
    candidates = [seq_dir]
    if os.path.isdir(seq_dir):
        for sub in os.listdir(seq_dir):
            sub_path = os.path.join(seq_dir, sub)
            if os.path.isdir(sub_path):
                candidates.append(sub_path)
                for sub2 in os.listdir(sub_path):
                    sub2_path = os.path.join(sub_path, sub2)
                    if os.path.isdir(sub2_path):
                        candidates.append(sub2_path)

    for c in candidates:
        if os.path.isdir(os.path.join(c, 'oxts', 'data')):
            return c

    print("\n[Hata] oxts/data klasoru bulunamadi.")
    print("  Mevcut klasorler:")
    if os.path.isdir(seq_dir):
        for item in os.listdir(seq_dir):
            print("    " + item)
    else:
        print("  '" + seq_dir + "' klasoru yok!")
    raise FileNotFoundError("oxts/data bulunamadi.")


def load_sequence(seq_dir):
    seq_dir  = find_sequence_dir(seq_dir)
    oxts_dir = os.path.join(seq_dir, 'oxts')
    data_dir = os.path.join(oxts_dir, 'data')
    ts_file  = os.path.join(oxts_dir, 'timestamps.txt')

    print("  Bulunan yol: " + seq_dir)

    if not os.path.isfile(ts_file):
        raise FileNotFoundError("timestamps.txt bulunamadi: " + ts_file)

    with open(ts_file) as f:
        ts_lines = [l.strip() for l in f if l.strip()]
    timestamps_ms = np.array([parse_kitti_timestamp(l) for l in ts_lines])

    data_files = sorted([
        os.path.join(data_dir, fn)
        for fn in os.listdir(data_dir)
        if fn.endswith('.txt')
    ])

    if len(data_files) != len(timestamps_ms):
        n = min(len(data_files), len(timestamps_ms))
        data_files    = data_files[:n]
        timestamps_ms = timestamps_ms[:n]

    rows = []
    for fp in data_files:
        with open(fp) as f:
            vals = list(map(float, f.read().split()))
        rows.append(vals)

    data = np.array(rows)
    return timestamps_ms, data


def upsample_to_100hz(timestamps_ms, data, target_hz=100):
    """
    KITTI (~10 Hz) verisini 100 Hz'e upsample et.
    - IMU: lineer interpolasyon
    - GPS: sadece orijinal frame pozisyonlarinda, arasi NaN (Istanbul formatiyla ayni)
    """
    orig_dt_ms = np.median(np.diff(timestamps_ms))
    new_dt_ms  = 1000.0 / target_hz
    factor     = int(round(orig_dt_ms / new_dt_ms))

    n_orig = len(timestamps_ms)
    n_new  = (n_orig - 1) * factor + 1

    t0        = timestamps_ms[0]
    new_ts    = t0 + np.arange(n_new) * new_dt_ms
    orig_trel = timestamps_ms - t0
    new_trel  = new_ts - t0

    # Tum kolonlari interpolasyon ile doldur (IMU icin)
    new_data = np.zeros((n_new, data.shape[1]))
    for col in range(data.shape[1]):
        new_data[:, col] = np.interp(new_trel, orig_trel, data[:, col])

    # GPS: sadece orijinal frame pozisyonlarinda gercek deger, digerleri NaN
    gps_valid = np.full((n_new, data.shape[1]), np.nan)
    gps_mask  = np.zeros(n_new, dtype=bool)
    for i in range(n_orig):
        idx = i * factor
        gps_valid[idx, :] = data[i, :]
        gps_mask[idx]     = True

    return new_ts, new_data, gps_valid, gps_mask, factor


def build_gps_csv(timestamps_ms, gps_valid, gps_mask, out_path):
    """
    Cikti kolonlari (pipeline'in beklediği format):
    pc_time, ts_ms, lat, lon, height_mm, velN, velE, velD, fixType, numSV, counter
    """
    ts_ms_rel = timestamps_ms - timestamps_ms[0]

    gps_rows = gps_valid[gps_mask]
    gps_ts   = ts_ms_rel[gps_mask]

    lat       = gps_rows[:, 0]
    lon       = gps_rows[:, 1]
    height_mm = (gps_rows[:, 2] * 1000).astype(int)
    velN      = (gps_rows[:, 6]  * 1000).round(1)
    velE      = (gps_rows[:, 7]  * 1000).round(1)
    velD      = (-gps_rows[:, 10] * 1000).round(1)
    navstat   = gps_rows[:, 25].astype(int)
    numsats   = gps_rows[:, 26].astype(int)
    fixtype   = np.array([navstat_to_fixtype(s) for s in navstat])

    base_unix = timestamps_ms[0] / 1000.0
    pc_times  = [
        datetime.utcfromtimestamp(base_unix + gps_ts[i] / 1000.0)
                .strftime('%Y-%m-%dT%H:%M:%S.%f')
        for i in range(len(gps_ts))
    ]

    df = pd.DataFrame({
        'pc_time'   : pc_times,
        'ts_ms'     : gps_ts.astype(int),
        'lat'       : lat,
        'lon'       : lon,
        'height_mm' : height_mm,
        'velN'      : velN,
        'velE'      : velE,
        'velD'      : velD,
        'fixType'   : fixtype,
        'numSV'     : numsats,
        'counter'   : np.arange(len(lat)),
    })

    df.to_csv(out_path, index=False, float_format='%.7f')
    print("  GPS  -> " + out_path + "  (" + str(len(df)) + " satir)")

    good  = (fixtype >= 2).sum()
    speed = np.sqrt(velN**2 + velE**2) / 1000
    print("         fixType>=2: " + str(good) + "/" + str(len(df)))
    print("         Bolge: lat=" + str(round(lat.mean(), 4)) + "  lon=" + str(round(lon.mean(), 4)))
    print("         Hiz ort/maks: " + str(round(speed.mean(), 1)) + " / " + str(round(speed.max(), 1)) + " m/s")
    return df


def build_imu_csv(timestamps_ms, new_data, out_path):
    """
    Cikti kolonlari (pipeline'in beklediği format):
    pc_time, ts_ms, gyroX[deg/s], gyroY, gyroZ, accelX[g], accelY, accelZ
    """
    ts_ms_rel = timestamps_ms - timestamps_ms[0]

    accelX = new_data[:, 11] / GRAVITY
    accelY = new_data[:, 12] / GRAVITY
    accelZ = new_data[:, 13] / GRAVITY
    gyroX  = new_data[:, 17] * RAD2DEG
    gyroY  = new_data[:, 18] * RAD2DEG
    gyroZ  = new_data[:, 19] * RAD2DEG

    base_unix = timestamps_ms[0] / 1000.0
    pc_times  = [
        datetime.utcfromtimestamp(base_unix + ts_ms_rel[i] / 1000.0)
                .strftime('%Y-%m-%dT%H:%M:%S.%f')
        for i in range(len(ts_ms_rel))
    ]

    df = pd.DataFrame({
        'pc_time': pc_times,
        'ts_ms'  : ts_ms_rel.astype(int),
        'gyroX'  : gyroX,
        'gyroY'  : gyroY,
        'gyroZ'  : gyroZ,
        'accelX' : accelX,
        'accelY' : accelY,
        'accelZ' : accelZ,
    })

    df.to_csv(out_path, index=False, float_format='%.6f')
    print("  IMU  -> " + out_path + "  (" + str(len(df)) + " satir)")

    az_mean = float(np.mean(accelZ))
    g_check = 'OK' if 0.7 < abs(az_mean) < 1.3 else 'Kontrol et'
    dt_ms   = np.diff(ts_ms_rel)
    hz      = 1000.0 / np.mean(dt_ms) if len(dt_ms) > 0 else 0
    print("         Frekans: " + str(round(hz, 1)) + " Hz")
    print("         accelZ ort: " + str(round(az_mean, 3)) + " g (" + g_check + ")")
    return df


def main():
    parser = argparse.ArgumentParser(
        description='KITTI Raw -> pipeline GPS/IMU CSV converter'
    )
    parser.add_argument(
        'seq_dir',
        help='KITTI sequence klasoru (orn: 2011_09_26_drive_0001_sync)'
    )
    parser.add_argument(
        '--out-prefix', default='kitti',
        help='Cikti dosya oneki (default: kitti)'
    )
    args = parser.parse_args()

    seq_dir = args.seq_dir.rstrip('/\\')
    seq_name = os.path.basename(seq_dir)
    gps_out = "gps_" + args.out_prefix + ".csv"
    imu_out = "imu_" + args.out_prefix + ".csv"

    print("\n[KITTI Converter]")
    print("  Sequence : " + seq_name)

    print("\n[1] oxts verisi okunuyor...")
    timestamps_ms, data = load_sequence(seq_dir)

    n     = len(timestamps_ms)
    dur_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    print("  " + str(n) + " frame  |  " + str(round(dur_s, 1)) + "s")

    print("\n[2] 100 Hz'e upsample ediliyor...")
    new_ts, new_data, gps_valid, gps_mask, factor = upsample_to_100hz(timestamps_ms, data)
    print("  " + str(n) + " frame x" + str(factor) + " = " + str(len(new_ts)) + " frame @ 100 Hz")

    print("\n[3] GPS CSV olusturuluyor...")
    build_gps_csv(new_ts, gps_valid, gps_mask, gps_out)

    print("\n[4] IMU CSV olusturuluyor...")
    build_imu_csv(new_ts, new_data, imu_out)

    print("\n" + "=" * 50)
    print("TAMAMLANDI -> " + imu_out + "  " + gps_out)
    print("=" * 50)


if __name__ == '__main__':
    main()
