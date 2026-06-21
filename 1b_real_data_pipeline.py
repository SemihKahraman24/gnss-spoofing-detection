"""
1b_real_data_pipeline.py - Gercek STM32 Sensor Verisi Pipeline
===============================================================
Gercek IMU ve GPS CSV lerini okur, birim donusumu yapar,
yazilimsal spoofing uygulayarak egitim/test dataseti uretir.

Cikti:
  data/dataset_real_train.csv  -> 4_train.py ile egit
  data/dataset_real_test.csv   -> 5_test.py ile test et

Girdi CSV Formatlari:
  IMU: pc_time, ts_ms, gyroX[deg/s], gyroY, gyroZ, accelX[g], accelY, accelZ
  GPS: pc_time, ts_ms, lat[deg], lon[deg], height_mm,
       velN[mm/s], velE, velD, fixType, numSV, counter

Kullanim:
  python 1b_real_data_pipeline.py imu.csv gps.csv
  python 1b_real_data_pipeline.py imu.csv gps.csv --imu2 imu2.csv --gps2 gps2.csv
"""

import numpy as np
import pandas as pd
import os
import argparse

GRAVITY      = 9.80665
DEG2RAD      = np.pi / 180.0
EARTH_R      = 6378137.0
IMU_HZ       = 100
CHUNK_SEC    = 150          # Her trajectory icin kac saniye kullanilacak
N_TRAIN      = 60
N_TEST       = 8


# ──────────────────────────────────────────────────────────────
# 1. OKUMA & BIRIM DONUSUMU
# ──────────────────────────────────────────────────────────────
def load_imu(csv_path):
    df  = pd.read_csv(csv_path)
    out = pd.DataFrame()
    out['timestamp'] = (df['ts_ms'] - df['ts_ms'].iloc[0]) / 1000.0
    out['ax'] = df['accelX'] * GRAVITY
    out['ay'] = df['accelY'] * GRAVITY
    out['az'] = df['accelZ'] * GRAVITY
    out['gx'] = df['gyroX'] * DEG2RAD
    out['gy'] = df['gyroY'] * DEG2RAD
    out['gz'] = df['gyroZ'] * DEG2RAD

    # Koordinat sistemi normalizasyonu:
    # KITTI: az ~ +9.8 m/s2 (Z asagi, NED)
    # Diger IMU (STM32 vb.): az ~ -9.8 m/s2 (Z yukari, ENU)
    # ENU verisini NED'e cevir: az ve gz isaretini tersine cevir
    az_mean = out['az'].mean()
    if az_mean < 0:  # ENU -> NED
        out['az'] = -out['az']
        out['gz'] = -out['gz']

    return out.reset_index(drop=True)


def load_gps(csv_path, imu_t0_ms=0.0):
    df      = pd.read_csv(csv_path)
    ref_lat = df['lat'].iloc[0]
    ref_lon = df['lon'].iloc[0]
    lat_rad = ref_lat * DEG2RAD

    out = pd.DataFrame()
    out['timestamp'] = (df['ts_ms'] - imu_t0_ms) / 1000.0
    out['gps_pn']    = (df['lat'] - ref_lat) * DEG2RAD * EARTH_R
    out['gps_pe']    = (df['lon'] - ref_lon) * DEG2RAD * EARTH_R * np.cos(lat_rad)
    out['gps_vn']    = df['velN'] / 1000.0
    out['gps_ve']    = df['velE'] / 1000.0

    if 'fixType' in df.columns:
        bad = (df['fixType'] < 2).values
        out.loc[bad, ['gps_pn','gps_pe','gps_vn','gps_ve']] = np.nan

    return out.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# 2. IMU + GPS BIRLESTIRME  (vektorel)
# ──────────────────────────────────────────────────────────────
def merge_sensors(imu_df, gps_df):
    merged = imu_df.copy()
    for col in ['gps_pn','gps_pe','gps_vn','gps_ve']:
        merged[col] = np.nan

    imu_ts = merged['timestamp'].values
    valid  = gps_df.dropna(subset=['gps_pn'])

    if len(valid) > 0:
        idxs = np.searchsorted(imu_ts, valid['timestamp'].values)
        idxs = np.clip(idxs, 0, len(imu_ts) - 1)
        for col in ['gps_pn','gps_pe','gps_vn','gps_ve']:
            loc = merged.columns.get_loc(col)
            merged.iloc[idxs, loc] = valid[col].values

    merged['label'] = -1
    return merged


# ──────────────────────────────────────────────────────────────
# 3. RASTGELE DILIM + AUGMENTASYON
# ──────────────────────────────────────────────────────────────
def sample_chunk(base_df, chunk_sec, rng):
    """Kayittan rastgele chunk_sec saniyelik bir dilim al."""
    total_dur = base_df['timestamp'].iloc[-1]
    max_start = max(0.0, total_dur - chunk_sec)
    t_start   = rng.uniform(0, max_start)
    t_end     = t_start + chunk_sec

    mask  = (base_df['timestamp'] >= t_start) & (base_df['timestamp'] < t_end)
    chunk = base_df[mask].copy()
    chunk['timestamp'] = chunk['timestamp'] - chunk['timestamp'].iloc[0]
    chunk = chunk.reset_index(drop=True)

    # IMU gurultusu
    n = len(chunk)
    chunk['ax'] += rng.normal(0, 0.03, n)
    chunk['ay'] += rng.normal(0, 0.03, n)
    chunk['az'] += rng.normal(0, 0.02, n)
    chunk['gx'] += rng.normal(0, 0.002, n)
    chunk['gy'] += rng.normal(0, 0.002, n)
    chunk['gz'] += rng.normal(0, 0.001, n)

    # GPS gurultusu
    gps_mask = chunk['gps_pn'].notna()
    n_gps    = gps_mask.sum()
    if n_gps > 0:
        chunk.loc[gps_mask, 'gps_pn'] += rng.normal(0, 1.5, n_gps)
        chunk.loc[gps_mask, 'gps_pe'] += rng.normal(0, 1.5, n_gps)
        chunk.loc[gps_mask, 'gps_vn'] += rng.normal(0, 0.05, n_gps)
        chunk.loc[gps_mask, 'gps_ve'] += rng.normal(0, 0.05, n_gps)

    return chunk


# ──────────────────────────────────────────────────────────────
# 4. SPOOFING UYGULAMA  (vektorel)
# ──────────────────────────────────────────────────────────────
def apply_spoofing(df, spoof_type, spoof_start, spoof_end,
                   drift_speed=2.0, jump_dist=300.0, replay_delay=60.0, seed=42):
    df       = df.copy()
    df['label'] = 0

    if spoof_type == 0:
        return df

    angle     = seed * 0.17
    t_vals    = df['timestamp'].values
    gps_mask  = df['gps_pn'].notna().values
    in_spoof  = (t_vals >= spoof_start) & (t_vals <= spoof_end)
    spoof_gps = gps_mask & in_spoof

    if spoof_type == 1:
        elapsed = t_vals[spoof_gps] - spoof_start
        df.loc[spoof_gps, 'gps_pn'] += elapsed * drift_speed * np.cos(angle)
        df.loc[spoof_gps, 'gps_pe'] += elapsed * drift_speed * np.sin(angle)
        df.loc[spoof_gps, 'gps_vn'] += drift_speed * np.cos(angle)
        df.loc[spoof_gps, 'gps_ve'] += drift_speed * np.sin(angle)

    elif spoof_type == 2:
        df.loc[spoof_gps, 'gps_pn'] += jump_dist * np.cos(angle)
        df.loc[spoof_gps, 'gps_pe'] += jump_dist * np.sin(angle)

    elif spoof_type == 3:
        gps_idxs  = np.where(gps_mask)[0]
        gps_times = t_vals[gps_idxs]
        pn_snap   = df['gps_pn'].values.copy()
        pe_snap   = df['gps_pe'].values.copy()
        vn_snap   = df['gps_vn'].values.copy()
        ve_snap   = df['gps_ve'].values.copy()
        for idx in np.where(spoof_gps)[0]:
            target = t_vals[idx] - replay_delay
            past   = gps_times[gps_times <= target]
            if len(past) > 0:
                old_pos = int(np.searchsorted(gps_times, past[-1]))
                old_row = gps_idxs[min(old_pos, len(gps_idxs)-1)]
                df.at[idx, 'gps_pn'] = pn_snap[old_row]
                df.at[idx, 'gps_pe'] = pe_snap[old_row]
                df.at[idx, 'gps_vn'] = vn_snap[old_row]
                df.at[idx, 'gps_ve'] = ve_snap[old_row]

    df.loc[in_spoof, 'label'] = spoof_type
    return df


# ──────────────────────────────────────────────────────────────
# 5. DATASET OLUSTURMA
# ──────────────────────────────────────────────────────────────
def build_dataset(base_df, n_trajectories, seed_offset=0):
    all_dfs  = []
    rng      = np.random.RandomState(seed_offset)
    names    = ['Normal', 'Yavas Kayma', 'Ani Atlama', 'Tekrar Oynat']
    total_dur = base_df['timestamp'].iloc[-1]
    chunk_sec = min(CHUNK_SEC, total_dur * 0.9)

    for i in range(n_trajectories):
        seed       = seed_offset + i * 13
        spoof_type = i % 4

        # Rastgele dilim + augmentasyon
        chunk     = sample_chunk(base_df, chunk_sec, rng)
        total_c   = chunk['timestamp'].iloc[-1]

        sf = rng.uniform(0.15, 0.45)
        ef = rng.uniform(0.20, 0.40)
        sp_start = total_c * sf
        sp_end   = total_c * (sf + ef)

        drift_speed  = rng.uniform(0.5, 5.0)
        jump_dist    = rng.uniform(50, 500)
        replay_delay = rng.uniform(10, max(11, sp_start * 0.8))

        chunk = apply_spoofing(chunk, spoof_type, sp_start, sp_end,
                               drift_speed, jump_dist, replay_delay, seed+1)
        chunk['trajectory_id'] = i
        chunk['spoof_type']    = spoof_type

        print(f"  {i+1:3d}/{n_trajectories} | {names[spoof_type]:14s} | "
              f"{sp_start:.0f}-{sp_end:.0f}s/{total_c:.0f}s | seed={seed}")
        all_dfs.append(chunk)

    return pd.concat(all_dfs, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# 6. KALITE KONTROLU
# ──────────────────────────────────────────────────────────────
def quality_check(imu_df, gps_df):
    print("\n[Kalite Kontrolu]")
    dt_imu = np.diff(imu_df['timestamp'].values)
    dt_gps = np.diff(gps_df['timestamp'].values)
    print(f"  IMU frekansi : {1/np.mean(dt_imu):.1f} Hz")
    print(f"  GPS frekansi : {1/np.mean(dt_gps):.1f} Hz")
    print(f"  Toplam sure  : {imu_df['timestamp'].iloc[-1]:.1f}s "
          f"({imu_df['timestamp'].iloc[-1]/60:.1f} dk)")
    az_mean = imu_df['az'].mean()
    g_ok    = 'OK' if 7 < abs(az_mean) < 12 else 'Kontrol et'
    print(f"  az ortalama  : {az_mean:.2f} m/s2 ({g_ok})")
    pn_std  = gps_df['gps_pn'].std()
    pe_std  = gps_df['gps_pe'].std()
    print(f"  GPS konum std: N={pn_std:.1f}m, E={pe_std:.1f}m "
          f"({'Hareketli' if pn_std > 5 else 'Statik'})")
    print()


# ──────────────────────────────────────────────────────────────
# 7. ANA FONKSIYON
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('imu_csv')
    parser.add_argument('gps_csv')
    parser.add_argument('--imu2', nargs='+', default=[])
    parser.add_argument('--gps2', nargs='+', default=[])
    parser.add_argument('--out-dir',    default='data')
    parser.add_argument('--train-traj', type=int, default=N_TRAIN)
    parser.add_argument('--test-traj',  type=int, default=N_TEST)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[1] Sensor verileri yukleniyor...")
    imu_raw = pd.read_csv(args.imu_csv)
    t0_ms   = float(imu_raw['ts_ms'].iloc[0])
    imu_df  = load_imu(args.imu_csv)
    gps_df  = load_gps(args.gps_csv, imu_t0_ms=t0_ms)
    print(f"  Kayit 1: {args.imu_csv}")

    for imu_p, gps_p in zip(args.imu2, args.gps2):
        extra_raw = pd.read_csv(imu_p)
        extra_imu = load_imu(imu_p)
        extra_gps = load_gps(gps_p, imu_t0_ms=float(extra_raw['ts_ms'].iloc[0]))
        t_off = imu_df['timestamp'].iloc[-1] + 1.0/IMU_HZ
        extra_imu['timestamp'] += t_off
        extra_gps['timestamp'] += t_off
        imu_df = pd.concat([imu_df, extra_imu], ignore_index=True)
        gps_df = pd.concat([gps_df, extra_gps], ignore_index=True)
        print(f"  + {imu_p}")

    quality_check(imu_df, gps_df)

    print("[2] Sensorler birlestiriliyor...")
    base_df = merge_sensors(imu_df, gps_df)
    total   = base_df['timestamp'].iloc[-1]
    print(f"  {len(base_df)} satir, {total:.0f}s ({total/60:.1f} dk)\n")

    print(f"[3] Egitim dataseti ({args.train_traj} trajectory, "
          f"her biri ~{CHUNK_SEC}s)...")
    train_df   = build_dataset(base_df, args.train_traj, seed_offset=100)
    train_path = os.path.join(args.out_dir, 'dataset_real_train.csv')
    train_df.to_csv(train_path, index=False, float_format='%.8f')

    labels = train_df[train_df['label'] >= 0]['label']
    names  = ['Normal','Yavas Kayma','Ani Atlama','Tekrar Oynat']
    print(f"\n  Toplam: {len(train_df)} satir -> {train_path}")
    print("  Etiket dagilimi:")
    for lbl in range(4):
        cnt = (labels == lbl).sum()
        pct = 100*cnt/len(labels) if len(labels) > 0 else 0
        print(f"    {lbl} ({names[lbl]:14s}): {cnt:7d} ({pct:.1f}%)")

    print(f"\n[4] Test dataseti ({args.test_traj} trajectory)...")
    test_df   = build_dataset(base_df, args.test_traj, seed_offset=900)
    test_path = os.path.join(args.out_dir, 'dataset_real_test.csv')
    test_df.to_csv(test_path, index=False, float_format='%.8f')
    print(f"  Toplam: {len(test_df)} satir -> {test_path}")

    print("\n" + "="*50)
    print("TAMAMLANDI")
    print("="*50)
    print(f"  Egitim : {train_path}")
    print(f"  Test   : {test_path}")
    print("\nSonraki adimlar:")
    print("  python 4_train.py --data data/dataset_real_train.csv")
    print("  python 5_test.py  --data data/dataset_real_test.csv")


if __name__ == '__main__':
    main()
