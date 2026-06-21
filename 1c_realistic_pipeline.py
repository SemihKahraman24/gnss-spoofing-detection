"""
1c_realistic_pipeline.py — GERCEKCI spoofing veri ureteci (NON-DESTRUCTIVE)
==========================================================================
Mevcut 1b_real_data_pipeline.py'a DOKUNMAZ. Onun okuma/merge/dilim
fonksiyonlarini yeniden kullanir; sadece SPOOFING'i gercekcilestirir ve
ciktiyi YENI dosyalara yazar:
    data/dataset_realistic_train.csv
    data/dataset_realistic_test.csv

Gercekci spoofing — her tur AYRI bir SUREGELEN imza birakir (boylece sadece
baslangic aninda degil, saldiri boyunca da ve TUR ayrimi icin sinyal olur):

  1) Yavas Kayma  : ivmeli drift. Konum hatasi KARESEL buyur (0.5*a*t^2),
                    hiz ofseti lineer buyur -> surekli buyuyen innovation +
                    kucuk sabit ivme uyumsuzlugu. (Eski: sabit hiz ofseti, ic
                    pencerede gorunmezdi.)
  2) Ani Atlama   : ani degil ~1 sn'lik HIZLI rampa, sonra plato. Gercek bir
                    spoofer'in devralmasi gibi; buyuk ama sonlu innovation sicramasi.
  3) Tekrar Oynat : artan gecikmeli replay -> konum yavasca gecmise "kayar",
                    drift'ten farkli salinimli bir innovation imzasi.

Geri donus: hicbir mevcut dosya degismedi. Eski duruma donmek = eski
dataset_real_*.csv ve checkpoints/*.pkl'i kullanmaya devam etmek.
"""
import sys, os, importlib
import numpy as np, pandas as pd

_1b = importlib.import_module("1b_real_data_pipeline")
load_imu      = _1b.load_imu
load_gps      = _1b.load_gps
merge_sensors = _1b.merge_sensors
sample_chunk  = _1b.sample_chunk
quality_check = _1b.quality_check
CHUNK_SEC     = _1b.CHUNK_SEC
IMU_HZ        = _1b.IMU_HZ

KITTI_IMU = [
    "imu_kitti_0015.csv","imu_kitti_0019.csv","imu_kitti_0022.csv","imu_kitti_0023.csv",
    "imu_kitti_0027.csv","imu_kitti_0028.csv","imu_kitti_0029.csv","imu_kitti_0032.csv",
    "imu_kitti_0035.csv","imu_kitti_0036.csv","imu_kitti_0039.csv","imu_kitti_0051.csv",
    "imu_kitti_0056.csv","imu_kitti_0059.csv","imu_kitti_0061.csv","imu_kitti_0064.csv",
    "imu_kitti_0070.csv","imu_kitti_0084.csv","imu_kitti_0086.csv","imu_kitti_0087.csv",
    "imu_kitti_0091.csv","imu_kitti_0101.csv","imu_kitti_09.csv","imu_kitti_103_0027.csv",
    "imu_kitti_103_0042.csv","imu_kitti_103_0047.csv","imu_kitti_14.csv","imu_kitti_30_0016.csv",
    "imu_kitti_30_0018.csv","imu_kitti_30_0020.csv","imu_kitti_30_0034.csv",
]
ISTANBUL_IMU = ["imu_20260519_220639.csv","imu_20260519_221756.csv","imu_20260519_223444.csv"]
N_TRAIN, N_TEST = 60, 8


def apply_spoofing_realistic(df, spoof_type, spoof_start, spoof_end,
                             drift_accel=0.15, jump_dist=300.0, replay_delay=60.0, seed=42):
    df = df.copy()
    df['label'] = 0
    if spoof_type == 0:
        return df

    angle    = seed * 0.17
    ca, sa   = np.cos(angle), np.sin(angle)
    t_vals   = df['timestamp'].values
    gps_mask = df['gps_pn'].notna().values
    in_spoof = (t_vals >= spoof_start) & (t_vals <= spoof_end)
    spoof_gps = gps_mask & in_spoof
    elapsed   = t_vals[spoof_gps] - spoof_start

    if spoof_type == 1:
        # Ivmeli drift: konum karesel, hiz lineer buyur
        pos = 0.5 * drift_accel * elapsed**2
        vel = drift_accel * elapsed
        df.loc[spoof_gps, 'gps_pn'] += pos * ca
        df.loc[spoof_gps, 'gps_pe'] += pos * sa
        df.loc[spoof_gps, 'gps_vn'] += vel * ca
        df.loc[spoof_gps, 'gps_ve'] += vel * sa

    elif spoof_type == 2:
        # ~1 sn hizli rampa sonra plato
        ramp = 1.0
        frac = np.clip(elapsed / ramp, 0.0, 1.0)
        pos  = jump_dist * frac
        df.loc[spoof_gps, 'gps_pn'] += pos * ca
        df.loc[spoof_gps, 'gps_pe'] += pos * sa
        # rampa sirasinda hiz sicramasi
        vspk = np.where(elapsed < ramp, jump_dist / ramp, 0.0)
        df.loc[spoof_gps, 'gps_vn'] += vspk * ca
        df.loc[spoof_gps, 'gps_ve'] += vspk * sa

    elif spoof_type == 3:
        # Artan gecikmeli replay: konum yavasca gecmise kayar
        gps_idxs  = np.where(gps_mask)[0]
        gps_times = t_vals[gps_idxs]
        pn = df['gps_pn'].values.copy(); pe = df['gps_pe'].values.copy()
        vn = df['gps_vn'].values.copy(); ve = df['gps_ve'].values.copy()
        for k, idx in enumerate(np.where(spoof_gps)[0]):
            el = t_vals[idx] - spoof_start
            growing_delay = replay_delay + 0.5 * el   # gecikme zamanla artar
            target = t_vals[idx] - growing_delay
            past = gps_times[gps_times <= target]
            if len(past) > 0:
                op = int(np.searchsorted(gps_times, past[-1]))
                orow = gps_idxs[min(op, len(gps_idxs)-1)]
                df.at[idx, 'gps_pn'] = pn[orow]; df.at[idx, 'gps_pe'] = pe[orow]
                df.at[idx, 'gps_vn'] = vn[orow]; df.at[idx, 'gps_ve'] = ve[orow]

    df.loc[in_spoof, 'label'] = spoof_type
    return df


def build_dataset_realistic(base_df, n_traj, seed_offset=0):
    rng = np.random.RandomState(seed_offset)
    names = ['Normal','Yavas Kayma','Ani Atlama','Tekrar Oynat']
    total_dur = base_df['timestamp'].iloc[-1]
    chunk_sec = min(CHUNK_SEC, total_dur * 0.9)
    out = []
    for i in range(n_traj):
        seed = seed_offset + i * 13
        st = i % 4
        chunk = sample_chunk(base_df, chunk_sec, rng)
        tc = chunk['timestamp'].iloc[-1]
        sf = rng.uniform(0.15, 0.45); ef = rng.uniform(0.20, 0.40)
        sp_s = tc * sf; sp_e = tc * (sf + ef)
        drift_accel  = rng.uniform(0.30, 1.20)  # tespit edilebilir aralik (egri: >=0.3 -> %100)      # m/s^2 (gercekci ivmeli drift)
        jump_dist    = rng.uniform(50, 500)
        replay_delay = rng.uniform(10, max(11, sp_s * 0.8))
        chunk = apply_spoofing_realistic(chunk, st, sp_s, sp_e,
                                         drift_accel, jump_dist, replay_delay, seed+1)
        chunk['trajectory_id'] = i; chunk['spoof_type'] = st
        print(f"  {i+1:3d}/{n_traj} | {names[st]:14s} | {sp_s:.0f}-{sp_e:.0f}s/{tc:.0f}s")
        out.append(chunk)
    return pd.concat(out, ignore_index=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__)) or "."
    os.chdir(here)
    all_imu = KITTI_IMU + ISTANBUL_IMU
    all_gps = [f.replace("imu_", "gps_") for f in all_imu]
    miss = [f for f in all_imu + all_gps if not os.path.exists(f)]
    if miss:
        print("[!] Eksik dosyalar:", miss[:5], "..."); sys.exit(1)

    print(f"[1] {len(all_imu)} kayit yukleniyor...")
    imu0 = pd.read_csv(all_imu[0])
    imu_df = load_imu(all_imu[0])
    gps_df = load_gps(all_gps[0], imu_t0_ms=float(imu0['ts_ms'].iloc[0]))
    for ip, gp in zip(all_imu[1:], all_gps[1:]):
        r = pd.read_csv(ip)
        ei = load_imu(ip); eg = load_gps(gp, imu_t0_ms=float(r['ts_ms'].iloc[0]))
        off = imu_df['timestamp'].iloc[-1] + 1.0/IMU_HZ
        ei['timestamp'] += off; eg['timestamp'] += off
        imu_df = pd.concat([imu_df, ei], ignore_index=True)
        gps_df = pd.concat([gps_df, eg], ignore_index=True)

    quality_check(imu_df, gps_df)
    base = merge_sensors(imu_df, gps_df)
    os.makedirs("data", exist_ok=True)

    print(f"[2] GERCEKCI egitim seti ({N_TRAIN} trajectory)...")
    tr = build_dataset_realistic(base, N_TRAIN, seed_offset=100)
    tr.to_csv("data/dataset_realistic_train.csv", index=False, float_format='%.8f')
    print(f"[3] GERCEKCI test seti ({N_TEST} trajectory)...")
    te = build_dataset_realistic(base, N_TEST, seed_offset=900)
    te.to_csv("data/dataset_realistic_test.csv", index=False, float_format='%.8f')
    print("\n[TAMAM] -> data/dataset_realistic_train.csv / _test.csv")


if __name__ == "__main__":
    main()
