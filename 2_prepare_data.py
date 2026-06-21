"""
ADIM 2: Veri Hazirlama (Spoofing Tespiti) - v4 (DUZELTILMIS)
============================================================
v4 degisiklikleri (kok-neden duzeltmeleri):

  1) KOORDINAT CERCEVESI DUZELTMESI (en kritik):
     Eski kod  acc_diff_n = gps_an - ax  yaziyordu. Ama ax/ay GOVDE
     cercevesinde, gps_an/gps_ae NED cercevesinde. Arac yonu 0 degilse
     bu cikarma anlamsizdi -> 4 kanal gurultuydu.
     Yeni: IMU ivmesi GPS heading'i ile NED'e dondurulup karsilastiriliyor.
     Ayrica frame-bagimsiz BUYUKLUK farki da ekli.

  2) TRAJECTORY-BAZLI BOLME:
     Eski kod tum (yari yariya ortusen) pencereleri rastgele karistirip
     train/val/test'e boluyordu -> ciddi sizinti. Yeni: once trajectory'ler
     bolunur, sonra her grup AYRI pencerelenir. Komsu pencere sizintisi yok.

  3) ETIKET SAFLIK FILTRESI:
     Pencere etiketi median ile veriliyordu -> sinir pencereleri %51/%49
     rastgele etiketleniyordu. Yeni: bir pencere ancak baskin etiket
     >= purity_frac orani kapliyorsa kullanilir, yoksa atilir.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# torch yalnizca egitim/test (4_, 5_) icin gerekli. Veri analizi/ablasyon
# ortaminda torch olmayabilir; bu yuzden import korumali.
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    Dataset = object


# =============================================================
# 1. OZELLIK CIKARMA (HER TRAJECTORY AYRI ISLENIR)
# =============================================================
def _extract_single_trajectory(tdf):
    df = tdf.copy().reset_index(drop=True)
    dt = 0.01
    n = len(df)

    # GPS enterpolasyon
    for col in ['gps_pn', 'gps_pe', 'gps_vn', 'gps_ve']:
        if col in df.columns:
            df[col + '_i'] = df[col].interpolate(method='linear', limit_direction='both')
        else:
            df[col + '_i'] = 0.0

    # HAM GPS ATLAMA (enterpolasyon ONCESI) - Ani Atlama icin kritik
    raw_pn = df['gps_pn'].values.copy()
    raw_pe = df['gps_pe'].values.copy()
    gps_valid_indices = np.where(~np.isnan(raw_pn))[0]
    if len(gps_valid_indices) > 1:
        gps_step = max(1, int(np.median(np.diff(gps_valid_indices))))
    else:
        gps_step = 100

    raw_jump = np.zeros(n)
    last_valid_pn = np.nan
    last_valid_pe = np.nan
    for i in range(n):
        if not np.isnan(raw_pn[i]):
            if not np.isnan(last_valid_pn):
                jump = np.sqrt((raw_pn[i] - last_valid_pn)**2 +
                              (raw_pe[i] - last_valid_pe)**2)
                end_idx = min(i + gps_step, n)
                raw_jump[i:end_idx] = jump
            last_valid_pn = raw_pn[i]
            last_valid_pe = raw_pe[i]
    df['raw_gps_jump'] = raw_jump

    raw_vn = df['gps_vn'].values.copy()
    raw_ve = df['gps_ve'].values.copy()
    raw_speed_jump = np.zeros(n)
    last_vn = np.nan
    last_ve = np.nan
    for i in range(n):
        if not np.isnan(raw_vn[i]):
            if not np.isnan(last_vn):
                dv = np.sqrt((raw_vn[i] - last_vn)**2 + (raw_ve[i] - last_ve)**2)
                end_idx = min(i + gps_step, n)
                raw_speed_jump[i:end_idx] = dv
            last_vn = raw_vn[i]
            last_ve = raw_ve[i]
    df['raw_speed_jump'] = raw_speed_jump

    # GPS ivmesi (NED cercevesinde)
    df['gps_speed'] = np.sqrt(df['gps_vn_i']**2 + df['gps_ve_i']**2)
    df['gps_an'] = df['gps_vn_i'].diff().fillna(0) / dt
    df['gps_ae'] = df['gps_ve_i'].diff().fillna(0) / dt
    df['gps_an'] = df['gps_an'].rolling(10, min_periods=1, center=True).mean()
    df['gps_ae'] = df['gps_ae'].rolling(10, min_periods=1, center=True).mean()

    # *** DUZELTME 1: IMU ivmesini GPS heading'i ile NED'e dondur ***
    #   a_n = ax*cos(t) - ay*sin(t) ;  a_e = ax*sin(t) + ay*cos(t)
    heading = np.arctan2(df['gps_ve_i'].values, df['gps_vn_i'].values)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    ax = df['ax'].values
    ay = df['ay'].values
    imu_an = ax * cos_h - ay * sin_h
    imu_ae = ax * sin_h + ay * cos_h

    df['acc_diff_n'] = df['gps_an'].values - imu_an
    df['acc_diff_e'] = df['gps_ae'].values - imu_ae
    df['acc_diff_mag'] = np.sqrt(df['acc_diff_n']**2 + df['acc_diff_e']**2)

    # Frame-bagimsiz yedek: ivme BUYUKLUGU farki (rotasyona bagli degil)
    imu_acc_mag = np.sqrt(ax**2 + ay**2)
    gps_acc_mag = np.sqrt(df['gps_an'].values**2 + df['gps_ae'].values**2)
    df['acc_mag_diff'] = np.abs(gps_acc_mag - imu_acc_mag)

    # GPS yon vs jiroskop
    df['gps_heading'] = np.arctan2(df['gps_ve_i'], df['gps_vn_i'])
    gps_hdg = df['gps_heading'].values.copy()
    for i in range(1, len(gps_hdg)):
        while gps_hdg[i] - gps_hdg[i-1] > np.pi: gps_hdg[i] -= 2*np.pi
        while gps_hdg[i] - gps_hdg[i-1] < -np.pi: gps_hdg[i] += 2*np.pi
    gps_hdg_rate = np.gradient(gps_hdg, dt)
    gps_hdg_rate_s = pd.Series(gps_hdg_rate).rolling(20, min_periods=1, center=True).mean().values
    df['heading_diff'] = gps_hdg_rate_s - df['gz'].values

    # Enterpolasyonlu GPS atlama
    dpn = df['gps_pn_i'].diff().fillna(0)
    dpe = df['gps_pe_i'].diff().fillna(0)
    df['gps_jump'] = np.sqrt(dpn**2 + dpe**2)
    df['gps_speed_diff_abs'] = df['gps_speed'].diff().fillna(0).abs()

    # Hareketli istatistikler
    w = 200
    df['acc_diff_mean'] = df['acc_diff_mag'].rolling(w, min_periods=1).mean()
    df['acc_diff_std'] = df['acc_diff_mag'].rolling(w, min_periods=1).std().fillna(0)
    df['heading_diff_mean'] = df['heading_diff'].rolling(w, min_periods=1).mean()
    df['jump_mean'] = df['gps_jump'].rolling(w, min_periods=1).mean()
    df['raw_jump_mean'] = df['raw_gps_jump'].rolling(w, min_periods=1).mean()
    df['raw_jump_max'] = df['raw_gps_jump'].rolling(w, min_periods=1).max()

    # Kisa sureli konum farki (IMU dead-reckoning NED ivmeden vs GPS)
    imu_pn = np.zeros(n)
    imu_pe = np.zeros(n)
    vn, ve, pn, pe = 0.0, 0.0, 0.0, 0.0
    for i in range(1, n):
        if i % 200 == 0:
            vn, ve, pn, pe = 0.0, 0.0, 0.0, 0.0
        vn += imu_an[i] * dt
        ve += imu_ae[i] * dt
        pn += vn * dt
        pe += ve * dt
        imu_pn[i] = pn
        imu_pe[i] = pe

    gps_dpn_2s = df['gps_pn_i'].diff(200).fillna(0)
    gps_dpe_2s = df['gps_pe_i'].diff(200).fillna(0)
    df['pos_diff_2s'] = np.sqrt(
        (gps_dpn_2s - pd.Series(imu_pn).diff(200).fillna(0))**2 +
        (gps_dpe_2s - pd.Series(imu_pe).diff(200).fillna(0))**2
    )

    # =============================================
    # *** YENI: INNOVATION (GPS-vs-IMU surekli residual) ***
    # Gercek GNSS spoofing tespitinin temeli. IMU dead-reckoning ile
    # tahmin edilen konum, GPS'ten ne kadar sapiyor? Sabit-ofset spoof'ta
    # baslangicta sicrar, drift'te SUREKLI buyur. Her 5sn'de GPS'e yeniden
    # capalanir (gercek tamamlayici filtre gibi) -> bias birikmez.
    # =============================================
    gpn_i = df['gps_pn_i'].values
    gpe_i = df['gps_pe_i'].values
    innov = np.zeros(n)
    vn = ve = ipn = ipe = 0.0
    anchor_n, anchor_e = gpn_i[0], gpe_i[0]
    last_anchor = 0
    for i in range(1, n):
        vn += imu_an[i] * dt
        ve += imu_ae[i] * dt
        ipn += vn * dt
        ipe += ve * dt
        innov[i] = np.sqrt((gpn_i[i] - (anchor_n + ipn))**2 +
                           (gpe_i[i] - (anchor_e + ipe))**2)
        if i - last_anchor >= 500:        # 5 sn'de bir yeniden capala
            anchor_n, anchor_e = gpn_i[i], gpe_i[i]
            vn = ve = ipn = ipe = 0.0
            last_anchor = i
    df['innov'] = innov
    df['innov_mean'] = pd.Series(innov).rolling(w, min_periods=1).mean().values

    df = df.fillna(0).replace([np.inf, -np.inf], 0)
    return df


def extract_features(df):
    """Her trajectory'yi ayri ayri isle, sonra birlestir."""
    if 'trajectory_id' in df.columns:
        parts = []
        for tid, group in df.groupby('trajectory_id'):
            processed = _extract_single_trajectory(group)
            processed['trajectory_id'] = tid
            parts.append(processed)
        return pd.concat(parts, ignore_index=True)
    else:
        return _extract_single_trajectory(df)


# 18 kanal ozellik (acc_mag_diff yeni; toplam 18)
FEATURE_COLS = [
    'ax', 'ay', 'az', 'gx', 'gy', 'gz',           # 6 IMU ham
    'acc_diff_n', 'acc_diff_e',                       # 2 ivme tutarsizligi (NED, DUZELTILDI)
    'acc_mag_diff',                                   # 1 frame-bagimsiz ivme farki (YENI)
    'heading_diff',                                   # 1 yon tutarsizligi
    'gps_jump',                                       # 1 GPS atlama
    'gps_speed_diff_abs',                             # 1 GPS hiz degisimi
    'acc_diff_std',                                   # 1 ivme istatistik
    'heading_diff_mean',                               # 1 yon trend
    'jump_mean',                                       # 1 atlama trend
    'pos_diff_2s',                                     # 1 kisa sureli konum farki
    'raw_jump_mean', 'raw_jump_max',                  # 2 ham GPS atlama istatistigi
    'innov', 'innov_mean',                            # 2 GPS-vs-IMU innovation (YENI, en onemli)
]
# Toplam 20 kanal


# =============================================================
# 2. PENCERELEME (etiket saflik filtresi ile)
# =============================================================
def create_windows(df, window_size=200, stride=50, binary=False, purity_frac=0.85,
                   label_mode='window'):
    """
    label_mode='window' : eski davranis. Baskin etiket >= purity_frac ise
                          o etiket; aksi halde pencere ATILIR.
    label_mode='onset'  : *** ONERILEN ***. Spoofing tespitini DEGISIM-ANI
                          tespiti olarak kurar. Pozitif = saldirinin pencere
                          icinde BASLADIGI (gecis iceren) pencere; Negatif =
                          tamamen normal pencere. Saldirinin ic kismi (sabit
                          ofset, fiziksel olarak ayirt edilemez) ATILIR.
                          Etiket daima binary (0/1) olur.
    """
    X_list, Y_list = [], []
    dropped = 0
    if 'trajectory_id' in df.columns:
        groups = df.groupby('trajectory_id')
    else:
        groups = [('single', df)]

    third = max(1, window_size // 3)
    for name, group in groups:
        group = group.reset_index(drop=True)
        features = group[FEATURE_COLS].values
        labels = group['label'].values
        n = len(group)
        for start in range(0, n - window_size, stride):
            end = start + window_size
            window_labels = labels[start:end]
            valid = window_labels[window_labels >= 0]
            if len(valid) == 0:
                continue

            if label_mode == 'onset':
                atk = (window_labels > 0)
                frac = atk.mean()
                head_atk = atk[:third].mean()
                tail_atk = atk[-third:].mean()
                if frac == 0:
                    label = 0                                  # tam normal
                elif head_atk < 0.2 and tail_atk > 0.5:
                    label = 1                                  # onset (gecis)
                else:
                    dropped += 1                               # ic-saldiri -> at
                    continue
                X_list.append(features[start:end]); Y_list.append(label)
                continue

            vals, counts = np.unique(valid, return_counts=True)
            dom = int(vals[np.argmax(counts)])
            dom_frac = counts.max() / len(valid)
            if binary:
                atk_frac = (valid > 0).mean()
                if atk_frac <= (1 - purity_frac):
                    label = 0
                elif atk_frac >= purity_frac:
                    label = 1
                else:
                    dropped += 1
                    continue
            else:
                if dom_frac < purity_frac:
                    dropped += 1
                    continue
                label = dom
            X_list.append(features[start:end])
            Y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.int64)
    return X, Y, dropped


# =============================================================
# 3. NORMALIZASYON
# =============================================================
class Scaler:
    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self._scaler.fit(flat)
        self._fitted = True

    def transform(self, X):
        shape = X.shape
        flat = X.reshape(-1, shape[-1])
        return self._scaler.transform(flat).reshape(shape).astype(np.float32)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._scaler, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self._scaler = pickle.load(f)
        self._fitted = True
        return self


# =============================================================
# 4. PYTORCH DATASET
# =============================================================
class SpoofDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =============================================================
# 5. TAM PIPELINE  (*** DUZELTME 2: TRAJECTORY-BAZLI BOLME ***)
# =============================================================
def prepare(csv_path, window_size=200, stride=20, batch_size=64, binary=False,
            purity_frac=0.85, seed=42, label_mode='window'):
    print(f"  Veri okunuyor: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"  Ozellikler cikariliyor (trajectory bazli)...")
    df = extract_features(df)

    if 'trajectory_id' not in df.columns:
        df['trajectory_id'] = 0
    tids = np.array(sorted(df['trajectory_id'].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(tids)
    n_t = len(tids)
    n_test = max(1, int(n_t * 0.15))
    n_val = max(1, int(n_t * 0.15))
    test_ids = set(tids[:n_test])
    val_ids = set(tids[n_test:n_test + n_val])
    train_ids = set(tids[n_test + n_val:])
    print(f"  Trajectory bolme: egitim={len(train_ids)}, dogrulama={len(val_ids)}, test={len(test_ids)}")

    def subset(ids):
        return df[df['trajectory_id'].isin(ids)]

    print(f"  Pencereler olusturuluyor (mod={label_mode}, saflik>={purity_frac})...")
    X_tr, Y_tr, d_tr = create_windows(subset(train_ids), window_size, stride, binary, purity_frac, label_mode)
    X_va, Y_va, d_va = create_windows(subset(val_ids), window_size, stride, binary, purity_frac, label_mode)
    X_te, Y_te, d_te = create_windows(subset(test_ids), window_size, stride, binary, purity_frac, label_mode)
    print(f"  Atilan (karisik) pencere: egitim={d_tr}, dogrulama={d_va}, test={d_te}")

    scaler = Scaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    train_ld = DataLoader(SpoofDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(SpoofDataset(X_va, Y_va), batch_size=batch_size, shuffle=False)
    test_ld = DataLoader(SpoofDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)

    print(f"  Pencere: egitim={len(X_tr)}, dogrulama={len(X_va)}, test={len(X_te)}")
    print(f"  Girdi: {X_tr.shape}")
    print(f"  Egitim sinif dagilimi: {dict(zip(*np.unique(Y_tr, return_counts=True)))}")
    return train_ld, val_ld, test_ld, scaler


if __name__ == '__main__':
    print("[*] Veri hazirlama testi\n")
    train_ld, val_ld, test_ld, scaler = prepare('data/dataset_real_train.csv')
    X, Y = next(iter(train_ld))
    print(f"\n  Batch: X={X.shape}, Y={Y.shape}")
    print("\n[TAMAM]")
