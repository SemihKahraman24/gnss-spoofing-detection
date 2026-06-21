"""
lstm_compare.py — BiLSTM (derin ogrenme) ile 4-sinif onset tespiti
===================================================================
Bu betik, makaledeki HGB/RF/MLP karsilastirmasina BiLSTM satirini eklemek
icindir. AYNI veri (gercekci, onset 4-sinif), AYNI yorunge-bazli bolme ve
AYNI metrikler kullanilir; cikti dogrudan karsilastirma tablosuna girer.

GEREKSINIM: PyTorch (CPU de calisir, GPU varsa otomatik kullanir).
  pip install torch
Calistirma:
  python lstm_compare.py
Cikti:
  results/lstm_results.csv  (model, heldout_acc, heldout_macroF1, ext_acc, ext_macroF1)
  ayrica sinif-bazli rapor ekrana yazilir.
"""
import os, sys, importlib, time
import numpy as np
import pandas as pd

# --- ozellik cikarma (2_prepare_data ile tutarli) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
m2 = importlib.import_module("2_prepare_data")
ef = m2.extract_features
FC = m2.FEATURE_COLS                       # 20 kanal
NAMES = ["Normal", "Yavas Kayma", "Ani Atlama", "Tekrar Oynat"]
WIN, STRIDE = 200, 25

import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────
# 1. Sekans + onset/hibrit etiket (HGB ile birebir ayni kural)
# ──────────────────────────────────────────────────────────────
def build(df):
    third = WIN // 3
    X, Y, TID = [], [], []
    for tid, g in df.groupby("trajectory_id"):
        g = g.reset_index(drop=True)
        fv = g[FC].values.astype(np.float32)
        lb = g["label"].values
        n = len(g)
        for s in range(0, n - WIN, STRIDE):
            wl = lb[s:s + WIN]; atk = (wl > 0); fr = atk.mean()
            if fr == 0:
                y = 0
            else:
                dom = int(np.median(wl[wl > 0]))
                onset = atk[:third].mean() < 0.2 and atk[-third:].mean() > 0.5
                if onset:
                    y = dom
                elif fr >= 0.8 and dom in (1, 3):
                    y = dom
                else:
                    continue
            X.append(fv[s:s + WIN]); Y.append(y); TID.append(tid)
    return np.array(X, np.float32), np.array(Y, np.int64), np.array(TID)


# ──────────────────────────────────────────────────────────────
# 2. BiLSTM modeli (3_model.py ile ayni mimari)
# ──────────────────────────────────────────────────────────────
class BiLSTM(nn.Module):
    def __init__(self, input_size=20, hidden=128, layers=2, num_classes=4, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def macro_f1(y, p, C=4):
    f = []
    for c in range(C):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        pr = tp / (tp + fp + 1e-9); rc = tp / (tp + fn + 1e-9)
        f.append(2 * pr * rc / (pr + rc + 1e-9))
    return float(np.mean(f)) * 100


def evaluate(model, X, Y, scaler):
    model.eval()
    Xs = ((X - scaler[0]) / scaler[1]).astype(np.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xs), 256):
            xb = torch.from_numpy(Xs[i:i + 256]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    p = np.concatenate(preds) if preds else np.array([])
    acc = (p == Y).mean() * 100
    return acc, macro_f1(Y, p, 4), p


def main():
    t0 = time.time()
    print(f"[i] Cihaz: {device}")
    print("[1] Veri + ozellik + sekans...")
    Xtr_all, Ytr_all, Ttr = build(ef(pd.read_csv("data/dataset_realistic_train.csv")))
    Xext, Yext, _ = build(ef(pd.read_csv("data/dataset_realistic_test.csv")))
    print(f"    egitim sekans: {Xtr_all.shape}, dis test: {Xext.shape}")

    # yorunge-bazli bolme (held-out)
    rng = np.random.RandomState(0); u = np.unique(Ttr); rng.shuffle(u)
    k = max(1, int(len(u) * 0.25)); held = np.isin(Ttr, u[:k])
    tr = ~held
    # train icinden kucuk val (early stopping)
    u2 = np.unique(Ttr[tr]); rng.shuffle(u2); kv = max(1, int(len(u2) * 0.2))
    val = np.isin(Ttr, u2[:kv]) & tr; trn = tr & ~val

    # normalizasyon (kanal bazli, egitimden)
    flat = Xtr_all[trn].reshape(-1, Xtr_all.shape[-1])
    mu = flat.mean(0); sd = flat.std(0) + 1e-6; scaler = (mu, sd)
    def norm(X): return ((X - mu) / sd).astype(np.float32)

    Xtrn = torch.from_numpy(norm(Xtr_all[trn])); Ytrn = torch.from_numpy(Ytr_all[trn])

    # sinif agirliklari (dengesizlik)
    cnt = np.bincount(Ytr_all[trn], minlength=4).astype(float)
    w = (1.0 / (cnt + 1e-6)); w = w / w.sum() * 4
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))

    model = BiLSTM(input_size=len(FC), num_classes=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("[2] Egitim (BiLSTM)...")
    EPOCHS, BS, PAT = 60, 64, 10
    best_f1, best_state, no_imp = -1, None, 0
    n = len(Xtrn)
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            xb = Xtrn[idx].to(device); yb = Ytrn[idx].to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        va_acc, va_f1, _ = evaluate(model, Xtr_all[val], Ytr_all[val], scaler)
        if va_f1 > best_f1:
            best_f1 = va_f1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
        print(f"  epoch {ep:3d} | val_acc={va_acc:.1f} val_macroF1={va_f1:.1f} | best={best_f1:.1f}")
        if no_imp >= PAT:
            print("  erken durdurma."); break

    model.load_state_dict(best_state)
    h_acc, h_f1, h_p = evaluate(model, Xtr_all[held], Ytr_all[held], scaler)
    e_acc, e_f1, e_p = evaluate(model, Xext, Yext, scaler)

    print("\n" + "=" * 52)
    print(f"  BiLSTM SONUC  (cihaz={device}, {time.time()-t0:.0f}s)")
    print("=" * 52)
    print(f"  Held-out : acc={h_acc:.1f}%  macroF1={h_f1:.1f}")
    print(f"  Dis test : acc={e_acc:.1f}%  macroF1={e_f1:.1f}")
    for tag, Y, P in [("HELD-OUT", Ytr_all[held], h_p), ("DIS TEST", Yext, e_p)]:
        print(f"\n  [{tag}] sinif-bazli (P/R/F1):")
        for c in range(4):
            tp = ((P == c) & (Y == c)).sum(); fp = ((P == c) & (Y != c)).sum(); fn = ((P != c) & (Y == c)).sum()
            pr = tp / (tp + fp + 1e-9) * 100; rc = tp / (tp + fn + 1e-9) * 100; f1 = 2*pr*rc/(pr+rc+1e-9)
            print(f"    {NAMES[c]:13s} P={pr:5.1f} R={rc:5.1f} F1={f1:5.1f}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame([{"model": "BiLSTM (derin ogrenme)",
                   "heldout_acc": round(h_acc, 1), "heldout_macroF1": round(h_f1, 1),
                   "ext_acc": round(e_acc, 1), "ext_macroF1": round(e_f1, 1)}]
                 ).to_csv("results/lstm_results.csv", index=False)
    print("\n-> results/lstm_results.csv yazildi. Bu satiri makaledeki Tablo'ya ekleyebilirsin.")


if __name__ == "__main__":
    main()
