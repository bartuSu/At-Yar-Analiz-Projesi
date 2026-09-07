"""
"value >= eşik" stratejisini BİRDEN FAZLA zaman diliminde (walk-forward)
test eder - tek bir test döneminde şansla çıkan bir sonucu, gerçek bir
örüntüden ayırt etmek için.

Kullanım:
    python walk_forward_value_backtest.py
"""
import numpy as np
import pandas as pd
import xgboost as xgb

import scraper_config

NUMERIC_FEATURES = [
    "kilo", "mesafe", "son_3_ort_sira", "gecmis_yaris_sayisi",
    "son_yaristan_gun_farki", "jokey_id_kazanma_orani",
    "antrenor_id_kazanma_orani", "sahip_id_kazanma_orani",
    "baba_kazanma_orani", "anne_kazanma_orani",
]
CATEGORICAL_FEATURES = ["sehir", "pist"]
TARGET = "birinci_mi"
N_FOLDS = 5
ESIKLER = [0.05, 0.10, 0.15, 0.20, 0.25]

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_child_weight": 10,
}


def load_and_prepare():
    df = pd.read_csv(scraper_config.PROCESSED_DIR / "features.csv", parse_dates=["tarih_dt"])
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dummy_na=True)
    cat_cols = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]
    feature_cols = NUMERIC_FEATURES + cat_cols
    df = df.sort_values(["tarih_dt", "kosu_kodu"]).reset_index(drop=True)
    return df, feature_cols


def gruplari_hesapla(df):
    return df.groupby("kosu_kodu", sort=False).size().tolist()


def walk_forward_foldlar(df, n_folds):
    unique_dates = df["tarih_dt"].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(unique_dates)
    sinirlar = [int(n * i / (n_folds + 1)) for i in range(n_folds + 2)]
    for i in range(1, n_folds + 1):
        val_tarihleri = unique_dates.iloc[sinirlar[i]:sinirlar[i + 1]]
        if val_tarihleri.empty:
            continue
        val_baslangic, val_bitis = val_tarihleri.iloc[0], val_tarihleri.iloc[-1]
        train_mask = df["tarih_dt"] < val_baslangic
        val_mask = (df["tarih_dt"] >= val_baslangic) & (df["tarih_dt"] <= val_bitis)
        yield df[train_mask], df[val_mask]


def _softmax(s: pd.Series) -> pd.Series:
    x = s.values - s.values.max()
    e = np.exp(x)
    return pd.Series(e / e.sum(), index=s.index)


def hesapla_value(val_df, model, feature_cols):
    val_df = val_df.copy()
    val_df["skor"] = model.predict(val_df[feature_cols])
    val_df["model_olasiligi"] = val_df.groupby("kosu_kodu")["skor"].transform(_softmax)

    ganyan_num = pd.to_numeric(val_df["ganyan"].astype(str).str.replace(",", "."), errors="coerce")
    val_df["ganyan_num"] = ganyan_num
    implied = np.where(ganyan_num > 0, 1 / ganyan_num, np.nan)
    val_df["implied_ham"] = implied
    toplam = val_df.groupby("kosu_kodu")["implied_ham"].transform("sum")
    val_df["piyasa_olasiligi"] = val_df["implied_ham"] / toplam

    val_df["value"] = val_df["model_olasiligi"] - val_df["piyasa_olasiligi"]
    return val_df


def backtest(val_df, esik):
    bahisler = []
    for _, grup in val_df.groupby("kosu_kodu"):
        if grup["birinci_mi"].sum() != 1:
            continue
        adaylar = grup[grup["value"] >= esik]
        if adaylar.empty:
            continue
        secim = adaylar.loc[adaylar["value"].idxmax()]
        kazandi = secim["birinci_mi"] == 1
        kar = (secim["ganyan_num"] - 1) if kazandi else -1
        bahisler.append({"kazandi": kazandi, "kar": kar})
    if not bahisler:
        return 0, 0.0, 0.0
    bdf = pd.DataFrame(bahisler)
    return len(bdf), bdf["kazandi"].mean(), bdf["kar"].sum() / len(bdf)


def main():
    df, feature_cols = load_and_prepare()

    print(f"{N_FOLDS} fold ile walk-forward doğrulama başlıyor...\n")
    tum_esik_sonuclari = {esik: [] for esik in ESIKLER}

    for fold_no, (train_df, val_df) in enumerate(walk_forward_foldlar(df, N_FOLDS), 1):
        if train_df.empty or val_df.empty:
            continue

        X_train, y_train = train_df[feature_cols], train_df[TARGET]
        train_gruplari = gruplari_hesapla(train_df)

        model = xgb.XGBRanker(objective="rank:ndcg", random_state=42, n_jobs=-1, **DEFAULT_PARAMS)
        model.fit(X_train, y_train, group=train_gruplari)

        val_scored = hesapla_value(val_df, model, feature_cols)

        print(f"--- Fold {fold_no} ({val_df['tarih_dt'].min().date()} -> {val_df['tarih_dt'].max().date()}) ---")
        for esik in ESIKLER:
            n, kazanma, roi = backtest(val_scored, esik)
            tum_esik_sonuclari[esik].append((n, kazanma, roi))
            print(f"  value >= {esik}: {n} bahis, kazanma {kazanma:.1%}, ROI {roi:+.1%}")
        print()

    print("=" * 60)
    print("TÜM FOLD'LAR BİRLEŞTİRİLMİŞ (havuzlanmış) SONUÇLAR")
    print("=" * 60)
    for esik, sonuclar in tum_esik_sonuclari.items():
        toplam_bahis = sum(s[0] for s in sonuclar)
        if toplam_bahis == 0:
            print(f"value >= {esik}: hiç bahis yok")
            continue
        toplam_kar_orani = sum(n * roi for n, _, roi in sonuclar) / toplam_bahis
        toplam_kazanma = sum(n * k for n, k, _ in sonuclar) / toplam_bahis
        print(f"value >= {esik}: TOPLAM {toplam_bahis} bahis, "
              f"kazanma {toplam_kazanma:.1%}, ROI {toplam_kar_orani:+.1%}")


if __name__ == "__main__":
    main()