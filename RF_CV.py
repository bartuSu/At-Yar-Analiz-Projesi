"""
XGBRanker için walk-forward (zamana göre ilerleyen) hiperparametre taraması.

Kullanım:
    python tune_model.py
    -> her parametre kombinasyonunun ortalama isabet oranını ve tied-score
       oranını konsola yazar
"""
import itertools

import numpy as np
import pandas as pd
import xgboost as xgb

import scraper_config

NUMERIC_FEATURES = [
    "kilo", "mesafe", "son_3_ort_sira", "gecmis_yaris_sayisi",
    "son_yaristan_gun_farki", "jokey_id_kazanma_orani",
    "antrenor_id_kazanma_orani", "sahip_id_kazanma_orani",
    "baba_kazanma_orani", "anne_kazanma_orani",
    #"ganyan",  # ganyansız versiyon için bu satırı kapat
]
CATEGORICAL_FEATURES = ["sehir", "pist"]
TARGET = "birinci_mi"

N_FOLDS = 5

PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 10],
}


def load_and_prepare():
    df = pd.read_csv(scraper_config.PROCESSED_DIR / "features.csv", parse_dates=["tarih_dt"])

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # DİKKAT: median doldurma YOK, NaN native bırakılıyor.

    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dummy_na=True)
    cat_cols = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]
    feature_cols = NUMERIC_FEATURES + cat_cols

    df = df.sort_values(["tarih_dt", "kosu_kodu"]).reset_index(drop=True)
    return df, feature_cols


def gruplari_hesapla(df: pd.DataFrame) -> list[int]:
    return df.groupby("kosu_kodu", sort=False).size().tolist()


def walk_forward_foldlar(df: pd.DataFrame, n_folds: int):
    unique_dates = df["tarih_dt"].drop_duplicates().sort_values().reset_index(drop=True)
    n = len(unique_dates)
    sinirlar = [int(n * i / (n_folds + 1)) for i in range(n_folds + 2)]

    for i in range(1, n_folds + 1):
        val_tarihleri = unique_dates.iloc[sinirlar[i]:sinirlar[i + 1]]
        if val_tarihleri.empty:
            continue
        val_baslangic = val_tarihleri.iloc[0]
        val_bitis = val_tarihleri.iloc[-1]
        train_mask = df["tarih_dt"] < val_baslangic
        val_mask = (df["tarih_dt"] >= val_baslangic) & (df["tarih_dt"] <= val_bitis)
        yield df[train_mask], df[val_mask]


def race_level_accuracy(val_df: pd.DataFrame, scores: np.ndarray) -> float:
    val_df = val_df.copy()
    val_df["skor"] = scores
    dogru, toplam = 0, 0
    for _, grup in val_df.groupby("kosu_kodu"):
        if grup["birinci_mi"].sum() != 1:
            continue
        toplam += 1
        secim = grup.loc[grup["skor"].idxmax()]
        dogru += int(secim["birinci_mi"] == 1)
    return dogru / toplam if toplam else 0.0


def tune():
    df, feature_cols = load_and_prepare()

    kombinasyonlar = list(itertools.product(*PARAM_GRID.values()))
    anahtarlar = list(PARAM_GRID.keys())
    print(f"Toplam {len(kombinasyonlar)} parametre kombinasyonu, {N_FOLDS} fold ile taranacak.")
    print("Bu uzun sürebilir...\n")

    sonuclar = []
    for i, degerler in enumerate(kombinasyonlar, 1):
        params = dict(zip(anahtarlar, degerler))
        fold_skorlari = []

        for train_df, val_df in walk_forward_foldlar(df, N_FOLDS):
            if train_df.empty or val_df.empty:
                continue

            X_train, y_train = train_df[feature_cols], train_df[TARGET]
            X_val = val_df[feature_cols]
            train_gruplari = gruplari_hesapla(train_df)

            model = xgb.XGBRanker(
                objective="rank:ndcg",
                random_state=42,
                n_jobs=-1,
                **params,
            )
            model.fit(X_train, y_train, group=train_gruplari)
            skorlar = model.predict(X_val)
            fold_skorlari.append(race_level_accuracy(val_df, skorlar))

        ortalama = np.mean(fold_skorlari) if fold_skorlari else 0.0
        sonuclar.append((params, ortalama, fold_skorlari))
        print(f"[{i}/{len(kombinasyonlar)}] {params} -> ort. isabet: {ortalama:.2%} "
              f"(fold'lar: {[f'{s:.2%}' for s in fold_skorlari]})")

    sonuclar.sort(key=lambda x: x[1], reverse=True)
    print("\n--- EN İYİ 5 KOMBİNASYON ---")
    for params, ortalama, fold_skorlari in sonuclar[:5]:
        print(f"{ortalama:.2%}  {params}")

    print("\nEn iyi parametreleri train_model.py'deki DEFAULT_PARAMS'a elle kopyala.")


if __name__ == "__main__":
    tune()