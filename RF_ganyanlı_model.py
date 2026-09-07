"""
XGBoost XGBRanker modeli eğitir (LightGBM yerine).

NEDEN XGBOOST'A GEÇTİK: LightGBM LambdaRank denemesinde "tied score"
sorunu yaşadık - düşük num_leaves/yüksek min_child_samples ayarları
birbirinden farklı atları aynı yaprağa itip yapay olarak eşit skor
veriyordu (softmax sonrası "1/N" gibi bariz eşit olasılıklar). XGBoost'un
level-wise ağaç büyütme stratejisi (LightGBM'in leaf-wise'ından farklı)
bu sorunu azaltabilir.

EKSİK DEĞER STRATEJİSİ DEĞİŞTİ: Artık son_3_ort_sira ve
son_yaristan_gun_farki gibi "ilk yarış" kaynaklı NaN'lar MEDYAN ile
doldurulmuyor - XGBoost'un native missing-value desteğine bırakılıyor
(hangi dala gideceğine kendisi, veriye bakarak karar veriyor). NOT:
hızlı bir sentetik testte bunun median'a göre kesin bir üstünlük
sağlamadığını gördük - kesin bir çözüm değil, ama daha ilkesel bir
yaklaşım (keyfi bir sayı uydurmuyoruz).

Kullanım:
    python train_model.py
    -> data/processed/model.pkl (ganyanlı) ve konsola metrikler
"""
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score

import scraper_config

NUMERIC_FEATURES = [
    "kilo", "mesafe", "son_3_ort_sira", "gecmis_yaris_sayisi",
    "son_yaristan_gun_farki", "jokey_id_kazanma_orani",
    "antrenor_id_kazanma_orani", "sahip_id_kazanma_orani",
    "baba_kazanma_orani", "anne_kazanma_orani",
    "ganyan",  # ganyansız model için bu satırı kapat, MODEL_KAYIT_ADI'nı değiştir
]
CATEGORICAL_FEATURES = ["sehir", "pist"]
TARGET = "birinci_mi"

# tune_model.py'nin çıktısına bakarak burayı elle güncelle
DEFAULT_PARAMS = {
    "n_estimators":400,
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_child_weight":10,
}

MODEL_KAYIT_ADI = "model.pkl"


def load_and_prepare():
    df = pd.read_csv(scraper_config.PROCESSED_DIR / "features.csv", parse_dates=["tarih_dt"])

    eksik_kolonlar = [c for c in NUMERIC_FEATURES if c not in df.columns]
    if eksik_kolonlar:
        raise SystemExit(
            f"HATA: features.csv'de şu kolonlar yok: {eksik_kolonlar}\n"
            "Önce build_features.py'yi çalıştırıp features.csv'yi güncelle."
        )

    # DİKKAT: median ile doldurma YOK artık - sadece tip dönüşümü yapılıyor,
    # NaN'lar olduğu gibi bırakılıyor, XGBoost native olarak yönetiyor.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, dummy_na=True)
    cat_cols = [c for c in df.columns if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_FEATURES)]
    feature_cols = NUMERIC_FEATURES + cat_cols

    # KRİTİK: ranker için veri, önce tarihe, sonra HER tarih içinde
    # kosu_kodu'ya göre ARDIŞIK sıralı olmalı - gruplar bölünmesin.
    df = df.sort_values(["tarih_dt", "kosu_kodu"]).reset_index(drop=True)

    return df, feature_cols


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Son %test_ratio'luk tarih aralığını test seti yapar (zamana göre split)."""
    unique_dates = df["tarih_dt"].drop_duplicates().sort_values()
    cutoff_idx = int(len(unique_dates) * (1 - test_ratio))
    cutoff_date = unique_dates.iloc[cutoff_idx]

    train = df[df["tarih_dt"] < cutoff_date].reset_index(drop=True)
    test = df[df["tarih_dt"] >= cutoff_date].reset_index(drop=True)
    print(f"Cutoff tarihi: {cutoff_date.date()}")
    print(f"Train: {len(train)} satır, {train['kosu_kodu'].nunique()} koşu")
    print(f"Test:  {len(test)} satır, {test['kosu_kodu'].nunique()} koşu")
    return train, test


def gruplari_hesapla(df: pd.DataFrame) -> list[int]:
    """kosu_kodu'ya göre ardışık grup boyutlarını döner."""
    return df.groupby("kosu_kodu", sort=False).size().tolist()


def evaluate_race_level(test_df: pd.DataFrame, scores: np.ndarray) -> dict:
    """
    Koşu bazında değerlendirme: modelin en yüksek skor verdiği at
    gerçekten kazandı mı? Piyasa favorisiyle (en düşük ganyan) kıyasla.
    """
    test_df = test_df.copy()
    test_df["model_skor"] = scores

    dogru_model = 0
    dogru_favori = 0
    toplam_kosu = 0

    for kosu_kodu, grup in test_df.groupby("kosu_kodu"):
        if grup["birinci_mi"].sum() != 1:
            continue
        toplam_kosu += 1

        model_secimi = grup.loc[grup["model_skor"].idxmax()]
        if model_secimi["birinci_mi"] == 1:
            dogru_model += 1

        favori = grup.loc[grup["ganyan"].idxmin()]
        if favori["birinci_mi"] == 1:
            dogru_favori += 1

    return {
        "toplam_kosu": toplam_kosu,
        "model_isabet_orani": dogru_model / toplam_kosu if toplam_kosu else None,
        "favori_isabet_orani": dogru_favori / toplam_kosu if toplam_kosu else None,
    }


def tied_score_orani(test_df: pd.DataFrame, scores: np.ndarray) -> float:
    """
    Teşhis amaçlı: bir koşudaki atların ortalama ne kadarı BENZERSİZ skor
    alıyor. 1.0'a yakınsa her at farklı, düşükse (0.3-0.6 gibi) atlar
    skor paylaşıyor demektir - LightGBM'de yaşadığımız sorunun kontrolü.
    """
    tmp = test_df.copy()
    tmp["skor"] = scores
    farkli = tmp.groupby("kosu_kodu")["skor"].nunique()
    toplam = tmp.groupby("kosu_kodu").size()
    return (farkli / toplam).mean()


def train():
    df, feature_cols = load_and_prepare()
    train_df, test_df = temporal_split(df)

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    train_gruplari = gruplari_hesapla(train_df)

    print(f"\nKullanılan parametreler: {DEFAULT_PARAMS}")
    print("Model eğitiliyor (XGBoost XGBRanker, rank:ndcg)...")
    model = xgb.XGBRanker(
        objective="rank:ndcg",
        random_state=42,
        n_jobs=-1,
        **DEFAULT_PARAMS,
    )
    model.fit(X_train, y_train, group=train_gruplari)

    raw_scores = model.predict(X_test)

    print("\n--- TİED-SCORE KONTROLÜ (LightGBM'de sorun yaşadığımız yer) ---")
    oran = tied_score_orani(test_df, raw_scores)
    print(f"Ortalama benzersiz-skor oranı: {oran:.2%} (1.0'a ne kadar yakınsa o kadar iyi)")

    print("\n--- KOŞU BAZINDA METRİKLER ---")
    race_metrics = evaluate_race_level(test_df, raw_scores)
    print(f"Test setindeki koşu sayısı: {race_metrics['toplam_kosu']}")
    print(f"Modelin isabet oranı: {race_metrics['model_isabet_orani']:.2%}")
    print(f"Piyasa favorisinin isabet oranı: {race_metrics['favori_isabet_orani']:.2%}")

    print("\n--- NDCG@1 / NDCG@3 ---")
    ndcg1_list, ndcg3_list = [], []
    test_df_scored = test_df.copy()
    test_df_scored["model_skor"] = raw_scores
    for kosu_kodu, grup in test_df_scored.groupby("kosu_kodu"):
        if len(grup) < 2 or grup["birinci_mi"].sum() != 1:
            continue
        y_true = grup["birinci_mi"].values.reshape(1, -1)
        y_score = grup["model_skor"].values.reshape(1, -1)
        ndcg1_list.append(ndcg_score(y_true, y_score, k=1))
        ndcg3_list.append(ndcg_score(y_true, y_score, k=min(3, grup.shape[0])))
    print(f"Ortalama NDCG@1: {np.mean(ndcg1_list):.4f}")
    print(f"Ortalama NDCG@3: {np.mean(ndcg3_list):.4f}")

    print("\n--- EN ÖNEMLİ 15 FEATURE ---")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(importances.head(15))

    model_path = scraper_config.PROCESSED_DIR / MODEL_KAYIT_ADI
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"\nModel kaydedildi: {model_path}")


if __name__ == "__main__":
    train()