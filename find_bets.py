"""
Piyasanın (ganyan) gözden kaçırdığı "value" adaylarını bulur ve basit
bir flat-betting backtest'i ile bu stratejinin geçmişte kâr getirip
getirmediğini test eder.

GÜNCELLEME: Artık LightGBM LambdaRank modeli kullanılıyor. Bu model
predict_proba değil, ham bir "skor" üretiyor - piyasa olasılığıyla
karşılaştırılabilir hale getirmek için, her koşu İÇİNDE skorlar
SOFTMAX ile normalize ediliyor (ranking skorlarını göreceli bir
olasılık dağılımına çeviren standart bir yöntem).

MANTIK:
- Ganyan -> implied_prob = 1/ganyan, koşu içinde normalize (bahis marjı çıkar).
- GANYANSIZ model -> softmax(ham skor) = piyasadan bağımsız olasılık.
- "Value" = model_olasiligi - piyasa_olasiligi.

UYARI: Bu bir yatırım/bahis tavsiyesi değildir. Backtest sonuçları
geçmişe dönüktür, gelecekteki performansı garanti etmez.

ÖN KOŞUL: train_model.py'nin GANYANSIZ versiyonu çalıştırılmış olmalı,
data/processed/model_ganyansiz.pkl bulunmalı.

Kullanım:
    python find_value_bets.py
"""
import joblib
import numpy as np
import pandas as pd

import scraper_config

MODEL_PATH = scraper_config.PROCESSED_DIR / "model_ganyansiz.pkl"
VALUE_ESIGI = 0.05


def load_model_and_features():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"HATA: {MODEL_PATH} bulunamadı.\n"
            "Önce train_model.py'yi GANYANSIZ ayarlarla (NUMERIC_FEATURES'dan "
            "'ganyan' çıkarılmış, MODEL_KAYIT_ADI='model_ganyansiz.pkl') çalıştır."
        )
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]


def prepare_features_for_prediction(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in feature_cols if not c.startswith(("sehir_", "pist_"))]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df = pd.get_dummies(df, columns=["sehir", "pist"], dummy_na=True)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df


def _softmax(s: pd.Series) -> pd.Series:
    x = s.values - s.values.max()
    e = np.exp(x)
    return pd.Series(e / e.sum(), index=s.index)


def softmax_normalize_within_race(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    """Ham ranking skorlarını, her koşu İÇİNDE softmax ile olasılığa çevirir."""
    df[out_col] = df.groupby("kosu_kodu")[score_col].transform(_softmax)
    return df


def normalize_market_within_race(df: pd.DataFrame) -> pd.DataFrame:
    ganyan_num = pd.to_numeric(df["ganyan"].astype(str).str.replace(",", "."), errors="coerce")
    df["implied_prob_ham"] = np.where(ganyan_num > 0, 1 / ganyan_num, np.nan)
    toplam = df.groupby("kosu_kodu")["implied_prob_ham"].transform("sum")
    df["piyasa_olasiligi"] = df["implied_prob_ham"] / toplam
    df["ganyan_num"] = ganyan_num
    return df


def temporal_test_split(df: pd.DataFrame, test_ratio: float = 0.2) -> pd.DataFrame:
    unique_dates = df["tarih_dt"].drop_duplicates().sort_values()
    cutoff_idx = int(len(unique_dates) * (1 - test_ratio))
    cutoff_date = unique_dates.iloc[cutoff_idx]
    return df[df["tarih_dt"] >= cutoff_date].copy()


def backtest_flat_betting(df: pd.DataFrame, value_esigi: float, stake: float = 1.0) -> dict:
    bahisler = []
    for kosu_kodu, grup in df.groupby("kosu_kodu"):
        if grup["birinci_mi"].sum() != 1:
            continue
        adaylar = grup[grup["value"] >= value_esigi]
        if adaylar.empty:
            continue
        secim = adaylar.loc[adaylar["value"].idxmax()]
        kazandi = secim["birinci_mi"] == 1
        kar_zarar = (secim["ganyan_num"] - 1) * stake if kazandi else -stake
        bahisler.append({
            "kosu_kodu": kosu_kodu, "tarih": secim["tarih"], "sehir": secim["sehir"],
            "at_isim": secim["at_isim"], "ganyan": secim["ganyan_num"],
            "model_olasiligi": secim["model_olasiligi"], "piyasa_olasiligi": secim["piyasa_olasiligi"],
            "value": secim["value"], "kazandi": kazandi, "kar_zarar": kar_zarar,
        })

    bahis_df = pd.DataFrame(bahisler)
    if bahis_df.empty:
        return {"bahis_sayisi": 0, "bahis_df": bahis_df}

    toplam_stake = len(bahis_df) * stake
    toplam_kar = bahis_df["kar_zarar"].sum()
    return {
        "bahis_sayisi": len(bahis_df),
        "kazanma_sayisi": int(bahis_df["kazandi"].sum()),
        "kazanma_orani": bahis_df["kazandi"].mean(),
        "toplam_stake": toplam_stake,
        "toplam_kar_zarar": toplam_kar,
        "roi": toplam_kar / toplam_stake,
        "bahis_df": bahis_df,
    }


def backtest_baseline(df: pd.DataFrame, strateji: str, stake: float = 1.0, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    bahisler = []
    for kosu_kodu, grup in df.groupby("kosu_kodu"):
        if grup["birinci_mi"].sum() != 1:
            continue
        if strateji == "favori":
            secim = grup.loc[grup["ganyan_num"].idxmin()]
        elif strateji == "rastgele":
            secim = grup.sample(n=1, random_state=rng.integers(0, 1_000_000)).iloc[0]
        else:
            raise ValueError(strateji)
        kazandi = secim["birinci_mi"] == 1
        kar_zarar = (secim["ganyan_num"] - 1) * stake if kazandi else -stake
        bahisler.append({"kazandi": kazandi, "kar_zarar": kar_zarar})

    bdf = pd.DataFrame(bahisler)
    toplam_stake = len(bdf) * stake
    return {
        "strateji": strateji, "bahis_sayisi": len(bdf),
        "kazanma_orani": bdf["kazandi"].mean(), "roi": bdf["kar_zarar"].sum() / toplam_stake,
    }


def main():
    print("Model ve feature listesi yükleniyor...")
    model, feature_cols = load_model_and_features()

    print("features.csv yükleniyor...")
    df = pd.read_csv(scraper_config.PROCESSED_DIR / "features.csv", parse_dates=["tarih_dt"])

    print("Test dönemi ayrılıyor...")
    test_df = temporal_test_split(df)
    print(f"Test seti: {len(test_df)} satır, {test_df['kosu_kodu'].nunique()} koşu.")

    print("Piyasa olasılıkları hesaplanıyor...")
    test_df = normalize_market_within_race(test_df)

    print("Model skorları üretiliyor (ranking skoru -> softmax ile olasılık)...")
    pred_input = prepare_features_for_prediction(test_df.copy(), feature_cols)
    test_df["model_skor_ham"] = model.predict(pred_input[feature_cols])
    test_df = softmax_normalize_within_race(test_df, "model_skor_ham", "model_olasiligi")

    test_df["value"] = test_df["model_olasiligi"] - test_df["piyasa_olasiligi"]

    print("\n--- EN YÜKSEK 'VALUE' GÖSTEREN 15 AT-KOŞU ---")
    en_iyi = test_df.sort_values("value", ascending=False).head(15)
    print(en_iyi[["tarih", "sehir", "at_isim", "ganyan_num", "piyasa_olasiligi",
                   "model_olasiligi", "value", "birinci_mi"]].to_string(index=False))

    print(f"\n--- BACKTEST: value >= {VALUE_ESIGI} olan atlara flat-bet stratejisi ---")
    sonuc = backtest_flat_betting(test_df, VALUE_ESIGI)

    print("\n--- REFERANS STRATEJİLER ---")
    favori_sonuc = backtest_baseline(test_df, "favori")
    print(f"Her koşuda favoriye oyna: {favori_sonuc['bahis_sayisi']} bahis, "
          f"kazanma {favori_sonuc['kazanma_orani']:.1%}, ROI {favori_sonuc['roi']:+.1%}")
    rastgele_sonuc = backtest_baseline(test_df, "rastgele")
    print(f"Her koşuda rastgele ata oyna: {rastgele_sonuc['bahis_sayisi']} bahis, "
          f"kazanma {rastgele_sonuc['kazanma_orani']:.1%}, ROI {rastgele_sonuc['roi']:+.1%}")

    if sonuc["bahis_sayisi"] == 0:
        print("\nBu eşikte hiç bahis fırsatı bulunamadı.")
        return

    print(f"\nToplam bahis sayısı: {sonuc['bahis_sayisi']}")
    print(f"Kazanma sayısı: {sonuc['kazanma_sayisi']} ({sonuc['kazanma_orani']:.1%})")
    print(f"ROI: {sonuc['roi']:+.1%}")

    print("\n--- EŞİK TARAMASI ---")
    for esik in [0.05, 0.10, 0.15, 0.20, 0.25]:
        sonuc_esik = backtest_flat_betting(test_df, esik)
        if sonuc_esik["bahis_sayisi"] == 0:
            print(f"value >= {esik}: hiç bahis yok")
            continue
        print(f"value >= {esik}: {sonuc_esik['bahis_sayisi']} bahis, "
              f"kazanma {sonuc_esik['kazanma_orani']:.1%}, ROI {sonuc_esik['roi']:+.1%}")

    out_path = scraper_config.PROCESSED_DIR / "value_bets_backtest.csv"
    sonuc["bahis_df"].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nDetaylı bahis listesi kaydedildi: {out_path}")

    print("\nNOT: Bu backtest kapanış ganyanı kullanıyor; gerçek bahis anındaki "
          "oran farklı olabilir. Sonuç, gelecekteki performansı garanti etmez.")


if __name__ == "__main__":
    main()