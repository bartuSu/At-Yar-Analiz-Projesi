"""
Verilen bütçeyle, 6 ayaklık (Altılı Ganyan) bir kupon için en yüksek
tahmini tutma olasılığına sahip at kombinasyonunu bulur.

GÜNCELLEME: LightGBM LambdaRank modelleri kullanılıyor. Ham ranking
skorları, her ayak İÇİNDE softmax ile göreceli olasılığa çevriliyor.

İKİ MODEL, İKİ FARKLI AMAÇLA KULLANILIYOR:
- GANYANLI model (model.pkl): genel/mutlak olasılık tahmini ve bütçe
  optimizasyonu için.
- GANYANSIZ model (model_ganyansiz.pkl): SADECE "sürpriz" at seçimi
  için - piyasadan bağımsız görüş, "value" hesaplaması bunun üstünden.

İKİ KUPON ÜRETİLİR:
1. GÜVENLİ kupon: her ayakta ganyanlı modelin en yüksek olasılık verdiği N at.
2. SÜRPRİZ kupon: aynı bütçe/yapı, en fazla SURPRIZ_AYAK_SAYISI ayakta,
   ganyansız modelin bulduğu en yüksek "value"lu at, TOPLAM OLASILIĞA
   EN AZ ZARAR VEREN ayaklarda zorla dahil edilir (dar ayaklarda swap
   yapılmaz, çünkü orantısız pahalıya patlar).

SINIRLAMA: Şu an sadece features.csv'de zaten bulunan (geçmiş,
tamamlanmış) yarışlar için çalışır.

Fiyat varsayımı: COMBINASYON_BIRIM_FIYATI aşağıda sabit tanımlı,
gerçek güncel fiyatı TJK'dan/bayiden doğrula.

ÖN KOŞUL: hem model.pkl (ganyanlı) hem model_ganyansiz.pkl (ganyansız)
data/processed/ altında, GÜNCEL ayarlarla eğitilmiş olmalı.

Kullanım:
    python generate_coupon.py --tarih 11/08/2026 --sehir Ankara --kosular 4 5 6 7 8 9 --butce 1000
"""
import argparse
import math

import joblib
import numpy as np
import pandas as pd

import scraper_config

COMBINASYON_BIRIM_FIYATI = 1.0  # TL - VARSAYIM, gerçek fiyatı doğrula
SURPRIZ_AYAK_SAYISI = 2


def load_model(dosya_adi: str):
    model_path = scraper_config.PROCESSED_DIR / dosya_adi
    if not model_path.exists():
        raise SystemExit(
            f"HATA: {model_path} bulunamadı. Önce train_model.py'yi ilgili "
            f"ayarlarla (ganyanlı için model.pkl, ganyansız için "
            f"model_ganyansiz.pkl) çalıştır."
        )
    bundle = joblib.load(model_path)
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


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def load_race_legs(
    tarih: str,
    sehir: str,
    kosu_no_list: list[int],
    model_ganyanli,
    feature_cols_ganyanli,
    model_ganyansiz,
    feature_cols_ganyansiz,
) -> list[dict]:
    """
    features.csv'den verilen tarih/şehir/koşu numaralarına ait atları çeker.
    Genel olasılık (ganyanlı model, softmax) VE bağımsız value sinyali
    (ganyansız model, softmax) ayrı ayrı hesaplanır.
    """
    df = pd.read_csv(scraper_config.PROCESSED_DIR / "features.csv")
    df = df[(df["tarih"] == tarih) & (df["sehir"] == sehir) & (df["kosu_no"].isin(kosu_no_list))]

    if df.empty:
        raise SystemExit(
            f"HATA: {tarih} {sehir} için verilen koşu numaralarında ({kosu_no_list}) "
            "features.csv'de veri bulunamadı."
        )

    pred_input_ganyanli = prepare_features_for_prediction(df.copy(), feature_cols_ganyanli)
    df["skor_ganyanli"] = model_ganyanli.predict(pred_input_ganyanli[feature_cols_ganyanli])

    pred_input_ganyansiz = prepare_features_for_prediction(df.copy(), feature_cols_ganyansiz)
    df["skor_bagimsiz"] = model_ganyansiz.predict(pred_input_ganyansiz[feature_cols_ganyansiz])

    ganyan_num = pd.to_numeric(df["ganyan"].astype(str).str.replace(",", "."), errors="coerce")
    df["ganyan_num"] = ganyan_num
    df["implied_prob_ham"] = (1 / ganyan_num).where(ganyan_num > 0)

    legs = []
    for kosu_no in sorted(kosu_no_list):
        grup = df[df["kosu_no"] == kosu_no].copy()
        if grup.empty:
            raise SystemExit(f"HATA: {kosu_no}. koşu için at bulunamadı.")

        grup["model_olasiligi"] = _softmax(grup["skor_ganyanli"].values)
        grup["piyasa_olasiligi"] = grup["implied_prob_ham"] / grup["implied_prob_ham"].sum()
        grup["model_olasiligi_bagimsiz"] = _softmax(grup["skor_bagimsiz"].values)
        grup["value"] = grup["model_olasiligi_bagimsiz"] - grup["piyasa_olasiligi"]
        grup = grup.sort_values("model_olasiligi", ascending=False).reset_index(drop=True)

        legs.append({
            "kosu_no": kosu_no,
            "atlar": grup["at_isim"].tolist(),
            "model_olasiliklari": grup["model_olasiligi"].tolist(),
            "piyasa_olasiliklari": grup["piyasa_olasiligi"].tolist(),
            "value_siralamasi": grup.sort_values("value", ascending=False)["at_isim"].tolist(),
            "value_degerleri": grup.sort_values("value", ascending=False)["value"].tolist(),
        })
    return legs


def en_iyi_tahsisi_bul(legs: list[dict], maks_kombinasyon: int) -> tuple[list[int], float]:
    """
    Çarpımsal knapsack (DP): her ayakta kaç at seçilirse toplam tutma
    olasılığı maksimize olur, bütçe (maks_kombinasyon) aşılmadan.
    GANYANLI modelin olasılıkları kullanılır.
    """
    dp = {1: (0.0, [])}

    for leg in legs:
        olasiliklar = leg["model_olasiliklari"]
        n_max = len(olasiliklar)
        kapsama = [sum(olasiliklar[:n]) for n in range(1, n_max + 1)]
        log_kapsama = [math.log(k) if k > 0 else float("-inf") for k in kapsama]

        yeni_dp = {}
        for k, (log_olasilik, gecmis) in dp.items():
            for n in range(1, n_max + 1):
                yeni_k = k * n
                if yeni_k > maks_kombinasyon:
                    break
                yeni_log_olasilik = log_olasilik + log_kapsama[n - 1]
                if yeni_k not in yeni_dp or yeni_log_olasilik > yeni_dp[yeni_k][0]:
                    yeni_dp[yeni_k] = (yeni_log_olasilik, gecmis + [n])
        dp = yeni_dp

    if not dp:
        raise SystemExit("HATA: Bu bütçeyle hiçbir kombinasyon mümkün değil (bütçe çok düşük).")

    en_iyi_k = max(dp.keys(), key=lambda k: dp[k][0])
    log_olasilik, tahsis = dp[en_iyi_k]
    return tahsis, math.exp(log_olasilik)


def guvenli_kupon_olustur(legs: list[dict], tahsis: list[int]) -> list[list[str]]:
    return [leg["atlar"][:n] for leg, n in zip(legs, tahsis)]


def surpriz_kupon_olustur(legs: list[dict], tahsis: list[int], surpriz_ayak_sayisi: int) -> tuple[list[list[str]], list[int]]:
    """
    Güvenli kuponla aynı yapı, ama GANYANSIZ modelin bulduğu pozitif
    value'lu ayaklar arasında, swap'in toplam olasılığa EN AZ zarar
    verdiği ayaklar seçilip değişiklik yapılır.
    """
    kupon = guvenli_kupon_olustur(legs, tahsis)
    degisen_ayaklar = []

    aday_swapler = []
    for i, (leg, secim, n) in enumerate(zip(legs, kupon, tahsis)):
        if leg["value_degerleri"][0] <= 0:
            continue
        surpriz_at = leg["value_siralamasi"][0]
        if surpriz_at in secim:
            continue

        at_index = {at: idx for idx, at in enumerate(leg["atlar"])}
        eski_toplam = sum(leg["model_olasiliklari"][at_index[at]] for at in secim)
        yeni_secim = secim[:-1] + [surpriz_at]
        yeni_toplam = sum(leg["model_olasiliklari"][at_index[at]] for at in yeni_secim)
        olasilik_kaybi_orani = (eski_toplam - yeni_toplam) / eski_toplam if eski_toplam > 0 else 1.0

        aday_swapler.append((i, olasilik_kaybi_orani, leg["value_degerleri"][0], yeni_secim))

    aday_swapler.sort(key=lambda x: x[1])

    for i, kayip_orani, value, yeni_secim in aday_swapler[:surpriz_ayak_sayisi]:
        kupon[i] = yeni_secim
        degisen_ayaklar.append(i)

    return kupon, degisen_ayaklar


def olasilik_hesapla(legs: list[dict], kupon: list[list[str]], key: str = "model_olasiliklari") -> float:
    toplam = 1.0
    for leg, secili_atlar in zip(legs, kupon):
        at_index = {at: idx for idx, at in enumerate(leg["atlar"])}
        toplam *= sum(leg[key][at_index[at]] for at in secili_atlar)
    return toplam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tarih", required=True, help="DD/MM/YYYY")
    parser.add_argument("--sehir", required=True)
    parser.add_argument("--kosular", required=True, type=int, nargs=6, help="6 koşu numarası, örn: 4 5 6 7 8 9")
    parser.add_argument("--butce", required=True, type=float)
    args = parser.parse_args()

    maks_kombinasyon = int(args.butce / COMBINASYON_BIRIM_FIYATI)

    print("Modeller yükleniyor...")
    model_ganyanli, feature_cols_ganyanli = load_model("model.pkl")
    model_ganyansiz, feature_cols_ganyansiz = load_model("model_ganyansiz.pkl")

    print(f"{args.tarih} {args.sehir} için {args.kosular} numaralı koşular yükleniyor...")
    legs = load_race_legs(
        args.tarih, args.sehir, args.kosular,
        model_ganyanli, feature_cols_ganyanli,
        model_ganyansiz, feature_cols_ganyansiz,
    )

    print(f"\nBütçe: {args.butce:.0f} TL -> maksimum {maks_kombinasyon} kombinasyon")
    print("En iyi tahsis hesaplanıyor...")
    tahsis, tahmini_olasilik = en_iyi_tahsisi_bul(legs, maks_kombinasyon)

    kombinasyon_sayisi = 1
    for n in tahsis:
        kombinasyon_sayisi *= n
    maliyet = kombinasyon_sayisi * COMBINASYON_BIRIM_FIYATI

    print(f"\n{'='*60}")
    print("GÜVENLİ KUPON (en yüksek tutma olasılığı, ganyanlı model)")
    print(f"{'='*60}")
    guvenli = guvenli_kupon_olustur(legs, tahsis)
    for leg, secim, n in zip(legs, guvenli, tahsis):
        print(f"  {leg['kosu_no']}. Koşu ({n} at): {', '.join(secim)}")
    print(f"\n  Toplam kombinasyon: {kombinasyon_sayisi}")
    print(f"  Maliyet: {maliyet:.0f} TL / Bütçe: {args.butce:.0f} TL")
    print(f"  Tahmini tutma olasılığı (model): {tahmini_olasilik:.2%}")

    piyasa_olasilik = olasilik_hesapla(legs, guvenli, "piyasa_olasiliklari")
    print(f"  (Karşılaştırma) Piyasa/ganyan bazlı tutma olasılığı: {piyasa_olasilik:.2%}")

    print(f"\n{'='*60}")
    print(f"SÜRPRİZ KUPON (en fazla {SURPRIZ_AYAK_SAYISI} ayakta, ganyansız modelin")
    print("bulduğu 'value' at, EN UCUZ BEDELLE zorlanır)")
    print(f"{'='*60}")
    surpriz, degisen_ayaklar = surpriz_kupon_olustur(legs, tahsis, SURPRIZ_AYAK_SAYISI)
    surpriz_olasilik = olasilik_hesapla(legs, surpriz, "model_olasiliklari")

    if not degisen_ayaklar:
        print("  Hiçbir ayakta uygun (pozitif value'lu ve makul bedelli) sürpriz bulunamadı.")
        print("  Güvenli kupon zaten en iyi seçenek.")
    else:
        for i, (leg, secim, n) in enumerate(zip(legs, surpriz, tahsis)):
            degisti = " <- SÜRPRİZ" if i in degisen_ayaklar else ""
            print(f"  {leg['kosu_no']}. Koşu ({n} at): {', '.join(secim)}{degisti}")

        print(f"\n  Toplam kombinasyon: {kombinasyon_sayisi} (aynı bütçe)")
        print(f"  Tahmini tutma olasılığı: {surpriz_olasilik:.2%}")

        fark = tahmini_olasilik - surpriz_olasilik
        if fark > 0:
            yuzde_kayip = fark / tahmini_olasilik * 100
            print(f"  Güvenli kupona göre kayıp: {fark:.2%} puan ({yuzde_kayip:.1f}% oransal düşüş)")
        else:
            print("  Güvenli kupona göre olasılık kaybı yok.")
        print("  NOT: Bu kombinasyonu muhtemelen daha az kişi oynuyor - tutarsa ödeme daha yüksek olabilir.")

    print(f"\n{'='*60}")
    print("UYARI: Bu bir yatırım/bahis tavsiyesi değildir. Tahmini olasılıklar")
    print("geçmiş veriye dayalı bir modelden geliyor, gerçek sonucu garanti etmez.")
    print("Kayıp riski gerçektir. Bütçenizi kaybetmeyi göze alabileceğiniz")
    print("miktarla sınırlı tutun.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()