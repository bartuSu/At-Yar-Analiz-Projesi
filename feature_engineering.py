"""
Ham JSON'lardan (günlük sonuçlar + at geçmişi) ML için düz bir feature
tablosu (pandas DataFrame / CSV) üretir.

KRİTİK KURAL: Bir koşu için feature hesaplarken SADECE o koşunun
tarihinden ÖNCEKİ veriyi kullan. Sonraki yarışları/deneyimi feature'a
karıştırmak "veri sızıntısı" (data leakage) olur, modeli gerçekte
olmayan bir başarıyla kandırır.

GÜNCELLEME: Jokey/antrenör/sahip/baba/anne kazanma oranları artık
"shrinkage" (Bayesian smoothing) ile hesaplanıyor. Az yarışlı bir
varlığın (örn. 2 yarışta 1 kazanç = ham %50) oranı, veri azken genel
ortalamaya doğru çekiliyor - böylece küçük örneklem gürültüsü modelin
"az bilinen atları abartması"na yol açmıyor.

ÖN KOŞUL: fetch_horses_for_dataset.py çalıştırılmış olmalı, yani
data/raw/at_gecmisi/ klasöründe dataset'teki atların bio bilgisi
(baba/anne dahil) bulunmalı.

Kullanım:
    python build_features.py
    -> data/processed/features.csv üretir
"""
import json
from datetime import datetime

import pandas as pd

import scraper_config

# Shrinkage gücü: "kaç sanal yarış kadar genel ortalamaya güven" demek.
# Küçükse (örn. 5) az veriye daha çabuk güvenir, büyükse (örn. 30) daha
# temkinli olur. 15 makul bir orta nokta.
SHRINKAGE_K = 15


def _parse_tarih(s: str) -> datetime:
    return datetime.strptime(s, "%d/%m/%Y")


def load_all_daily_results() -> pd.DataFrame:
    """Tüm günlük sonuç JSON'larını tek bir düz tabloya indirger (1 satır = 1 at + 1 koşu)."""
    rows = []
    for json_path in scraper_config.RAW_RESULTS_DIR.glob("*.json"):
        races = json.loads(json_path.read_text(encoding="utf-8"))
        for race in races:
            for at in race["atlar"]:
                if at.get("at_id") is None:
                    continue  # "koşmaz" gibi eksik kayıtları atla
                rows.append({
                    "kosu_kodu": race["kosu_kodu"],
                    "tarih": race["tarih"],
                    "tarih_dt": _parse_tarih(race["tarih"]),
                    "sehir": race["sehir"],
                    "kosu_no": race.get("kosu_no"),
                    "mesafe": race.get("mesafe"),
                    "pist": race.get("pist"),
                    "at_id": at["at_id"],
                    "at_isim": at["at_isim"],
                    "kilo": at.get("kilo"),
                    "jokey_id": at.get("jokey_id"),
                    "antrenor_id": at.get("antrenor_id"),
                    "sahip_id": at.get("sahip_id"),
                    "ganyan": at.get("ganyan"),
                    "start_no": at.get("start_no"),
                    "sira_no": at.get("sira_no"),
                    "birinci_mi": 1 if str(at.get("sira_no")) == "1" else 0,
                })
    return pd.DataFrame(rows)


def add_gecmis_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her at+koşu satırı için, o koşudan ÖNCEKİ yarışlarına bakarak
    'son N yarışta ortalama bitiş sırası' gibi feature'lar ekler.
    """
    df = df.sort_values(["at_id", "tarih_dt"]).reset_index(drop=True)

    df["son_3_ort_sira"] = (
        df.groupby("at_id")["sira_no"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    df["gecmis_yaris_sayisi"] = df.groupby("at_id").cumcount()
    df["son_yaristan_gun_farki"] = (
        df.groupby("at_id")["tarih_dt"].diff().dt.days
    )

    return df


def _smoothed_expanding_rate(df: pd.DataFrame, group_col: str, target_col: str, global_rate: float, k: float) -> pd.Series:
    """
    Bayesian shrinkage ile 'o koşudan ÖNCEKİ' kazanma oranını hesaplar.
    Az yarışlı gruplar (örn. yeni jokey) genel ortalamaya yakın kalır,
    çok yarışlı gruplar kendi gerçek oranına yakınsar.

    df 'group_col, tarih_dt' sırasına göre sıralı olmalı (çağıran fonksiyon garantiliyor).
    """
    grouped = df.groupby(group_col)[target_col]
    onceki_kazanc = grouped.apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=0, drop=True)
    onceki_yaris = grouped.apply(lambda s: s.shift(1).expanding().count()).reset_index(level=0, drop=True)

    return (onceki_kazanc + k * global_rate) / (onceki_yaris + k)


def add_jokey_antrenor_basari(df: pd.DataFrame) -> pd.DataFrame:
    """Jokey/antrenör/sahip için, shrinkage'li geçmiş kazanma oranı."""
    genel_ortalama = df["birinci_mi"].mean()
    print(f"  Genel kazanma oranı (prior): {genel_ortalama:.4f}")

    for col in ["jokey_id", "antrenor_id", "sahip_id"]:
        df = df.sort_values([col, "tarih_dt"])
        df[f"{col}_kazanma_orani"] = _smoothed_expanding_rate(df, col, "birinci_mi", genel_ortalama, SHRINKAGE_K)

    df = df.sort_values(["at_id", "tarih_dt"]).reset_index(drop=True)
    return df


def load_bio_data() -> dict:
    """Tüm at bio JSON'larını at_id -> {baba, anne} sözlüğüne indirger."""
    bio_map = {}
    for json_path in scraper_config.RAW_HORSE_DIR.glob("*.json"):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        bio_map[data["at_id"]] = {
            "baba": data["bio"].get("baba"),
            "anne": data["bio"].get("anne"),
        }
    return bio_map


def add_orijin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Baba/anne bilgisini ekler + shrinkage'li geçmiş kazanma oranı hesaplar."""
    bio_map = load_bio_data()
    eslesen = df["at_id"].isin(bio_map.keys()).sum()
    print(f"  {eslesen}/{len(df)} satır için bio verisi bulundu ({eslesen/len(df):.1%}).")

    df["baba"] = df["at_id"].map(lambda x: bio_map.get(x, {}).get("baba"))
    df["anne"] = df["at_id"].map(lambda x: bio_map.get(x, {}).get("anne"))

    genel_ortalama = df["birinci_mi"].mean()

    for col in ["baba", "anne"]:
        df = df.sort_values([col, "tarih_dt"])
        df[f"{col}_kazanma_orani"] = _smoothed_expanding_rate(df, col, "birinci_mi", genel_ortalama, SHRINKAGE_K)

    df = df.sort_values(["at_id", "tarih_dt"]).reset_index(drop=True)
    return df


def build():
    print("Günlük sonuçlar yükleniyor...")
    df = load_all_daily_results()
    print(f"{len(df)} at-koşu satırı bulundu.")

    if df.empty:
        print("UYARI: Hiç veri yok. Önce backfill.py'yi çalıştırıp veri topla.")
        return

    print("Geçmiş form feature'ları hesaplanıyor...")
    df = add_gecmis_form_features(df)

    print("Jokey/antrenör/sahip başarı oranları hesaplanıyor (shrinkage'li)...")
    df = add_jokey_antrenor_basari(df)

    print("Orijin (baba/anne) feature'ları hesaplanıyor (shrinkage'li)...")
    df = add_orijin_features(df)

    out_path = scraper_config.PROCESSED_DIR / "features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Kaydedildi: {out_path} ({len(df)} satır, {len(df.columns)} kolon)")


if __name__ == "__main__":
    build()