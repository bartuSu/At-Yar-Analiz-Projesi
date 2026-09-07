"""
At Yarışı Kupon Önericisi - Ana Dashboard (CSV tabanlı, sadeleştirilmiş)

Kullanıcı:
1. TJK'nın resmi günlük yarış programı CSV'sini yükler (PDF DEĞİL - çok
   daha güvenilir, yapılandırılmış veri, manuel düzeltmeye gerek kalmıyor)
2. Hangi altılıyı oynayacağını seçer (CSV'den otomatik tespit edilir,
   bulunamazsa elle işaretleme seçeneği var)
3. AGF kaynağını seçer: CSV'nin içindeki (otomatik) ya da TJK AGF
   sayfasından kopyaladığı güncel metin
4. Bütçesini girer

Sistem:
- Ganyanlı model (model.pkl) ile bütçe optimizasyonu yapıp GÜVENLİ kupon oluşturur
- Ganyansız model (model_ganyansiz.pkl) ile "value" hesaplayıp, en az
  bedelli 1-2 ayakta SÜRPRİZ at zorlayarak TEK BİR birleşik kupon üretir
- Kullanıcıya bu birleşik kuponu "at no - isim" formatında ve tahmini
  tutma olasılığını gösterir

Çalıştırma:
    streamlit run main_dashboard.py
"""
import json
import math
import re
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import scraper_config
from parse_program_csv import parse_program_csv, sehir_ve_tarih_tahmin_et, altili_secenekleri_bul

SURPRIZ_AYAK_SAYISI = 2
COMBINASYON_BIRIM_FIYATI = 1.0  # TL - VARSAYIM, gerçek fiyatı doğrula

# ============================================================
# 1) AGF METİN PARSER (web sitesinden yapıştırma seçeneği için)
# ============================================================

AYAK_BASLIK_RE = re.compile(r"(\d+)\s*\.\s*AYAK", re.IGNORECASE)
AT_YUZDE_RE = re.compile(r"(\d+)\s*\(\s*%\s*([\d]+(?:[.,]\d+)?)\s*\)")


def _yuzde_to_float(s: str) -> float:
    return float(s.replace(",", "."))


def parse_agf_text(ham_metin: str) -> dict[int, dict]:
    """{ayak_no: {at_no: yuzde}} döner - sadece koşacak atlar (koşmaz hariç)."""
    baslik_eslesmeleri = list(AYAK_BASLIK_RE.finditer(ham_metin))
    if not baslik_eslesmeleri:
        raise ValueError("Metinde 'N. AYAK' başlığı bulunamadı. Doğru kopyaladığından emin ol.")

    sonuc = {}
    for i, esleme in enumerate(baslik_eslesmeleri):
        ayak_no = int(esleme.group(1))
        blok_baslangic = esleme.end()
        blok_bitis = baslik_eslesmeleri[i + 1].start() if i + 1 < len(baslik_eslesmeleri) else len(ham_metin)
        blok = ham_metin[blok_baslangic:blok_bitis]
        kosanlar_metni = blok.split("KOŞMAZ", 1)[0] if "KOŞMAZ" in blok else blok
        sonuc[ayak_no] = {int(at_no): _yuzde_to_float(yuzde) for at_no, yuzde in AT_YUZDE_RE.findall(kosanlar_metni)}
    return sonuc


def normalize_isim(isim: str) -> str:
    isim = re.sub(r"\(ap\.?\)", "", isim, flags=re.IGNORECASE)
    ceviri = str.maketrans("çğıöşüİ", "cgiosui")
    isim = isim.lower().translate(ceviri)
    return re.sub(r"[^a-z]", "", isim)


# ============================================================
# 2) VERİ VE MODEL YÜKLEME (Streamlit cache ile)
# ============================================================

@st.cache_resource
def load_models():
    ganyanli = joblib.load(scraper_config.PROCESSED_DIR / "model.pkl")
    ganyansiz = joblib.load(scraper_config.PROCESSED_DIR / "model_ganyansiz.pkl")
    return ganyanli["model"], ganyanli["feature_cols"], ganyansiz["model"], ganyansiz["feature_cols"]


@st.cache_data
def load_name_cache():
    path = scraper_config.PROCESSED_DIR / "name_id_cache.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_features_lookup():
    """features.csv'den her varlık için EN GÜNCEL satırı ve genel medyanları çıkarır."""
    df = pd.read_csv(scraper_config.PROCESSED_DIR / "features.csv", parse_dates=["tarih_dt"])
    df = df.sort_values("tarih_dt")

    at_son = df.drop_duplicates("at_id", keep="last").set_index("at_id")
    jokey_son = df.dropna(subset=["jokey_id"]).drop_duplicates("jokey_id", keep="last").set_index("jokey_id")
    antrenor_son = df.dropna(subset=["antrenor_id"]).drop_duplicates("antrenor_id", keep="last").set_index("antrenor_id")
    sahip_son = df.dropna(subset=["sahip_id"]).drop_duplicates("sahip_id", keep="last").set_index("sahip_id")
    # YENİ: baba/anne isim bazlı lookup - CSV'den doğrudan baba/anne ismi geldiği için,
    # at_id eşleşmesi başarısız olsa bile soy hattı bilgisiyle tahmin yapabiliyoruz.
    baba_son = df.dropna(subset=["baba"]).drop_duplicates("baba", keep="last").set_index("baba")
    anne_son = df.dropna(subset=["anne"]).drop_duplicates("anne", keep="last").set_index("anne")

    medyanlar = {
        "son_3_ort_sira": df["son_3_ort_sira"].median(),
        "gecmis_yaris_sayisi": df["gecmis_yaris_sayisi"].median(),
        "son_yaristan_gun_farki": df["son_yaristan_gun_farki"].median(),
        "jokey_id_kazanma_orani": df["jokey_id_kazanma_orani"].median(),
        "antrenor_id_kazanma_orani": df["antrenor_id_kazanma_orani"].median(),
        "sahip_id_kazanma_orani": df["sahip_id_kazanma_orani"].median(),
        "baba_kazanma_orani": df["baba_kazanma_orani"].median(),
        "anne_kazanma_orani": df["anne_kazanma_orani"].median(),
    }
    return at_son, jokey_son, antrenor_son, sahip_son, baba_son, anne_son, medyanlar


# ============================================================
# 3) CANLI FEATURE ÜRETİMİ
# ============================================================

def at_id_bul(at_isim: str, at_son_df: pd.DataFrame):
    hedef = at_isim.strip().upper()
    eslesenler = at_son_df[at_son_df["at_isim"].str.strip().str.upper() == hedef]
    if eslesenler.empty:
        return None
    return eslesenler.index[0]


def canli_satir_olustur(at: dict, name_cache, at_son, jokey_son, antrenor_son, sahip_son,
                         baba_son, anne_son, medyanlar) -> dict:
    satir = {"kilo": at["kilo"]}

    at_id = at_id_bul(at["isim"], at_son)
    if at_id is not None:
        gecmis = at_son.loc[at_id]
        satir["son_3_ort_sira"] = gecmis.get("son_3_ort_sira", medyanlar["son_3_ort_sira"])
        satir["gecmis_yaris_sayisi"] = gecmis.get("gecmis_yaris_sayisi", medyanlar["gecmis_yaris_sayisi"]) + 1
        satir["son_yaristan_gun_farki"] = medyanlar["son_yaristan_gun_farki"]
        satir["_at_bulundu"] = True
    else:
        satir["son_3_ort_sira"] = medyanlar["son_3_ort_sira"]
        satir["gecmis_yaris_sayisi"] = 0
        satir["son_yaristan_gun_farki"] = medyanlar["son_yaristan_gun_farki"]
        satir["_at_bulundu"] = False

    # Baba/anne kazanma oranı: CSV'den gelen isimle DOĞRUDAN ara (at_id eşleşmesine
    # bağımlı değil - yeni/bilinmeyen bir at bile, ebeveyni tanınıyorsa bundan faydalanır)
    for alan, son_df, kolon in [
        (at.get("baba"), baba_son, "baba_kazanma_orani"),
        (at.get("anne"), anne_son, "anne_kazanma_orani"),
    ]:
        if alan and alan in son_df.index:
            satir[kolon] = son_df.loc[alan, kolon]
        else:
            satir[kolon] = medyanlar[kolon]

    for alan, cache_key, son_df, kolon in [
        ("jokey", "jokey_adi_to_id", jokey_son, "jokey_id_kazanma_orani"),
        ("antrenor", "antrenor_adi_to_id", antrenor_son, "antrenor_id_kazanma_orani"),
        ("sahip", "sahip_adi_to_id", sahip_son, "sahip_id_kazanma_orani"),
    ]:
        isim_norm = normalize_isim(at.get(alan, "") or "")
        id_deger = name_cache.get(cache_key, {}).get(isim_norm) if name_cache else None
        if id_deger is not None and id_deger in son_df.index:
            satir[kolon] = son_df.loc[id_deger, kolon]
        else:
            satir[kolon] = medyanlar[kolon]

    return satir


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def prepare_for_model(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in feature_cols if not c.startswith(("sehir_", "pist_"))]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = pd.get_dummies(df, columns=["sehir", "pist"], dummy_na=True)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df


def kosu_verisi_hazirla(kosu_no_list, altili_no, kosular_parsed, sehir,
                         agf_kaynagi, agf_yapistirilan,
                         model_g, fc_g, model_gs, fc_gs,
                         name_cache, at_son, jokey_son, antrenor_son, sahip_son,
                         baba_son, anne_son, medyanlar):
    legs = []
    bulunamayan_atlar = []

    for ayak_index, kosu_no in enumerate(kosu_no_list, start=1):
        kosu = kosular_parsed[kosu_no]
        koşacak_atlar = [a for a in kosu["atlar"] if not a["kosmaz"]]

        if not koşacak_atlar:
            raise ValueError(f"{kosu_no}. koşuda koşacak at bulunamadı.")

        satirlar = []
        for at in koşacak_atlar:
            satir = canli_satir_olustur(at, name_cache, at_son, jokey_son, antrenor_son,
                                         sahip_son, baba_son, anne_son, medyanlar)
            satir["at_no"] = at["at_no"]
            satir["at_isim"] = at["isim"]
            satir["mesafe"] = kosu.get("mesafe") or 1400
            satir["pist"] = kosu.get("pist") or "Çim"
            satir["sehir"] = sehir

            # AGF kaynağı: web'den yapıştırılan varsa onu tercih et, yoksa CSV'nin içindeki
            if agf_kaynagi == "yapistir" and agf_yapistirilan:
                agf_yuzde = agf_yapistirilan.get(ayak_index, {}).get(at["at_no"])
            else:
                degerler = at.get("agf_degerleri") or []
                if len(degerler) > 1:
                    # bu koşu birden fazla altılıya ait (çakışma bölgesi) - doğru altılının
                    # AGF'sini seçmemiz lazım. Değerlerin sırası CSV'de altılı sırasına göre.
                    idx = 0 if altili_no == 1 else min(altili_no - 1, len(degerler) - 1)
                    agf_yuzde = degerler[idx]
                elif len(degerler) == 1:
                    agf_yuzde = degerler[0]
                else:
                    agf_yuzde = None

            if agf_yuzde and agf_yuzde > 0:
                satir["ganyan"] = 100 / agf_yuzde
                satir["agf_yuzde"] = agf_yuzde
            else:
                satir["ganyan"] = 50.0
                satir["agf_yuzde"] = 0.0

            if not satir.pop("_at_bulundu"):
                bulunamayan_atlar.append(at["isim"])

            satirlar.append(satir)

        leg_df = pd.DataFrame(satirlar)
        leg_df["at_etiket"] = leg_df["at_no"].astype(int).astype(str) + " - " + leg_df["at_isim"]

        pred_g = prepare_for_model(leg_df.copy(), fc_g)
        leg_df["model_olasiligi"] = _softmax(model_g.predict(pred_g[fc_g]))

        pred_gs = prepare_for_model(leg_df.copy(), fc_gs)
        leg_df["model_olasiligi_bagimsiz"] = _softmax(model_gs.predict(pred_gs[fc_gs]))

        toplam_agf = leg_df["agf_yuzde"].sum()
        leg_df["piyasa_olasiligi"] = leg_df["agf_yuzde"] / toplam_agf if toplam_agf > 0 else 1 / len(leg_df)
        leg_df["value"] = leg_df["model_olasiligi_bagimsiz"] - leg_df["piyasa_olasiligi"]

        leg_df = leg_df.sort_values("model_olasiligi", ascending=False).reset_index(drop=True)

        legs.append({
            "kosu_no": kosu_no,
            "atlar": leg_df["at_etiket"].tolist(),
            "model_olasiliklari": leg_df["model_olasiligi"].tolist(),
            "piyasa_olasiliklari": leg_df["piyasa_olasiligi"].tolist(),
            "value_siralamasi": leg_df.sort_values("value", ascending=False)["at_etiket"].tolist(),
            "value_degerleri": leg_df.sort_values("value", ascending=False)["value"].tolist(),
        })

    return legs, list(set(bulunamayan_atlar))


# ============================================================
# 4) KUPON OPTİMİZASYONU
# ============================================================

def en_iyi_tahsisi_bul(legs, maks_kombinasyon):
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
                yeni_log = log_olasilik + log_kapsama[n - 1]
                if yeni_k not in yeni_dp or yeni_log > yeni_dp[yeni_k][0]:
                    yeni_dp[yeni_k] = (yeni_log, gecmis + [n])
        dp = yeni_dp
    if not dp:
        return None, 0.0
    en_iyi_k = max(dp.keys(), key=lambda k: dp[k][0])
    log_olasilik, tahsis = dp[en_iyi_k]
    return tahsis, math.exp(log_olasilik)


def guvenli_kupon_olustur(legs, tahsis):
    return [leg["atlar"][:n] for leg, n in zip(legs, tahsis)]


def surpriz_kupon_olustur(legs, tahsis, surpriz_ayak_sayisi):
    """Birleşik kupon: güvenli seçimler + en ucuz bedelli ayaklarda value-bazlı sürpriz."""
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
        kayip_orani = (eski_toplam - yeni_toplam) / eski_toplam if eski_toplam > 0 else 1.0
        aday_swapler.append((i, kayip_orani, leg["value_degerleri"][0], yeni_secim))

    aday_swapler.sort(key=lambda x: x[1])
    for i, kayip, value, yeni_secim in aday_swapler[:surpriz_ayak_sayisi]:
        kupon[i] = yeni_secim
        degisen_ayaklar.append(i)

    return kupon, degisen_ayaklar


def olasilik_hesapla(legs, kupon, key="model_olasiliklari"):
    toplam = 1.0
    for leg, secili in zip(legs, kupon):
        at_index = {at: idx for idx, at in enumerate(leg["atlar"])}
        toplam *= sum(leg[key][at_index[at]] for at in secili)
    return toplam


# ============================================================
# 5) STREAMLIT ARAYÜZ
# ============================================================

st.set_page_config(page_title="At Yarışı Kupon Önericisi", layout="wide")
st.title("🐎 At Yarışı Kupon Önericisi")

# --- CSV YÜKLEME ---
st.sidebar.header("1) Resmi Program CSV'sini Yükle")
csv_dosya = st.sidebar.file_uploader("TJK Günlük Yarış Programı CSV'si", type=["csv"])

if csv_dosya is None:
    st.info("👈 Önce soldan TJK resmi program CSV'ini yükle (GunlukYarisProgrami-TR.csv).")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
    tmp.write(csv_dosya.read())
    tmp_path = tmp.name

with st.spinner("CSV parse ediliyor..."):
    kosular_parsed = parse_program_csv(tmp_path)

if not kosular_parsed:
    st.error("CSV'den hiç koşu bulunamadı - dosyanın doğru formatta olduğundan emin ol.")
    st.stop()

sehir_tahmin, tarih_tahmin = sehir_ve_tarih_tahmin_et(csv_dosya.name)

st.sidebar.success(f"✅ {len(kosular_parsed)} koşu başarıyla okundu.")
with st.sidebar.expander("Koşu özeti"):
    for kosu_no in sorted(kosular_parsed):
        k = kosular_parsed[kosu_no]
        kosan = sum(1 for a in k["atlar"] if not a["kosmaz"])
        st.write(f"{kosu_no}. Koşu [{k['saat']}] - {kosan} at")

st.sidebar.header("2) Şehir / Tarih")
sehir = st.sidebar.text_input("Şehir", value=sehir_tahmin or "")
tarih = st.sidebar.text_input("Tarih (GG/AA/YYYY)", value=tarih_tahmin or "")

# --- ALTILI SEÇİMİ (CSV'den otomatik tespit + elle işaretleme yedeği) ---
st.sidebar.header("3) Altılı Seç")
altili_secenekleri = altili_secenekleri_bul(kosular_parsed)

secenek_listesi = list(altili_secenekleri.keys()) + ["✏️ Elle seç"]
altili_secim = st.sidebar.radio("Hangi 6'lı ganyan?", secenek_listesi)

if altili_secim == "✏️ Elle seç":
    st.sidebar.caption("Aşağıdan TAM OLARAK 6 koşu işaretle.")
    secilenler = [
        kosu_no for kosu_no in sorted(kosular_parsed)
        if st.sidebar.checkbox(f"{kosu_no}. Koşu", key=f"manuel_{kosu_no}")
    ]
    if len(secilenler) != 6:
        st.sidebar.warning(f"Şu an {len(secilenler)} koşu işaretli - tam 6 tane olmalı.")
        st.stop()
    kosu_no_list = sorted(secilenler)
    altili_no = 1  # elle seçimde tek AGF değeri varsayılır
else:
    altili_no, kosu_no_list = altili_secenekleri[altili_secim]

# --- AGF KAYNAĞI ---
st.sidebar.header("4) AGF Kaynağı")
agf_kaynagi_secim = st.sidebar.radio(
    "Hangi AGF kullanılsın?",
    ["CSV'deki AGF (otomatik)", "TJK web sayfasından güncel AGF yapıştır"],
)
agf_kaynagi = "csv" if agf_kaynagi_secim.startswith("CSV") else "yapistir"

agf_yapistirilan = None
if agf_kaynagi == "yapistir":
    st.sidebar.caption("tjk.org/AGFv2/... sayfasından Ctrl+A, Ctrl+C ile kopyala")
    agf_metin = st.sidebar.text_area("AGF tablosu (düz metin)", height=200)
    if agf_metin.strip():
        try:
            agf_yapistirilan = parse_agf_text(agf_metin)
        except ValueError as e:
            st.sidebar.error(str(e))

# --- BÜTÇE ---
st.sidebar.header("5) Bütçe")
butce = st.sidebar.number_input("Bütçe (TL)", min_value=1.0, value=1000.0, step=50.0)

calistir = st.sidebar.button("🎯 Kupon Oluştur", type="primary", use_container_width=True)

# ============================================================
# 6) KUPON HESAPLAMA VE GÖSTERİM
# ============================================================

if calistir:
    if agf_kaynagi == "yapistir" and not agf_yapistirilan:
        st.error("AGF metni boş ya da hatalı - önce yapıştırman lazım, ya da 'CSV'deki AGF'yi kullan' seç.")
        st.stop()

    with st.spinner("Modeller ve geçmiş veri yükleniyor..."):
        model_g, fc_g, model_gs, fc_gs = load_models()
        name_cache = load_name_cache()
        at_son, jokey_son, antrenor_son, sahip_son, baba_son, anne_son, medyanlar = load_features_lookup()

    if name_cache is None:
        st.warning("name_id_cache.json bulunamadı.")

    with st.spinner("Koşu verileri hazırlanıyor, model tahminleri üretiliyor..."):
        try:
            legs, bulunamayanlar = kosu_verisi_hazirla(
                kosu_no_list, altili_no, kosular_parsed, sehir,
                agf_kaynagi, agf_yapistirilan,
                model_g, fc_g, model_gs, fc_gs,
                name_cache, at_son, jokey_son, antrenor_son, sahip_son,
                baba_son, anne_son, medyanlar,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if bulunamayanlar:
        st.info(f"Geçmiş veride bulunamayan {len(bulunamayanlar)} at için medyan/soy tahmini kullanıldı: "
                f"{', '.join(bulunamayanlar)}")

    maks_kombinasyon = int(butce / COMBINASYON_BIRIM_FIYATI)
    tahsis, tahmini_olasilik = en_iyi_tahsisi_bul(legs, maks_kombinasyon)

    if tahsis is None:
        st.error("Bu bütçeyle hiçbir kombinasyon mümkün değil.")
        st.stop()

    kombinasyon_sayisi = 1
    for n in tahsis:
        kombinasyon_sayisi *= n
    maliyet = kombinasyon_sayisi * COMBINASYON_BIRIM_FIYATI

    guvenli = guvenli_kupon_olustur(legs, tahsis)
    piyasa_olasilik = olasilik_hesapla(legs, guvenli, "piyasa_olasiliklari")

    surpriz, degisen_ayaklar = surpriz_kupon_olustur(legs, tahsis, SURPRIZ_AYAK_SAYISI)
    surpriz_olasilik = olasilik_hesapla(legs, surpriz, "model_olasiliklari")

    st.header("🎟️ Önerilen Kupon")

    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Kombinasyon", f"{kombinasyon_sayisi}")
    col2.metric("Maliyet", f"{maliyet:.0f} TL")
    col3.metric("Kalan Bütçe", f"{butce - maliyet:.0f} TL")

    st.subheader("Kupon (Güvenli seçimler + Sürpriz atlar birleşik)")
    st.metric("Tahmini Tutma Olasılığı", f"%{surpriz_olasilik * 100:.2f}",
              delta=f"{(surpriz_olasilik - tahmini_olasilik) * 100:+.2f} puan (güvenliye göre)")
    st.caption(f"(Karşılaştırma) Piyasa/AGF bazlı tutma olasılığı: %{piyasa_olasilik * 100:.2f}")

    for i, (leg, secim, n) in enumerate(zip(legs, surpriz, tahsis)):
        etiket = " 🎲 SÜRPRİZ İÇERİYOR" if i in degisen_ayaklar else ""
        st.write(f"**{leg['kosu_no']}. Koşu** ({n} at seçildi){etiket}")
        st.write(" | ".join(secim))

    st.divider()
    st.warning(
        "⚠️ Bu bir yatırım/bahis tavsiyesi değildir. Tahmini olasılıklar geçmiş "
        "veriye dayalı bir modelden geliyor, gerçek sonucu garanti etmez. "
        "Kayıp riski gerçektir. Bütçenizi kaybetmeyi göze alabileceğiniz "
        "miktarla sınırlı tutun."
    )
else:
    st.info("Soldaki adımları tamamlayıp 'Kupon Oluştur' butonuna bas.")