"""
At Yarışı Kupon Önericisi - Ana Dashboard

Kullanıcı:
1. TJK Resmi Program PDF'ini yükler (otomatik parse edilir, başarısız
   olan koşular için elle düzeltme tablosu gösterilir)
2. 1. Altılı mı 2. Altılı mı seçer (PDF'ten otomatik tespit edilir)
3. TJK AGF sayfasından kopyaladığı metni yapıştırır (OCR yok, düz metin)
4. Bütçesini girer

Sistem:
- AGF yüzdesini "ganyan"a çevirir (pseudo_ganyan = 100/agf_yuzde)
- At/jokey/antrenör/sahip isimlerini geçmiş veriden ID'lerine eşler
- Ganyanlı model (model.pkl) ile bütçe optimizasyonu yapıp GÜVENLİ kupon oluşturur
- Ganyansız model (model_ganyansiz.pkl) ile "value" hesaplayıp, en az
  bedelli 1-2 ayakta SÜRPRİZ at zorlayarak TEK BİR birleşik kupon üretir
- Kullanıcıya bu birleşik kuponu "at no - isim" formatında ve tahmini
  tutma olasılığını gösterir

Çalıştırma:
    python -m pip install streamlit pdfplumber
    python -m streamlit run main_dashboard.py
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
from parse_program_csv import parse_pdf, sehir_ve_tarih_tahmin_et

SURPRIZ_AYAK_SAYISI = 2
COMBINASYON_BIRIM_FIYATI = 1.0  # TL - VARSAYIM, gerçek fiyatı doğrula

# ============================================================
# 1) AGF METİN PARSER
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
    return at_son, jokey_son, antrenor_son, sahip_son, medyanlar


# ============================================================
# 3) CANLI FEATURE ÜRETİMİ
# ============================================================

def at_id_bul(at_isim: str, at_son_df: pd.DataFrame):
    hedef = at_isim.strip().upper()
    eslesenler = at_son_df[at_son_df["at_isim"].str.strip().str.upper() == hedef]
    if eslesenler.empty:
        return None
    return eslesenler.index[0]


def canli_satir_olustur(at: dict, name_cache, at_son, jokey_son, antrenor_son, sahip_son, medyanlar) -> dict:
    satir = {"kilo": at["kilo"]}

    at_id = at_id_bul(at["isim"], at_son)
    if at_id is not None:
        gecmis = at_son.loc[at_id]
        satir["son_3_ort_sira"] = gecmis.get("son_3_ort_sira", medyanlar["son_3_ort_sira"])
        satir["gecmis_yaris_sayisi"] = gecmis.get("gecmis_yaris_sayisi", medyanlar["gecmis_yaris_sayisi"]) + 1
        satir["son_yaristan_gun_farki"] = medyanlar["son_yaristan_gun_farki"]
        satir["baba_kazanma_orani"] = gecmis.get("baba_kazanma_orani", medyanlar["baba_kazanma_orani"])
        satir["anne_kazanma_orani"] = gecmis.get("anne_kazanma_orani", medyanlar["anne_kazanma_orani"])
        satir["_at_bulundu"] = True
    else:
        satir["son_3_ort_sira"] = medyanlar["son_3_ort_sira"]
        satir["gecmis_yaris_sayisi"] = 0
        satir["son_yaristan_gun_farki"] = medyanlar["son_yaristan_gun_farki"]
        satir["baba_kazanma_orani"] = medyanlar["baba_kazanma_orani"]
        satir["anne_kazanma_orani"] = medyanlar["anne_kazanma_orani"]
        satir["_at_bulundu"] = False

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


def kosu_verisi_hazirla(kosu_no_list, kosu_atlari, kosu_meta, sehir, agf_verisi,
                         model_g, fc_g, model_gs, fc_gs,
                         name_cache, at_son, jokey_son, antrenor_son, sahip_son, medyanlar):
    """
    kosu_atlari: {kosu_no: [{"at_no", "isim", "kilo", "jokey", "sahip", "antrenor"}, ...]}
    kosu_meta:   {kosu_no: {"mesafe": int, "pist": str}}
    """
    legs = []
    bulunamayan_atlar = []

    for i, kosu_no in enumerate(kosu_no_list):
        atlar = kosu_atlari.get(kosu_no, [])
        meta = kosu_meta.get(kosu_no, {})
        agf_ayak = agf_verisi.get(i + 1, {})  # AGF ayak no'su seçilen altılı içinde 1'den başlar

        if not atlar:
            raise ValueError(f"{kosu_no}. koşu için at verisi yok - PDF'ten parse edilemedi ve elle de girilmedi.")

        satirlar = []
        for at in atlar:
            satir = canli_satir_olustur(at, name_cache, at_son, jokey_son, antrenor_son, sahip_son, medyanlar)
            satir["at_no"] = at["at_no"]
            satir["at_isim"] = at["isim"]
            satir["mesafe"] = meta.get("mesafe") or 1400
            satir["pist"] = meta.get("pist") or "Kum"
            satir["sehir"] = sehir

            agf_yuzde = agf_ayak.get(at["at_no"])
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

# --- PDF YÜKLEME ---
st.sidebar.header("1) Resmi Programı Yükle")
pdf_dosya = st.sidebar.file_uploader("TJK Resmi Program PDF'i", type=["pdf"])

if pdf_dosya is None:
    st.info("👈 Önce soldan TJK resmi program PDF'ini yükle.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(pdf_dosya.read())
    tmp_path = tmp.name

with st.spinner("PDF parse ediliyor..."):
    kosular_parsed = parse_pdf(tmp_path)

sehir_tahmin, tarih_tahmin = sehir_ve_tarih_tahmin_et(pdf_dosya.name)

st.sidebar.subheader("Koşu Durumu")
for kosu_no in sorted(kosular_parsed):
    durum = kosular_parsed[kosu_no]["durum"]
    isaret = {"ok": "✅", "kilo_uyusmuyor": "⚠️", "manuel_gerekli": "❌"}.get(durum, "❓")
    st.sidebar.write(f"{isaret} {kosu_no}. Koşu")

st.sidebar.header("2) Şehir / Tarih")
sehir = st.sidebar.text_input("Şehir", value=sehir_tahmin or "")
tarih = st.sidebar.text_input("Tarih (GG/AA/YYYY)", value=tarih_tahmin or "")

# --- KOŞU VERİLERİNİ GÖSTER / DÜZENLE ---
st.header("📋 Koşu Verilerini Kontrol Et / Düzenle")

manuel_gereken = [k for k, v in kosular_parsed.items() if v["durum"] != "ok"]
if manuel_gereken:
    st.warning(
        f"{len(manuel_gereken)} koşu otomatik parse edilemedi: {sorted(manuel_gereken)}. "
        "Aşağıdaki tablolara elle gir (At No, İsim, Kilo, Jokey, Sahip, Antrenör dolu olmalı)."
    )

kosu_atlari_final = {}
kosu_meta_final = {}

for kosu_no in sorted(kosular_parsed):
    kosu = kosular_parsed[kosu_no]
    kosu_meta_final[kosu_no] = {"mesafe": kosu.get("mesafe"), "pist": kosu.get("pist")}

    baslik = f"{kosu_no}. Koşu"
    if kosu.get("mesafe"):
        baslik += f" ({kosu['mesafe']}m, {kosu.get('pist', '?')})"
    if kosu.get("altili_baslangic"):
        baslik += f" — {kosu['altili_baslangic']} 6'lı burdan başlıyor"

    with st.expander(baslik, expanded=(kosu["durum"] != "ok")):
        if kosu["durum"] == "ok":
            df_goster = pd.DataFrame([
                {"At No": at["at_no"], "İsim": at["isim"], "Kilo": at["kilo_metin"],
                 "Jokey": at.get("jokey"), "Sahip": at.get("sahip"), "Antrenör": at.get("antrenor")}
                for at in kosu["atlar"]
            ])
        else:
            st.caption("⚠️ Otomatik parse edilemedi - elle gir (satır eklemek için tablonun altına tıkla)")
            df_goster = pd.DataFrame(columns=["At No", "İsim", "Kilo", "Jokey", "Sahip", "Antrenör"])

        duzenlenmis = st.data_editor(
            df_goster, num_rows="dynamic", key=f"editor_{kosu_no}", use_container_width=True,
        )

        kosu_atlari_final[kosu_no] = [
            {"at_no": int(r["At No"]), "isim": str(r["İsim"]), "kilo": float(r["Kilo"]),
             "jokey": str(r["Jokey"]) if pd.notna(r["Jokey"]) else "",
             "sahip": str(r["Sahip"]) if pd.notna(r["Sahip"]) else "",
             "antrenor": str(r["Antrenör"]) if pd.notna(r["Antrenör"]) else ""}
            for _, r in duzenlenmis.iterrows()
            if pd.notna(r["At No"]) and pd.notna(r["İsim"])
        ]


# --- ALTILI SEÇİMİ (otomatik tespit + elle işaretleme seçeneği) ---
st.sidebar.header("3) Altılı Seç")

ETIKET_HARITASI = {
    "BİRİNCİ": "1. Altılı", "İKİNCİ": "2. Altılı", "ÜÇÜNCÜ": "3. Altılı",
    "KARMA": "Karma Altılı", "TEK": "Altılı",
}

altili_secenekleri = {}
for kosu_no, kosu in kosular_parsed.items():
    etiket = kosu.get("altili_baslangic")
    if etiket:
        ad = ETIKET_HARITASI.get(etiket, etiket)
        altili_secenekleri[f"{ad} ({kosu_no}-{kosu_no + 5}. koşular) [otomatik bulundu]"] = list(range(kosu_no, kosu_no + 6))

secenek_listesi = list(altili_secenekleri.keys()) + ["✏️ Elle seç"]
altili_secim = st.sidebar.radio("Hangi 6'lı ganyan?", secenek_listesi)

if altili_secim == "✏️ Elle seç":
    st.sidebar.caption(
        "PDF'te altılının hangi koşudan başladığı otomatik bulunamadı (ya da "
        "farklı bir kombinasyon istiyorsun) - aşağıdan TAM OLARAK 6 koşu işaretle."
    )
    tum_kosular = sorted(kosular_parsed.keys())
    secilenler = []
    for kosu_no in tum_kosular:
        varsayilan = kosu_no in secilenler  # önceki seçimi hatırlamaya çalış
        if st.sidebar.checkbox(f"{kosu_no}. Koşu", key=f"manuel_altili_{kosu_no}"):
            secilenler.append(kosu_no)

    if len(secilenler) != 6:
        st.sidebar.warning(f"Şu an {len(secilenler)} koşu işaretli - tam olarak 6 tane işaretlemen lazım.")
        st.stop()
    kosu_no_list = sorted(secilenler)
else:
    kosu_no_list = altili_secenekleri[altili_secim]

# --- AGF METNİ ---
st.sidebar.header("4) AGF Metnini Yapıştır")
st.sidebar.caption("TJK AGF sayfasından Ctrl+A, Ctrl+C ile kopyala")
agf_metin = st.sidebar.text_area("AGF tablosu (düz metin)", height=200)

# --- BÜTÇE ---
st.sidebar.header("5) Bütçe")
butce = st.sidebar.number_input("Bütçe (TL)", min_value=1.0, value=1000.0, step=50.0)

calistir = st.sidebar.button("🎯 Kupon Oluştur", type="primary", use_container_width=True)

# ============================================================
# 6) KUPON HESAPLAMA VE GÖSTERİM
# ============================================================

if calistir:
    eksik_kosular = [k for k in kosu_no_list if not kosu_atlari_final.get(k)]
    if eksik_kosular:
        st.error(f"Şu koşularda at verisi eksik: {eksik_kosular}. Yukarıdaki tabloları doldur.")
        st.stop()

    if not agf_metin.strip():
        st.error("AGF metni boş - önce yapıştırman lazım.")
        st.stop()

    try:
        agf_verisi = parse_agf_text(agf_metin)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Modeller ve geçmiş veri yükleniyor..."):
        model_g, fc_g, model_gs, fc_gs = load_models()
        name_cache = load_name_cache()
        at_son, jokey_son, antrenor_son, sahip_son, medyanlar = load_features_lookup()

    if name_cache is None:
        st.warning("name_id_cache.json bulunamadı - `python build_name_id_cache.py` çalıştırmanı öneririm.")

    with st.spinner("Koşu verileri hazırlanıyor, model tahminleri üretiliyor..."):
        try:
            legs, bulunamayanlar = kosu_verisi_hazirla(
                kosu_no_list, kosu_atlari_final, kosu_meta_final, sehir, agf_verisi,
                model_g, fc_g, model_gs, fc_gs,
                name_cache, at_son, jokey_son, antrenor_son, sahip_son, medyanlar,
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if bulunamayanlar:
        st.info(f"Geçmiş veride bulunamayan {len(bulunamayanlar)} at için medyan değer kullanıldı: "
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