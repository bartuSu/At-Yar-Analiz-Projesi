"""
TJK Resmi Program PDF'ini parse eder. Tablo çıkarma (pdfplumber) + metin
regex'i BİRLİKTE kullanılıyor, kilo değeri çapraz kontrol için kullanılıyor.

DÜRÜST UYARI: TJK'nın PDF üretim şablonu tutarlı değil - bazı koşularda
harfler birbirinden ayrı basılıyor (x_tolerance=10 ile büyük ölçüde
düzeltiliyor), bazı koşularda at ismi satırın SONUNDA (normalde BAŞINDA)
oluyor (iki farklı desen deneniyor), bazı koşularda tablo çizgileri hiç
algılanamıyor. Bu durumlarda o koşu "manuel_gerekli" olarak işaretlenir -
dashboard kullanıcıya elle giriş imkanı sunar.

Kullanım:
    from parse_program_pdf import parse_pdf, sehir_ve_tarih_tahmin_et
    kosular = parse_pdf("program.pdf")
"""
import re

import pdfplumber

X_TOLERANCE = 10  # varsayılan (3) bazı PDF'lerde harfleri gereksiz ayırıyor

RACE_HEADER_RE = re.compile(r"(\d{1,2})\s+(\d)\s*(YŞ|YK)\.", re.MULTILINE)
YAS_TOKEN_RE = re.compile(r"\b(\d{1,2})y\s+([a-zçğıöşü]{1,3})\b", re.IGNORECASE)
PIST_MESAFE_RE = re.compile(r"(KUM|ÇİM)\s*P[İI]ST\s*/\s*(\d+)\s*METRE", re.IGNORECASE)
ALTILI_BASLAR_RE = re.compile(r"(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ)\s+6['’]L[İI]\s+GANYAN\s+bu\s+koşudan\s+başlar", re.IGNORECASE)


def parse_isim_bastan(satir: str) -> dict | None:
    """'1 ADI YAŞASIN 3y ad Baba - Anne 57 ...' formatı (at ismi baştaysa)."""
    satir = satir.strip()
    m_at_no = re.match(r"^(\d{1,2})\s+", satir)
    if not m_at_no:
        return None
    at_no = int(m_at_no.group(1))
    kalan = satir[m_at_no.end():]
    m_yas = YAS_TOKEN_RE.search(kalan)
    if not m_yas or m_yas.start() > 60:
        return None
    isim = re.sub(r"[\d.]+\s*₺", "", kalan[:m_yas.start()]).strip()
    if not isim or not re.match(r"^[A-ZÇĞİÖŞÜ]", isim):
        return None
    sonrasi = kalan[m_yas.end():].strip()
    if " - " not in sonrasi:
        return None
    baba, kalan2 = sonrasi.split(" - ", 1)
    m_kilo = re.match(r"^(.+?)\s+(\d{2,3}(?:[.,]\d)?)\s+", kalan2.strip())
    if not m_kilo:
        return None
    return {
        "at_no": at_no, "isim": isim, "baba": baba.strip(),
        "anne": m_kilo.group(1).strip(),
        "kilo_metin": float(m_kilo.group(2).replace(",", ".")),
    }


def parse_isim_sonda(satir: str) -> dict | None:
    """'3y de Mendip (USA) - Mucizem 58 N. Avci ... S. Çelikman 1 APOLLO MAXIMUS' (at ismi sonda)."""
    satir = satir.strip()
    m_yas = YAS_TOKEN_RE.match(satir)
    if not m_yas:
        return None
    sonrasi = satir[m_yas.end():].strip()
    if " - " not in sonrasi:
        return None
    baba, kalan2 = sonrasi.split(" - ", 1)
    m_kilo = re.match(r"^(.+?)\s+(\d{2,3}(?:[.,]\d)?)\s+(.*)$", kalan2.strip())
    if not m_kilo:
        return None
    anne = m_kilo.group(1).strip()
    kilo = float(m_kilo.group(2).replace(",", "."))
    kalan3 = m_kilo.group(3).strip()
    m_son = re.search(r"(\d{1,2})\s+([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]+)$", kalan3)
    if not m_son:
        return None
    return {
        "at_no": int(m_son.group(1)), "isim": m_son.group(2).strip(),
        "baba": baba.strip(), "anne": anne, "kilo_metin": kilo,
    }


def find_race_header_positions(sayfa) -> list[tuple[int, float]]:
    """Her koşu başlığının (kosu_no, y_pozisyonu) listesini kelime konumlarından bulur."""
    kelimeler = sayfa.extract_words(x_tolerance=X_TOLERANCE)
    sonuclar = []
    for i, kelime in enumerate(kelimeler):
        if not re.match(r"^\d{1,2}$", kelime["text"]):
            continue
        for j in range(i + 1, min(i + 3, len(kelimeler))):
            sonraki = kelimeler[j]["text"]
            if re.match(r"^\d\s*Y[ŞK]\.?$", sonraki) or (
                re.match(r"^\d$", sonraki) and j + 1 < len(kelimeler)
                and re.match(r"^Y[ŞK]\.?", kelimeler[j + 1]["text"])
            ):
                kosu_no = int(kelime["text"])
                if 1 <= kosu_no <= 20:
                    sonuclar.append((kosu_no, kelime["top"]))
                break
    return sonuclar


def parse_pdf(pdf_path: str) -> dict:
    """
    Döner: {kosu_no: {durum, format, pist, mesafe, altili_baslangic, atlar: [...]}}
    durum: "ok" (kilo eşleşti, güvenilir) / "kilo_uyusmuyor" (şüpheli)
           / "manuel_gerekli" (otomatik parse başarısız, elle gir)
    """
    kosular = {}

    with pdfplumber.open(pdf_path) as pdf:
        for sayfa in pdf.pages:
            tam_metin = sayfa.extract_text(x_tolerance=X_TOLERANCE) or ""
            baslik_pozisyonlari = sorted(find_race_header_positions(sayfa), key=lambda x: x[1])
            tablo_objeleri = sayfa.find_tables()
            tablo_pozisyonlari = sorted([(t.bbox[1], t) for t in tablo_objeleri], key=lambda x: x[0])
            metin_baslik_eslesmeleri = list(RACE_HEADER_RE.finditer(tam_metin))

            for kosu_no, y_pos in baslik_pozisyonlari:
                if kosu_no in kosular:
                    continue

                esleme = next((m for m in metin_baslik_eslesmeleri if int(m.group(1)) == kosu_no), None)
                if not esleme:
                    kosular[kosu_no] = {
                        "durum": "manuel_gerekli", "format": "baslik_metinde_yok",
                        "pist": None, "mesafe": None, "altili_baslangic": None, "atlar": [],
                    }
                    continue

                idx = metin_baslik_eslesmeleri.index(esleme)
                blok_bitis = (metin_baslik_eslesmeleri[idx + 1].start()
                              if idx + 1 < len(metin_baslik_eslesmeleri) else len(tam_metin))
                blok = tam_metin[esleme.start():blok_bitis]

                pist_m = PIST_MESAFE_RE.search(blok)
                pist = pist_m.group(1).title() if pist_m else None
                mesafe = int(pist_m.group(2)) if pist_m else None
                altili_m = ALTILI_BASLAR_RE.search(blok)
                altili_baslangic = (altili_m.group(1) or "TEK") if altili_m else None

                satirlar = [s.strip() for s in blok.split("\n") if s.strip()]

                uygun_tablo = None
                for t_top, t_obj in tablo_pozisyonlari:
                    if t_top >= y_pos - 5:
                        uygun_tablo = t_obj
                        break

                beklenen_sayi = None
                tablo_veri = None
                if uygun_tablo:
                    tablo_veri = uygun_tablo.extract()
                    if len(tablo_veri) >= 2:
                        beklenen_sayi = len((tablo_veri[1][3] or "").split("\n"))

                atlari_bastan = [at for s in satirlar if (at := parse_isim_bastan(s))]
                atlari_sondan = [at for s in satirlar if (at := parse_isim_sonda(s))]

                if beklenen_sayi and len(atlari_bastan) == beklenen_sayi:
                    metin_atlari, format_tipi = atlari_bastan, "bastan"
                elif beklenen_sayi and len(atlari_sondan) == beklenen_sayi:
                    metin_atlari, format_tipi = atlari_sondan, "sondan"
                else:
                    metin_atlari, format_tipi = [], "yok"

                durum = "manuel_gerekli"
                if uygun_tablo and metin_atlari and tablo_veri:
                    veri_satiri = tablo_veri[1]
                    kilo_liste = [x.strip() for x in (veri_satiri[2] or "").split("\n")]
                    jokey_liste = [x.strip() for x in (veri_satiri[3] or "").split("\n")]
                    sahip_liste = [x.strip() for x in (veri_satiri[4] or "").split("\n")]
                    antrenor_liste = [x.strip() for x in (veri_satiri[5] or "").split("\n")]

                    hepsi_eslesti = True
                    for idx2, at in enumerate(metin_atlari):
                        tk = kilo_liste[idx2] if idx2 < len(kilo_liste) else None
                        eslesme = (tk is not None and abs(float(tk.replace(",", ".")) - at["kilo_metin"]) < 0.01)
                        if not eslesme:
                            hepsi_eslesti = False
                        at["jokey"] = jokey_liste[idx2]
                        at["sahip"] = sahip_liste[idx2] if idx2 < len(sahip_liste) else None
                        at["antrenor"] = antrenor_liste[idx2] if idx2 < len(antrenor_liste) else None
                    durum = "ok" if hepsi_eslesti else "kilo_uyusmuyor"

                if durum == "manuel_gerekli":
                    metin_atlari = []

                kosular[kosu_no] = {
                    "durum": durum, "format": format_tipi, "pist": pist, "mesafe": mesafe,
                    "altili_baslangic": altili_baslangic, "atlar": metin_atlari,
                }

    return kosular


def sehir_ve_tarih_tahmin_et(dosya_adi: str) -> tuple[str | None, str | None]:
    """Dosya adından şehir/tarih tahmin eder (örn. '20260903-resmi-program-İzmir.pdf')."""
    import os
    dosya_adi = os.path.basename(dosya_adi)
    m = re.match(r"(\d{4})(\d{2})(\d{2}).*?-([A-ZÇĞİÖŞÜa-zçğıöşü]+)\.pdf", dosya_adi)
    if m:
        yil, ay, gun, sehir = m.groups()
        return sehir, f"{gun}/{ay}/{yil}"
    return None, None