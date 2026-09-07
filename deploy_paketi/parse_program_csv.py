"""
TJK'nın resmi günlük yarış programı CSV'sini parse eder
(örn. "07_09_2026-Bursa-GunlukYarisProgrami-TR.csv").

Bu, PDF parser'a göre ÇOK daha güvenilir - TJK'nın kendi yapılandırılmış
verisi, ';' ile ayrılmış temiz sütunlar, hiçbir OCR/tablo tahmini yok.
AGF verisi dosyanın içinde zaten var - TJK web sayfasından ayrıca
kopyalamaya gerek KALMAYABİLİR (ama daha güncel/canlı AGF istersen web
sayfasından yapıştırma seçeneği de dashboard'da bırakılabilir).

Kullanım:
    from parse_program_csv import parse_program_csv
    kosular = parse_program_csv("program.csv")
"""
import re

KOSU_BASLIK_RE = re.compile(r"^(\d+)\.\s*Kosu\s*:\s*([\d.]+);(.+)$")
ALTILI_RE = re.compile(r"(\d+)\.\s*6['’]L[İI]\s*GANYAN", re.IGNORECASE)
TAKI_KODLARI = {"KG", "DB", "SK", "SKG", "GKR", "K", "BB", "ÇABA", "SGKR", "DS", "YP", "TGK", "ÖG"}


def _isim_temizle(ham_isim: str) -> tuple[str, bool]:
    """At isminden takı kodlarını ve '(Koşmaz)' ibaresini ayıklar."""
    kosmaz = bool(re.search(r"\(Ko[şs]maz\)", ham_isim, re.IGNORECASE))
    isim = re.sub(r"\(Ko[şs]maz\)", "", ham_isim, flags=re.IGNORECASE).strip()
    parcalar = isim.split()
    while parcalar and parcalar[-1].upper() in TAKI_KODLARI:
        parcalar.pop()
    return " ".join(parcalar), kosmaz


def _kilo_parse(ham_kilo: str) -> float | None:
    """'57' ya da '57 +0.60' (fazla kilo dahil) ya da '54,5 +2.00' formatlarını çözer."""
    ham_kilo = ham_kilo.strip()
    m = re.match(r"^([\d,\.]+)(?:\s*\+([\d,\.]+))?", ham_kilo)
    if not m:
        return None
    baz = float(m.group(1).replace(",", "."))
    fazla = float(m.group(2).replace(",", ".")) if m.group(2) else 0.0
    return baz + fazla


def _agf_parse(ham_agf: str) -> list[float]:
    """'%6.66(6)' ya da çift altılı çakışmasında '%5.75(8)  %5.14(8)' formatını çözer."""
    return [float(x) for x in re.findall(r"%([\d,\.]+)\(", ham_agf.replace(",", "."))]


def parse_program_csv(dosya_yolu_veya_metin, metin_mi: bool = False) -> dict:
    """
    dosya_yolu_veya_metin: dosya yolu (str) ya da (metin_mi=True ise) CSV içeriğinin kendisi.
    Döner: {kosu_no: {saat, kosu_tipi, yas_grubu, mesafe, pist, atlar, altili_baslangic}}
    """
    if metin_mi:
        icerik = dosya_yolu_veya_metin
    else:
        with open(dosya_yolu_veya_metin, "r", encoding="utf-8-sig") as f:
            icerik = f.read()

    satirlar = icerik.split("\r\n") if "\r\n" in icerik else icerik.split("\n")

    kosular = {}
    i = 0
    while i < len(satirlar):
        m = KOSU_BASLIK_RE.match(satirlar[i])
        if not m:
            i += 1
            continue

        kosu_no = int(m.group(1))
        saat = m.group(2)
        detay = m.group(3).split(";")
        kosu_tipi = detay[0].strip() if len(detay) > 0 else None
        yas_grubu = detay[1].strip() if len(detay) > 1 else None
        mesafe_m = re.search(r"(\d+)m", m.group(3))
        mesafe = int(mesafe_m.group(1)) if mesafe_m else None
        pist = "Çim" if "Çim" in m.group(3) else ("Kum" if "Kum" in m.group(3) else None)

        j = i + 1
        while j < len(satirlar) and not satirlar[j].startswith("At No;"):
            j += 1
        j += 1

        atlar = []
        while j < len(satirlar) and re.match(r"^\d+;", satirlar[j]):
            alanlar = satirlar[j].split(";")
            isim, kosmaz = _isim_temizle(alanlar[1])
            atlar.append({
                "at_no": int(alanlar[0]),
                "isim": isim,
                "kosmaz": kosmaz,
                "kilo": _kilo_parse(alanlar[5]) if len(alanlar) > 5 else None,
                "jokey": alanlar[6].strip() if len(alanlar) > 6 else None,
                "sahip": alanlar[7].strip() if len(alanlar) > 7 else None,
                "antrenor": alanlar[8].strip() if len(alanlar) > 8 else None,
                "baba": alanlar[3].strip() if len(alanlar) > 3 else None,
                "anne": alanlar[4].strip() if len(alanlar) > 4 else None,
                "agf_degerleri": _agf_parse(alanlar[10]) if len(alanlar) > 10 else [],
            })
            j += 1

        altili_baslangic = []
        if j < len(satirlar):
            altili_baslangic = [int(x) for x in ALTILI_RE.findall(satirlar[j])]

        kosular[kosu_no] = {
            "saat": saat, "kosu_tipi": kosu_tipi, "yas_grubu": yas_grubu,
            "mesafe": mesafe, "pist": pist, "atlar": atlar,
            "altili_baslangic": altili_baslangic,
        }
        i = j

    return kosular


def sehir_ve_tarih_tahmin_et(dosya_adi: str) -> tuple[str | None, str | None]:
    """Dosya adından şehir/tarih çıkarır (örn. '07_09_2026-Bursa-GunlukYarisProgrami-TR.csv')."""
    import os
    dosya_adi = os.path.basename(dosya_adi)
    m = re.match(r"(\d{2})_(\d{2})_(\d{4})-([A-ZÇĞİÖŞÜa-zçğıöşü]+)-", dosya_adi)
    if m:
        gun, ay, yil, sehir = m.groups()
        return sehir, f"{gun}/{ay}/{yil}"
    return None, None


def altili_secenekleri_bul(kosular: dict) -> dict:
    """{'1. Altılı (1-6. koşular)': (altili_no, [kosu_no_list]), ...} döner."""
    baslangiclar = {}
    for kosu_no, kosu in sorted(kosular.items()):
        for altili_no in kosu["altili_baslangic"]:
            baslangiclar[altili_no] = kosu_no

    secenekler = {}
    for altili_no, baslangic_kosu in baslangiclar.items():
        kosu_no_list = list(range(baslangic_kosu, baslangic_kosu + 6))
        if all(k in kosular for k in kosu_no_list):
            secenekler[f"{altili_no}. Altılı ({baslangic_kosu}-{baslangic_kosu + 5}. koşular)"] = (altili_no, kosu_no_list)
    return secenekler