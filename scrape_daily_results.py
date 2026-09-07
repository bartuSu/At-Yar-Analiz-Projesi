"""
Bir şehrin bir günkü TÜM yarış sonuçlarını çeker ve parse eder.
Server-side render - JS/Playwright gerekmez.
"""
import argparse
import json
import re
from datetime import date, timedelta

from bs4 import BeautifulSoup

import scraper_config
from scrapper_http import get, logger

COLUMN_CLASS_MAP = {
    "sira": "gunluk-GunlukYarisSonuclari-SONUCNO",
    "at_adi_raw": "gunluk-GunlukYarisSonuclari-AtAdi3",
    "yas": "gunluk-GunlukYarisSonuclari-Yas",
    "orijin_raw": "gunluk-GunlukYarisSonuclari-Baba",
    "kilo_raw": "gunluk-GunlukYarisSonuclari-Kilo",
    "jokey_raw": "gunluk-GunlukYarisSonuclari-JokeAdi",
    "sahip_raw": "gunluk-GunlukYarisSonuclari-SahipAdi",
    "antrenor_raw": "gunluk-GunlukYarisSonuclari-AntronorAdi",
    "derece": "gunluk-GunlukYarisSonuclari-Derece",
    "ganyan_raw": "gunluk-GunlukYarisSonuclari-Gny",
    "agf_raw": "gunluk-GunlukYarisSonuclari-AGFORAN",
    "start_no": "gunluk-GunlukYarisSonuclari-StartId",
    "fark": "gunluk-GunlukYarisSonuclari-Fark",
    "gec_cikis": "gunluk-GunlukYarisSonuclari-GecCikis",
    "handikap_puani": "gunluk-GunlukYarisSonuclari-Hc",
}


def _text(cell) -> str:
    return cell.get_text(" ", strip=True) if cell else ""


def parse_horse_row(row) -> dict:
    cells = {key: row.find("td", class_=cls) for key, cls in COLUMN_CLASS_MAP.items()}

    at_cell = cells["at_adi_raw"]
    at_link = at_cell.find("a") if at_cell else None
    at_id_match = re.search(r"QueryParameter_AtId=(\d+)", at_link["href"]) if at_link else None
    at_isim = at_link.get_text(strip=True) if at_link else _text(at_cell)
    isim_match = re.match(r"^(.+?)\((\d+)\)$", at_isim)

    orijin_links = cells["orijin_raw"].find_all("a") if cells["orijin_raw"] else []
    baba = orijin_links[0].get_text(strip=True) if len(orijin_links) > 0 else None
    anne = orijin_links[1].get_text(strip=True) if len(orijin_links) > 1 else None
    kisrak_babasi = orijin_links[2].get_text(strip=True) if len(orijin_links) > 2 else None

    def _id_and_name(cell, id_param):
        link = cell.find("a") if cell else None
        if not link:
            return None, None
        m = re.search(rf"{id_param}=(\d+)", link.get("href", ""))
        return (int(m.group(1)) if m else None), link.get_text(strip=True)

    jokey_id, jokey_adi = _id_and_name(cells["jokey_raw"], "QueryParameter_JokeyId")
    sahip_id, sahip_adi = _id_and_name(cells["sahip_raw"], "QueryParameter_SahipId")
    antrenor_id, antrenor_adi = _id_and_name(cells["antrenor_raw"], "QueryParameter_AntrenorId")

    kilo_text = _text(cells["kilo_raw"])
    kilo_match = re.match(r"^([\d,]+)", kilo_text)
    kilo = float(kilo_match.group(1).replace(",", ".")) if kilo_match else None

    ganyan_text = _text(cells["ganyan_raw"])
    ganyan = float(ganyan_text.replace(",", ".")) if ganyan_text else None

    agf_links = cells["agf_raw"].find_all("a") if cells["agf_raw"] else []
    agf_values = []
    for a in agf_links:
        title = a.get("title", "")
        m = re.search(r"%([\d,]+)\((\d+)\)", title)
        if m:
            agf_values.append({"yuzde": float(m.group(1).replace(",", ".")), "sira": int(m.group(2))})

    return {
        "sira_no": _text(cells["sira"]) or None,
        "at_id": int(at_id_match.group(1)) if at_id_match else None,
        "at_isim": isim_match.group(1) if isim_match else at_isim,
        "kulvan_no": int(isim_match.group(2)) if isim_match else None,
        "yas_cinsiyet": _text(cells["yas"]),
        "baba": baba,
        "anne": anne,
        "kisrak_babasi": kisrak_babasi,
        "kilo": kilo,
        "jokey_id": jokey_id,
        "jokey_adi": jokey_adi,
        "sahip_id": sahip_id,
        "sahip_adi": sahip_adi,
        "antrenor_id": antrenor_id,
        "antrenor_adi": antrenor_adi,
        "derece": _text(cells["derece"]) or None,
        "ganyan": ganyan,
        "agf": agf_values,
        "start_no": _text(cells["start_no"]) or None,
        "fark": _text(cells["fark"]) or None,
        "gec_cikis": _text(cells["gec_cikis"]) or None,
        "handikap_puani": _text(cells["handikap_puani"]) or None,
    }


def parse_race_metadata(race_div) -> dict:
    details = race_div.find("div", class_="race-details")
    if not details:
        return {}

    header_text = _text(details.find("h3", class_="race-no"))
    m = re.match(r"(\d+)\.\s*Koşu\s+([\d.]+)", header_text)
    kosu_no = int(m.group(1)) if m else None
    saat = m.group(2) if m else None

    config_text = _text(details.find("h3", class_="race-config"))
    mesafe_match = re.search(r"\b(\d{3,4})\b(?!\s*kg)", config_text)
    mesafe = int(mesafe_match.group(1)) if mesafe_match else None

    if "Çim" in config_text:
        pist = "Çim"
    elif "Kum" in config_text:
        pist = "Kum"
    elif "Sentetik" in config_text:
        pist = "Sentetik"
    else:
        pist = None

    return {
        "kosu_no": kosu_no,
        "saat": saat,
        "mesafe": mesafe,
        "pist": pist,
        "aciklama_ham": config_text,
    }


def parse_betting_results(race_div) -> list[dict]:
    results = []
    for card in race_div.find_all("div", class_="bahisSonucCard"):
        spans = card.find_all("span")
        if len(spans) >= 3:
            results.append({
                "bahis_tipi": spans[0].get_text(strip=True),
                "kazanan": spans[1].get_text(strip=True),
                "tutar": spans[2].get_text(strip=True),
            })
    return results


def parse_city_page(html: str, tarih: str, sehir: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    races = []

    for race_div in soup.find_all("div", id=re.compile(r"^\d+$")):
        if race_div.get("sehir") is None:
            continue

        kosu_kodu = int(race_div["id"])
        meta = parse_race_metadata(race_div)
        table = race_div.find("table", class_="tablesorter")
        atlar = []
        if table:
            tbody = table.find("tbody")
            if tbody:
                for row in tbody.find_all("tr"):
                    atlar.append(parse_horse_row(row))

        races.append({
            "kosu_kodu": kosu_kodu,
            "tarih": tarih,
            "sehir": sehir,
            **meta,
            "atlar": atlar,
            "bahis_sonuclari": parse_betting_results(race_div),
        })

    return races


def fetch_city_results(sehir_id: int, sehir_adi: str, tarih: date, era: str = "past") -> list[dict]:
    tarih_str = tarih.strftime("%d/%m/%Y")
    resp = get(scraper_config.URLS["gunluk_sonuclar_sehir"], params={
        "SehirId": sehir_id,
        "QueryParameter_Tarih": tarih_str,
        "SehirAdi": sehir_adi,
        "Era": era,
    })

    raw_path = scraper_config.RAW_RESULTS_DIR / f"{tarih.isoformat()}_{sehir_adi}.html"
    raw_path.write_text(resp.text, encoding="utf-8")

    races = parse_city_page(resp.text, tarih_str, sehir_adi)
    json_path = scraper_config.RAW_RESULTS_DIR / f"{tarih.isoformat()}_{sehir_adi}.json"
    json_path.write_text(json.dumps(races, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("%s %s: %d koşu bulundu.", tarih_str, sehir_adi, len(races))
    return races


def fetch_all_cities(tarih: date, era: str = "past") -> dict:
    all_results = {}
    for sehir_adi, sehir_id in scraper_config.SEHIR_IDS.items():
        json_path = scraper_config.RAW_RESULTS_DIR / f"{tarih.isoformat()}_{sehir_adi}.json"
        if json_path.exists():
            logger.info("%s %s zaten çekilmiş, atlanıyor.", tarih, sehir_adi)
            all_results[sehir_adi] = json.loads(json_path.read_text(encoding="utf-8"))
            continue
        try:
            all_results[sehir_adi] = fetch_city_results(sehir_id, sehir_adi, tarih, era)
        except Exception as e:
            logger.error("%s %s çekilirken hata: %s", tarih, sehir_adi, e)
    return all_results


def daterange(start: date, end: date):
    for n in range((end - start).days + 1):
        yield start + timedelta(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tarih", help="Tek gün: DD/MM/YYYY")
    parser.add_argument("--start", help="Aralık başı: DD/MM/YYYY")
    parser.add_argument("--end", help="Aralık sonu: DD/MM/YYYY")
    parser.add_argument("--sehir", help="Sadece bu şehir (boşsa hepsi çekilir)")
    parser.add_argument("--era", default="past", choices=["today", "lastMonth", "past"])
    args = parser.parse_args()

    def parse_ddmmyyyy(s):
        d, m, y = s.split("/")
        return date(int(y), int(m), int(d))

    if args.tarih:
        dates = [parse_ddmmyyyy(args.tarih)]
    elif args.start and args.end:
        dates = list(daterange(parse_ddmmyyyy(args.start), parse_ddmmyyyy(args.end)))
    else:
        parser.error("--tarih ya da --start/--end vermelisin")
        return

    for tarih in dates:
        if args.sehir:
            fetch_city_results(scraper_config.SEHIR_IDS[args.sehir], args.sehir, tarih, args.era)
        else:
            fetch_all_cities(tarih, args.era)


if __name__ == "__main__":
    main()