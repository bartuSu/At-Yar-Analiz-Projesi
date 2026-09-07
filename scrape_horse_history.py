"""
Bir atın (AtId ile) bio bilgisini ve TÜM geçmiş koşularını çeker.
"""
import argparse
import json
import re

from bs4 import BeautifulSoup

import scraper_config
from scrapper_http import get, logger


def parse_bio(soup: BeautifulSoup) -> dict:
    """
    Bio bilgisini span.key / span.value çiftlerinden çeker (regex yerine
    gerçek HTML yapısını kullanıyor - sayfanın üst kısmındaki arama
    menüsündeki benzer kelimelerle karışmıyor).
    """
    container = soup.select_one("div.grid_14.kunye") or soup
    pairs = {}
    for key_span in container.find_all("span", class_="key"):
        value_span = key_span.find_next_sibling("span", class_="value")
        if value_span:
            pairs[key_span.get_text(strip=True)] = value_span.get_text(" ", strip=True)

    anne_raw = pairs.get("Anne", "")
    anne_parcalari = [p.strip() for p in anne_raw.split("/")] if anne_raw else []

    def _extract_id(span_key, id_param):
        key_span = container.find("span", class_="key", string=span_key)
        if not key_span:
            return None
        value_span = key_span.find_next_sibling("span", class_="value")
        link = value_span.find("a") if value_span else None
        if not link:
            return None
        m = re.search(rf"{id_param}=(\d+)", link.get("href", ""))
        return int(m.group(1)) if m else None

    return {
        "isim": pairs.get("İsim"),
        "yas_cinsiyet": pairs.get("Yaş"),
        "dogum_tarihi": pairs.get("Doğ. Trh"),
        "handikap_puani": pairs.get("Handikap P.") or None,
        "baba": pairs.get("Baba"),
        "anne": anne_parcalari[0] if len(anne_parcalari) > 0 else None,
        "kisrak_babasi": anne_parcalari[1] if len(anne_parcalari) > 1 else None,
        "antrenor": pairs.get("Antrenör"),
        "antrenor_id": _extract_id("Antrenör", "QueryParameter_AntrenorId"),
        "gercek_sahip": pairs.get("Gerçek Sahip"),
        "uzerine_kosan_sahip": pairs.get("Üzerine Koşan Sahip"),
        "sahip_id": _extract_id("Üzerine Koşan Sahip", "QueryParameter_SahipId"),
        "yetistirici": pairs.get("Yetiştirici"),
        "yetistirici_id": _extract_id("Yetiştirici", "QueryParameter_YetistiriciId"),
    }


RACE_HISTORY_COLUMNS = [
    "tarih", "sehir", "mesafe", "pist", "sira", "derece", "siklet", "taki",
    "jokey", "start_no", "ganyan", "grup", "kosu_no_gunun", "kosu_cinsi",
    "antrenor", "sahip", "handikap_puani", "ikramiye", "s20", "video_url", "foto_url",
]


def parse_race_history(soup: BeautifulSoup) -> list[dict]:
    """
    Koşu geçmişi tablosunu id="queryTable" ile bulur (thead'de <tr> YOK,
    bu yüzden eski "başlık satırında Tarih/Şehir ara" mantığı çalışmıyordu).
    tbody satırlarını sabit kolon sırasına göre parse eder.
    """
    table = soup.find("table", id="queryTable")
    if not table:
        return []

    tbody = table.find("tbody")
    if not tbody:
        return []

    results = []
    for row in tbody.find_all("tr", recursive=False):
        tds = row.find_all("td", recursive=False)
        # "Toplam N sonuçtan..." gibi özet/loading satırlarını atla
        if len(tds) < len(RACE_HISTORY_COLUMNS) - 2:
            continue

        record = {}
        for name, td in zip(RACE_HISTORY_COLUMNS, tds):
            if name in ("jokey", "antrenor", "sahip"):
                link = td.find("a")
                record[name] = link.get_text(strip=True) if link else (td.get_text(strip=True) or None)
                id_param = {
                    "jokey": "QueryParameter_JokeyId",
                    "antrenor": "QueryParameter_AntrenorId",
                    "sahip": "QueryParameter_SahipId",
                }[name]
                m = re.search(rf"{id_param}=(\d+)", link.get("href", "")) if link else None
                record[f"{name}_id"] = int(m.group(1)) if m else None
            elif name in ("video_url", "foto_url"):
                link = td.find("a")
                record[name] = link.get("href") if link else None
            else:
                record[name] = td.get_text(" ", strip=True) or None

        results.append(record)

    return results


def parse_summary_stats(soup: BeautifulSoup) -> list[dict]:
    """TOPLAM / Çim / Kum / Sentetik özet istatistik tablosu."""
    results = []
    for table in soup.find_all("table", class_="tablesorter"):
        header = table.find("thead")
        if not header or "K." not in header.get_text():
            continue
        tbody = table.find("tbody")
        if not tbody:
            continue
        for row in tbody.find_all("tr"):
            tds = row.find_all("td")
            if len(tds) < 8:
                continue
            results.append({
                "pist": tds[0].get_text(strip=True),
                "kosu_sayisi": tds[1].get_text(strip=True),
                "1inci": tds[2].get_text(strip=True),
                "2nci": tds[3].get_text(strip=True),
                "3uncu": tds[4].get_text(strip=True),
                "4uncu": tds[5].get_text(strip=True),
                "5inci": tds[6].get_text(strip=True),
                "kazanc": tds[7].get_text(" ", strip=True),
            })
        break  # ilk uygun tablo yeterli
    return results


def fetch_horse(at_id: int) -> dict | None:
    raw_path = scraper_config.RAW_HORSE_DIR / f"at_{at_id}.html"
    json_path = scraper_config.RAW_HORSE_DIR / f"at_{at_id}.json"

    if json_path.exists():
        logger.info("At %s zaten çekilmiş, atlanıyor.", at_id)
        return json.loads(json_path.read_text(encoding="utf-8"))

    resp = get(scraper_config.URLS["at_kosu_bilgileri"], params={"1": "1", "QueryParameter_AtId": at_id})
    raw_path.write_text(resp.text, encoding="utf-8")

    soup = BeautifulSoup(resp.text, "lxml")
    if "İsim" not in soup.get_text():
        logger.warning("AtId %s için veri bulunamadı.", at_id)
        return None

    data = {
        "at_id": at_id,
        "bio": parse_bio(soup),
        "ozet_istatistik": parse_summary_stats(soup),
        "kosu_gecmisi": parse_race_history(soup),
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("AtId %s kaydedildi: %d geçmiş koşu.", at_id, len(data["kosu_gecmisi"]))
    return data


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--at-id", type=int)
    group.add_argument("--at-id-range", type=int, nargs=2, metavar=("BASLA", "BITIR"))
    args = parser.parse_args()

    if args.at_id:
        fetch_horse(args.at_id)
        return

    start, end = args.at_id_range
    ok, empty = 0, 0
    for at_id in range(start, end + 1):
        try:
            result = fetch_horse(at_id)
            ok += 1 if result else 0
            empty += 0 if result else 1
        except Exception as e:
            logger.error("AtId %s çekilirken hata: %s", at_id, e)
        if at_id % 50 == 0:
            logger.info("İlerleme: %d/%d (bulunan: %d, boş: %d)", at_id - start, end - start, ok, empty)

    logger.info("Bitti. Toplam bulunan: %d, boş: %d", ok, empty)


if __name__ == "__main__":
    main()