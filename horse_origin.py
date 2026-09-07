"""
Günlük sonuçlarda geçen TÜM benzersiz atların bio bilgisini (baba/anne dahil)
çeker. scrape_horse_history.fetch_horse zaten cache'li çalıştığı için,
daha önce çekilmiş atları otomatik atlar - yarıda kesilirse güvenle
tekrar çalıştırabilirsin.

UYARI: Dataset'te muhtemelen binlerce benzersiz at var. Rate limit
(1.5 sn/istek) yüzünden bu SAATLERCE sürebilir. Gece çalıştırmayı düşün.

Kullanım:
    python fetch_horses_for_dataset.py
"""
import json

import scraper_config
from scrape_horse_history import fetch_horse
from scrapper_http import logger


def collect_unique_at_ids() -> set[int]:
    """Tüm günlük sonuç JSON'larından benzersiz at_id'leri toplar."""
    at_ids = set()
    for json_path in scraper_config.RAW_RESULTS_DIR.glob("*.json"):
        races = json.loads(json_path.read_text(encoding="utf-8"))
        for race in races:
            for at in race["atlar"]:
                if at.get("at_id"):
                    at_ids.add(at["at_id"])
    return at_ids


def main():
    at_ids = sorted(collect_unique_at_ids())
    logger.info("Toplam %d benzersiz at bulundu.", len(at_ids))

    ok, empty, hata = 0, 0, 0
    for i, at_id in enumerate(at_ids, 1):
        try:
            result = fetch_horse(at_id)
            ok += 1 if result else 0
            empty += 0 if result else 1
        except Exception as e:
            logger.error("AtId %s çekilirken hata: %s", at_id, e)
            hata += 1

        if i % 100 == 0:
            logger.info("İlerleme: %d/%d (ok: %d, boş: %d, hata: %d)", i, len(at_ids), ok, empty, hata)

    logger.info("Bitti. Toplam: %d, ok: %d, boş: %d, hata: %d", len(at_ids), ok, empty, hata)


if __name__ == "__main__":
    main()