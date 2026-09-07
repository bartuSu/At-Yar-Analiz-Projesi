"""
Çok yıllık geçmiş veri taraması için üst seviye orkestratör.
"""
import argparse
from datetime import date, timedelta

from scrape_daily_results import daterange, fetch_all_cities
from scrapper_http import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--era", default="past", choices=["today", "lastMonth", "past"])
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=365 * args.years)

    logger.info("Backfill başlıyor: %s -> %s (%d yıl)", start, end, args.years)
    logger.info(
        "Not: At bio/geçmiş verisi için ayrıca "
        "`python scrape_horse_history.py --at-id-range 1 N` çalıştırman gerekiyor."
    )

    total_days = (end - start).days
    for i, tarih in enumerate(daterange(start, end)):
        fetch_all_cities(tarih, args.era)
        if i % 30 == 0:
            logger.info("İlerleme: %d/%d gün", i, total_days)

    logger.info("Backfill (günlük sonuçlar) tamamlandı.")


if __name__ == "__main__":
    main()