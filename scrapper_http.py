"""
Ortak HTTP client: her istekte otomatik rate-limit + retry uygular.
"""
import time
import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import scraper_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tjk_scraper")

_session = requests.Session()
_session.headers.update({
    "User-Agent": scraper_config.USER_AGENT,
    "Accept-Language": "tr-TR,tr;q=0.9",
})

_last_request_time = 0.0


def _throttle():
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    wait = scraper_config.REQUEST_DELAY_SECONDS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


@retry(
    stop=stop_after_attempt(scraper_config.MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def get(url: str, params: dict | None = None) -> requests.Response:
    _throttle()
    logger.info("GET %s params=%s", url, params)
    resp = _session.get(url, params=params, timeout=scraper_config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


@retry(
    stop=stop_after_attempt(scraper_config.MAX_RETRIES),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
    reraise=True,
)
def post(url: str, data: dict | None = None) -> requests.Response:
    _throttle()
    logger.info("POST %s data=%s", url, data)
    resp = _session.post(url, data=data, timeout=scraper_config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp