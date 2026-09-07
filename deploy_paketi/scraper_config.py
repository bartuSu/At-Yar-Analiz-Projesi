"""
Merkezi ayarlar. Buradaki değerleri değiştirerek script'in davranışını
kontrol edersin. Yeni bir hipodrom/koşu tipi keşfedince buraya ekle.
"""
from pathlib import Path

BASE_URL = "https://www.tjk.org"

URLS = {
    "at_kosu_bilgileri": BASE_URL + "/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri",
    "gunluk_sonuclar_sehir": BASE_URL + "/TR/yarissever/Info/Sehir/GunlukYarisSonuclari",
    "at_sorgulama": BASE_URL + "/TR/YarisSever/Query/Page/Atlar",
    "kosu_sorgulama": BASE_URL + "/TR/YarisSever/Query/Page/KosuSorgulama",
}

# Bilinen şehir ID'leri - tarayıcı DevTools'tan doğrulanarak bulundu.
# Eksik olanları (İstanbul, İzmir, vs.) o günün bülten sayfasını açıp
# şehir sekmesinin href'indeki SehirId= değerinden bulup buraya ekle.
SEHIR_IDS = {
     "Ankara": 5,
     "Kocaeli": 9,
     "Karma": 17,
     "İstanbul":3,
     "İzmir":2,
     "Bursa":4,
     "Elazığ":7,
     "Diyarbakır":8,
}

# TJK sunucusuna nazik davranmak için istekler arası bekleme (saniye).
# Bunu 1'in altına DÜŞÜRME.
REQUEST_DELAY_SECONDS = 1.5

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT = 20  # saniye
MAX_RETRIES = 3

# Veri klasörleri
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_RESULTS_DIR = RAW_DIR / "gunluk_sonuclar"
RAW_HORSE_DIR = RAW_DIR / "at_gecmisi"

for d in (RAW_RESULTS_DIR, RAW_HORSE_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)