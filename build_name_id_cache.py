"""
Jokey/antrenör/sahip isim -> ID eşlemesini, tüm ham günlük sonuç
JSON'larından TEK SEFERLİK tarayıp çıkarır, data/processed/name_id_cache.json
olarak kaydeder. Dashboard bunu okur, ham JSON'ları taramaz (14 bin+ dosya
her seferinde taramak çok yavaş olurdu).

Aynı isim birden fazla ID'ye denk geliyorsa (nadir ama olabilir - iki farklı
kişinin kısaltması aynı çıkabiliyor), EN SIK GEÇEN ID seçilir.

Kullanım:
    python build_name_id_cache.py
    -> data/processed/name_id_cache.json üretir (birkaç dakika sürebilir)
"""
import json
import re
from collections import Counter

import scraper_config


def normalize_isim(isim: str) -> str:
    """
    'M. M. Bilgin', '(Ap.) K. Yılmaz', 'M.A.SOLMAZ' gibi farklı formatlardaki
    isimleri KARŞILAŞTIRILABİLİR tek bir forma indirger: sadece harfler,
    büyük harf, Türkçe karakterler ASCII'ye çevrilmiş.
    """
    if not isim:
        return ""
    isim = re.sub(r"\(ap\.?\)", "", isim, flags=re.IGNORECASE)  # "(Ap.)" apranti etiketini at
    ceviri = str.maketrans("çğıöşüİ", "cgiosui")
    isim = isim.lower().translate(ceviri)
    isim = re.sub(r"[^a-z]", "", isim)  # sadece harfler kalsın
    return isim


def main():
    jokey_sayaci: dict[str, Counter] = {}
    antrenor_sayaci: dict[str, Counter] = {}
    sahip_sayaci: dict[str, Counter] = {}

    json_dosyalari = list(scraper_config.RAW_RESULTS_DIR.glob("*.json"))
    print(f"{len(json_dosyalari)} dosya taranacak...")

    for i, json_path in enumerate(json_dosyalari, 1):
        try:
            races = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for race in races:
            for at in race.get("atlar", []):
                for isim_alani, id_alani, sayac in [
                    ("jokey_adi", "jokey_id", jokey_sayaci),
                    ("antrenor_adi", "antrenor_id", antrenor_sayaci),
                    ("sahip_adi", "sahip_id", sahip_sayaci),
                ]:
                    isim = at.get(isim_alani)
                    id_deger = at.get(id_alani)
                    if not isim or not id_deger:
                        continue
                    norm = normalize_isim(isim)
                    if not norm:
                        continue
                    sayac.setdefault(norm, Counter())[id_deger] += 1

        if i % 2000 == 0:
            print(f"  {i}/{len(json_dosyalari)} dosya tarandı...")

    def en_sik_id(sayac: dict[str, Counter]) -> dict[str, int]:
        return {isim: sayaç.most_common(1)[0][0] for isim, sayaç in sayac.items()}

    cache = {
        "jokey_adi_to_id": en_sik_id(jokey_sayaci),
        "antrenor_adi_to_id": en_sik_id(antrenor_sayaci),
        "sahip_adi_to_id": en_sik_id(sahip_sayaci),
    }

    out_path = scraper_config.PROCESSED_DIR / "name_id_cache.json"
    out_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nKaydedildi: {out_path}")
    print(f"  {len(cache['jokey_adi_to_id'])} benzersiz jokey ismi")
    print(f"  {len(cache['antrenor_adi_to_id'])} benzersiz antrenör ismi")
    print(f"  {len(cache['sahip_adi_to_id'])} benzersiz sahip ismi")


if __name__ == "__main__":
    main()