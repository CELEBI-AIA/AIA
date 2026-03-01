# 🛩️ TEKNOFEST 2026 — Havacılıkta Yapay Zeka

<div align="center">

**Otonom hava araçları için gerçek zamanlı nesne tespiti, görsel odometri ve referans obje eşleştirme sistemi**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://docs.ultralytics.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Mimari](#-mimari)
- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Yapılandırma](#-yapılandırma)
- [Görev 3 Parametre Dosyası](#-görev-3-parametre-dosyası)
- [Tutarlılık ve Tekrarlanabilirlik (Best-Effort)](#-tutarlılık-ve-tekrarlanabilirlik-best-effort)
- [Dosya Yapısı](#-dosya-yapısı)
- [Audit & Sağlamlaştırma](#-audit--sağlamlaştırma)
- [Yarışma Kuralları](#-yarışma-kuralları)
- [Görev 1 Temporal Karar Mantığı](#-görev-1-temporal-karar-mantığı)
- [Eğitim ve Test Veri Setleri](#-eğitim-ve-test-veri-setleri)

---

## 🎯 Proje Hakkında

Bu proje, **TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması** kapsamında geliştirilmiştir. Sistem üç ana görevi yerine getirir:

1. **Nesne Tespiti (Görev 1):** Drone kamera görüntülerinden taşıt, insan, UAP (Uçan Araba Park) ve UAİ (Uçan Ambulans İniş) alanlarını gerçek zamanlı tespit eder. Taşıtlar için hareket durumunu, UAP/UAİ için iniş uygunluk durumunu belirler.

2. **Pozisyon Kestirimi (Görev 2):** GPS sinyali kesildiğinde (ilk 1 dk sağlıklı, sonra değişken) görsel odometri (optik akış) ile hava aracının X, Y, Z konumunu metre cinsinden kestirir.

3. **Referans Obje Tespiti (Görev 3):** Oturum başında verilen referans obje fotoğraflarını video karelerinde tespit eder. ORB/SIFT feature matching + homography ile bounding box koordinatlarını raporlar.

---

## 🏗️ Mimari

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              main.py                                    │
│  FPS sayacı • Graceful shutdown • Hata yönetimi • Görev 1/2/3 orkestra  │
│  Bootstrap: runtime_profile.py | Sim: data_loader.py                    │
└───┬──────────┬──────────┬──────────┬──────────────┬─────────────────────┘
    │          │          │          │              │
┌───▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼──────────┐ ┌▼──────────────┐
│network│ │detectn.│ │movemnt.│ │localization  │ │image_matcher  │
│  .py  │ │  .py   │ │  .py   │ │    .py       │ │    .py        │
│ HTTP  │ │ YOLOv8 │ │Temporal│ │GPS + Optik   │ │ORB/SIFT       │
│ Retry │ │ SAHI   │ │ karar  │ │Akış hibrit   │ │Feature Match  │
│ JSON  │ │ İniş   │ │ motion │ │Lucas-Kanade  │ │Homography     │
│ log   │ │ durumu │ │ status │ │Odometri      │ │Ref. Obje      │
└───┬───┘ └───┬────┘ └───┬────┘ └──────┬───────┘ └──────┬────────┘
    │         │          │             │                │
    └─────────┴──────────┴─────────────┴────────────────┘
                          │
     ┌────────────────────▼────────────────────────────────────┐
     │              config/settings.py                          │
     │   Merkezi yapılandırma • Sınıf eşleştirme • Görev 3     │
     └────────────────────┬────────────────────────────────────┘
                          │
     ┌────────────────────▼────────────────────────────────────┐
     │  src/utils.py • src/resilience.py • src/send_state.py    │
     │  Renkli Logger • Visualizer • Circuit Breaker • JSON log │
     └─────────────────────────────────────────────────────────┘
```

---

## ✨ Özellikler

| Özellik | Detay |
|---------|-------|
| **Model** | YOLOv8 (Ultralytics) — COCO/VisDrone → TEKNOFEST sınıf eşleştirmesi, custom eğitim destekli |
| **Hız** | FP16 half-precision + model warmup → **~33 FPS** (RTX 3060) |
| **İniş Tespiti** | Intersection-over-area + kenar temas kontrolü + perspektif marjı |
| **Hareket Tespiti** | Temporal pencere tabanlı karar + kamera hareket kompanzasyonu |
| **Lokalizasyon** | Hibrit GPS + Lucas-Kanade optik akış + Z ekseni scale tahmini + EMA yumuşatma |
| **Referans Obje** | ORB/SIFT feature matching + homography + degenerate guard (Görev 3) |
| **Ağ** | Otomatik retry, timeout yönetimi, circuit breaker, idempotency guard |
| **Debug** | Renkli konsol çıktısı, tespit görselleştirme, periyodik kayıt |
| **Güvenilirlik** | Global hata yakalama, SIGINT/SIGTERM handler, degrade mode, OOM koruması |
| **Offline** | İnternet bağlantısı gerektirmez — yarışma kurallarına uygun (şartname 6.2) |
| **Test** | 45 birim testi, pytest-timeout (10s), tek dosyada (`tests/test_all.py`) |

---

## 🚀 Kurulum

### Gereksinimler

- **Python** 3.10+
- **NVIDIA GPU** (önerilen) + CUDA 12.x
- **İşletim Sistemi:** Linux, Windows (test edildi)

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/siimsek/HavaciliktaYZ.git
cd HavaciliktaYZ

# 2. Sanal ortam oluştur
python -m venv venv

# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# 3. Bağımlılıkları kur (requirements.txt PyTorch CUDA URL içerir)
pip install -r requirements.txt

# 4. Model dosyası: model/ dizinine .pt dosyası yerleştir
# Varsayılan: model/best_mAP50-0.923_mAP50-95-0.766.pt
mkdir model
# Custom eğitilmiş model veya YOLOv8 base model kullanılabilir
```

---

## 💻 Kullanım

### Varsayılan (Yarışma Modu, Non-Interactive)

```bash
python main.py
```

### CLI-First Modlar

```bash
# Yarışma modu
python main.py --mode competition --deterministic-profile max

# Otonom test (VID — Görev 1 + 2 + 3)
python main.py --mode simulate_vid --show

# Otonom test (DET — Görev 1)
python main.py --mode simulate_det --save

# Eski menüyü kullanmak isterseniz
python main.py --interactive
```

### Mock Server ile Yerel Test

```bash
# Terminal 1: Yarışma sunucusunu simüle et
python tools/mock_server.py

# Terminal 2: Competition modunda tam test
python main.py --mode competition
```

Desteklenen tutarlılık profilleri:
- `off`
- `balanced` (simülasyon/iterasyon için önerilen varsayılan)
- `max` (competition modunda daha kararlı sonuç davranışı için önerilir)

### Çıktı Formatı (Sunucuya Gönderilen JSON — Şartname Bölüm 3)

```json
{
  "id": 123,
  "user": "Takim_ID",
  "frame": "/api/frames/123",
  "detected_objects": [
    {
      "cls": 0,
      "landing_status": -1,
      "motion_status": 1,
      "top_left_x": 150,
      "top_left_y": 200,
      "bottom_right_x": 400,
      "bottom_right_y": 350
    }
  ],
  "detected_translations": [
    {
      "translation_x": 1.25,
      "translation_y": -0.43,
      "translation_z": 0.0
    }
  ],
  "detected_undefined_objects": [
    {
      "object_id": 1,
      "top_left_x": 320,
      "top_left_y": 180,
      "bottom_right_x": 480,
      "bottom_right_y": 340
    }
  ]
}
```

---

> **Not:** `detected_objects` → Görev 1 sonuçları, `detected_translations` → Görev 2, `detected_undefined_objects` → Görev 3 referans obje tespitleri. Hareket alanı sunucuda `motion_status` adıyla iletilir.

## ⚙️ Yapılandırma

Tüm ayarlar [`config/settings.py`](config/settings.py) içinde merkezi olarak yönetilir:

### Genel / Çalışma Modları

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `BASE_URL` | `http://127.0.0.1:5000` | Sunucu adresi (yarışma günü güncellenir) |
| `TEAM_NAME` | `"Takim_ID"` | Takım kimliği (yarışma günü güncellenir) |
| `SIMULATION_MODE` | `True` | Legacy simülasyon bayrağı (runtime CLI-first çalışır) |
| `DEBUG` | `True` | Detaylı log + görsel çıktı |
| `MAX_FRAMES` | `2250` | Yarışma karesi limiti (sunucudan dinamik alınabilir) |

### Model Ayarları

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `CONFIDENCE_THRESHOLD` | `0.40` | Minimum tespit güven eşiği |
| `NMS_IOU_THRESHOLD` | `0.15` | NMS IoU eşiği (çift tespit bastırma) |
| `INFERENCE_SIZE` | `1280` | Inference çözünürlüğü (piksel) |
| `HALF_PRECISION` | `True` | FP16 hızlandırma (CUDA) |
| `AGNOSTIC_NMS` | `True` | Sınıflar arası NMS (farklı sınıf çakışmalarını bastırır) |
| `MAX_DETECTIONS` | `300` | Maksimum tespit sayısı (SAHI ile artar) |
| `AUGMENTED_INFERENCE` | `False` | TTA — deterministiklik için kapalı |
| `WARMUP_ITERATIONS` | `3` | Model ısınma tekrarı |

### CLAHE (Ön-İşleme)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `CLAHE_ENABLED` | `True` | Kontrast iyileştirme (karanlık bölgeler) |
| `CLAHE_CLIP_LIMIT` | `2.0` | CLAHE kontrast sınırı |
| `CLAHE_TILE_SIZE` | `8` | CLAHE tile boyutu (piksel) |

### SAHI (Slicing Aided Hyper Inference)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `SAHI_ENABLED` | `True` | Parçalı inference (küçük nesneler için) |
| `SAHI_SLICE_SIZE` | `640` | Parça boyutu (piksel) |
| `SAHI_OVERLAP_RATIO` | `0.35` | Parçalar arası örtüşme oranı |
| `SAHI_MERGE_IOU` | `0.25` | Birleştirme NMS IoU eşiği |

### Bbox Filtreleri

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `MIN_BBOX_SIZE` | `20` | Minimum bbox boyutu (px) |
| `MAX_BBOX_SIZE` | `9999` | Maksimum bbox boyutu (px) |

### Simülasyon (datasets/)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `DATASETS_DIR` | `datasets` | Simülasyon görüntü kök dizini |
| `IMAGE_EXTENSIONS` | `(.jpg, .jpeg, .png, .bmp, .tif, .tiff)` | Recursive taranacak uzantılar |
| `SIMULATION_DET_SAMPLE_SIZE` | `100` | simulate_det modunda rastgele seçilecek görüntü sayısı |

### Görev 2 (Pozisyon Kestirimi)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `FOCAL_LENGTH_PX` | `800.0` | Kamera odak uzunluğu (px) — yarışma kamera parametreleriyle güncellenmeli |
| `DEFAULT_ALTITUDE` | `50.0` | Optik akış fallback irtifası (m) |
| `LATENCY_COMP_ENABLED` | `False` | GPS=0 için submit öncesi latency compensation/projeksiyon bayrağı |
| `LATENCY_COMP_MAX_MS` | `120.0` | Ölçülen fetch→submit gecikmesi için üst sınır (ms) |
| `LATENCY_COMP_MAX_DELTA_M` | `2.0` | Frame başına maksimum projeksiyon mesafesi clamp (m) |
| `LATENCY_COMP_EMA_ALPHA` | `0.35` | Hız (v_t) EMA yumuşatma katsayısı |

> Not: Compensation sadece `gps_health=0` olduğunda çalışır, fetch zamanı monotonic olarak ölçülür ve payload şeması değiştirilmez.

### Görev 3 (Referans Obje Tespiti)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `TASK3_ENABLED` | `True` | Görev 3 modülünü aç/kapat |
| `TASK3_REFERENCE_DIR` | `datasets/task3_references` | Referans obje dizini (veya sunucudan alınır) |
| `TASK3_SIMILARITY_THRESHOLD` | `0.72` | Feature matching onay eşiği |
| `TASK3_FALLBACK_THRESHOLD` | `0.66` | Fallback sweep kabul eşiği |
| `TASK3_FALLBACK_INTERVAL` | `5` | Fallback her N karede tetiklenir |
| `TASK3_FEATURE_METHOD` | `"ORB"` | Feature metodu (`"ORB"` veya `"SIFT"`) |

### Movement (Temporal Karar — Görev 1)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `MOVEMENT_WINDOW_FRAMES` | `24` | Temporal pencere boyutu |
| `MOVEMENT_MIN_HISTORY` | `6` | Karar için minimum geçmiş frame sayısı |
| `MOVEMENT_THRESHOLD_PX` | `12.0` | Hareket eşiği (piksel) |
| `MOVEMENT_MATCH_DISTANCE_PX` | `80.0` | Frame arası bbox eşleştirme mesafesi |
| `MOVEMENT_MAX_MISSED_FRAMES` | `8` | Takip kaybı toleransı |

### Motion Compensation (Kamera Hareket Ayırma)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `MOTION_COMP_ENABLED` | `True` | Kamera hareket kompanzasyonunu aç/kapat |
| `MOTION_COMP_MIN_FEATURES` | `40` | Güvenilir global flow için minimum köşe sayısı |
| `MOTION_COMP_MAX_CORNERS` | `200` | Shi-Tomasi ile çıkarılacak maksimum köşe |
| `MOTION_COMP_QUALITY_LEVEL` | `0.01` | Köşe kalite eşiği |
| `MOTION_COMP_MIN_DISTANCE` | `20` | Köşeler arası minimum mesafe |
| `MOTION_COMP_WIN_SIZE` | `21` | LK optik akış pencere boyutu |

### Tutarlılık ve Tekrarlanabilirlik (Best-Effort)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `DETERMINISM_SEED` | `42` | Run-to-run varyansını azaltmak için global seed |
| `DETERMINISM_CPU_THREADS` | `1` | CPU thread sabitleme |

---

## 🎛️ Görev 3 Parametre Dosyası

`config/task3_params.yaml` dosyası **opsiyonel** olarak yüklenir. Dosya mevcutsa, içindeki değerler `Settings` üzerine yazılır. YAML yoksa veya hata varsa `config/settings.py` sabitleri kullanılır.

| Settings Parametresi | Varsayılan | YAML Anahtarı |
|---------------------|------------|---------------|
| `TASK3_SIMILARITY_THRESHOLD` | `0.72` | `t_confirm` |
| `TASK3_FALLBACK_THRESHOLD` | `0.66` | `t_fallback` |
| `TASK3_FALLBACK_INTERVAL` | `5` | `n_fallback_interval` |
| `TASK3_GRID_STRIDE` | `32` | `grid_stride` |

---

## 🔒 Tutarlılık ve Tekrarlanabilirlik (Best-Effort)

Sistem çıktılarında run-to-run varyansını azaltmak için aşağıdaki pratikler önerilir:

1. **Seed Sabitleme (numpy/torch/random):**
   - Aynı senaryolarda karşılaştırılabilir sonuçlar için sabit seed kullanılması önerilir.
   - `numpy`, `torch`, `random` için tek noktadan seed ataması pratik bir yaklaşımdır.

2. **Model Eval Mode:**
   - İnference öncesi modellerin `eval` modunda çalıştırılması önerilir.
   - Dropout ve BatchNorm gibi katmanların eğitim davranışını kapatmak sonuç stabilitesine yardımcı olur.

3. **Sabit Sürüm Pinleme:**
   - `torch`, `torchvision`, `ultralytics`, CUDA ve cuDNN sürümlerini pinlemek önerilir.
   - Üretim ortamında sürüm kaymasını azaltmak için aynı bağımlılık seti korunmalıdır.

4. **JSON Sırası ve Kararlı Serileştirme:**
   - Çıktı JSON'larını kararlı anahtar sırası ile üretmek (`sort_keys=True` veya sabit alan sırası) tavsiye edilir.
   - Sayısal formatlama ve alan sırasını sürümler arasında korumak entegrasyon riskini azaltır.

5. **Frame-Index Tabanlı Karar Kuralları:**
   - Adaptasyonları wall-clock yerine frame index/pencere kuralına bağlamak daha tutarlı sonuç üretir.
   - Bu yaklaşım farklı donanımlarda karar sapmasını azaltır.
   - Not: Wall-clock kullanımı ağ dayanıklılığı orkestrasyonu (circuit breaker/degrade) için kullanılabilir; model karar mantığında (`motion_status`, `landing_status`, sınıf çıktıları) frame-index yaklaşımı tercih edilir.

6. **Runtime Profil Kullanımı:**
   - Competition çalıştırmalarında `max` profil daha kararlı davranış için önerilir.
   - `max`: seed + deterministic backend + TTA kapalı + FP16 kapalı (FP32); sınır vakalarda run-to-run farkını azaltmayı hedefler.
   - `balanced`: seed + deterministic backend + TTA kapalı, FP16 açık; simülasyon ve hızlı iterasyon için uygundur.

---

## 📂 Dosya Yapısı

```
HavaciliktaYZ/
├── main.py                  # Ana giriş noktası (Görev 1 + 2 + 3 orkestra)
├── requirements.txt         # Python bağımlılıkları
├── README.md               # Bu dosya
├── .gitignore              # Git hariç tutma kuralları
│
├── config/
│   ├── __init__.py
│   ├── settings.py         # Merkezi yapılandırma (tüm görevler)
│   └── task3_params.yaml   # Görev 3 parametreleri (opsiyonel override)
│
├── src/
│   ├── __init__.py
│   ├── detection.py        # Görev 1: YOLOv8 nesne tespiti + iniş durumu
│   ├── frame_context.py    # Ortak gri dönüşüm (detection/movement/localization)
│   ├── movement.py         # Görev 1: Temporal hareket kararı + kamera kompanzasyonu
│   ├── localization.py     # Görev 2: GPS + optik akış + EMA pozisyon kestirimi
│   ├── image_matcher.py    # Görev 3: ORB/SIFT referans obje eşleştirme
│   ├── network.py          # Sunucu iletişimi + retry + idempotency + payload guard
│   ├── resilience.py       # Circuit breaker + degrade mode kontrolü
│   ├── data_loader.py      # Simülasyon veri yükleme (VID/DET)
│   ├── runtime_profile.py  # Deterministik profil uygulaması
│   ├── send_state.py       # SendResultStatus enum tanımları
│   └── utils.py            # Logger, Visualizer, yardımcı araçlar
│
├── tools/
│   └── mock_server.py      # Yerel mock sunucu (yarışma formatı test)
│
├── tests/
│   ├── conftest.py         # ML mock'ları + 10s global timeout
│   └── test_all.py         # 47 konsolide birim testi
│
├── model/
│   └── best_*.pt           # Eğitilmiş YOLOv8 modeli (Git'e dahil değil)
│
├── datasets/
│   ├── task3_references/    # Görev 3 referans obje resimleri
│   └── (herhangi alt klasör)  # Simülasyon: recursive taranır (.jpg, .png vb.); VID/DET modu için
│
├── sartname/
│   └── sartname.md         # Yarışma birleşik teknik şartnamesi
│
├── logs/                   # Çalışma zamanı logları (otomatik)
└── debug_output/           # Debug görselleri (otomatik)
```

---

## 🛡️ Audit & Sağlamlaştırma

Sistem kapsamlı bir audit sürecinden geçirilmiş ve aşağıdaki iyileştirmeler uygulanmıştır:

| # | İyileştirme | Dosya | Detay |
|---|------------|-------|-------|
| 1 | **Optik akış EMA yumuşatma** | `localization.py` | Frame-to-frame gürültüyü α=0.4 EMA ile bastırma + son GPS irtifası fallback |
| 2 | **Exception sağlamlaştırma** | `detection.py` | OOM ayrı handle, `SystemExit`/`KeyboardInterrupt` yeniden raise |
| 3 | **Kararlı sıralama** | `detection.py` | NMS ve containment suppression'da `kind="stable"` |
| 4 | **Float birikim sınırı** | `movement.py` | `_cam_total_x/y` ±1e6 ile sınırlandı |
| 5 | **GPS simülasyonu** | `data_loader.py` | Deterministik döngü yerine %33 rastgele degradasyon |
| 6 | **Homography koruması** | `image_matcher.py` | Dejenere/koliner nokta kontrolü + fallback bounding rect |
| 7 | **task3_params.yaml** | `config/settings.py` | YAML opsiyonel yükleme; mevcutsa Görev 3 parametrelerini override eder |
| 8 | **Fallback pozisyon** | `main.py` | Görüntü indirilemezse son bilinen pozisyon (0,0,0 yerine) |
| 9 | **Circuit breaker** | `resilience.py` | Oturum iptali yok; degrade modunda devam, ağ düzelince toparlanma |

### Testler

```bash
# Tüm testleri çalıştır (pytest-timeout 10s)
python -m pytest tests/test_all.py -v
```

Gereksinimler: `pytest`, `pytest-timeout`, `PyYAML` (`requirements.txt` içinde)

---

## 📏 Yarışma Kuralları (Şartname Özeti)

### Görev 1: Tespit Edilecek Nesneler

| Sınıf | ID | İniş Durumu | Hareket Durumu | Açıklama |
|-------|----|-------------|----------------|----------|
| **Taşıt** | 0 | -1 | 0 veya 1 | Otomobil, motosiklet, otobüs, kamyon, tren, deniz taşıtı, traktör |
| **İnsan** | 1 | -1 | -1 | Ayakta/oturur tüm insanlar |
| **UAP** | 2 | 0 veya 1 | -1 | Uçan Araba Park alanı (4,5 m çap, mavi daire) |
| **UAİ** | 3 | 0 veya 1 | -1 | Uçan Ambulans İniş alanı (4,5 m çap, kırmızı daire) |

### İniş Uygunluk Kuralları (Şartname 3.1.3)

- **Uygun (1):** UAP/UAİ alanı **tamamen** kadraj içinde VE üzerinde hiçbir nesne yok
- **Uygun Değil (0):** Alan kısmen kadraj dışı VEYA üzerinde nesne var (perspektif yanılsaması dahil)
- Perspektif etkisi: Alana yakın cisimler üstünde olmasa bile perspektiften dolayı öyle görünüyorsa → Uygun Değil (0)

### Özel Etiketleme Kuralları (Şartname 3.1.2)

- **Bisiklet/motosiklet sürücüsü:** İnsan değil, taşıtla birlikte yalnız "taşıt" olarak etiketlenir
- **Scooter:** Sürücüsüz → "taşıt", sürücülü → "insan" (uygulamada rider suppression yaklaşımı)
- **Tren:** Lokomotif ve her vagon ayrı birer obje olarak tespit edilir
- **Kısmi görünürlük:** Tamamı görünmeyen veya kısmen örtülen nesneler de tespit edilmelidir
- **Kamera hareketi:** Sabit taşıtlar kamera hareketi nedeniyle hareketli görünebilir, ayırt edilmeli

### Görev 2: Pozisyon Kestirimi (Şartname 3.2)

- İlk 1 dakika (450 kare) GPS kesinlikle sağlıklı
- Son 4 dakikada GPS sağlıksız olabilir → görsel odometri devreye girer
- GPS sağlıksız olduğunda sunucu `translation_x/y/z = "NaN"` gönderir
- Çıktı: X, Y, Z eksenleri metre cinsinden

### Görev 3: Referans Obje Tespiti (Şartname 3.3)

- Oturum başlangıcında 1+ referans obje fotoğrafı verilir
- Farklı kamera (termal, RGB), farklı açı/irtifa veya uydu görüntüsü olabilir
- Referans obje her karede aranır ve bulunursa `detected_undefined_objects` ile raporlanır

## ⏱️ Görev 1 Temporal Karar Mantığı

Görev 1 kararları tek frame üzerinden verilmez. Tüm hareket ve iniş uygunluk çıktıları pencere (window) tabanlı temporal birikim ile üretilir.

### 1) Window (Pencere) Yapısı

- Her hedef nesne/alan için son `W` frame tutulur (örnek: `W=24`).
- `W` değeri sabit konfigürasyon parametresidir; çalışma sırasında dinamik değiştirilmez.
- Karar, tek bir frame yerine pencere içindeki kanıtların birleşimi ile verilir.

### 2) Decay (Ağırlıklandırma)

- Yakın frame'lere daha yüksek, eski frame'lere daha düşük ağırlık verilir.
- Örnek ağırlık şeması: üstel veya doğrusal decay (`w_t`) ve normalize toplam.
- Amaç kısa süreli gürültü/yanlış tespitten etkilenmeden stabil karar üretmektir.

### 3) Threshold (Karar Eşiği)

- Pencere boyunca biriken temporal skor `S` hesaplanır.
- `S >= T_move` ise taşıt için `movement_status=1`, aksi halde `movement_status=0`.
- Runtime çıktısında bu alan şartname uyumu için `motion_status` adıyla gönderilir.
- Kamera hareketi, global median optical-flow kompanzasyonu ile ayrıştırılır.
- UAP/UAİ için `S >= T_land` ise `landing_status=1`, aksi halde `landing_status=0`.
- `T_move` ve `T_land` kalibrasyon testleri ile sabitlenir.

### 4) Tek-Frame Karar Yasağı

- Tek frame ile doğrudan `movement_status` veya `landing_status` kararı verilmez.
- Anlık kararlar yalnızca geçici kanıt olarak temporal havuza yazılır; nihai karar pencere sonunda üretilir.

### Teknik Kısıtlamalar (Şartname 6.2 / 8.1)

- 📡 İnternet bağlantısı **kesinlikle yasak** (offline çalışma zorunlu, şartname 6.2)
- 🎬 Oturum başına **2250 kare** (5 dk, 7.5 FPS — uçuş süresi değişkenlik gösterebilir)
- 📐 Çözünürlük: 1920×1080 (Full HD) veya 3840×2160 (4K)
- 🖼️ Kare formatı: JPG, PNG vb. (herhangi bir görüntü formatı olabilir)
- 📊 Görev 1 değerlendirme: mAP (IoU ≥ 0.5)
- 📊 Görev 2 değerlendirme: Konum kestirimi doğruluğu (metre cinsinden hata)
- 🎥 Video koşulları: Gece/gündüz, kar/yağmur, şehir/orman/deniz, 70–90° kamera açısı
- 🌡️ Kamera: RGB veya termal kamera

---

## 📊 Eğitim ve Test Veri Setleri

Yarışma öncesi modeli eğitmek ve sistemi test etmek için kullanılabilecek veri setleri:

### Önerilen Veri Setleri

| Dataset | İçerik | Neden Uygun? | Link |
|---------|--------|-------------|------|
| **VisDrone** | 260K+ kare, insan + araç | Drone perspektifi, çeşitli ortamlar | [GitHub](https://github.com/VisDrone/VisDrone-Dataset) |
| **UAVDT** | 80K kare, araç tespiti | UAV yükseklik çeşitliliği | [Site](https://sites.google.com/view/grli-uavdt) |
| **TEKNOFEST Resmi** | Örnek video (Mart 2026) | Yarışma formatı ile birebir uyumlu | [GitHub](https://github.com/TEKNOFEST-YARISMALAR/havacilikta-yapay-zeka-yarismasi) |

### VisDrone ile Eğitim

VisDrone sınıfları TEKNOFEST'e doğrudan eşleştirilebilir:

```
VisDrone → TEKNOFEST
──────────────────────
pedestrian    → İnsan (1)
people        → İnsan (1)
car           → Taşıt (0)
van           → Taşıt (0)
truck         → Taşıt (0)
bus           → Taşıt (0)
motor         → Taşıt (0)
bicycle       → Taşıt (0)
tricycle      → Taşıt (0)
```

> ⚠️ **Not:** TEKNOFEST resmi örnek video dağıtım tarihi **10-28 Mart 2026**'tir. [Resmi repo](https://github.com/TEKNOFEST-YARISMALAR/havacilikta-yapay-zeka-yarismasi) takip edilmelidir.

---

## 📜 Lisans

MIT License — Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

<div align="center">

**TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması** için geliştirilmiştir 🇹🇷

</div>
