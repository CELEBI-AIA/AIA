# 🛩️ TEKNOFEST 2026 — Havacılıkta Yapay Zeka

<div align="center">

**Otonom hava araçları için gerçek zamanlı nesne tespiti ve görsel odometri sistemi**

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
- [Deterministiklik Sözleşmesi](#-deterministiklik-sözleşmesi)
- [Dosya Yapısı](#-dosya-yapısı)
- [Yarışma Kuralları](#-yarışma-kuralları)
- [Görev 1 Temporal Karar Mantığı](#-görev-1-temporal-karar-mantığı)
- [Eğitim ve Test Veri Setleri](#-eğitim-ve-test-veri-setleri)

---

## 🎯 Proje Hakkında

Bu proje, **TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması** kapsamında geliştirilmiştir. Sistem iki ana görevi yerine getirir:

1. **Nesne Tespiti (Görev 1):** Drone kamera görüntülerinden taşıt, insan, UAP (Uçan Araba Park) ve UAİ (Uçan Ambulans İniş) alanlarını gerçek zamanlı tespit eder. İniş alanlarının uygunluk durumunu belirler.

2. **Pozisyon Kestirimi (Görev 2):** GPS sinyali kesildiğinde görsel odometri (optik akış) ile hava aracının konumunu kestirir.

---

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main.py                                      │
│  FPS sayacı • Graceful shutdown • Hata yönetimi                       │
│  Bootstrap: runtime_profile.py | Sim: data_loader.py                 │
└────┬────────────┬────────────┬────────────┬──────────────────────────┘
     │            │            │            │
┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌────▼────────────┐
│network │  │detection│  │movement │  │ localization     │
│  .py   │  │  .py    │  │  .py    │  │    .py           │
│ HTTP   │  │ YOLOv8  │  │ Temporal│  │ GPS + Optik      │
│ Retry  │  │ FP16    │  │ karar   │  │ Akış hibrit     │
│ JSON   │  │ İniş    │  │ movement│  │ Lucas-Kanade     │
│ log    │  │ durumu  │  │ status  │  │ Odometri         │
└────┬───┘  └────┬────┘  └────┬────┘  └────────┬────────┘
     │           │             │                 │
     └───────────┴─────────────┴─────────────────┘
                          │
     ┌────────────────────▼─────────────────────────────┐
     │              config/settings.py                    │
     │   Merkezi yapılandırma • Sınıf eşleştirme          │
     │   Kamera parametreleri • Ağ ayarları               │
     └────────────────────┬──────────────────────────────┘
                          │
     ┌────────────────────▼──────────────────────────────┐
     │              src/utils.py                          │
     │   Renkli Logger • Visualizer • JSON log            │
     └───────────────────────────────────────────────────┘
```

---

## ✨ Özellikler

| Özellik | Detay |
|---------|-------|
| **Model** | YOLOv8m (Ultralytics) — COCO → TEKNOFEST sınıf eşleştirmesi |
| **Hız** | FP16 half-precision + model warmup → **~33 FPS** (RTX 3060) |
| **İniş Tespiti** | Intersection-over-area + kenar temas kontrolü |
| **Lokalizasyon** | Hibrit GPS + Lucas-Kanade optik akış |
| **Ağ** | Otomatik retry, timeout yönetimi, JSON traffic logging |
| **Debug** | Renkli konsol çıktısı, tespit görselleştirme, periyodik kayıt |
| **Güvenilirlik** | Global hata yakalama, SIGINT/SIGTERM handler, asla çökmez |
| **Offline** | İnternet bağlantısı gerektirmez — yarışma kurallarına uygun |

---

## 🚀 Kurulum

### Gereksinimler

- **Python** 3.10+
- **NVIDIA GPU** (önerilen) + CUDA 12.x
- **İşletim Sistemi:** Linux (Ubuntu 22.04 test edildi)

### Adımlar

```bash
# 1. Repoyu klonla
git clone https://github.com/siimsek/HavaciliktaYZ.git
cd HavaciliktaYZ

# 2. Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate

# 3. PyTorch'u CUDA ile kur (önce)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Diğer bağımlılıkları kur
pip install -r requirements.txt

# 5. Model dosyasını indir (eğer yoksa)
# YOLOv8m modeli models/ dizinine yerleştirilmeli
mkdir -p models
# https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt
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
python main.py --mode competition --deterministic-profile balanced

# Otonom test (VID)
python main.py --mode simulate_vid --show

# Otonom test (DET)
python main.py --mode simulate_det --save

# Eski menüyü kullanmak isterseniz
python main.py --interactive
```

Desteklenen deterministik profiller:
- `off`
- `balanced` (önerilen, varsayılan)
- `max`

### Çıktı Formatı (Sunucuya Gönderilen JSON)

```json
{
  "id": 123,
  "user": "Takim_ID",
  "frame": "/api/frames/123",
  "detected_objects": [
    {
      "cls": "0",
      "landing_status": "-1",
      "motion_status": "1",
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
  "detected_undefined_objects": []
}
```

---

Not: Runtime, taslak şartnameyle uyumlu geniş şema gönderir. Hareket alanı sunucuda `motion_status` adıyla iletilir.

## ⚙️ Yapılandırma

Tüm ayarlar [`config/settings.py`](config/settings.py) içinde merkezi olarak yönetilir:

### Genel / Çalışma Modları

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `SIMULATION_MODE` | `True` | Legacy simülasyon bayrağı (runtime CLI-first çalışır) |
| `DEBUG` | `True` | Detaylı log + görsel çıktı |
| `MAX_FRAMES` | `2250` | Yarışma karesi limiti |

### Model Ayarları

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `CONFIDENCE_THRESHOLD` | `0.20` | Minimum tespit güven eşiği |
| `NMS_IOU_THRESHOLD` | `0.35` | NMS IoU eşiği (çift tespit bastırma) |
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
| `SAHI_MERGE_IOU` | `0.35` | Birleştirme NMS IoU eşiği |

### Bbox Filtreleri

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `MIN_BBOX_SIZE` | `10` | Minimum bbox boyutu (px) — altındakiler elenir |
| `MAX_BBOX_SIZE` | `300` | Maksimum bbox boyutu (px) — bina/çatı filtreleme |

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

### Rider Suppression (Bisiklet/Motosiklet Sürücüsü)

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `RIDER_SUPPRESS_ENABLED` | `True` | Sürücü suppression kuralını aç/kapat |
| `RIDER_OVERLAP_THRESHOLD` | `0.35` | Person-overlap oranı eşiği |
| `RIDER_IOU_THRESHOLD` | `0.15` | IoU yedek eşiği |
| `RIDER_SOURCE_CLASSES` | `(1, 3, 10)` | Two-wheeler kaynak sınıf ID'leri (COCO/VisDrone) |

### Deterministiklik

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `DETERMINISM_SEED` | `42` | Tekrarlanabilirlik için global seed |
| `DETERMINISM_CPU_THREADS` | `1` | CPU thread sabitleme |

---

## 🎛️ Görev 3 Parametre Dosyası

Görev 3 (dinamik referans obje tespiti) için tüm kritik eşikler tek bir dosyada tanımlanır:

- Dosya: [`config/task3_params.yaml`](config/task3_params.yaml)
- Amaç: `T_confirm`, `T_fallback`, `N`, `grid stride` değerlerini merkezi ve denetlenebilir tutmak

### Parametreler

| Parametre | Dosya Anahtarı | Açıklama |
|-----------|----------------|----------|
| `T_confirm` | `t_confirm` | Stage-2 aday doğrulama minimum benzerlik eşiği |
| `T_fallback` | `t_fallback` | Stage-3 fallback sweep kabul eşiği |
| `N` | `n_fallback_interval` | Stage-3 fallback'in her kaç frame'de bir tetikleneceği |
| `grid stride` | `grid_stride` | Stage-3 grid/sliding-window tarama adımı (piksel) |

### Örnek İçerik

```yaml
t_confirm: 0.72
t_fallback: 0.66
n_fallback_interval: 5
grid_stride: 32
```

Not: Bu değerler çalışma sırasında dinamik değiştirilmemelidir; deterministik ve tekrarlanabilir karar için oturum başında sabitlenmelidir.

---

## 🔒 Deterministiklik Sözleşmesi

Sistem çıktılarının tekrarlanabilir olması için aşağıdaki kurallar zorunludur:

1. **Seed Sabitleme (numpy/torch/random):**
   - Tüm çalıştırmalarda aynı seed kullanılmalıdır.
   - Öneri: `numpy`, `torch`, `random` için tek noktadan seed ataması yapılmalı.

2. **Model Eval Mode:**
   - İnference öncesi tüm modeller `eval` modunda çalıştırılmalıdır.
   - Dropout ve BatchNorm gibi katmanların eğitim davranışı kapatılmalıdır.

3. **Sabit Sürüm Pinleme:**
   - `torch`, `torchvision`, `ultralytics`, CUDA ve cuDNN sürümleri pinlenmelidir.
   - Üretim ortamında sürüm kayması engellenmeli, aynı bağımlılık seti korunmalıdır.

4. **JSON Sırası ve Kararlı Serileştirme:**
   - Çıktı JSON'ları kararlı anahtar sırası ile üretilmelidir (`sort_keys=True` veya sabit alan sırası).
   - Sayısal formatlama ve alan sırası sürümler arasında değiştirilmemelidir.

5. **Frame-Index Tabanlı Karar Kuralları:**
   - Adaptasyonlar wall-clock süreye göre değil, frame index/pencere kuralına göre yapılmalıdır.
   - Bu yaklaşım farklı donanımlarda aynı karar davranışını korur.

6. **Runtime Profil Kullanımı:**
   - Yarışma için `--deterministic-profile balanced` önerilir.
   - `balanced`: seed + deterministic backend + TTA kapalı, FP16 açık kalır.

---

## 📂 Dosya Yapısı

```
HavaciliktaYZ/
├── main.py                  # Ana giriş noktası
├── requirements.txt         # Python bağımlılıkları
├── README.md               # Bu dosya
├── .gitignore              # Git hariç tutma kuralları
│
├── config/
│   ├── __init__.py
│   ├── settings.py         # Merkezi yapılandırma
│   └── task3_params.yaml   # Görev 3 eşik ve tarama parametreleri
│
├── src/
│   ├── __init__.py
│   ├── detection.py        # YOLOv8 nesne tespiti + iniş durumu
│   ├── movement.py        # Temporal karar mantığı + kamera kompanzasyonu (motion_status)
│   ├── data_loader.py     # Simülasyon veri yükleme (VID/DET)
│   ├── runtime_profile.py # Deterministik profil uygulaması
│   ├── network.py         # Sunucu iletişimi + retry + simülasyon
│   ├── localization.py    # GPS + optik akış pozisyon kestirimi
│   └── utils.py           # Logger, Visualizer, yardımcı araçlar
│
├── models/
│   └── yolov8m.pt          # YOLOv8 medium modeli (Git'e dahil değil)
│
├── sim_data/
│   └── test_frame.jpg      # Simülasyon test görseli
│
├── sartname/
│   └── teknofest_context.md # Yarışma şartname özeti
│
├── logs/                   # Çalışma zamanı logları (otomatik)
└── debug_output/           # Debug görselleri (otomatik)
```

---

## 📏 Yarışma Kuralları (Özet)

### Tespit Edilecek Nesneler

| Sınıf | ID | İniş Durumu | Açıklama |
|-------|----|-------------|----------|
| **Taşıt** | 0 | -1 | Otomobil, motosiklet, otobüs, tren, deniz taşıtı |
| **İnsan** | 1 | -1 | Ayakta/oturur tüm insanlar |
| **UAP** | 2 | 0 veya 1 | Uçan Araba Park alanı |
| **UAİ** | 3 | 0 veya 1 | Uçan Ambulans İniş alanı |

### İniş Uygunluk Kuralları

- **Uygun (1):** Alan tamamen kadraj içinde VE üzerinde hiçbir nesne yok
- **Uygun Değil (0):** Alan kısmen kadraj dışı VEYA üzerinde nesne var
- Bisiklet/motosiklet sürücüleri "insan" değil, taşıtla birlikte "taşıt" olarak etiketlenir
- Scooter için ayrı sınıf sinyali veri setinde yoksa two-wheeler suppression yaklaşımı kullanılır (yaklaşımsal kural).

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

### Teknik Kısıtlamalar

- 📡 İnternet bağlantısı **yasak** (offline çalışma zorunlu)
- 🎬 Oturum başına **2250 kare** (5 dk, 7.5 FPS)
- 📐 Çözünürlük: 1920×1080 veya 3840×2160
- 📊 Değerlendirme: mAP (IoU ≥ 0.5)

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
