"""
TEKNOFEST Havacılıkta Yapay Zeka Yarışması - Merkezi Konfigürasyon Dosyası
===========================================================================
Tüm sistem parametreleri bu dosyada tanımlanır. Yarışma günü yalnızca
bu dosyadaki değerler güncellenerek sisteme adapte olunur.

Kullanım:
    from config.settings import Settings
    print(Settings.BASE_URL)
"""

from pathlib import Path
import os


# =============================================================================
#  PROJE KÖK DİZİNİ
# =============================================================================
# Bu dosyanın iki üst dizini = proje kökü (HavaciliktaYZ/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings:
    """
    Tüm sistem ayarlarını barındıran merkezi konfigürasyon sınıfı.

    Yarışma günü değiştirilmesi gereken parametreler en üstte,
    nadiren değişen parametreler altta gruplandırılmıştır.
    """

    # =========================================================================
    #  YARIŞMA GÜNÜ DEĞİŞECEK PARAMETRELER
    # =========================================================================

    # Sunucu Bağlantısı - Yarışma günü güncellenecek
    BASE_URL: str = "http://127.0.0.1:5000"

    # API Endpoint'leri
    ENDPOINT_NEXT_FRAME: str = "/next_frame"
    ENDPOINT_SUBMIT_RESULT: str = "/submit_result"

    # Takım Bilgileri - Yarışma günü güncellenecek
    TEAM_NAME: str = "Takim_ID"

    # Çalışma Modları
    SIMULATION_MODE: bool = True    # True: Yerel test, False: Yarışma
    DEBUG: bool = True              # True: Detaylı log + görsel çıktı

    # =========================================================================
    #  MODEL AYARLARI
    # =========================================================================

    # YOLOv8 Model Dosyası (yerel diskten yüklenir - OFFLINE MODE)
    # Custom trained model (YOLOv11m finetuned on UAI, UAP, car, human)
    MODEL_PATH: str = os.path.join(str(PROJECT_ROOT), "model", "best_mAP50-0.923_mAP50-95-0.766.pt")

    # Tespit Güven Eşiği (0.0 - 1.0)
    # 0.40 = Increased confidence threshold to prevent false positives like poles detected as humans or random background objects as vehicles
    CONFIDENCE_THRESHOLD: float = 0.40

    # NMS IoU Eşiği (Non-Maximum Suppression)
    # 0.25 = Very aggressive suppression to merge multi-part human/vehicle detections (e.g. legs + torso, or hood + bumper)
    NMS_IOU_THRESHOLD: float = 0.25

    # Cihaz Seçimi (cuda: GPU, cpu: İşlemci)
    DEVICE: str = "cuda"

    # FP16 Yarı Hassasiyet — RTX 3060'ta ~%40 hız artışı sağlar
    HALF_PRECISION: bool = True

    # Inference Çözünürlüğü (piksel)
    # 1280 = drone görüntüleri için en iyi (uzaktaki insanlar/araçlar)
    # Yarışma offline — gerçek zamanlı hız kısıtı yok, kalite öncelikli
    INFERENCE_SIZE: int = 1280

    # Sınıflar arası NMS — aynı bölgede farklı sınıf çakışmalarını da bastırır
    AGNOSTIC_NMS: bool = True

    # Maksimum tespit sayısı (SAHI ile daha fazla sonuç gelir)
    MAX_DETECTIONS: int = 300

    # Test-Time Augmentation (çoklu ölçekte inference → mAP artışı)
    # Deterministiklik için yarışma profilinde kapatılır.
    AUGMENTED_INFERENCE: bool = False

    # Ön-İşleme: CLAHE Kontrast İyileştirme (drone görüntülerinde
    # karanlık/düşük kontrastlı bölgelerdeki nesneleri ortaya çıkarır)
    CLAHE_ENABLED: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_TILE_SIZE: int = 8

    # Minimum bbox boyutu (piksel) — altındakiler false positive sayılır
    # Increased to 20px to effectively filter out tiny noise (like individual legs/feet or small poles)
    MIN_BBOX_SIZE: int = 20

    # Maksimum bbox boyutu (piksel) — şartname büyük nesneleri de tespit etmeyi zorunlu kılıyor
    # (otobüs, tren, gemi vb.) Bu nedenle pratikte devre dışı bırakıldı.
    MAX_BBOX_SIZE: int = 9999

    # =========================================================================
    #  SAHI (Slicing Aided Hyper Inference) — Tepeden Görünüm İyileştirmesi
    # =========================================================================
    # Görüntüyü örtüşen parçalara böler → her parçada ayrı inference → birleştir
    # Küçük nesneleri (tepeden araç/insan) dramatik şekilde iyileştirir
    SAHI_ENABLED: bool = True
    SAHI_SLICE_SIZE: int = 640       # Her parçanın boyutu (piksel)
    SAHI_OVERLAP_RATIO: float = 0.35 # Parçalar arası örtüşme (%35) - Kenarları yakalar
    SAHI_MERGE_IOU: float = 0.25     # Birleştirme NMS IoU eşiği (Çift tespitleri önler)

    # Model ısınma tekrar sayısı (ilk kare gecikmesini önler)
    WARMUP_ITERATIONS: int = 3

    # =========================================================================
    #  GÖREV 3 — REFERANS OBJE TESPİTİ (Image Matching)
    # =========================================================================
    TASK3_ENABLED: bool = True
    TASK3_REFERENCE_DIR: str = str(PROJECT_ROOT / "datasets" / "task3_references")
    TASK3_SIMILARITY_THRESHOLD: float = 0.72    # task3_params.yaml: t_confirm
    TASK3_FALLBACK_THRESHOLD: float = 0.66      # task3_params.yaml: t_fallback
    TASK3_FALLBACK_INTERVAL: int = 5            # Her N karede fallback sweep
    TASK3_GRID_STRIDE: int = 32                 # Sliding window adımı (piksel)
    TASK3_MAX_REFERENCES: int = 10              # Oturum başına maks referans obje
    TASK3_FEATURE_METHOD: str = "ORB"           # "ORB" veya "SIFT"

    # =========================================================================
    #  SINIF TANIMLARI (TEKNOFEST Şartname)
    # =========================================================================

    # TEKNOFEST Sınıf ID'leri
    CLASS_TASIT: int = 0       # Taşıt
    CLASS_INSAN: int = 1       # İnsan
    CLASS_UAP: int = 2         # Uçan Araba Park Alanı
    CLASS_UAI: int = 3         # Uçan Ambulans İniş Alanı

    # İniş Durumu Kodları
    LANDING_NOT_AREA: str = "-1"    # İniş alanı değil (Taşıt/İnsan)
    LANDING_NOT_SUITABLE: str = "0" # İniş için uygun değil
    LANDING_SUITABLE: str = "1"     # İniş için uygun

    # COCO → TEKNOFEST Sınıf Eşleştirme Tablosu
    # COCO Dataset Sınıf Numaraları:
    #   0=person, 1=bicycle, 2=car, 3=motorcycle, 4=airplane,
    #   5=bus, 6=train, 7=truck, 8=boat, 9=traffic light ...
    #
    # Şartname Kuralları:
    #   - Tüm motorlu karayolu taşıtları → Taşıt (0)
    #   - Raylı taşıtlar (tren, tramvay) → Taşıt (0)
    #   - Tüm deniz taşıtları → Taşıt (0)
    #   - Bisiklet/motosiklet sürücüsü → Taşıt (sürücüyle birlikte bütün)
    #   - Tüm insanlar → İnsan (1)
    COCO_TO_TEKNOFEST: dict = {
        0: 1,    # person → İnsan
        1: 0,    # bicycle → Taşıt (sürücüsüyle birlikte)
        2: 0,    # car → Taşıt
        3: 0,    # motorcycle → Taşıt
        5: 0,    # bus → Taşıt
        6: 0,    # train → Taşıt (vagonlar dahil)
        7: 0,    # truck → Taşıt
        8: 0,    # boat → Taşıt (deniz taşıtı)
    }

    # VisDrone → TEKNOFEST Sınıf Eşleştirme Tablosu
    # VisDrone sınıfları:
    #   0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
    #   6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others
    VISDRONE_TO_TEKNOFEST: dict = {
        1: 1,     # pedestrian → İnsan
        2: 1,     # people → İnsan
        3: 0,     # bicycle → Taşıt
        4: 0,     # car → Taşıt
        5: 0,     # van → Taşıt
        6: 0,     # truck → Taşıt
        7: 0,     # tricycle → Taşıt
        8: 0,     # awning-tricycle → Taşıt
        9: 0,     # bus → Taşıt
        10: 0,    # motor → Taşıt
    }

    # İniş alanı üzerine nesne kontrolü için kesişim eşiği
    # Şartname: "Herhangi bir nesne varsa iniş için uygun değildir"
    # Bu yüzden eşik 0.0 — herhangi bir kesişim = uygun değil
    LANDING_IOU_THRESHOLD: float = 0.0

    # Perspektif toleransı: iniş alanı bbox'ını genişleterek yakın nesneleri yakalar
    # Şartname 4.6: "Çekim açısına bağlı yanıltıcı durumda iniş uygun değildir"
    # 0.10 = bbox her yönde %10 genişletilir — 70-90° kamera açısı toleransı
    # 200px UAP'da ~20px ≈ gerçek dünyada ~0.5m buffer (sahada kalibrasyon önerilir)
    LANDING_PROXIMITY_MARGIN: float = 0.10

    # Kenar temas kontrolü margin oranı (çözünürlüğe göre ölçeklenir)
    # 1920px'de ~8px, 3840px'de ~15px — çözünürlük bağımsız davranış
    EDGE_MARGIN_RATIO: float = 0.004

    # Haritalanamayan model sınıflarını iniş engeli olarak değerlendir
    # Şartname 4.6: "tespit edilen veya tespit edilemeyen herhangi bir nesne"
    UNKNOWN_OBJECTS_AS_OBSTACLES: bool = True

    # UAP/UAİ sınıfları için daha düşük containment suppression eşiği
    # SAHI duplikasyonlarını temizler (tipik %80+ örtüşme) ama
    # yan yana gerçek alanları korur (standart: 0.85)
    LANDING_ZONE_CONTAINMENT_IOU: float = 0.70

    # =========================================================================
    #  KAMERA PARAMETRELERİ (Yarışma günü kalibrasyon ile güncellenecek)
    # =========================================================================

    # Odak Uzaklığı (piksel cinsinden) - Kamera modeline göre değişir
    FOCAL_LENGTH_PX: float = 800.0

    # Görüntü Merkez Noktası (piksel)
    CAMERA_CX: float = 960.0   # 1920 / 2
    CAMERA_CY: float = 540.0   # 1080 / 2

    # Varsayılan İrtifa (metre) - Optik akış hesabında fallback
    DEFAULT_ALTITUDE: float = 50.0

    # =========================================================================
    #  AĞ AYARLARI
    # =========================================================================

    # HTTP İstek Timeout (saniye)
    REQUEST_TIMEOUT: int = 5
    # Yeni timeout ayrıştırması için fallback taban değer (geriye uyumluluk)
    REQUEST_CONNECT_TIMEOUT_SEC: float = 1.5
    REQUEST_READ_TIMEOUT_SEC_FRAME_META: float = 2.5
    REQUEST_READ_TIMEOUT_SEC_IMAGE: float = 4.0
    REQUEST_READ_TIMEOUT_SEC_SUBMIT: float = 3.5

    # Bağlantı Hatası Retry Sayısı
    MAX_RETRIES: int = 3

    # Retry Arası Bekleme (saniye)
    RETRY_DELAY: float = 1.0
    BACKOFF_BASE_SEC: float = 0.4
    BACKOFF_MAX_SEC: float = 5.0
    BACKOFF_JITTER_RATIO: float = 0.25
    SEEN_FRAME_LRU_SIZE: int = 512
    IDEMPOTENCY_KEY_PREFIX: str = "aia"

    # Circuit breaker transient pencere süresi (saniye)
    CB_TRANSIENT_WINDOW_SEC: float = 30.0

    # Aynı pencere içindeki transient olay limiti (open tetikleme)
    CB_TRANSIENT_MAX_EVENTS: int = 12

    # Breaker OPEN bekleme (cooldown) süresi (saniye)
    CB_OPEN_COOLDOWN_SEC: float = 8.0

    # Oturumda izin verilen maksimum breaker OPEN çevrimi
    CB_MAX_OPEN_CYCLES: int = 6

    # Oturum genelinde transient/degrade toplam duvar saati limiti (saniye)
    CB_SESSION_MAX_TRANSIENT_SEC: float = 120.0

    # Degrade modunda fetch-only yaklaşımı (ağ toparlanana kadar ağır inference azaltılır)
    DEGRADE_FETCH_ONLY_ENABLED: bool = True

    # Degrade modunda ağır inference deneme aralığı (her N karede bir)
    DEGRADE_SEND_INTERVAL_FRAMES: int = 3

    # =========================================================================
    #  DOSYA YOLLARI
    # =========================================================================

    # Log Dizini
    LOG_DIR: str = os.path.join(str(PROJECT_ROOT), "logs")

    # Geçici Kare Dosyası
    TEMP_FRAME_PATH: str = os.path.join(str(PROJECT_ROOT), "temp_frame.jpg")

    # Debug Çıktı Dizini
    DEBUG_OUTPUT_DIR: str = os.path.join(str(PROJECT_ROOT), "debug_output")

    # Veri Seti Dizini
    DATASETS_DIR: str = os.path.join(str(PROJECT_ROOT), "datasets")

    # Simülasyon DET modu: rastgele seçilecek fotoğraf sayısı
    SIMULATION_DET_SAMPLE_SIZE: int = 100

    # =========================================================================
    #  PERFORMANS AYARLARI
    # =========================================================================

    # FPS Raporlama Aralığı (her N karede bir)
    FPS_REPORT_INTERVAL: int = 10
    # Yarışma modunda kısa KPI sonuç satırı log aralığı (her N karede bir)
    COMPETITION_RESULT_LOG_INTERVAL: int = 10

    # Ana Döngü Bekleme Süresi (saniye) - 0 = maksimum hız
    LOOP_DELAY: float = 0.0

    # GPU Bellek Temizleme Aralığı (her N karede bir)
    GPU_CLEANUP_INTERVAL: int = 200

    # Debug görsel kaydetme aralığı (her N karede diske yaz)
    DEBUG_SAVE_INTERVAL: int = 50

    # JSON log performans ayarları
    ENABLE_JSON_LOGGING: bool = True
    JSON_LOG_EVERY_N_FRAMES: int = 10
    LOG_MAX_FILES: int = 2000

    # Deterministik çalışma profili (startup'ta override edilebilir)
    DETERMINISM_SEED: int = 42
    DETERMINISM_CPU_THREADS: int = 1

    # Taşıt hareketlilik kestirimi (movement_status) parametreleri
    MOVEMENT_WINDOW_FRAMES: int = 24
    MOVEMENT_MIN_HISTORY: int = 6
    MOVEMENT_THRESHOLD_PX: float = 12.0
    MOVEMENT_MATCH_DISTANCE_PX: float = 80.0
    MOVEMENT_MAX_MISSED_FRAMES: int = 8

    # Hareket eşiği referans çözünürlüğü — bu genişlikte MOVEMENT_THRESHOLD_PX geçerli
    # 3840px frame'de otomatik 2× ölçeklenir, 1080p'de 1× kalır
    MOVEMENT_THRESHOLD_REF_WIDTH: int = 1920

    # Frozen frame tespiti — ardışık kareler arasındaki ortalama piksel farkı
    # (0-255 ölçeği) bu eşiğin altındaysa frame donmuş sayılır.
    # Piksel bazlı kontrol: kamera stabil ama araç hareketli → fark > 1.0 → NOT frozen
    # Gerçek donmuş/duplicate frame → fark ≈ 0 → frozen
    FROZEN_FRAME_DIFF_THRESHOLD: float = 1.0

    # Kamera hareket kompanzasyonu (motion_status)
    MOTION_COMP_ENABLED: bool = True
    MOTION_COMP_MIN_FEATURES: int = 40
    MOTION_COMP_MAX_CORNERS: int = 200
    MOTION_COMP_QUALITY_LEVEL: float = 0.01
    MOTION_COMP_MIN_DISTANCE: int = 20
    MOTION_COMP_WIN_SIZE: int = 21

    # Sürücü suppression (bisiklet/motosiklet üzerindeki insanı insan olarak sayma)
    # Şartname: Bisiklet/motosiklet sürücüsü → taşıt olarak etiketlenir
    # Scooter kuralı: COCO'da ayrı sınıf yok, bu yüzden rider suppression
    # ile yaklaşımsal olarak kapsanır. Veri setinde scooter ayrı sınıf
    # olarak gelirse aşağıdaki RIDER_SOURCE_CLASSES'a eklenmeli.
    RIDER_SUPPRESS_ENABLED: bool = True
    RIDER_OVERLAP_THRESHOLD: float = 0.35
    RIDER_IOU_THRESHOLD: float = 0.15
    # COCO: bicycle=1, motorcycle=3 | VisDrone: bicycle=3, motor=10
    RIDER_SOURCE_CLASSES: tuple = (1, 3, 10)

    # =========================================================================
    #  YARIŞMA LİMİTLERİ (Şartname)
    # =========================================================================

    # Oturum başına toplam kare sayısı (şartnameye göre 2250)
    MAX_FRAMES: int = 2250

    # Sonuç payload nesne limiti (deterministik cap)
    RESULT_MAX_OBJECTS: int = 100
    # Sınıf bazlı kota (cls: quota)
    RESULT_CLASS_QUOTA: dict = {
        "0": 40,
        "1": 40,
        "2": 10,
        "3": 10,
    }
