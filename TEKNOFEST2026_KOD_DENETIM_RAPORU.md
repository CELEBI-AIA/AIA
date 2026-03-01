# TEKNOFEST 2026 — Havacılıkta Yapay Zeka  
## Rekabet Sistemi Kod Denetim Raporu

**Denetim Tarihi:** 01.03.2026  
**Denetçi Rolü:** Kıdemli Yazılım Mimarı / Rekabet Sistemi Denetçisi  
**Kapsam:** Statik inceleme + mantıksal çıkarım (kod çalıştırılmamıştır)  
**Tek Doğruluk Kaynağı:** TEKNOFEST 2026 Birleşik Teknik Şartnamesi (sartname.md)

---

## 1. Özet Bulgular

Aşağıda, puana doğrudan etkisi olan ve/veya oturum stabilitesini tehdit eden bulgular öncelik sırasıyla listelenmektedir:

1. **`detection.py` — `_is_touching_edge` çift tanımlanmış; `_is_touching_edge_raw` her zaman devreye giriyor (5 px sabit margin, çözünürlükten bağımsız).** `_determine_landing_status` mantığı, EDGE_MARGIN_RATIO'yu hiçbir zaman uygulamıyor; 4K görüntülerde kenar temas tespiti yetersiz kalarak yanlış `landing_status = 1` üretebilir. Görev 1 mAP'ini doğrudan etkiler.

2. **`main.py` — Competition loop'ta CPU busy-wait (spin-lock).** `submit_future` tamamlanmadan ve `pending_result is not None` iken herhangi bir `time.sleep` çağrısı yok; ana thread %100 CPU kullanımıyla döner. Inference ve ağ thread'leri baskılanabilir, kare başı gecikme artar; termal throttle riski taşır.

3. **`config/settings.py` yüklenmemiş; kritik parametreler doğrulanamıyor.** `FOCAL_LENGTH_PX`, `EDGE_MARGIN_RATIO`, `DEFAULT_ALTITUDE`, `BASE_URL`, `TASK3_SIMILARITY_THRESHOLD` gibi değerler görülmeden risk analizi koşullu kalmaktadır.

4. **`localization.py` — `FOCAL_LENGTH_PX = 800.0` varsayılanı, TBD-010 kamera parametresi kesinleşmediğinde GPS=0 modunda piksel→metre dönüşümünü doğrudan hatalı yapacak.** Her GPS=0 karesinde sistematik Öklid hata birikimi; Görev 2 puanına doğrudan yansır.

5. **`network.py` — JSON payload'da alan adı `motion_status`; şartname JSON şeması TBD-001 kapsamında kesinleşmemiş.** Yarışma sunucusunun `movement_status` adını beklemesi halinde tüm taşıt hareket durumu verisi geçersiz sayılır ve taşıt Average Precision (AP) değeri sıfıra düşer.

6. **`task3_params.yaml` — Hiçbir Python dosyası tarafından okunmuyor.** `requirements.txt`'te `PyYAML>=6.0` var; YAML dosyası bakılmıyor. Görev 3 parametreleri `config/settings.py`'daki statik değerlerden alınıyor; YAML ile senkronizasyon yok, konfigürasyon yanıltıcı.

7. **`detection.py` — Scooter + sürücü → "insan" etiketleme kuralı implement edilmemiş.** Şartname §3.1.2: "Scooter: sürücüsü olduğu zamanlarda 'insan' olarak etiketlenmelidir." `_suppress_rider_persons` yalnızca taşıt üzerindeki insanı siler; scooter+sürücü senaryosunu tersine işlemez.

8. **`main.py` — `_submit_competition_step` dönüş tipinde `str | tuple` belirsizliği; kırılgan `action_result` unpack.** `action_result == ("process", success_info)` varsayımıyla `_, success_info = action_result` yapılıyor. Beklenmedik bir değer durumunda `ValueError` → kare kaybı.

9. **`resilience.py` — `should_abort()` her zaman `None` döndürüyor; kalıcı ağ arızasında döngü durmaz.** Oturum boyunca `PERMANENT_REJECTED` veya sürekli zaman aşımı görülse bile döngü çalışmaya devam eder; 60 dakikalık oturumda gerçek anlamda geri kurtarma mekanizması yok.

10. **`localization.py` + `movement.py` — `FrameContext` yeniden dönüşüm optimizasyonu uygulanıyor ama `movement.py`, `frame_ctx` parametresi `np.ndarray` ise ayrıca `cv2.cvtColor` çağırıyor.** `FrameContext.gray` lazy property'si yarım kalmış; `movement.py` hem `isinstance(frame_ctx, np.ndarray)` hem de `.gray` yolunu destekliyor, bu tutarsızlık test kapsamı dışı kalıyor.

---

## 2. Şartname Uyumluluk Analizi

### 2.1 Görev 1 — Nesne Tespiti

| Gereksinim | Kod Durumu | Risk |
|---|---|---|
| FR-001: Taşıt(0), İnsan(1), UAP(2), UAİ(3) tespiti | `detection.py` YOLO tabanlı sınıf eşleme ile karşılanıyor | Orta — `_class_map_mode` dallarından `name_based` dalı yetersiz eşleme yapabilir |
| FR-004: UAP/UAİ iniş uygunluğu zorunlu | `_determine_landing_status` uygulanmış | **Yüksek** — `_is_touching_edge_raw` (5 px) kenar kontrolü çözünürlüğe göre ölçeklenmiyor (bkz. Kritik Risk #1) |
| FR-005: Taşıt hareket durumu — kamera hareketi ayırt edilmeli | `movement.py` optik akış + medyan kamera kayması uygulamış | Orta — `_cam_shift_hist` birikimi `_CAM_TOTAL_CAP` ile sınırlandırılmış ama uçuş boyunca uzun vadeli drift riski var |
| §3.1.2: Scooter+sürücü → "insan" | **Eksik** — `_suppress_rider_persons` yalnızca person→vehicle suppress yapıyor | Orta |
| §3.1.2: Tren vagonları ayrı ayrı etiketlenmeli | YOLO modeli bu ayrımı öğrenmişse karşılanır; kod seviyesinde ayrıca kontrol yok | Koşullu |
| FR-008: Kare atlanamaz, her kareye tam 1 sonuç | Loop yapısı sıralı gönderim zorluyor; fallback payload mekanizması var | Düşük — fallback `detected_objects: []` boş gönderiyor (AP kaybı ama kural ihlali yok) |

Metrik Notu: `motion_status` JSON alan adı TBD-001 belirsizliği nedeniyle yüksek risk taşıyor (bkz. Özet Bulgu #5). Şartname "hareket durumu" için kesin alan adını belirtmiyor; uyumsuzluk taşıt AP'sini tamamen geçersiz kılabilir.

---

### 2.2 Görev 2 — Pozisyon Tespiti

| Gereksinim | Kod Durumu | Risk |
|---|---|---|
| FR-011: GPS=0'da kendi algoritmasi gönderilmeli | `localization.py` GPS=0'da optik akış kullanıyor | Orta — `FOCAL_LENGTH_PX` doğru değilse sistematik hata (bkz. Özet Bulgu #4) |
| FR-012: GPS=1'de referans veya kendi değeri | Sistem GPS=1'de sunucu değerini doğrudan alıyor (FR-012'nin izin verdiği yol) | Düşük |
| §3.2.2: GPS=1 → ilk 450 kare kesinlikle sağlıklı | `_validate_frame_data` GPS değerlerini normalize ediyor; 450 kare boyunca GPS=1 beklentisi karşılanıyor | Düşük |
| Başlangıç: x₀=y₀=z₀=0 | `VisualOdometry.__init__` (0,0,0) ile başlıyor | Uyumlu |
| GPS=0 sonrası koordinat bütünlüğü | GPS→OF geçişinde son GPS pozisyonundan optik akış deltaları ekleniyor | Orta — EMA sıfırlama (`_ema_dx/dy = 0`) geçişte tutarlı, ancak `scale_ratio` Z tahmini deneysel |

Kritik Not: `_update_from_gps` her GPS=1 karesinde `_ema_dx = 0.0; _ema_dy = 0.0` yapıyor. GPS=1→0→1→0 döngülerinde EMA biriktirilen momentum her geçişte temizlendiğinden GPS=0 dönemlerinin başı optik akışta geçici duyarsızlığa neden olabilir.

---

### 2.3 Görev 3 — Görüntü Eşleme

| Gereksinim | Kod Durumu | Risk |
|---|---|---|
| FR-014: Oturum başında referans obje yüklenmeli | `load_references_from_directory()` uygulanmış | Orta — yarışmada referanslar sunucudan mı yoksa yerel dizinden mi gelecek belirsiz |
| FR-015: object_id + bounding box gönderilmeli | `image_matcher.match()` çıktısı `detected_undefined_objects` olarak payload'a ekleniyor | **Yüksek** — TBD-001: JSON alan adları kesinleşmemiş; `object_id`, `top_left_x` vb. uyumsuz olabilir |
| FR-016: Obje tüm akışta mevcut olmayabilir | `if not self.references: return []` ile ele alınmış | Uyumlu |
| task3_params.yaml kullanımı | **YAML okunmuyor** — parametre değerleri `Settings` üzerinden alınıyor | Orta — konfigürasyon karışıklığı |

ORB ile termal→RGB veya uydu→hava kamera geçişlerinde eşleştirme başarısı düşük olabilir; şartname §3.3 bu senaryoları açıkça sayıyor. Bu bir mimari risk olup doğrudan puanlama etkisi taşıyor.

---

### 2.4 Genel Döngü

| Gereksinim | Kod Durumu | Risk |
|---|---|---|
| Her kareye tam 1 sonuç | Sıralı fetch→process→submit döngüsü mevcut | Orta — thread pool yapısı CPU spin-wait riski taşıyor |
| İnternet erişimi yasak (HW-004) | Tüm HTTP istekleri `self.base_url` üzerinden; YOLO model `os.path.exists` ile diskten yükleniyor | Uyumlu |
| HW-009: Konfigürasyon değişkeni ile mod geçişi | `--mode competition/simulate_vid/simulate_det` argümanı var | Uyumlu |
| Süre dışı gönderimler geçersiz (FR-020) | Döngü gecikme biriktirirse sonuç süre dışına taşabilir | Orta — spin-wait + ağ zaman aşımları birleşince kritik olabilir |

---

## 3. Kritik Riskler

### Risk K-1: `_is_touching_edge_raw` — Kenar Temas Tespitinde Çözünürlükten Bağımsız Sabit Margin

**Risk Tanımı:**  
`detection.py` içinde `_determine_landing_status` metodu, UAP/UAİ alanlarının frame kenarına değip değmediğini kontrol etmek için `getattr(self, "_is_touching_edge_raw", None)` ile her zaman `_is_touching_edge_raw` metodunu çağırıyor. Bu metod yalnızca `margin = 5` piksel kullanıyor. `_is_touching_edge` metodunun iki tanımı da `EDGE_MARGIN_RATIO` tabanlı ölçeklenen margin hesaplamaktadır; ancak `_determine_landing_status`'taki `else` dalı asla çalışmıyor çünkü `_is_touching_edge_raw` her zaman mevcuttur (statik metod, `getattr` sonucu her zaman truthy).

**Şartname İhlali / Puan Etkisi:**  
§3.1.3: "İniş durumunun 'Uygun (1)' olabilmesi için UAP/UAİ alanının TAMAMI kare içinde bulunmalıdır." 4K (3840 px) görüntülerde `EDGE_MARGIN_RATIO × 3840 ≈ 15–20 px` olması beklenirken 5 px margin kullanılıyor. 6–19 px aralığındaki sınır temaslı iniş alanları "uygun" (`landing_status = 1`) olarak raporlanır; gerçekte "uygun değil" olduğundan false positive → AP değeri düşer.

**Olasılık:** Yüksek  
**Tetikleyici Senaryo:** Drone alçalma/yükselme sırasında UAP/UAİ alanı frame kenarına 6–19 px mesafede görünüyor.  
**Tespit Yöntemi:** Tespit edilen her `landing_status=1` nesnesinin kenar temas kontrolü için manuel log filtresi; 4K kayıtlarda beklenenden fazla "uygun" sayısı.  
**Mimari Düzeyde İyileştirme Yönü:** `_determine_landing_status` içindeki `getattr` koşulunu kaldırıp sadece `_is_touching_edge` (tek tanım, EDGE_MARGIN_RATIO ile) çağırmak. İki farklı `_is_touching_edge` tanımını birleştirmek; `_is_touching_edge_raw` metodunu kaldırmak veya yeniden adlandırmak.

---

### Risk K-2: Competition Loop — CPU Busy-Wait (Spin-Lock)

**Risk Tanımı:**  
`main.py`, `run_competition` fonksiyonunda `submit_future` tamamlanmadan ve `pending_result is not None` iken while döngüsünde hiçbir `time.sleep` veya `future.result(timeout=...)` çağrısı yok. `if submit_future is not None and submit_future.done()` False döndüğünde, `elif pending_result is None` dalı da geçilemiyor; döngü hemen bir sonraki iterasyona geçiyor. Bu durum ana thread'i %100 CPU kullanımında spin ettiriyor.

**Şartname İhlali / Puan Etkisi:**  
HW-007: Minimum 1 FPS işleme kapasitesi. CPU spin-wait nedeniyle GPU CUDA kernels'in scheduled edilmesi gecikebilir; inference latency artışı → kare başı süre artışı → FR-020 süre dışı sonuç gönderim riski. 60 dakikalık oturumda (2250 kare) ısı birikimi ve termal throttle olasılığı artar.

**Olasılık:** Yüksek  
**Tetikleyici Senaryo:** Ağ gecikme yüksekken veya yoğun GPU yükünde `submit_future` geç tamamlanıyor; her aradaki döngü iterasyonu CPU'yu boşu boşuna tüketiyor.  
**Tespit Yöntemi:** Yarışma oturumunda `htop`/`top` ile CPU kullanım grafiği; bir core'un sürekli %100 görünmesi.  
**Mimari Düzeyde İyileştirme Yönü:** `submit_future is not None and not submit_future.done()` durumunda kısa bir `time.sleep` veya `concurrent.futures.wait([submit_future], timeout=0.01)` çağrısı ile meşgul beklemeyi bloke beklemeye çevirmek.

---

### Risk K-3: `FOCAL_LENGTH_PX = 800.0` Varsayılanı ile Optik Akış Hata Birikimi

**Risk Tanımı:**  
`localization.py`, `VisualOdometry.__init__` içinde `Settings.FOCAL_LENGTH_PX == 800.0` için uyarı veriyor: "TBD-010 kamera parametreleri yayımlandığında config/settings.py güncellenmeli." GPS=0 modunda piksel→metre dönüşümü `dx_m = dx_px * altitude / focal` formulüyle yapılıyor. Gerçek odak uzunluğu 800'den farklıysa, her optik akış karesinde sistematik yüzde hatası birikecek.

**Şartname İhlali / Puan Etkisi:**  
§10.2: Görev 2 Ortalama 3D Öklid Hatası — GPS=0 olan her kare için yanlış focal length orantılı hata üretiyor. Örneğin gerçek focal 600 px ise, tüm X/Y çıktıları %33 şişirilmiş; 1800 GPS=0 karede kümülatif büyük Öklid hatası.

**Olasılık:** Orta (TBD-010 paylaşımından önce yarışmaya girilirse Yüksek)  
**Tetikleyici Senaryo:** Kamera parametreleri yarışma gününe kadar paylaşılmıyor veya settings.py güncellenmeden gidiliyor.  
**Tespit Yöntemi:** GPS=1 karelerde sunucu pozisyonu ile kendi çıktısı tutarlı görünüyor; GPS=0 karelerde kademeli kayma başlıyor.  
**Mimari Düzeyde İyileştirme Yönü:** TBD-010 verisini beklemek; parametreler geldiğinde settings.py'yi güncellemek; ayrıca `FOCAL_LENGTH_PX=800.0` iken çalışmayı engelleyen bir assertion veya uyarı ekleyerek istemeden yarışmaya girilmesini önlemek.

---

### Risk K-4: JSON `motion_status` Alan Adı — TBD-001 Uyumsuzluk Riski

**Risk Tanımı:**  
`network.py`, `build_competition_payload` ve `_preflight_validate_and_normalize_payload` içinde taşıt hareket durumu `"motion_status"` anahtarıyla JSON'a yazılıyor. Şartname §15 ve §3.1.2'de alan adları verilmiyor; TBD-001 kapsamında kesin JSON şeması ayrıca açıklanacak. Sartname.md §16 bu riskin farkında ve "şartname revizyonlarında kesin formatın paylaşılacağını" belirtiyor.

**Şartname İhlali / Puan Etkisi:**  
Yarışma sunucusu `"movement_status"` bekliyorsa tüm taşıtlarda hareket durumu değerlendirilemez; taşıt AP'si için hareket durumu yanlış değerlendirilir ve puan kaybı oluşur.

**Olasılık:** Orta  
**Tetikleyici Senaryo:** TBD-001 ile sunucunun beklediği alan adları `"motion_status"` değil.  
**Tespit Yöntemi:** Sunucu yanıtlarını `log_json_to_disk` ile takip etmek; taşıt AP değerinin hareket durumu olmayan sınıflara kıyasla düşük çıkması.  
**Mimari Düzeyde İyileştirme Yönü:** TBD-001 açıklandığında `build_competition_payload` ve `_preflight_validate_and_normalize_payload` içindeki alan adlarını güncellemek; alan adını settings.py'deki bir sabitten okuyarak tek noktadan yönetmek.

---

### Risk K-5: `task3_params.yaml` — Ölü Konfigürasyon

**Risk Tanımı:**  
`task3_params.yaml` (`t_confirm: 0.72`, `t_fallback: 0.66`, `n_fallback_interval: 5`, `grid_stride: 32`) dosyası hiçbir Python modülü tarafından import edilmiyor veya okunmuyor. `requirements.txt`'te `PyYAML>=6.0` bağımlılığı var; bu da YAML okumanın planlandığını ama implementasyonunun eksik kaldığını gösteriyor. `image_matcher.py` parametrelerini `Settings.TASK3_SIMILARITY_THRESHOLD`, `Settings.TASK3_FALLBACK_THRESHOLD`, `Settings.TASK3_FALLBACK_INTERVAL` üzerinden alıyor.

**Şartname İhlali / Puan Etkisi:**  
Doğrudan ihlal yok; ancak yarışma öncesinde YAML üzerinden eşik değerini ayarlamak isteyen ekip üyesi Settings.py'nin etkisiz olduğunu sanarak yanlış değerle yarışmaya girebilir. Görev 3 mAP'ini dolaylı etkiler.

**Olasılık:** Orta  
**Tetikleyici Senaryo:** Ekip üyesi `task3_params.yaml`'ı düzenliyor ama kod `Settings` değerlerini kullandığından değişiklik geçerli olmuyor.  
**Tespit Yöntemi:** Debug loglarında `similarity_threshold` değerinin YAML'daki değerle eşleşmemesi.  
**Mimari Düzeyde İyileştirme Yönü:** Ya YAML'ı `Settings` üzerine yüklemek (oturum başında `task3_params.yaml` parse edip `Settings.TASK3_*` değerlerini güncellemek) ya da YAML dosyasını kaldırıp settings.py'yi tek kaynak olarak tutmak.

---

## 4. Orta ve Düşük Öncelikli Riskler

### Risk O-1: Scooter + Sürücü Etiketleme Kuralı Eksik

**Risk Tanımı:** Şartname §3.1.2: "Scooter: sürücüsü olduğu zamanlarda 'insan' (ID:1) olarak etiketlenmelidir." `detection.py`, `_suppress_rider_persons` içinde tüm taşıt (cls=0) üzerindeki insanları bastırıyor. Scooter+sürücü kombinasyonunda ise bu insanın "insan" olarak raporlanması gerekiyor. Mevcut kod tam tersi davranır.

**Şartname İhlali:** FR-003 kısmen. **Olasılık:** Orta. **Puan Etkisi:** Orta — scooter sürücüsü hem insan hem taşıt olarak (veya hiçbiri olarak) yanlış sayılabilir.  
**Tespit Yöntemi:** Scooter içeren test karelerinde false negative / false positive sayısı.  
**Mimari Düzeyde İyileştirme Yönü:** Scooter sınıfı model tarafından ayrı öngörülüyorsa `_suppress_rider_persons` içine sürücü varlığını kontrol eden ve uygun taşıt türüne göre insan etiketi ekleyen bir dal eklemek.

---

### Risk O-2: `localization.py` — GPS=1→0→1→0 Geçişlerinde EMA Sürekli Sıfırlanıyor

**Risk Tanımı:** `_update_from_gps` her çağrıda `self._ema_dx = 0.0; self._ema_dy = 0.0` yapıyor. GPS=1 kareleri sürekli EMA'yı sıfırlıyor; GPS=0 moduna her girişte EMA sıfırdan başlıyor, yumuşatma birkaç kare içinde etkisiz kalıyor.

**Şartname İhlali:** Dolaylı — FR-011. **Olasılık:** Orta. **Puan Etkisi:** GPS=0 başlangıcında ani hareket fazla veya az tahmin edilebilir; birkaç kare için Öklid hatası yüksek çıkabilir.  
**Tespit Yöntemi:** GPS geçiş noktasında X/Y çıktısı anomalisi.  
**Mimari Düzeyde İyileştirme Yönü:** EMA sıfırlamayı yalnızca GPS→OF geçiş anında (zaten `_was_gps_healthy = False` geçiş bloğunda yapılıyor) yapmak; GPS=1 her karesinde sıfırlamayı kaldırmak.

---

### Risk O-3: `main.py` — `action_result` Dönüş Tipi Kırılganlığı

**Risk Tanımı:** `_submit_competition_step` üç farklı `action` değeri döndürüyor: `"break"`, `"continue"`, veya `("process", success_info)` tuple. `run_competition` içinde `_, success_info = action_result` ile tuple unpacking yapılıyor. Bağlamın dışına çıkan başka bir durum — örneğin `_submit_competition_step` içindeki bir hata değeri — `ValueError: not enough values to unpack` veya `TypeError` üretir. Bu hata `except Exception` tarafından yakalanarak 0.5 saniyelik uyku ile devam ediyor; `pending_result` takılı kalıyor ve kare kaybı oluşuyor.

**Olasılık:** Düşük (beklenen akışta oluşmaz; test kapsamı dışı edge case). **Puan Etkisi:** Orta (kare kaybolabilir).  
**Tespit Yöntemi:** Runtime loglarında `ValueError` veya `TypeError` izleri; `send_fail` KPI sayacında beklenmedik artış.  
**Mimari Düzeyde İyileştirme Yönü:** `action_result` için bir dataclass veya Enum tanımlamak; tip güvenli ayrıştırma yapmak.

---

### Risk O-4: `detection.py` — GPU OOM Sonrası Boş Tespit Listesi

**Risk Tanımı:** `detect` metodu `torch.cuda.OutOfMemoryError` yakaladıktan sonra boş liste döndürüyor. Bu kare için `detected_objects: []` gönderiyor. Şartname FR-008 kare atlamayı yasaklamıyor, ancak boş sonuç Görev 1 AP'sini düşürür.

**Şartname İhlali:** FR-008 ihlali yok; Görev 1 mAP etkisi var. **Olasılık:** Düşük (SAHI + 4K giriş + büyük model). **Puan Etkisi:** Orta.  
**Tetikleyici Senaryo:** SAHI aktif, 4K görüntü, tüm tile'ların batch inference yaptığı yoğun kare.  
**Tespit Yöntemi:** Log'da `GPU OOM hatası!`; sıfır nesne gönderilen karelerde ani düşüş.  
**Mimari Düzeyde İyileştirme Yönü:** OOM sonrası SAHI devre dışı bırakarak full-frame inference ile yeniden denemek; SAHI tile sayısını dinamik olarak azaltmak.

---

### Risk O-5: `image_matcher.py` — ORB/SIFT Yöntemi Şartnamenin Zorlu Çapraz-Modal Senaryolarında Yetersiz

**Risk Tanımı:** Şartname §3.3: Referans objeler termal kameradan, uydu görüntüsünden, farklı açıdan veya çeşitli görüntü işlemeden geçmiş olabilir. `ImageMatcher` ORB (varsayılan) veya SIFT feature descriptor kullanıyor. Bu metodların termal→RGB veya uydu→hava görüntü modalite geçişlerinde düşük eşleştirme performansı gösterdiği bilinmektedir.

**Şartname İhlali:** FR-014/FR-015. **Olasılık:** Yüksek (şartname bu senaryoları açıkça belirtiyor). **Puan Etkisi:** Yüksek — Görev 3 mAP.  
**Tespit Yöntemi:** Termal referans objeli test senaryolarında sıfır eşleşme.  
**Mimari Düzeyde İyileştirme Yönü:** Görünürlük-agnostik descriptor (SuperPoint, DISK) veya siamese network tabanlı eşleme yaklaşımı değerlendirmek.

---

### Risk O-6: `network.py` — `build_competition_payload` ile `_preflight_validate` Arasında `cls` Alan Tipi Tutarsızlığı

**Risk Tanımı:** `build_competition_payload`, `clean_objects` içinde `"cls": int(cls)` olarak int kaydediyor. `_preflight_validate_and_normalize_payload` ise `str(obj.get("cls", ""))` ile `{"0","1","2","3"}` setinde arıyor. `str(1) == "1"` True döndüğünden runtime hatası oluşmuyor; ancak son `clean_capped` içinde `"cls": obj["cls"]` ifadesi string değeri koruyor (preflight çıktısı). Payload'da `cls` tipi (int mi string mi) serbest kalıyor.

**Olasılık:** Düşük. **Puan Etkisi:** TBD-001 şemasına göre değişir. **Mimari Düzeyde İyileştirme Yönü:** Tek bir payload builder üzerinden çalışmak; `cls` tipini tüm akışta tutarlı (tek bir yerde int'e çevirmek, string→int→string dönüşümünü önlemek).

---

### Risk D-1: `mock_server.py` — Thread-Safe Değil (Test Etkisi)

**Risk Tanımı:** `MockServerHandler.current_index`, `results_received` ve `session_start` class-level değişkenler; `HTTPServer` concurrent request'larda aynı anda erişilebilir.

**Olasılık:** Düşük (test ortamında). **Puan Etkisi:** Yok (yarışma modunu etkilemez; test güvenilirliğini azaltır).  
**Mimari Düzeyde İyileştirme Yönü:** `threading.Lock` ile state koruma veya `http.server.ThreadingHTTPServer` yerine `HTTPServer` ile tek thread.

---

### Risk D-2: `data_loader.py` — Rastgele Sekans Seçimi Determinizm Kırar

**Risk Tanımı:** `_load_video_sequence` içinde `random.choice(sequence_dirs)` kullanılıyor. Her çalıştırmada farklı sekans seçiliyor; simülasyon sonuçları tekrarlanamıyor.

**Olasılık:** Yüksek. **Puan Etkisi:** Yok (simülasyon modu). **Yarışma Etkisi:** Regression testi güvenilirliğini azaltır.  
**Mimari Düzeyde İyileştirme Yönü:** Simülasyon modunda seed-ile deterministic seçim; varsa sekans adını CLI parametresiyle belirtme seçeneği eklemek.

---

### Risk D-3: `runtime_profile.py` — Settings Class Attribute'larını Doğrudan Değiştirme

**Risk Tanımı:** `apply_runtime_profile` içinde `Settings.AUGMENTED_INFERENCE = False` gibi class attribute'ları doğrudan değiştiriliyor. `Settings` singleton değilse ve birden fazla yerde import edilmişse tutarsız görünebilir.

**Olasılık:** Düşük. **Puan Etkisi:** Muhtemelen yok. **Tespit Yöntemi:** `Settings.AUGMENTED_INFERENCE`'ın beklenmedik True olduğu durumlar.

---

## 5. Performans ve Kaynak Değerlendirmesi

### 5.1 Kare Başı Latency Analizi

HW-007: Minimum 1 FPS. 60 dakika × 7,5 FPS = 2250 kare → kare başı maksimum budget `1/7.5 ≈ 133 ms`. Sistemin kare başı işlem adımları:

- **Görüntü indirme (ağ):** `REQUEST_READ_TIMEOUT_SEC_IMAGE` bağımlı. Retry'larda bütçe aşılabilir.
- **Ön işleme (CLAHE + Unsharp Mask):** CPU üzerinde. 4K için ~10–20 ms bekleniyor.
- **YOLO Inference:** GPU'da. Full-frame `INFERENCE_SIZE` bağımlı.
- **SAHI (aktifse):** Birden fazla tile × YOLO inference = `N_tile × inference_latency`. 4K'da tile sayısı çok artabilir; toplam süre 133 ms'yi aşabilir.
- **Optik Akış (`movement.py` + `localization.py`):** `goodFeaturesToTrack` + LK flow ≈ 5–20 ms.
- **Payload build + HTTP submit:** Ağa bağımlı.

**Risk:** SAHI aktif + 4K + uzak sunucu yanıt süresi birleşince kare başı budget aşılması kuvvetle muhtemel. CPU spin-wait (Risk K-2) bu riski büyütüyor.

### 5.2 GPU Bellek Kullanımı

SAHI aktifken `_sliced_inference` her tile için ayrı `model.predict` çağrısı yapıyor; `with torch.no_grad()` bloğu var ancak GPU intermediate tensors tile başına serbest bırakılabiliyor. OOM riski Risk O-4'te ele alındı.

### 5.3 CPU Kullanımı

`movement.py` ve `localization.py` optik akış için aynı gri görüntüyü ayrı ayrı hesaplıyor. `FrameContext.gray` lazy property ile paylaşılabiliyor; `movement.py` `frame_ctx.gray` yolunu kullanıyor (doğru). Ancak `localization.py` içinde `if isinstance(frame_ctx, np.ndarray): gray = cv2.cvtColor(...)` dalı da var; `FrameContext` nesnesi geçildiğinde `frame_ctx.gray` kullanılıyor. Bu ikili yol tutarlı ama bakımı güçleştirir.

---

## 6. Belirsizlikler ve Koşullu Riskler

Aşağıdaki bulgular görülemeyen dosyalar veya şartnamenin henüz kesinleşmemiş maddeleri nedeniyle koşullu belirsizlik taşımaktadır.

**Varsayım V-1:** `config/settings.py` incelemeye dahil değil. `EDGE_MARGIN_RATIO`, `FOCAL_LENGTH_PX`, `DEFAULT_ALTITUDE`, `TASK3_SIMILARITY_THRESHOLD`, `MAX_FRAMES`, `BASE_URL`, `TEAM_NAME`, timeout ve backoff parametreleri doğrulanamadı. Bu dosyadaki yanlış bir parametre Risk K-1, K-3 ve K-4'ü direkt tetikler.

**Varsayım V-2:** TBD-001 kesinleşmediğinden `motion_status` ve `detected_undefined_objects` içindeki `object_id`/`top_left_x` vb. alan adları şartname uyumunu teyit edemez. Sunucu şema geçersizliği durumunda her kare yanlış veya eksik değerlendirilebilir.

**Varsayım V-3:** TBD-010 kamera parametreleri paylaşılmadığından `FOCAL_LENGTH_PX=800.0` ile çalışılması halinde Görev 2 hata birikimi öngörülemiyor. Gerçek focal uzaklığı ne kadar sapıyorsa Görev 2 puanı o oranda etkileniyor.

**Varsayım V-4:** `network.py`, `start_session` içinde sunucu yanıtında `task3_references` alanı varsa log yazıyor ancak `image_matcher.load_references` için bu listeyi kullanmıyor. Görev 3 referans dağıtım mekanizması yarışma gününe kadar belirsiz.

**Varsayım V-5:** `_validate_frame_data` içinde `"nan"` string değeri `0.0`'a çevriliyor. Ancak sunucu `"NaN"` (büyük harf) gönderirse: `str("NaN").strip().lower() == "nan"` True → 0.0'a çevriliyor. Doğru. Ama `float("NaN")` Python'da `math.isnan` ile yakalanıyor — bu yol da var. Tutarlı.

**Varsayım V-6:** `_build_idempotency_key` her `send_result` çağrısında `uuid.uuid4().hex[:8]` ile bir `_run_uuid` oluşturuyor (lazy, ilk çağrıda bir kere). Bu idempotency key mekanizması sunucu tarafının bu header'ı tanıyıp tanımadığına bağlı olarak etkili veya etkisiz olacak.

---

## 7. Genel Sağlık Skoru

**Skor: 6.1 / 10**

### Gerekçe:

1. **Şartname uyumu eksikleri (−1.2):** `_is_touching_edge_raw` kenar tespiti (K-1), scooter kuralı eksikliği (O-1), `task3_params.yaml` ölü konfigürasyon (K-5) doğrudan şartname maddeleriyle çelişiyor.

2. **Kritik parametre belirsizliği (−0.8):** `FOCAL_LENGTH_PX=800.0` ve `config/settings.py`'nin erişilememesi; TBD-010 ve TBD-001 riskleri çözülmeden yarışmaya girişte Görev 2 ve olası alan adı uyumsuzlukları önemli puan kayıplarına yol açabilir.

3. **CPU spin-wait sorunu (−0.5):** Competition loop'taki busy-wait (K-2) sadece performans değil; termal throttle ve latency etkisiyle süre dışı gönderim riskine taşınıyor.

4. **Güçlü ağ katmanı (+0.8):** `network.py` kapsamlı retry, fallback payload, idempotency, payload preflight, circuit breaker entegrasyonu ile güvenilir bir altyapı sunuyor.

5. **İyi modüler tasarım (+0.6):** `FrameContext`, `MovementEstimator`, `VisualOdometry`, `ImageMatcher` temiz ayrıştırılmış; `resilience.py` circuit breaker pattern doğru uygulanmış.

6. **Test altyapısı mevcut (+0.4):** `test_all.py`, `conftest.py`, `mock_server.py` ile kapsamlı test yapısı var; ancak Görev 3 eşleştirme testleri ve landing_status senaryoları için birim test eksik görünüyor.

7. **Ölü kod ve konfigürasyon karmaşası (−0.4):** Çift `_is_touching_edge` tanımı, `task3_params.yaml` ölü dosyası, `motion_status` alan adı belirsizliği bakım zorluğu yaratıyor ve yarışma günü hızlı düzeltmeleri güçleştiriyor.

8. **Görev 3 modalite dayanıklılığı eksik (−0.4):** ORB/SIFT şartnamenin sayıkladığı termal→RGB ve uydu→hava senaryolarında zayıf; Görev 3'ten alınabilecek puan sınırlı kalacak.

---

*Bu rapor yalnızca statik inceleme ve mantıksal çıkarım yöntemiyle üretilmiştir. Kod çalıştırılmamış, simülasyon yapılmamıştır. Tüm risk değerlendirmeleri şartname maddeleri ile gözlemlenen kod davranışının karşılaştırılmasına dayanmaktadır.*
