# TEKNOFEST 2026 Havacılıkta Yapay Zeka — Rekabet Sistemi Kod Denetim Raporu

---

## 1. Özet Bulgular

1. **Fallback payload FR-011 ihlali riski:** Görüntü indirme başarısız olduğunda (`use_fallback=True`), `odometry.get_position()` kullanılıyor; bu pozisyon önceki kareden alınır. GPS=0 iken önceki kare GPS=1 ise sunucu referansı gönderilmiş olur — şartname FR-011'e göre GPS=0 durumunda kendi algoritma çıktısı zorunludur.
2. **Aynı nesne için çoklu bbox:** SAHI_MERGE_IOU (0.15) düşük kalabilir; özellikle farklı slice'lardan gelen benzer tespitler NMS'ten geçebilir. Şartname: aynı nesne için birden fazla bbox false positive sayılır, AP düşer.
3. **Görev 3 referans yüklemesi:** Yarışma modunda referanslar yalnızca `Settings.TASK3_REFERENCE_DIR`'den yükleniyor; sunucunun `task3_references` döndürmesi `start_session()` içinde loglanıyor ancak `ImageMatcher` bu veriyi kullanmıyor. Oturum başında sunucudan referans gelirse eşleştirme yapılmıyor.
4. **TBD-001 JSON şema belirsizliği:** Payload'da `cls` integer, şartname taslağında string; `object_id` tipi ve `detected_undefined_objects` formatı TBD-001'e bağlı. Kesin şema farklıysa geçersiz cevap ve puan kaybı riski var.
5. **İlk 450 kare GPS garantisi:** Şartname ilk 450 karede GPS'i sağlıklı sayıyor. Mock server ve data_loader buna uyuyor; gerçek sunucu farklı davranırsa optik akış referansı yanlış kareden başlayabilir (hata birikimi).
6. **Degrade modunda frame atlama:** `DEGRADE_SEND_INTERVAL_FRAMES=3` ile her 3 karede bir ağır inference yapılıyor; diğer kareler fallback ile gönderiliyor. Bu karelerde detection/odometry çalışmıyor; şartname her kareye tam 1 sonuç ve tüm karelerin işlenmesini bekliyor.
7. **Duplicate frame işleme:** `is_duplicate=True` olsa bile frame işlenip gönderiliyor. Sunucu aynı frame_id'yi tekrar gönderirse sonsuz döngü riski; idempotency ile submit kabul edilir ama sıradaki kare alınamaz.
8. **İnternet erişimi:** `requirements.txt` içinde PyTorch için `--extra-index-url https://download.pytorch.org/whl/cu121` var; bu kurulum sırasında kullanılır. Çalışma zamanında internet kullanımı yok (model yerel, requests yalnızca yerel ağ sunucusuna).
9. **Kamera parametreleri:** `FOCAL_LENGTH_PX=800` varsayılan; TBD-010'a göre kamera parametreleri yayımlanacak. Bu değer yanlışsa optik akış metre dönüşümü hatalı olur, 3D Öklid hatası artar.
10. **Circuit breaker ve oturum süresi:** `CB_SESSION_MAX_TRANSIENT_SEC=120` saniye degrade/transient limiti var; `should_abort()` her zaman `None` döndürüyor. Ağ uzun süre bozuksa sistem durmadan bekleyebilir; 60 dakikalık oturum süresi içinde çok kare kaçabilir.

---

## 2. Şartname Uyumluluk Analizi

### 2.1 Görev 1 — Nesne Tespiti


| Gereksinim                          | Durum      | Notlar                                                                |
| ----------------------------------- | ---------- | --------------------------------------------------------------------- |
| Taşıt/İnsan/UAP/UAİ (0,1,2,3)       | Uyumlu     | detection.py sınıf eşlemesi, COCO/VisDrone dönüşümü                   |
| Taşıt için movement_status (0,1)    | Uyumlu     | movement.py centroid takibi + kamera kompanzasyonu                    |
| UAP/UAİ için landing_status (0,1)   | Uyumlu     | _determine_landing_status: kenar, engel, proximity kontrolü           |
| Aynı nesne çoklu bbox               | Kısmi risk | NMS/containment suppression var; SAHI_MERGE_IOU eşiği düşük kalabilir |
| Yanlış landing/movement → AP düşüşü | Uyumlu     | Mantık şartnameyle uyumlu                                             |


### 2.2 Görev 2 — Pozisyon Kestirimi


| Gereksinim                             | Durum  | Notlar                                                                                                                                                        |
| -------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPS=1: Kendi veya referans değer       | Uyumlu | localization.py sunucu verisini kullanıyor                                                                                                                    |
| GPS=0: Kendi algoritma çıktısı zorunlu | Risk   | Normal akışta optik akış kullanılıyor. Fallback (görüntü yok) durumunda `get_position()` önceki kareden pozisyon alıyor; GPS=0 iken bu FR-011 ihlali olabilir |
| X/Y/Z metre cinsinden                  | Uyumlu | `_pixel_to_meter`, `get_position`                                                                                                                             |
| İlk 450 kare GPS sağlıklı              | Uyumlu | Mock/data_loader bu kurala uyuyor; gerçek sunucuya bağlı                                                                                                      |


### 2.3 Görev 3 — Referans Obje Eşleme


| Gereksinim                        | Durum  | Notlar                                                                |
| --------------------------------- | ------ | --------------------------------------------------------------------- |
| Oturum başında referans bildirimi | Eksik  | Sunucudan `task3_references` alınıyor ama ImageMatcher'a aktarılmıyor |
| Video akışında eşleştirme         | Uyumlu | ORB/SIFT, homografi, benzerlik eşiği                                  |
| Obje akışta olmayabilir           | Uyumlu | match() boş liste dönebilir                                           |
| mAP (Görev 1 benzeri)             | TBD    | Şartname TBD-003'te detay verecek                                     |


### 2.4 Genel Döngü


| Gereksinim                         | Durum  | Notlar                                       |
| ---------------------------------- | ------ | -------------------------------------------- |
| Her kareye tam 1 sonuç             | Uyumlu | pending_result → submit → sonra fetch        |
| Sıradaki kare önceki sonuç sonrası | Uyumlu | submit_future tamamlanmadan fetch başlamıyor |
| Süre dışı sonuç geçersiz           | Uyumlu | Timeout/retry mantığı var                    |
| İnternet yasağı                    | Uyumlu | Çalışma zamanında sadece yerel ağ            |


---

## 3. Kritik Riskler

### R1: Fallback Payload — GPS=0'da FR-011 İhlali

- **Risk:** Görüntü indirme başarısız olunca (`main.py` satır 506–516) `odometry.get_position()` kullanılıyor. Bu, son işlenen karenin pozisyonudur. O kare GPS=1 ise sunucu referansı kullanılmış olur. GPS=0 olan karede görüntü yoksa, kendi algoritma çalışmıyor; FR-011'e göre bu zorunlu.
- **Şartname / Puan:** FR-011; Görev 2 %40 puan, 3D Öklid hatası.
- **Olasılık:** Orta (ağ kopması, timeout, sunucu hataları).
- **Tetikleyici:** GPS=0 olan karede image download başarısız.
- **Tespit:** `send_fallback_ok` KPI'sı, `gps_health=0` ve `use_fallback=True` log çakışması.
- **Mimari:** Fallback senaryosunda GPS=0 ise optik akış için mevcut referans kareden tahmini pozisyon üretme veya uygun bir varsayılan (örn. son OF çıktısı) kullanma; sunucu referansını doğrudan kullanmama.

---

### R2: Görev 3 Sunucu Referanslarının Kullanılmaması

- **Risk:** `network.start_session()` sunucudan `task3_references` alıyor (satır 105–108) ancak bu veri `ImageMatcher`'a geçirilmiyor. Yarışma oturumunda referanslar sunucudan gelirse `load_references_from_directory()` boş kalır, Görev 3 pasif olur.
- **Şartname / Puan:** FR-014, FR-015; Görev 3 %25.
- **Olasılık:** Yüksek (sunucu tasarımına bağlı).
- **Tetikleyici:** Oturum başında sunucu `task3_references` döndürüyor; yerel dizinde referans yok.
- **Tespit:** Log: "Sunucudan X Görev 3 referansı alındı" ama "Referans obje bulunamadı" veya 0 eşleşme.
- **Mimari:** Sunucudan gelen referans formatını parse edip `ImageMatcher.load_references()` ile yükleyecek entegrasyon eklenmeli.

---

### R3: SAHI Duplikasyonu — Aynı Nesne Çoklu Bbox

- **Risk:** SAHI ile full-frame + sliced inference sonuçları `SAHI_MERGE_IOU=0.15` ile birleştiriliyor. Parça örtüşmelerinden dolayı aynı nesneye yakın ama farklı bbox'lar kalabilir. Şartname: aynı nesne için birden fazla bbox false positive.
- **Şartname / Puan:** 10.1.1 Örnek 4; Görev 1 mAP.
- **Olasılık:** Orta.
- **Tetikleyici:** Slice sınırlarındaki nesneler, birden fazla parçada tespit.
- **Tespit:** Aynı sınıfta IoU>0.5 örtüşen bbox sayısı; mAP düşüşü.
- **Mimari:** Sınıf bazlı NMS IoU eşiğini artırma veya per-class IoU; containment suppression'ı güçlendirme.

---

### R4: Degrade Modunda Kare Atlama

- **Risk:** `DEGRADE_SEND_INTERVAL_FRAMES=3` ile degrade modunda her 3 karede bir gerçek inference, diğerlerinde fallback (boş detection, son bilinen pozisyon) gönderiliyor. Şartname her kare için tam sonuç bekliyor.
- **Şartname / Puan:** FR-008, FR-019; eksik/yanlış sonuçlar mAP ve Öklid hatasını etkiler.
- **Olasılık:** Orta (uzun ağ kesintilerinde).
- **Tetikleyici:** Circuit breaker DEGRADED, `degrade_frames` yüksek.
- **Tespit:** `degrade_frames` KPI; fallback gönderim oranı.
- **Mimari:** Degrade stratejisini şartnameyle uyumlu hale getirme: her karede mümkün olan en iyi sonucu üretme veya sunucu ile kare atlama politikasını netleştirme.

---

### R5: Duplicate Frame Sonsuz Döngü

- **Risk:** `is_duplicate=True` olsa bile frame işlenip gönderiliyor (`main.py` 468–469). Sunucu aynı `frame_id`'yi tekrar gönderirse, sürekli aynı frame işlenir; sıradaki kare hiç gelmez. Idempotency ile submit kabul edilir ama ilerleme olmaz.
- **Şartname / Puan:** FR-019; oturum süresinde kare kaybı, puan kaybı.
- **Olasılık:** Düşük (sunucu hatası).
- **Tetikleyici:** Sunucu aynı frame'i tekrar tekrar döndürüyor.
- **Tespit:** `frame_duplicate_drop` KPI artışı; aynı `frame_id` için tekrarlayan fetch.
- **Mimari:** Duplicate frame'lerde submit edip bir sonraki frame isteğini yapma; ardışık N duplicate'te uyarı/abort mekanizması.

---

## 4. Orta ve Düşük Öncelikli Riskler

### R6: detection.py — Duplicate _is_touching_edge Tanımı

- **Risk:** `_is_touching_edge` iki kez tanımlı (659 ve 710); ikincisi birincisini ezer. `_is_touching_edge_raw` (margin=5) ve `_is_touching_edge` (EDGE_MARGIN_RATIO) farklı. `getattr(self, "_is_touching_edge_raw", None)` ile raw kullanılıyor; 5px çözünürlükten bağımsız.
- **Etki:** 4K'da 5px çok dar; 1080p'de makul. Yanlış kenar tespiti → landing_status hataları.
- **Olasılık:** Düşük.
- **Tetikleyici:** UAP/UAİ bbox kenarlara yakın.
- **Tespit:** Birim testte farklı çözünürlüklerde kenar davranışı.
- **Mimari:** Tek, tutarlı kenar kontrolü; margin'i çözünürlüğe göre ölçeklendirme.

---

### R7: localization.py — GPS=0 İlk Kare

- **Risk:** Oturumun ilk karesi GPS=0 ise (şartnameye aykırı ama sunucu hatası olabilir) `_prev_gray` yok; `_update_from_optical_flow` atlanır, pozisyon (0,0,0) korunur. Şartname ilk 450 kareyi sağlıklı kabul ettiği için pratikte nadir.
- **Etki:** İlk karede yanlış pozisyon.
- **Olasılık:** Düşük.
- **Tetikleyici:** Sunucu ilk karede gps_health=0 gönderir.
- **Tespit:** "GPS mevcut değil ve henüz referans kare oluşmadı" logu.
- **Mimari:** İlk karede GPS=0 ise uyarı ve güvenli fallback (örn. 0,0,0) politika olarak netleştirilmeli.

---

### R8: network.py — Payload cls Veri Tipi

- **Risk:** `build_competition_payload` içinde `cls` integer (örn. 0, 1); şartname taslağında string ("0", "1"). TBD-001 kesin şemayı belirleyecek. Tip uyumsuzluğu red veya yanlış değerlendirmeye yol açabilir.
- **Etki:** 4xx reject, geçersiz sonuç.
- **Olasılık:** Orta (TBD-001 çıktısına bağlı).
- **Tetikleyici:** Kesin şema string bekliyor.
- **Tespit:** Sunucu 4xx, preflight reject.
- **Mimari:** Konfigüre edilebilir çıktı formatı (int/str); kesin şema gelince hızlı uyarlama.

---

### R9: requirements.txt — Kurulum Sırasında İnternet

- **Risk:** `--extra-index-url https://download.pytorch.org/whl/cu121` kurulumda kullanılıyor. Yarışma ortamında paketler önceden kurulmuş olmalı; kurulum sırasında internet istenmemeli (HW-004, HW-006).
- **Etki:** Kurulum aşamasında sorun.
- **Olasılık:** Düşük (genelde ön kurulum yapılır).
- **Tetikleyici:** Yarışma bilgisayarında ilk kez pip install.
- **Tespit:** Kurulum dokümantasyonu kontrolü.
- **Mimari:** Offline kurulum paketleri veya yerel PyTorch indirme talimatı.

---

## 5. Performans ve Kaynak Değerlendirmesi

- **Gerçek zamanlı döngü:** Fetch ve submit paralel (ThreadPoolExecutor, max_workers=2); sıralama korunuyor (submit bitmeden fetch yok). Kare başına latency: detection (YOLOv8+SAHI) + movement + image_matcher + odometry.
- **HW-007 (min 1 FPS):** 7.5 fps video için kare başına ~1.33 s sınır. SAHI (çoklu inference), CLAHE, optik akış ek yük getiriyor. GPU zorunlu; RTX 3060 için FP16 ve SAHI slice boyutu performans kritik.
- **Bellek:** YOLOv8, frame buffer, movement track geçmişi, OF önceki kare. `MOVEMENT_WINDOW_FRAMES=24`, `SEEN_FRAME_LRU_SIZE=512` sınırlı.
- **Circuit breaker:** Transient storm'da OPEN, cooldown sonrası DEGRADED. `should_abort()` iptal üretmiyor; uzun ağ kesintisinde süre boşa gidebilir.
- **Degrade modu:** `DEGRADE_SEND_INTERVAL_FRAMES=3` inference'ı azaltıyor; fallback gönderim oranı artıyor.

---

## 6. Belirsizlikler ve Koşullu Riskler

### TBD-001 JSON Şeması (Varsayım)

- Kesin alan adları, tipler ve yapı henüz net değil. Mevcut payload taslak şemaya göre; farklılık red veya puan kaybına yol açabilir. **Varsayım:** Şema büyük ölçüde taslakla uyumlu.

### TBD-003 Görev 3 Puanlama (Varsayım)

- mAP hesaplama detayı (IoU eşiği, sınıflandırma) bilinmiyor. **Varsayım:** object_id ve bbox formatı Görev 1'e benzer.

### TBD-010 Kamera Parametreleri (Varsayım)

- `FOCAL_LENGTH_PX=800` varsayılan. Gerçek değer farklıysa optik akış metre dönüşümü hatalı olur. **Varsayım:** Parametreler yarışma öncesi paylaşılacak ve config güncellenecek.

### Sunucu API Davranışı (Varsayım)

- `task3_references` formatı, `next_frame` yanıt yapısı, 204 end-of-stream garantisi tam dokümante edilmemiş. **Varsayım:** Mock server davranışı gerçek sunucuya yakın.

---

## 7. Genel Sağlık Skoru (0–10)

**Skor: 6.5/10**

**Gerekçe:**

1. Görev 1 mimarisi sağlam; landing_status, movement_status, NMS mevcut; SAHI duplikasyon riski var.
2. Görev 2 normal akışta FR-011'e uyuyor; fallback senaryosunda ihlal riski.
3. Görev 3 sunucu referansları entegre değil; yerel dizin varsa çalışıyor.
4. Döngü sıralaması doğru; her kareye 1 sonuç prensibi korunuyor.
5. Degrade modu şartnameyle tam uyumlu değil; kare atlama/fallback stratejisi gözden geçirilmeli.
6. TBD'ler (şema, kamera, puanlama) ek belirsizlik getiriyor.
7. Circuit breaker oturum iptal etmiyor; uzun kesintilerde zaman kaybı riski.
8. Config yapısı (settings.py, BASE_URL) yarışma günü uyarlamaya uygun.

**Yarışma performansına etkisi:** Kritik noktalar Fallback FR-011 (Görev 2 puanı), Görev 3 referans entegrasyonu ve degrade stratejisi. Bu alanlar iyileştirilirse skor 7.5–8'e çıkabilir.

---

## Dosya Referansları


| Dosya                | İlgili Alanlar                                       |
| -------------------- | ---------------------------------------------------- |
| main.py              | Döngü, fetch/submit, fallback mantığı                |
| src/network.py       | Payload, validation, GPS normalize, task3_references |
| src/detection.py     | Tespit, landing_status, NMS, SAHI                    |
| src/localization.py  | GPS/OF hibrit, X/Y/Z                                 |
| src/movement.py      | motion_status, kamera kompanzasyonu                  |
| src/image_matcher.py | Görev 3 eşleştirme                                   |
| src/resilience.py    | Circuit breaker, degrade                             |
| config/settings.py   | Parametreler, TBD override'lar                       |


