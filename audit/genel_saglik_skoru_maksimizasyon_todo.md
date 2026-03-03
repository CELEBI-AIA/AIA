# TEKNOFEST Genel Sağlık Skoru Maksimizasyon To-Do

Bu plan, aşağıdaki raporların kod üzerinden çapraz doğrulanmış ve conflict çözümü yapılmış tek birleşik sürümüdür ve aşağıdaki raporlar silinmiştir tek olarak şu anki mevcut dosya bulunmaktadır.:
- `audit/teknofest_audit_raporu_20260303_132740.md`
- `audit/teknofest_audit_raporu_20260303_133107.md`
- `audit/audit_report_20260303_103659.md`
- `audit/auditcodx.md`

- Audit skorları: `4.3/10`, `7.0/10`, `6.0/10`, `6.8/10`
- Conflict çözümü sonrası birleşik baz sağlık skoru: `6.0/10`
- Hedef sağlık skoru: `>=9.0/10`
- Çalışma modu politikası: Varsayılan her zaman `test`; `competition` yalnızca tüm checklist tamamlanıp Go/No-Go geçildikten sonra manuel açılır.

## 1. Önceliklendirilmiş Uygulama Planı

### Adım 0 - Baseline ve Çalışma Çerçevesi
- [x] Baseline runtime metriklerini kaydet: frame kayıp oranı, submit reject oranı, fallback oranı, ortalama FPS (`run_summary.baseline_metrics`).
- [ ] Dataset/GPU gerektiren baseline metriklerini kaydet: Görev 1 mAP@0.5, Görev 2 ort. 3D hata, Görev 3 AP.
- [x] Günlük geliştirme/doğrulama için tek komutlu `test` çalışma profili sabitle.
- [x] `competition` modunu varsayılandan çıkar; yalnızca final onayda manuel aç.
- [x] Koşu sonunda otomatik metrik raporu üret (JSON + özet).

Tamamlanma kriteri:
- [ ] En az 3 ardışık koşuda metrikler tutarlı ve karşılaştırılabilir.

### Adım 1 - Kare Kapanış Garantisi (KR-01)
- [x] `frame_state_machine` uygula: `FETCHED -> PROCESSED -> SUBMITTING -> ACKED`.
- [x] `ACK` gelmeden yeni frame fetch edilmesini kesin engelle.
- [x] `pending_ttl` ile frame drop davranışını kaldır.
- [x] `permanent_rejected` durumunda sessiz drop yerine kontrollü yeniden deneme + güvenli terminal kararı uygula.
- [x] İdempotent gönderimi koru ve frame kapanışını sadece `ACK` ile tamamla.

Tamamlanma kriteri:
- [ ] Frame drop: `%0`
- [x] `ack_before_next_fetch` tüm hata yollarında sağlanıyor.

### Adım 2 - Debug/Visualizer Çökmesini Sıfırla (KR-02)
- [x] `Visualizer.__init__` içinde `self._save_counter = 0` başlat.
- [x] Debug çizim/kayıt hatalarını ana rekabet döngüsünden izole et (`fail-open`).
- [x] Yarışma koşusunda debug I/O maliyetini sınırlı profile taşı.

Tamamlanma kriteri:
- [x] `DEBUG=True` testinde uzun koşuda exception kaynaklı döngü durması yok.

### Adım 3 - Duplicate Storm ve Döngü Kilitlenmesini Kapat (KR-03)
- [x] Duplicate eşik sonrası sonsuz `continue` yerine deterministik terminal eylem uygula.
- [x] Duplicate fırtınası için yeniden-senkronizasyon veya güvenli oturum kapanış modu ekle.
- [x] Duplicate kararlarını telemetri ile logla.

Tamamlanma kriteri:
- [x] Duplicate fırtınasında kilitlenme yok, kararlar deterministik.

### Adım 4 - Resilience Abort Politikasını Düzelt (KR-04)
- [x] `should_abort` için soft/hard limit ayrımı yap.
- [x] Aktif `pending_result` varken hard abort yapma; önce gönderim kapanışı dene.
- [x] Degrade modunda akış sürekliliği ve skor korumasını ayrı karar matrisiyle yönet.

Tamamlanma kriteri:
- [x] Transient ağ bozulmasında erken/kontrolsüz oturum kesilmesi yok.

### Adım 5 - Görev 2 3B Doğruluk Sertleştirmesi (KR-05 + KR-GPS)
- [x] `gps_health=0` döneminde Z sabitlemeyi kaldır; dinamik 3B durum modeli uygula.
- [x] `FOCAL_LENGTH_PX`, `CAMERA_CX`, `CAMERA_CY` için yarışma öncesi zorunlu kalibrasyon doğrulaması ekle.
- [x] `gps_health` için tek kaynaklı semantik uygula: `0/1/unknown` tri-state; `unknown` değeri doğrudan `0` veya `1`e zorlanmasın.
- [x] `NaN`/bozuk translation verisinde “origin’e sert sıfırlama” yerine güvenli geçiş kuralı uygula.
- [x] GPS geri geldiğinde yumuşak re-anchor ve drift sınırlama uygula.

Tamamlanma kriteri:
- [ ] GPS kesintili senaryolarda ort. 3D hata en az `%20` iyileşiyor.

### Adım 6 - Görev 1 Recall/mAP ve Motion Doğruluğu (KR-06 + KR-MOTION)
- [x] Class-agnostic NMS yerine class-aware/hibrit NMS uygula.
- [x] Boyut/aspect filtrelerini sınıf bazlı adaptif hale getir.
- [x] `MOVEMENT_MIN_HISTORY` öncesi aşırı “0” eğilimini azaltacak erken hareket sinyali ekle.
- [x] Yeni giren taşıtlarda histerezis/güven skoru ile motion kararı stabilize et.
- [ ] Parametre taraması ile mAP@0.5 odaklı tuning pipeline işlet.

Tamamlanma kriteri:
- [ ] Görev 1 mAP@0.5 artışı kalıcı (`>= %8`).
- [ ] Taşıt `motion_status` doğruluğu artışı kalıcı (`>= %10`).

### Adım 7 - Fallback/Degrade Kaynaklı Puan Kırılmasını Azalt (KR-07)
- [x] Degrade modunda boş tespit gönderimini son çareye indir.
- [x] Hafif inference profili ile asgari algı sürekliliği sağla.
- [x] Fallback yoğunluğu için eşik-temelli skor koruma politikası uygula.

Tamamlanma kriteri:
- [ ] Uzun transient dönemde toplu false-negative etkisi belirgin azalıyor.

### Adım 8 - Payload Sözleşmesi ve Tip Uyumu (KR-PAYLOAD)
- [x] `landing_status` ve `motion_status` için tip profili ekle (`string`/`int`), tek konfigürasyon noktasından yönet.
- [x] Nihai JSON’a uyum için versioned payload adapter katmanı tamamla.
- [x] `_clamp_bbox` için tek kaynak fonksiyon kullan; çift implementasyonu kaldır.
- [x] Payload preflight için tip/şema regresyon testleri ekle.

Tamamlanma kriteri:
- [ ] Şema değişimi testlerinde submit reject oranı hedef altında.

### Adım 9 - Görev 3 Eşleştirme Dayanıklılığı
- [x] Referans doğrulamayı tek merkezde birleştir (main + matcher tekrarını kaldır).
- [x] `TASK3_MAX_REFERENCES` aşımında sessiz drop yerine önceliklendirme + raporlama + batch stratejisi ekle.
- [x] `detected_undefined_objects` için kalite/confidence sinyali (veya explicit quality flag) ekle.
- [x] Termal/uydu benzeri domain farklarında fallback descriptor stratejisi tanımla.

Tamamlanma kriteri:
- [ ] Görev 3 AP düşüşü referans yoğun senaryoda anlamlı şekilde azalıyor.

### Adım 10 - Ağ Politikası ve Operasyonel Güvenlik
- [x] Yalnızca izinli yerel host/URL allowlist politikası uygula.
- [x] `BASE_URL` için yarışma modunda yanlış konfigürasyona karşı startup guard ekle.
- [x] `frame_key` çakışma riskini azaltmak için kimlikleme politikasını güçlendir (gerekirse birleşik anahtar stratejisi).

Tamamlanma kriteri:
- [x] Yanlış ağ hedefi ile yarışma koşusu başlatılamıyor.

### Adım 11 - Performans ve 1 FPS Güvencesi
- [x] `FPS < 1` durumunda otomatik koruma profiline geç.
- [x] SAHI/full-frame inference tutarlılığını hizala (`augment` ve eşik davranışı).
- [x] Uzun koşuda periyodik GPU bellek/telemetri bakımını ekle.
- [x] JSON loglama frekansını performans profiline göre dinamikleştir.

Tamamlanma kriteri:
- [ ] Uzun koşuda FPS sürdürülebilir ve minimum eşik altı süre sınır içinde.

### Adım 12 - Go/No-Go Kapısı
- [ ] Zorunlu checklist: frame drop=0, duplicate kilitlenme=0, erken abort=0, payload reject hedef altında, mAP/3D hata hedefleri sağlandı.
- [ ] 4 oturum prova (her biri 60 dk / nominal 2250 kare) tamamla.
- [ ] Son sürümü dondur: config hash, model sürümü, payload profil sürümü, metrik özeti.

Tamamlanma kriteri:
- [ ] Go kriterleri sağlanmadan `competition` moduna geçiş yok.

## 2. Hızlı Kazanımlar (İlk 24 Saat)
- [x] `Visualizer._save_counter` hatasını düzelt.
- [x] `pending_ttl` drop yolunu kapat.
- [x] `network._apply_object_caps` değerlerini `Settings.RESULT_CLASS_QUOTA` + `Settings.RESULT_MAX_OBJECTS` ile gerçek limite çek.
- [x] Varsayılan modu `test` sabitle.
- [x] Payload tip profili anahtarını ekle (string/int uyum anahtarı).
- [x] `gps_health` semantiğini tek yerde birleştir (unknown dahil).

Beklenen etki:
- [ ] Sağlık skoru kısa vadede `6.0 -> ~7.8/10`.

## 3. Hedef Metrikler
- [ ] Frame kaybı: `%0`
- [ ] Submit başarı oranı: `>= %99`
- [ ] Submit reject oranı: `<= %1`
- [ ] Duplicate kilitlenme vakası: `0`
- [ ] Görev 1 mAP@0.5: baseline üstüne `>= %8`
- [ ] Taşıt `motion_status` doğruluğu: baseline üstüne `>= %10`
- [ ] Görev 2 ort. 3D hata: baseline’a göre `>= %20` düşüş
- [ ] Payload nesne üst limiti ihlali: `0`
- [ ] Uzun koşu stabilitesi: `60 dk` kesintisiz, kural ihlalsiz

## 4. Tek Satır Plan
1. Döngü güvenliği ve frame kapanış garantisi
2. Görev 2 (3B) ve Görev 1 (mAP/motion) skor üretimi
3. Payload sözleşmesi, Görev 3 dayanıklılığı ve operasyonel sertleştirme

## 5. Nihai Conflict Çözüm Kararları (Kod Doğrulamalı)
- [x] `network` preflight/idempotency gucludur; `pending_ttl` frame-drop yolu kaldirildi ve KR-01 guclendirildi.
- [x] Duplicate akışı deterministik terminal aksiyona bağlandı; KR-03 kapatıldı.
- [x] Wall-clock abort politikasi soft/hard limite ayrildi; aktif pending varken hard-abort ertelenerek KR-04 uygulandi.
- [x] `gps_health` varsayılan semantiği modüller arasında tutarsızdır (`network` zorla `0`, `main` eksikte `1`); tek kaynaklı tri-state politika seçilir.
- [x] `Z` ekseni OF sabitlemesi kaldirildi; dinamik 3B guncelleme + yumusak re-anchor uygulandi.
- [x] Payload alan tipleri (int/string) nihai şema belirsizliğine açıktır; tek konfigürasyonlu tip profili seçilir.
- [x] `object cap` devre dışı bırakma (`100000`) yarışma riski üretir; `Settings` tabanlı gerçek limit politikası seçilir.

## 6. Sprint Backlog Donusumu
- [x] Kalan `[ ]` maddeler sprint backlog formatina cevrildi: `audit/sprint_backlog_kalan_maddeler.md`

## 7. 2026-03-03 Kod + Şartname Doğrulama Güncellemesi
- [x] Şartname akış kuralı (aynı kare ACK almadan yeni kare fetch edilmez) explicit frame state machine ile zorunlu hale getirildi.
- [x] Duplicate storm için deterministik terminal aksiyon ve telemetri olayı eklendi (`event=duplicate_storm_terminal_action`).
- [x] Degrade fetch davranışı tek karar matrisi fonksiyonuna alındı (`full_frame` / `fallback_only`).
- [x] Yarışma başlangıcında kamera kalibrasyon guard eklendi (`FOCAL_LENGTH_PX`, `CAMERA_CX`, `CAMERA_CY` doğrulaması).
- [x] Görev 1 için class-aware/hybrid NMS ve sınıf-bazlı adaptif bbox filtreleri etkinleştirildi.
- [x] Hareket tespitinde erken sinyal + histerezis eklendi (özellikle `MOVEMENT_MIN_HISTORY` öncesi).
- [x] Task3 referans yüklemede `TASK3_MAX_REFERENCES` aşımı için önceliklendirme + batch + raporlama eklendi.
- [x] `detected_undefined_objects` kalite sinyali desteği eklendi (konfigürasyonla kontrollü gönderim).
- [x] Task3 referans doğrulama policy'si tek noktaya alındı (`src/task3_reference_policy.py`) ve `main` + `image_matcher` aynı doğrulama akışını kullanır hale getirildi.
- [x] Domain farkları için çoklu descriptor fallback stratejisi eklendi (primary + domain fallback descriptor).
- [x] Degrade akışında detection replay (TTL + kapasite) eklendi; boş payload gönderimi son çareye indirildi.
- [x] Versioned payload adapter katmanı eklendi (`v1`, `v1_legacy`, `v2_int`) ve gönderim öncesi adaptasyon zorunlu hale getirildi.
- [x] Performans guard eklendi: düşük FPS koruma profili, dinamik JSON log interval, periyodik GPU bakım telemetrisi.
- [x] Doğrulama: `PYTHONPATH=.` ile `pytest -q` çalıştırıldı, `83 passed`.
