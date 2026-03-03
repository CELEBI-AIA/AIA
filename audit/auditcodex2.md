# 1. Özet Bulgular

Bu denetim, yalnızca şartname metni (`/sartname/sartname.md`) ve kaynak kodun statik incelenmesi üzerinden yapılmıştır. Dinamik doğrulama yapılmamıştır.

Genel sonuç:
- Sistem mimarisi, üç görevi tek döngüde birleştiren ve “önce sonuç gönder, sonra yeni kare al” kuralını gözeten bir iskelet sunmaktadır.
- Ancak puan kaybına ve oturum stabilitesine etki edebilecek kritik seviyede davranışlar mevcuttur.
- Özellikle Görev 1’de sonuç kırpma/filtreleme yaklaşımı, Görev 2’de konfigürasyon-temelli çalışma modu riski ve Görev 3’ün kontrollü pasif moda düşebilmesi yarışma skorunu doğrudan etkileyebilir.

Kritik düzeyde öne çıkan başlıklar:
1. GPS=0 senaryosunda yanlış çalışma modu ile referans değer gönderimi riski (operasyonel risk).
2. Sonuç nesne kotası/kırpma nedeniyle mAP düşüş riski.
3. Görev 3’ün referans bütünlüğü sorununda tamamen pasifleştirilmesiyle recall çöküş riski.
4. Degrade modunda algı yürütülmeyen karelerde sistematik boş/eksik tespit gönderimi.

---

# 2. Şartname Uyumluluk Analizi

## 2.1 Görev 1 — Nesne Tespiti

### Bulgu T1-1
- **Dosya / Fonksiyon-Sınıf:** `src/detection.py` / `ObjectDetector.detect`, `_determine_landing_status`; `src/movement.py` / `MovementEstimator.annotate`; `src/payload_schema.py` / `CompetitionPayloadSchema.normalize_object`
- **Risk Tanımı:** Tespit edilen nesnelerde `landing_status` ve `motion_status` alanları zorunlu şemaya normalize edilmekte; taşıt dışı sınıflar için `motion_status=-1`, taşıt için hareket etiketi üretilmektedir. Kural desteği mevcut olsa da hareket/iniş kararları tek-kare semantik ve heuristiklere dayalı olduğundan yanlış durum etiketi riski devam eder.
- **Şartname İhlali / Puan Etkisi:** Yanlış `motion_status` veya `landing_status`, mAP/AP düşüşüne yol açar.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Kamera hareketi yüksek, perspektif yanılsaması güçlü, UAP/UAİ kenar temaslı veya kısmi görünürlük içeren kareler.
- **Tespit Yöntemi:** Statik akış incelemesi; iniş durumunun kenar/proximity/engel kesişimi ile belirlendiği, hareketin track penceresiyle çıkarıldığı görüldü.
- **Mimari Düzeyde İyileştirme Yönü:** Durum etiketlerini çoklu-kanıt (zamansal tutarlılık + sahne bağlamı) politikasıyla tek bir karar katmanında birleştiren doğrulama katmanı.

### Bulgu T1-2
- **Dosya / Fonksiyon-Sınıf:** `src/network.py` / `NetworkManager._apply_object_caps`
- **Risk Tanımı:** Sınıf başına kota ve global üst sınır nedeniyle tespitler yük öncesinde kırpılıyor.
- **Şartname İhlali / Puan Etkisi:** Aynı karede gerçek nesne sayısı kotayı aşarsa geri çağırma (recall) düşer; mAP azalır.
- **Olasılık:** Yüksek
- **Tetikleyici Senaryo:** Kalabalık şehir/deniz üstü sahnelerde çok sayıda taşıt/insan nesnesi.
- **Tespit Yöntemi:** Sonuç yükü hazırlanırken sınıf bazlı sıralama ve kesme uygulanması statik olarak izlendi.
- **Mimari Düzeyde İyileştirme Yönü:** Kotayı sabit kesme yerine yarışma metriklerine duyarlı, sahne yoğunluğu-adaptif çıktı bütçeleme yaklaşımı.

### Bulgu T1-3
- **Dosya / Fonksiyon-Sınıf:** `src/detection.py` / `_post_filter`, `_merge_detections_nms`, `_suppress_contained`
- **Risk Tanımı:** Min boyut, oran, kapsama baskılama ve NMS bir arada agresif filtreleme üretebilir.
- **Şartname İhlali / Puan Etkisi:** Eksik bbox -> false negative; fazla baskılama -> recall kaybı; düşük IoU -> AP düşüşü.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Küçük hedef, kısmi görünür hedef veya yakın konumlu çoklu nesneler.
- **Tespit Yöntemi:** Filtreleme zinciri ve eşik-temelli eleme adımlarının statik okunması.
- **Mimari Düzeyde İyileştirme Yönü:** Kararların tekil eşik yerine belirsizlik bandı ve sahne tipi profili ile yönetildiği katmanlı filtre politikası.

## 2.2 Görev 2 — GPS Kestirimi

### Bulgu T2-1
- **Dosya / Fonksiyon-Sınıf:** `config/settings.py` / `Settings.SIMULATION_MODE`; `main.py` / `parse_args`, `main`; `src/network.py` / `NetworkManager.__init__`
- **Risk Tanımı:** Varsayılan çalışma modu simülasyon. Operasyon sırasında mod yanlış bırakılırsa yarışma döngüsü yerine simülasyon akışı çalışabilir.
- **Şartname İhlali / Puan Etkisi:** Yarışma sunucusundan gerçek frame/health akışı alınmazsa tüm görev çıktıları geçersiz hale gelebilir.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Oturum başlangıcında komut satırı parametresi/konfigürasyon doğru set edilmez.
- **Tespit Yöntemi:** Başlangıç parametreleri ve varsayılan mod akışı statik olarak incelendi.
- **Mimari Düzeyde İyileştirme Yönü:** Yarışma-öncesi zorunlu çalışma modu doğrulama geçidi (session handshake bağlılığı).

### Bulgu T2-2
- **Dosya / Fonksiyon-Sınıf:** `src/localization.py` / `VisualOdometry.update`, `_update_from_optical_flow`, `predict_without_measurement`; `main.py` / `_fetch_competition_step`
- **Risk Tanımı:** GPS=0 ve görüntü/optik akış başarısızlığı durumunda son durumdan projeksiyonla devam ediliyor; bu, uzun kesitlerde drift birikimine açık.
- **Şartname İhlali / Puan Etkisi:** Ortalama 3B Öklid hatası kümülatif artabilir.
- **Olasılık:** Yüksek
- **Tetikleyici Senaryo:** Uzun süreli GPS arızası + düşük tekstür + ardışık frame indirme/takip kaybı.
- **Tespit Yöntemi:** Ölçümsüz tahmin akışı ve hız tabanlı ileri projeksiyon akışı statik izlendi.
- **Mimari Düzeyde İyileştirme Yönü:** Drift bütçesini yöneten, güven puanına dayalı durum makinesi ve yeniden senkronizasyon politikası.

### Bulgu T2-3
- **Dosya / Fonksiyon-Sınıf:** `src/network.py` / `_validate_frame_data`
- **Risk Tanımı:** `translation_*` alanlarında NaN/bozuk değerler sıfıra zorlanıyor.
- **Şartname İhlali / Puan Etkisi:** **Varsayım:** Sunucu tarafı NaN bilgisini “GPS sağlıksız” ayrımında semantik olarak kullanıyorsa, sıfırlama alt sistemlerde yanlış güven algısı üretebilir; hata metriğini dolaylı artırabilir.
- **Olasılık:** Düşük-Orta
- **Tetikleyici Senaryo:** Bozuk veri yoğunluğu veya format geçişlerinde alanların beklenmedik tipte gelmesi.
- **Tespit Yöntemi:** Veri temizleme adımlarının tip dönüşümü mantığı statik okundu.
- **Mimari Düzeyde İyileştirme Yönü:** Ham veri semantiğini koruyan ve “bilinmiyor/bozuk” durumunu ayrı taşıyan durum modeli.

## 2.3 Görev 3 — Referans Obje Eşleştirme

### Bulgu T3-1
- **Dosya / Fonksiyon-Sınıf:** `main.py` / `_validate_task3_references`, `run_competition`; `src/image_matcher.py` / `load_references`, `match`
- **Risk Tanımı:** Referans ID çakışma oranı kritik eşiği aşarsa Görev 3 tamamen pasif moda alınabiliyor.
- **Şartname İhlali / Puan Etkisi:** Tüm oturum segmentinde `detected_undefined_objects` boş kalabilir; Görev 3 mAP’inde ciddi recall kaybı.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Sunucu referans setinde duplicate/bozuk kayıt yoğunluğu.
- **Tespit Yöntemi:** Duplicate kritik eşik kontrolünden sonra matcher’ın devre dışı bırakıldığı kod yolu incelendi.
- **Mimari Düzeyde İyileştirme Yönü:** Tam kapatma yerine sınırlı ama güvenli devamı mümkün kılan çok-katmanlı bütünlük modları.

### Bulgu T3-2
- **Dosya / Fonksiyon-Sınıf:** `src/image_matcher.py` / `_match_reference`
- **Risk Tanımı:** Eşleşme, feature-benzerlik + homografi ile yürütülüyor; farklı modalite/kamera koşullarında yanlış eşleşme veya kaçırma riski yüksek.
- **Şartname İhlali / Puan Etkisi:** Yanlış ID veya eksik eşleşme Görev 3 mAP skorunu düşürür.
- **Olasılık:** Orta-Yüksek
- **Tetikleyici Senaryo:** Termal-RGB çapraz eşleme, yüksek açı farkı, düşük detaylı obje.
- **Tespit Yöntemi:** Eşleşme pipeline’ı (ratio test, homography, convexity) statik incelendi.
- **Mimari Düzeyde İyileştirme Yönü:** Tek yöntem yerine çoklu eşleşme kanıtlarını birleştiren kimlik tutarlılık katmanı.

## 2.4 Genel Rekabet Döngüsü ve Protokol

### Bulgu G-1
- **Dosya / Fonksiyon-Sınıf:** `main.py` / `run_competition` döngüsü, `_fetch_competition_step`, `_submit_competition_step`
- **Risk Tanımı:** Döngü tasarımı bir kare için sonucu ACK almadan yeni kare fetch etmemeye odaklı; bu yönüyle şartname sıralama kuralını destekliyor.
- **Şartname İhlali / Puan Etkisi:** İhlal görünmüyor; pozitif uyumluluk bulgusu.
- **Olasılık:** Düşük
- **Tetikleyici Senaryo:** Yüksek gecikme altında eşzamanlı thread kullanımında beklenmeyen yarış koşulları.
- **Tespit Yöntemi:** `pending_result` doluyken submit önceliği, `pending_result is None` iken fetch koşulu statik doğrulandı.
- **Mimari Düzeyde İyileştirme Yönü:** Durum makinesi geçişlerinin tek-kaynak doğrulama ile formel hale getirilmesi.

### Bulgu G-2
- **Dosya / Fonksiyon-Sınıf:** `src/network.py` / `send_result`, `_build_safe_fallback_payload`; `main.py` / `_fetch_competition_step`
- **Risk Tanımı:** Degrade/fallback akışında boş nesne listesiyle sonuç gönderimi sıklaşabilir.
- **Şartname İhlali / Puan Etkisi:** “Her kareye bir sonuç” korunur; ancak nesne tespit görevlerinde sistematik boş çıktı mAP’i düşürür.
- **Olasılık:** Yüksek
- **Tetikleyici Senaryo:** Bağlantı dalgalanması, image indirme başarısızlığı, circuit breaker degrade periyodu.
- **Tespit Yöntemi:** Fallback payload üretimi ve degrade slot mantığı statik izlendi.
- **Mimari Düzeyde İyileştirme Yönü:** Servis sürekliliği ile skor kaybı arasında denge kuran görev-bazlı degrade politikası.

### Bulgu G-3
- **Dosya / Fonksiyon-Sınıf:** `config/settings.py` / `BASE_URL`; `src/network.py` / `NetworkManager`
- **Risk Tanımı:** Kod interneti teknik olarak engelleyen bir “deny-by-default” ağ politikası içermiyor; sadece konfigürasyona güveniyor.
- **Şartname İhlali / Puan Etkisi:** **Varsayım:** Yanlış URL konfigürasyonu veya operatör hatasında internet yasağına uyumsuzluk riski oluşabilir.
- **Olasılık:** Düşük-Orta
- **Tetikleyici Senaryo:** Yarışma günü base URL’in yerel ağ dışına işaret etmesi.
- **Tespit Yöntemi:** Ağ erişim noktalarının merkezi konfigürasyona bağlı olduğu, ek erişim kısıtlayıcı katman bulunmadığı gözlendi.
- **Mimari Düzeyde İyileştirme Yönü:** Çalışma zamanında hedef ağ alanını doğrulayan güvenlik-kısıt katmanı.

## 2.5 JSON Uyum ve Gelecek Format Değişimi

### Bulgu J-1
- **Dosya / Fonksiyon-Sınıf:** `src/payload_schema.py` / `validate_top_level_payload`, `canonicalize_objects`; `src/network.py` / `build_competition_payload`, `_preflight_validate_and_normalize_payload`
- **Risk Tanımı:** Taslak şema alanları güçlü biçimde garanti ediliyor; ancak nihai format değişikliğinde bu sıkı doğrulama rejimi kırılgan olabilir.
- **Şartname İhlali / Puan Etkisi:** Format güncellemesi sonrası preflight reject/fallback oranı artarsa skor kaybı ve kalıcı 4xx reddi oluşabilir.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Yarışma öncesi nihai JSON’da alan adı/tip/değer alanı değişimi.
- **Tespit Yöntemi:** Zorunlu alan listesi ve canonical alan adı bağımlılığı statik incelendi.
- **Mimari Düzeyde İyileştirme Yönü:** Sürümlemeli sözleşme katmanı ve geriye uyumlu adaptör stratejisi.

---

# 3. Kritik Riskler

1. **Görev 2 operasyonel mod riski (SIMULATION_MODE varsayılanı)**
   - Yanlış mod ile yarışma akışına girilememe riski; tüm puanlama etkilenir.
2. **Görev 1 nesne kırpma/kota riski**
   - Yoğun sahnede geri çağırma düşüşü; mAP üzerinde doğrudan negatif etki.
3. **Görev 3’ün kritik duplicate durumunda pasifleştirilmesi**
   - Undefined object tespitinin kesilmesiyle Görev 3 skor çöküşü.
4. **Degrade/fallback boş çıktı yoğunluğu**
   - Döngü ayakta kalsa da görev puanları sistematik düşebilir.
5. **GPS=0 uzun kesitte ölçümsüz projeksiyon drifti**
   - Öklid hata metriğinde birikimli bozulma.

---

# 4. Orta ve Düşük Öncelikli Riskler

- Orta: Yanlış `motion_status` / `landing_status` etiketleme kaynaklı AP kayıpları.
- Orta: Feature-temelli Görev 3 eşleştirme yaklaşımının modalite farkında kararsız kalması.
- Orta: Nihai JSON formatı yayınlandığında sıkı şema doğrulamasının uyum kırılması.
- Düşük-Orta (**Varsayım**): Bozuk telemetri alanlarının sıfıra zorlanmasının semantik yan etkileri.
- Düşük-Orta (**Varsayım**): İnternet erişimi için uygulama içinde aktif engelleyici güvenlik katmanı olmaması.

---

# 5. Performans ve Kaynak Değerlendirmesi

- Sistem, thread tabanlı fetch/submit ayrımı, timeout sayaçları ve circuit breaker ile oturumu hayatta tutmaya odaklıdır.
- Bu dayanıklılık yaklaşımı, 1 FPS asgari gereksinimini korumaya yardımcı olacak şekilde tasarlanmıştır; ancak degrade modunda algılama azaltıldığında skor/fps dengesi skor aleyhine kayabilir.
- YOLO + SAHI + optik akış + feature matching kombinasyonu kaynak tüketimi açısından ağırdır; buna karşı degrade modları mevcuttur.
- Sonuç: Mimari “oturumu sürdürme” açısından güçlü, “skor sürekliliği” açısından koşullu risklidir.

---

# 6. Belirsizlikler ve Koşullu Riskler

1. **Nihai JSON formatı henüz taslak dışı yayınlanmadı**
   - Şema kırılması riski koşulludur.
2. **Görev 3 puanlama detayları TBD**
   - Duplicate/pasif mod etkisinin tam puan etkisi belirsizdir.
3. **Kamera kalibrasyon parametrelerinin yarışma sürümü**
   - Görsel odometri hata davranışı kalibrasyon doğruluğuna koşulludur.
4. **Sunucu yanıt politikaları (4xx/5xx, idempotency kabulü)**
   - Fallback/yeniden deneme akışının gerçek yarışma sunucusundaki etkisi koşulludur.

---

# 7. Genel Sağlık Skoru (0–10)

**6.8 / 10**

Gerekçe (özet):
- Artılar: Döngü disiplini, sözleşme doğrulama, dayanıklılık/circuit breaker, zorunlu alan üretimi.
- Eksiler: Görev puanını doğrudan etkileyen agresif fallback/kırpma davranışları, operasyonel mod riski, Görev 3 pasifleşme olasılığı, GPS=0 drift kırılganlığı.

