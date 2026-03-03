# 1. Özet Bulgular

Bu denetim, yalnızca `sartname/sartname.md` ve kaynak kodun statik incelemesine dayanır. Çalıştırma, simülasyon veya deney yapılmamıştır. İnceleme sonucunda sistemin çekirdek yarışma döngüsünü korumaya dönük önemli emniyet mekanizmaları içerdiği görülmekle birlikte, puan kaybı ve oturum bütünlüğü açısından kritik riskler tespit edilmiştir.

Öne çıkan sonuçlar:
- Görev 1/2/3 için temel boru hattı mevcut; ancak düşürme (fallback), clipping ve kontrollü pasif mod kararları doğrudan skor kaybına dönüşebilecek biçimde tasarlanmıştır.
- “Her kareye bir sonuç” ilkesini zorlayabilecek oturum sonlandırma ve ardışık duplicate kesme kararları mevcuttur.
- JSON şema doğrulama ve kanonik alan isimleri güçlüdür; fakat alan tipi/format belirsizliği (taslak format) karşısında bazı koşullarda kırılganlık riski vardır.
- İnternet yasağına karşı uygulama içinde açık bir çıkış engelleme politikası yoktur.

# 2. Şartname Uyumluluk Analizi

## 2.1 Görev 1 (Nesne tespiti + landing_status + motion_status)
- Dört sınıfa (0/1/2/3) eşleme ve tespit akışı bulunmaktadır. Dosya/Fonksiyon: `src/detection.py` / `ObjectDetector.detect`, `_map_model_class_to_teknofest`, `_determine_landing_status`.
- `landing_status` üretimi UAP/UAİ için kenar teması, engel örtüşmesi, perspektif-proximity kontrolü ile ele alınmıştır. Dosya/Fonksiyon: `src/detection.py` / `_determine_landing_status`.
- `motion_status` taşıtlar için ayrı modülde üretilmektedir. Dosya/Fonksiyon: `src/movement.py` / `MovementEstimator.annotate`.
- Duplike bbox riskini azaltmak için sınıf bazlı NMS ve containment suppression uygulanmaktadır. Dosya/Fonksiyon: `src/detection.py` / `_merge_detections_nms`, `_suppress_contained`.

## 2.2 Görev 2 (GPS kestirimi)
- `gps_health == 1` iken sunucu çevirileri kullanılmakta, `gps_health == 0` iken optik akış/predict-only hattına geçilmektedir. Dosya/Fonksiyon: `src/localization.py` / `VisualOdometry.update`, `_update_from_gps`, `_update_from_optical_flow`, `predict_without_measurement`.
- Ortalama 3D hata metriği açısından drift sınırlamaya yönelik EMA/cap yaklaşımı var; ancak bunun gerçek hata etkisi çalışma yapılmadan doğrulanamaz.
- Frame alınamazsa son bilinen/predicted pozisyon gönderimi ile döngü devam ettirilmektedir. Dosya/Fonksiyon: `main.py` / `_fetch_competition_step`.

## 2.3 Görev 3 (Referans obje eşleştirme)
- Referansları yükleme, ID doğrulama, duplicate quarantine ve ORB/SIFT tabanlı eşleştirme mevcut. Dosya/Fonksiyon: `src/image_matcher.py` / `load_references`, `match`; `main.py` / `_validate_task3_references`.
- Sunucu referanslarında duplicate oranı kritik eşiği aşınca Task3 kontrollü pasif moda alınmaktadır. Dosya/Fonksiyon: `main.py` / `run_competition`.

## 2.4 Rekabet döngüsü ve sonuç gönderimi
- Bekleyen sonuç gönderilmeden yeni frame işleme akışına geçilmemesi için `pending_result` kapısı kullanılmıştır. Dosya/Fonksiyon: `main.py` / `run_competition`.
- Tekrarlı gönderim ve kalıcı red senaryoları için durum makinesi vardır. Dosya/Fonksiyon: `src/network.py` / `send_result`; `src/send_state.py` / `apply_send_result_status`.
- Şema self-check ve preflight payload doğrulama bulunur. Dosya/Fonksiyon: `src/network.py` / `assert_contract_ready`, `_preflight_validate_and_normalize_payload`; `src/payload_schema.py`.

# 3. Kritik Riskler

## Bulgu K1
- Risk Tanımı: Görüntü indirilemediğinde sistemin boş `detected_objects` ve boş `detected_undefined_objects` ile fallback sonuç göndermesi, ilgili karelerde Görev 1 ve Görev 3 için sistematik false negative üretir.
- Şartname İhlali / Puan Etkisi: “Her kareye sonuç” korunurken, mAP tarafında recall düşüşü kaynaklı doğrudan puan kaybı oluşturur (Görev 1 ve 3).
- Olasılık: Yüksek
- Tetikleyici Senaryo: Ağ gecikmesi/zaman aşımı/bozuk görsel nedeniyle `download_image` başarısız olduğunda fallback yoluna girilmesi.
- Tespit Yöntemi: `main.py` içinde `_fetch_competition_step` fallback dalında `detected_objects: []` ve `detected_undefined_objects: []` atanması; `src/network.py` içinde fallback payload’da da nesnelerin boşaltılması.
- Mimari Düzeyde İyileştirme Yönü: Fallback modunda dahi bilgi korunumunu önceleyen bir “kademeli çıktı güvenliği” stratejisiyle minimum algı bilgisinin taşınması.
- İlgili Kod: `main.py` / `_fetch_competition_step`; `src/network.py` / `_build_safe_fallback_payload`.

## Bulgu K2
- Risk Tanımı: Payload clipping (global ve sınıf kotası) nedeniyle tespitlerin bir kısmı kesilerek gönderiliyor; yoğun sahnelerde özellikle küçük/kalabalık nesnelerde sistematik recall kaybı oluşabilir.
- Şartname İhlali / Puan Etkisi: Şartname mAP değerlendirmesinde “eksik bbox” doğrudan AP düşürür; bu mekanizma bazı karelerde kontrollü ama kalıcı puan kaybına neden olur.
- Olasılık: Yüksek
- Tetikleyici Senaryo: Bir karede tespit sayısının sınıf kotası veya global limite taşması.
- Tespit Yöntemi: `_apply_object_caps` fonksiyonunun düşen tespitleri loglayıp yalnızca üst sıralı alt kümeyi göndermesi.
- Mimari Düzeyde İyileştirme Yönü: Skor-optimizasyon yerine şartname skor metriklerini hedefleyen adaptif gönderim politikası ve sınıf-adil bütçeleme.
- İlgili Kod: `src/network.py` / `_apply_object_caps`, `_preflight_validate_and_normalize_payload`.

## Bulgu K3
- Risk Tanımı: Ardışık duplicate frame sayısı eşik aşınca oturum döngüsü bilinçli olarak sonlandırılıyor.
- Şartname İhlali / Puan Etkisi: Erken çıkış halinde oturumdaki kalan karelere sonuç gitmemesi nedeniyle doğrudan kapsamlı puan kaybı (her görevde).
- Olasılık: Orta
- Tetikleyici Senaryo: Sunucudan ardışık duplicate metadata dönmesi veya frame ID bütünlüğü bozulması.
- Tespit Yöntemi: `consecutive_duplicate_frames` sayaç eşiği aşımında `break` ile loop sonlandırma.
- Mimari Düzeyde İyileştirme Yönü: Oturum sürekliliğini önceleyen, kesmeden ayrıştıran ve güvenli ilerleme sağlayan duplicate-yönetim politikası.
- İlgili Kod: `main.py` / `run_competition` (duplicate abort bloğu).

## Bulgu K4
- Risk Tanımı: Görev 3 referanslarında duplicate kritikleştiğinde Task3 tamamen pasifleniyor; tüm akış boyunca `detected_undefined_objects` boş kalabiliyor.
- Şartname İhlali / Puan Etkisi: Referans obje eşleştirme metriğinde sistematik kaçırma (FN) ve ciddi mAP kaybı.
- Olasılık: Orta
- Tetikleyici Senaryo: Oturum başı referans listesinin duplicate/bozuk gelmesi ve kritik oran eşiğinin aşılması.
- Tespit Yöntemi: `_validate_task3_references` sonucu `duplicate_critical` olduğunda matcher devre dışı bırakılması.
- Mimari Düzeyde İyileştirme Yönü: Tam pasifleştirme yerine güvenilir alt-küme ile devam eden “kısmi görev devamlılığı” mimarisi.
- İlgili Kod: `main.py` / `_validate_task3_references`, `run_competition`.

## Bulgu K5
- Risk Tanımı: Uygulama içinde internet çıkışını engelleyen açık bir politika/koruma bulunmuyor; `BASE_URL` harici adrese yönlenebilir.
- Şartname İhlali / Puan Etkisi: İnternet yasağı (genel kurallar, HW-004/HW-005/HW-006) bağlamında operasyonel diskalifiye riski.
- Olasılık: Orta
- Tetikleyici Senaryo: Yanlış konfigürasyonla public IP/domain’e bağlantı denemesi.
- Tespit Yöntemi: Ağ erişimi `requests.Session()` üzerinden genel URL’ye açık; yalnızca yerel ağ zorlaması yok.
- Mimari Düzeyde İyileştirme Yönü: Ağ katmanında çevrimdışı/yerel ağ zorlamasını mimari güvenlik ilkesi haline getirme.
- İlgili Kod: `config/settings.py` / `BASE_URL`; `src/network.py` / `NetworkManager.__init__`, `start_session`, `get_frame`, `download_image`, `send_result`.

# 4. Orta ve Düşük Öncelikli Riskler

## Bulgu O1
- Risk Tanımı: `gps_health` bozuk/eksik geldiğinde zorla `0` kabul edilerek vision-only hatta düşülüyor; ilk 450 karede sağlık değeri 1 kesinliği ile çelişen veri gelirse sistem beklenmedik moda geçebilir.
- Şartname İhlali / Puan Etkisi: Görev 2’de gereksiz mode switch ile hata birikimi ve Öklid hata artışı riski.
- Olasılık: Orta
- Tetikleyici Senaryo: Sunucu metadata format sapması (`unknown`, `null`, `NaN` vb.).
- Tespit Yöntemi: `_validate_frame_data` içinde `gps_health` değerinin hatada `0`’a zorlanması.
- Mimari Düzeyde İyileştirme Yönü: Sağlık sinyali güvenilirliği için kaynak-doğrulama ve oturum-faz bağlamı kullanan karar katmanı.
- İlgili Kod: `src/network.py` / `_validate_frame_data`; `src/localization.py` / `update`.

## Bulgu O2
- Risk Tanımı: `detected_objects` alanında `landing_status` ve `motion_status` tiplerinin integer/string dönüşümü taslak formatla tam uyumsuzlaşabilir.
- Şartname İhlali / Puan Etkisi: Nihai JSON format değişikliğinde parser uyumsuzluğu riski; sonuçların geçersiz sayılma riski.
- Olasılık: Orta
- Tetikleyici Senaryo: Yarışma günü nihai API’nin tip katılığı göstermesi.
- Tespit Yöntemi: Şema normalizasyonunda alanların int’e çevrilmesi; örnek şartname gösteriminde string kullanımı.
- Mimari Düzeyde İyileştirme Yönü: JSON şemasını tip-güvenli ama format-evrimine dayanıklı sözleşme katmanı olarak sürdürme.
- İlgili Kod: `src/payload_schema.py` / `normalize_object`; `sartname/sartname.md` / 9.2 örnek JSON.
- Varsayım: Nihai API’nin tipleri katı şekilde doğruladığı varsayılmıştır.

## Bulgu O3
- Risk Tanımı: Taşıt hareketlilik kararı kısa geçmişte “0” eğilimli olabilir; ilk yakalanan taşıtlarda geçici yanlış `motion_status` riski var.
- Şartname İhlali / Puan Etkisi: Yanlış motion_status, Görev 1 AP düşüşüne doğrudan etki eder.
- Olasılık: Orta
- Tetikleyici Senaryo: Yeni beliren taşıt, kısa iz geçmişi, kamera hareketi/bulanıklık.
- Tespit Yöntemi: `MovementEstimator._status` içinde düşük history’de varsayılan sabit karar eğilimi.
- Mimari Düzeyde İyileştirme Yönü: Hareket kararını tek modüle bağımlı olmaktan çıkarıp zaman tutarlılığı ağırlıklı karar füzyonuna taşımak.
- İlgili Kod: `src/movement.py` / `annotate`, `_status`.

## Bulgu O4
- Risk Tanımı: Oturum yönetimi tek-oturum döngüsü olarak kurgulanmış; 4 oturumun uçtan uca operasyonel otomasyonu uygulama seviyesinde görünür değil.
- Şartname İhlali / Puan Etkisi: Manuel süreç hatalarında oturum hazırlık/yarışma geçişlerinde operasyonel risk oluşur.
- Olasılık: Düşük
- Tetikleyici Senaryo: Peş peşe oturumlarda insan müdahalesi ve konfigürasyon atlama.
- Tespit Yöntemi: Uygulama akışında çoklu oturum orkestrasyonu bulunmaması.
- Mimari Düzeyde İyileştirme Yönü: Oturum yaşam döngüsünü (hazırlık/yarışma/sonlandırma) açık bir orkestrasyon katmanına taşımak.
- İlgili Kod: `main.py` / `main`, `run_competition`.
- Varsayım: Organizasyon tarafı her oturumu ayrı süreç başlatımıyla yürütmüyorsa bu risk anlamlıdır.

# 5. Performans ve Kaynak Değerlendirmesi

- Sistem performansını ölçen FPS sayaçları mevcut, ortalama FPS raporlanıyor ve donanım profilleme destekleniyor.
- Ancak HW-007 (min 1 FPS) şartını garanti eden açık bir kabul/red kapısı yok; yalnızca gözlemsel loglama var.
- Detection tarafında SAHI + yüksek çözünürlük + çoklu ön işleme kombinasyonu ağır sahnelerde throughput düşüşü yaratabilir.
- Ağ tarafında timeout/backoff/circuit-breaker ile ayakta kalma hedeflenmiş; buna karşılık degrade/fallback kararları skor kaybı pahasına süreklilik sağlıyor.
- Görev 2 tarafında drift baskılama (EMA/cap) var; fakat oturum sonu 1800 kareye uzayan GPS=0 fazında kümülatif hata davranışı yalnızca statik okumayla güvence altına alınamaz.

# 6. Belirsizlikler ve Koşullu Riskler

- Varsayım: Yarışma günü nihai JSON formatı taslaktan anlamlı saparsa mevcut şema katmanı ek adaptasyon gerektirebilir.
- Varsayım: Sunucu duplicate frame veya bozuk metadata üretirse, mevcut abort/pasifleme kararları puan yerine güvenliğe öncelik vererek skor düşürür.
- Varsayım: Kullanılan modelin (ağırlık dosyası) sınıf alanı UAP/UAİ için yeterli temsil üretmiyorsa eşleme katmanı doğru olsa da tespit başarımı düşer.
- Varsayım: Yarışma altyapısı idempotency header’ını dikkate almıyorsa tekrar gönderimlerde yalnızca “ilk sonuç geçerli” kuralı nedeniyle efektif throughput etkilenebilir.

# 7. Genel Sağlık Skoru (0–10)

**6.4 / 10**

Gerekçe (özet):
- Artılar: Sözleşme doğrulama, hata sınıflandırma, circuit-breaker/degrade, idempotent gönderim ve görev bazlı modülerlik güçlü.
- Eksiler: Fallback/clip/pasifleme kararlarının mAP ve Öklid hata puanına doğrudan olumsuz etkisi yüksek; bazı koruma stratejileri oturum tamamlama ile skor optimizasyonu arasında sert trade-off yaratıyor; internet yasağına karşı kod seviyesinde zorlayıcı güvenlik sınırı görünmüyor.
