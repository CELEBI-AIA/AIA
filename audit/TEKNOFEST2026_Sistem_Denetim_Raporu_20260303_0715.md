1. Özet Bulgular
=====================================
TEKNOFEST 2026 Havacılıkta Yapay Zeka yarışması kod tabanı (versiyon 1.0 - 21.02.2026 şartnamesine göre) statik olarak incelenmiştir. Genel mimari (modüller arası ayrım, resilience mantığı, network katmanı) sağlam temellere oturtulmuş ve yarışmanın 4 oturumlu (2250 kare/oturum) offline yapısına uygun tasarlanmıştır.

Bununla birlikte, puan kaybına (Öklid hatasının birikmesi, false-positive IoU cezaları) yol açabilecek çeşitli algoritmik ve mimari riskler tespit edilmiştir. İniş alanı (UAP/UAİ) kontrolleri ve Görev 2 optik akış mekanizmasındaki bazı ölçeklendirme/drift durumları öne çıkmaktadır. İnternet yasağı, lokal çalışma zorunluluğu ve API JSON çıktı şemalarına (taslaklara) tam uyum gösterilmektedir.

2. Şartname Uyumluluk Analizi
=====================================
- **Görev 1 (Nesne Tespiti):** Taşıt (0), İnsan (1), UAP (2), UAİ (3) sınıfları tanımlı. Hareketlilik (motion_status) temporal olarak inceleniyor. Kısmi görünürlük, kamera hareketi kompanzasyonu ve iniş durumu kontrolleri (kenar teması, engel örtüşmesi) koda entegre edilmiş durumda. UAP/UAİ için IoU < 0.5 durumlarına veya yanlış `landing_status` gönderimlerine karşı mantıksal savunmalar var ancak risk taşıyor.
- **Görev 2 (GPS Kestirimi):** GPS sağlıklı olduğunda sunucu değeri doğrudan alınıyor. GPS sağlıksızken (GPS=0) Lucas-Kanade algoritması ile Optik Akışa (Visual Odometry) geçiliyor. Bu şartname FR-011 ve FR-012 maddelerini karşılıyor.
- **Görev 3 (Görüntü Eşleme):** ORB/SIFT kullanılarak referans objeler eşleştiriliyor. Farklı açılardan veya kameralardan eşleştirme için esnek eşikleme yapılmış, JSON formatı `detected_undefined_objects` ile doğru şekilde raporlanıyor.
- **Genel Akış ve Kısıtlamalar:** Offline, karesel senkronizasyon ve saniyede minimum 1 FPS (HW-007) mimariyle sağlanıyor. İnternetsiz çalışma ve hardcoded parametrelerin konfigürasyon üzerinden yönetimi (HW-009) başarılmış.


3. Kritik Riskler
=====================================
**Risk Tanımı:** Görev 2 Optik Akışta (Visual Odometry) İrtifa (Altitude) Sabitlenmesi Nedeniyle Kestirim Hatası (Drift)
- **Şartname İhlali / Puan Etkisi:** GPS koptuğunda (GPS=0) altitude bilgisi sunucudan gelmezse veya NaN olursa, sistem son bilinen GPS irtifasını sabit kabul ediyor. Araç irtifa değiştirirse pixel-to-meter çevirisi tamamen yanlış hesaplanır, bu durum 3D Öklid hatasını dramatik şekilde artırarak puan kaybına yol açar.
- **Olasılık:** Yüksek
- **Tetikleyici Senaryo:** 4 dakikalık GPS kopması sırasında uçağın irtifasının (z ekseni) ciddi oranda değişmesi.
- **Tespit Yöntemi:** `src/localization.py` içindeki `_update_from_optical_flow` ve `_pixel_to_meter` fonksiyonlarının incelenmesi. Scale_ratio üzerinden z ekseni hareketi bulunmaya çalışılsa da güvenilirliği belirsiz.
- **Mimari Düzeyde İyileştirme Yönü:** Ölçek değişimini (`scale_ratio`) daha sağlam bir derinlik kestirimi veya affine transform ayrıştırması (monocular depth, ya da homography matrisi) ile güçlendirerek irtifanın (z ekseninin) daha güvenilir tahmin edilmesi sağlanmalıdır.

**Risk Tanımı:** UAP/UAİ İniş Uygunluğu (Landing Status) Hesaplamasında Engel Olarak Kendi Sınıfını (UAP/UAİ) Filtrelememe
- **Şartname İhlali / Puan Etkisi:** UAP veya UAİ tespit edildiğinde `landing_status` hesaplanıyor. Engel (obstacle) kontrolünde (`_determine_landing_status`), engel listesine (obstacles) kendi `landing_zone_ids` (2 ve 3) dışındaki sınıflar ekleniyor. Ancak UAP/UAİ nesnesi tespitinde bounding box'lar birbiriyle veya kendileriyle overlap durumunda false negative üretme riski taşıyor. Ayrıca modelin taşıt olarak tanımladığı ama aslında UAP üzerinde park halindeki nesnelerin doğru sınıflandırılmama ihtimali. **Varsayım:** Sahi/sliced inference nedeniyle birden fazla parça/box oluşursa bu durum çakışmalara yol açabilir. Puan kaybı (AP düşüşü).
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Aynı iniş alanı için parça parça veya birden çok (overlap) bbox üretilmesi durumu.
- **Tespit Yöntemi:** `src/detection.py` içindeki `_determine_landing_status` analizi.
- **Mimari Düzeyde İyileştirme Yönü:** NMS (Non-Maximum Suppression) sonrasında aynı sınıf UAP/UAİ objeleri arasında ekstra overlap kontrolleri yapılmalı ve sadece taşıt (0) veya insan (1) sınıfları mutlak engel sayılmalı. UAP/UAİ'nin kendi içindeki çakışmalar engel değerlendirmesi dışında tutulmalıdır.

4. Orta ve Düşük Öncelikli Riskler
=====================================
**Risk Tanımı:** Görev 3 Referans Obje Eşleştirmede Feature Sayısı Yetersizliği (ORB / SIFT)
- **Şartname İhlali / Puan Etkisi:** Termal kamera, düşük ışık veya bulanık görüntülerde ORB'nin yetersiz feature çıkarması referans objenin bulunamamasına (false negative) ve mAP düşüşüne yol açar.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Termal veya çok farklı açılardan/kaliteden referans verilmesi (Şartname 3.3).
- **Tespit Yöntemi:** `src/image_matcher.py` incelenmesi. Feature extraction (ORB default) salt yoğunluk bazlı çalışıyor.
- **Mimari Düzeyde İyileştirme Yönü:** Termal vs RGB gibi çok modaliteli (cross-modal) eşleştirmeler için geleneksel SIFT/ORB yerine Deep Learning tabanlı eşleştiricilerin (SuperGlue, LoFTR vb.) entegrasyonu (inference hızını koruyarak) düşünülebilir. Veya SIFT'in zorunlu etkinleştirilmesi.

**Risk Tanımı:** Temporal Movement Karar Mantığında Bounding Box Takip Hatası
- **Şartname İhlali / Puan Etkisi:** Taşıtın hareket durumunun (0 veya 1) hatalı verilmesi AP düşüşü yaratır. Bounding box takip mantığı Euclidean mesafeye (`MOVEMENT_MATCH_DISTANCE_PX`) dayanıyor. Kalabalık araç gruplarında ID geçişleri (ID switch) oluşabilir.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Yüksek hızda ilerleyen drone altında yan yana giden veya park etmiş çok sayıda taşıtın frame'den frame'e karışması.
- **Tespit Yöntemi:** `src/movement.py` `_match` metodu incelendi. Sadece merkez mesafesi kontrol ediliyor, IoU veya DeepSORT benzeri görsel görünüm (appearance) özellikleri kullanılmıyor.
- **Mimari Düzeyde İyileştirme Yönü:** Track eşleştirmede (assignment) mesafe ile birlikte IoU (Intersection over Union) kullanılarak ID eşleşmelerinin sağlamlaştırılması. Kalman filtresi eklentisi düşünülebilir.

**Risk Tanımı:** Çakışan Konfigürasyon ve Model Performans Limiti
- **Şartname İhlali / Puan Etkisi:** Donanım düşük performansta çalışırsa sistem circuit breaker `timeout` yüzünden kare atlayabilir/gecikme telafisine (latency_comp) düşebilir. Karesel ilerleme kuralına uyulmazsa sunucu timeout hatası üretir.
- **Olasılık:** Düşük
- **Tetikleyici Senaryo:** OOM (Out Of Memory) veya SAHI+YOLOv8 kombinasyonunun 1 FPS altına düşmesi.
- **Tespit Yöntemi:** `main.py` ve `network.py` timeout mekanizmaları analizi.
- **Mimari Düzeyde İyileştirme Yönü:** FPS aşırı düşerse SAHI geçici olarak (veya dinamik olarak) devreden çıkarılabilmeli (degrade mode detection'a da entegre edilebilir).


5. Performans ve Kaynak Değerlendirmesi
=====================================
- **Donanım Uyumluluğu:** GPU `cuda` olarak ayarlanmış, `HALF_PRECISION=True` kullanımı performansı artırıcı iyi bir pratik. 1 FPS barajını rahat geçecek (30+ FPS öngörülmüş) bir donanım optimizasyonu var.
- **Hafıza Yönetimi:** YOLOv8 için Memory leak engellemek amacıyla `torch.cuda.empty_cache()` kullanımı var, OOM hatalarına karşı try-catch blokları ile programın çökmesi (`FatalSystemError`) engelleniyor ve bir sonraki kareye geçiş (`degrade` mod) başarılı şekilde tasarlanmış.
- **Ağ Dayanıklılığı:** Circuit Breaker (`src/resilience.py`) ve Idempotency key ile yarışma sunucusu çökmelerine, gecikmelerine karşı sağlam bir defans kurulmuş.

6. Belirsizlikler ve Koşullu Riskler
=====================================
- **Varsayım:** Şartname Madde 17 (TBD) JSON formatlarının kesin olmadığını belirtiyor. Sistemin `payload_schema.py` esnekliği iyi olmakla birlikte, sunucudan gelecek yeni bir key/alan yapısında kodun çökmeme (exception swallowing) veya silent failure üretme riski bulunuyor. `CompetitionPayloadSchema.validate_top_level_payload` strict olarak tasarlandığından yeni zorunlu alanlar eklenirse `DataContractError` fırlatacak ve sistem durabilecektir.
- **Varsayım:** Kamera Parametreleri. `FOCAL_LENGTH_PX = 800.0` hardcode. Yarışma gününden önce gerçek kamera parametreleri verilip de bu değer config'den (veya TBD-010'dan) alınmazsa Görev 2 tamamen hatalı mesafe (drift) ölçecektir.

7. Genel Sağlık Skoru (0–10)
=====================================
**8 / 10**

**Gerekçe:** Mimari yapı, hata kontrolü (resilience), ağ yapısı ve modülerlik TEKNOFEST şartnamesi (1 FPS, internetsiz çalışma, JSON paketlemeleri, rekabet döngüsü) ile tam uyumludur. Temel YOLO nesne tespiti entegrasyonu sağlamdır. Puan kıran unsurlar; özellikle GPS kopması anında irtifa (z) kestirimi mantığındaki zayıflık ve çoklu araçlarda basit Euclidean takibin getirebileceği yanlış `motion_status` risklerinden kaynaklanmaktadır. Bu alanlardaki ufak algoritmik güncellemeler ile sistem kolaylıkla 10/10 seviyesine çıkabilir.