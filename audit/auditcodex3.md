# 1. Özet Bulgular

Bu denetimde `sartname/sartname.md` tek doğruluk kaynağı alınarak kod tabanı statik olarak incelenmiştir. Sistem, üç görevi aynı çıktı sözleşmesinde birleştiren mimariyi kurmuş olsa da yarışma puanını ve oturum sürekliliğini doğrudan etkileyen kritik riskler bulunmaktadır.

1. Şartname gereksinimleri kısmen karşılanıyor; özellikle Görev 1/2/3 için temel üretim zincirleri mevcut.
2. Aynı kareye çoklu sonuç gönderimi, boş fallback çıktı gönderimi ve agresif oturum sonlandırma koşulları puan kaybı riski oluşturuyor.
3. Döngü kilitlenmesinden çok döngüden erken çıkma/oturum sonlandırma riski öne çıkıyor.
4. Modüller arası bazı davranışlar tutarsız; özellikle idempotency niyeti ile fiili tekrar gönderim davranışı ayrışıyor.
5. İnternet yasağını kod seviyesinde zorlayan bir teknik kilit bulunmuyor; çalışma güvenliği konfigürasyona bırakılmış.

# 2. Şartname Uyumluluk Analizi

| Alan | Kanıt Durumu | Değerlendirme |
|---|---|---|
| Görev 1 sınıf kapsamı (0/1/2/3) | `src/detection.py::_configure_class_mapping`, `detect` | Karşılanıyor (model sınıfı eşleme ve 4 sınıfa indirgeme mevcut). |
| Görev 1 zorunlu durum alanları | `src/detection.py::_determine_landing_status`, `src/movement.py::annotate`, `src/payload_schema.py::normalize_object` | Kısmen karşılanıyor (normal akışta alanlar üretiliyor; fallback/boş obje akışında doğal olarak gönderilmiyor). |
| Görev 1 duplikasyon riski kontrolü | `src/detection.py::_merge_detections_nms`, `_suppress_contained` | Karşılanıyor (NMS + containment suppression var), ancak submit öncesi ayrıca obje kırpma (cap) nedeniyle recall riski var. |
| Görev 2 GPS sağlık koşullu davranış | `src/localization.py::update`, `_update_from_gps`, `_update_from_optical_flow`, `predict_without_measurement` | Kısmen karşılanıyor (GPS=0 için görsel kestirim/predict-only yolu var). |
| Görev 2 ilk 450 kare sağlık=1 | Doğrudan istemci doğrulaması yok | **Varsayım:** Sunucu bu kuralı garanti ettiği için istemci tarafında ek kural konmamış. |
| Görev 3 referans eşleştirme | `src/image_matcher.py::load_references`, `match`, `_match_reference`; `main.py::_validate_task3_references` | Karşılanıyor (ID doğrulama + eşleştirme zinciri var), fakat kritik senaryoda modül pasifleşebiliyor. |
| Her kareye 1 sonuç, sıradaki kareyi önceki sonuçtan sonra alma | `main.py::run_competition` (pending/fetch/submit state machine) | Kısmen karşılanıyor (akış niyeti doğru), duplicate metadata ve tekrar submit durumlarında kural riski doğuyor. |
| JSON taslak uyumu | `src/network.py::build_competition_payload`, `src/payload_schema.py::validate_top_level_payload` | Karşılanıyor (taslak alanlar mevcut), nihai format TBD olduğu için kırılganlık riski sürüyor. |
| İnternet yasağı / tam yerel çalışma | `config/settings.py::BASE_URL`, `main.py::apply_runtime_overrides` | Uyum garanti edilmiyor (internet erişimini teknik olarak engelleyen zorlayıcı katman yok). |
| 1 FPS minimum (HW-007) | `main.py::FPSCounter`, `_print_summary` | İzleme var, zorlayıcı garanti yok. |

# 3. Kritik Riskler

### Bulgu K1

- Dosya: `main.py`, `src/network.py`, `tests/test_all.py`
- Fonksiyon/Sınıf: `_fetch_competition_step`, `NetworkManager.send_result`, `TestIdempotencySubmit.test_second_submit_same_frame_is_blocked`
- Risk Tanımı: Duplicate `frame_id` geldiğinde kare işlenmeye devam ediyor ve daha önce gönderilmiş kare için tekrar POST atılabiliyor.
- Şartname İhlali / Puan Etkisi: Aynı kareye birden fazla sonuç gönderme riski doğar; şartnameye göre yalnızca ilk sonuç geçerlidir. 4 oturum x nominal 2250 kare ölçeğinde tekrarlar, gönderim kapasitesi/engelleme riski nedeniyle dolaylı geçersiz kare artışı ve mAP + Öklid hata puanında toplam kayıp üretir.
- Olasılık: Yüksek
- Tetikleyici Senaryo: Sunucunun duplicate metadata döndürmesi veya aynı kare için istemci yeniden gönderim döngüsüne girmesi.
- Tespit Yöntemi: Statik akış incelemesi; `_was_already_submitted` sadece uyarı logu üretip gönderimi durdurmuyor, testte ikinci çağrıda da POST sayısı artıyor.
- Mimari Düzeyde İyileştirme Yönü: Kare kimliği bazlı tek-yazım (single-write) garanti katmanı ve rekabet döngüsünde “frame lifecycle state” zorunlu geçiş mimarisi.

### Bulgu K2

- Dosya: `main.py`, `src/network.py`
- Fonksiyon/Sınıf: `_fetch_competition_step`, `NetworkManager._build_safe_fallback_payload`, `NetworkManager.send_result`
- Risk Tanımı: Degrade/fallback yollarında `detected_objects` ve `detected_undefined_objects` boşaltılarak sonuç gönderiliyor.
- Şartname İhlali / Puan Etkisi: Her kareye sonuç gönderme kuralı korunurken içerik boşaldığı için Görev 1 ve Görev 3’te recall düşüşü ve mAP kaybı oluşur. GPS=0 döneminde görsel bilgi kaybı zincirleme şekilde Görev 2 stabilitesini de etkileyebilir.
- Olasılık: Orta-Yüksek
- Tetikleyici Senaryo: Geçici ağ hatası, görüntü indirilememesi, preflight reject sonrası güvenli fallback’a zorlanma.
- Tespit Yöntemi: Statik kod incelemesi; fallback payload’da obje listeleri sıfırlanıyor, degrade modunda belirli slotlarda doğrudan fallback gönderiliyor.
- Mimari Düzeyde İyileştirme Yönü: “Süreklilik modu” içinde semantik çıktıyı tamamen sıfırlamak yerine görev bazlı minimum bilgi korunumu sağlayan ayrı bir yarışma-degrade mimarisi.

### Bulgu K3

- Dosya: `src/resilience.py`, `main.py`
- Fonksiyon/Sınıf: `SessionResilienceController.should_abort`, `run_competition` (duplicate/permanent reject abort eşikleri)
- Risk Tanımı: Geçici sorunlarda oturumu erken sonlandıran çoklu abort koşulları bulunuyor.
- Şartname İhlali / Puan Etkisi: Oturum içinde gönderilemeyen kareler doğrudan geçersizdir; 60 dakikalık pencere içinde erken çıkış, kalan karelerin puanlanamamasına ve toplam puanın hızla düşmesine neden olur.
- Olasılık: Orta-Yüksek
- Tetikleyici Senaryo: Uzayan transient süre, art arda duplicate frame, art arda permanent reject.
- Tespit Yöntemi: Statik karar ağacı incelemesi; transient wall-time limiti, duplicate abort sayacı ve permanent reject abort sayacı ayrı ayrı “break” üretiyor.
- Mimari Düzeyde İyileştirme Yönü: Oturum sonlandırmayı son çare haline getiren, görev bazlı izolasyonlu ve geri-kazanım öncelikli bir hata dayanıklılık orkestrasyonu.

### Bulgu K4

- Dosya: `main.py`
- Fonksiyon/Sınıf: `run_competition`, `_validate_task3_references`
- Risk Tanımı: Referans ID bütünlüğü kritik eşik aşımında Görev 3 eşleştirici tamamen pasifleştiriliyor.
- Şartname İhlali / Puan Etkisi: Oturum boyunca `detected_undefined_objects` üretilememesi Görev 3 mAP skorunu sistematik olarak düşürür.
- Olasılık: Orta
- Tetikleyici Senaryo: Oturum başında sunucudan gelen referans setinde duplicate/quarantine oranının kritik eşiği geçmesi.
- Tespit Yöntemi: Statik akış incelemesi; `duplicate_critical` durumunda `image_matcher = None` atanıyor.
- Mimari Düzeyde İyileştirme Yönü: Referans bütünlüğü bozulsa da görevi tamamen kapatmak yerine kısmi güvenilir referans kümesiyle sürdürülebilen dayanıklı eşleştirme katmanı.

### Bulgu K5

- Dosya: `src/network.py`, `config/settings.py`
- Fonksiyon/Sınıf: `NetworkManager._apply_object_caps`, `Settings.RESULT_MAX_OBJECTS`, `Settings.RESULT_CLASS_QUOTA`
- Risk Tanımı: Submit öncesi global ve sınıf bazlı obje kırpma nedeniyle tespitler sistematik düşürülebiliyor.
- Şartname İhlali / Puan Etkisi: Aynı karede çok sayıda gerçek nesne bulunduğunda tespitlerin bir kısmı bilinçli olarak düşürülür; bu durum false negative artışıyla Görev 1/Görev 3 mAP skorlarını aşağı çeker.
- Olasılık: Yüksek
- Tetikleyici Senaryo: Kalabalık sahneler, toplu taşıt/insan yoğunluğu, çoklu referans görünürlüğü.
- Tespit Yöntemi: Statik kod incelemesi; `raw_count` > kota durumunda kırpma uygulanıyor ve loglanıyor.
- Mimari Düzeyde İyileştirme Yönü: Sunucu sınırlarını korurken puan metriklerine duyarlı, senaryo-bağımlı yük yönetimi ve önceliklendirme mimarisi.

# 4. Orta ve Düşük Öncelikli Riskler

### Bulgu M1

- Dosya: `config/settings.py`, `main.py`, `src/network.py`
- Fonksiyon/Sınıf: `Settings.BASE_URL`, `apply_runtime_overrides`, `NetworkManager.__init__`
- Risk Tanımı: İnternet yasağını teknik olarak zorlayan (hard block) bir güvenlik katmanı bulunmuyor; endpoint tamamen runtime override ile değiştirilebiliyor.
- Şartname İhlali / Puan Etkisi: Yanlış ortam konfigürasyonu ile internete çıkış riski oluşur; bu durum yarışma kural ihlali ve diskalifikasyon riski doğurur.
- Olasılık: Orta
- Tetikleyici Senaryo: Yarışma günü hatalı `--base-url` veya `AIA_BASE_URL` kullanımı.
- Tespit Yöntemi: Statik konfigürasyon/başlatma akışı incelemesi.
- Mimari Düzeyde İyileştirme Yönü: Yarışma modunda ağ hedeflerini sadece yerel segmentle sınırlandıran politika-katı yaklaşımı.

### Bulgu M2

- Dosya: `src/payload_schema.py`, `src/network.py`
- Fonksiyon/Sınıf: `CompetitionPayloadSchema.validate_top_level_payload`, `normalize_object`, `NetworkManager.build_competition_payload`
- Risk Tanımı: JSON taslağına güçlü bağlılık var; nihai format değişiminde alan adı/tip uyumsuzluğu riski yüksek.
- Şartname İhlali / Puan Etkisi: Nihai format yayınlandığında tip/alan farkı oluşursa gönderimler 4xx alabilir; art arda reddedilen kareler puan kaybı ve oturum kesintisi üretir.
- Olasılık: Orta
- Tetikleyici Senaryo: Yarışma öncesi nihai JSON şemasında alan tipi veya ad değişikliği.
- Tespit Yöntemi: Statik sözleşme incelemesi; required alanlar ve tip normalizasyonu hard-coded.
- Mimari Düzeyde İyileştirme Yönü: Şema sürümleme ve sözleşme-adapter mimarisiyle format evrimine kontrollü geçiş.

### Bulgu M3

- Dosya: `main.py`
- Fonksiyon/Sınıf: `FPSCounter`, `_print_summary`, `run_competition`
- Risk Tanımı: 1 FPS minimum gereksinimi izleniyor fakat zorlayıcı runtime emniyet mekanizması bulunmuyor.
- Şartname İhlali / Puan Etkisi: Uzun oturumlarda hız düşerse süre içinde tamamlanamayan kareler geçersiz kalır; toplam puanı doğrudan düşürür.
- Olasılık: Orta
- Tetikleyici Senaryo: Yüksek çözünürlük, SAHI + matcher yükü, CPU fallback, disk I/O artışı.
- Tespit Yöntemi: Statik performans akışı incelemesi; ölçüm/log var, otomatik uyum koruma yok.
- Mimari Düzeyde İyileştirme Yönü: Görevler arası hesaplama bütçesini gerçek zamanlı dengeleyen performans yönetişim katmanı.

### Bulgu M4

- Dosya: `src/localization.py`, `src/network.py`
- Fonksiyon/Sınıf: `VisualOdometry.update`, `NetworkManager._validate_frame_data`
- Risk Tanımı: GPS sağlık verisi bozuk/eksik geldiğinde istemci tarafı değeri 0’a zorlayarak görsel kestirime geçiyor.
- Şartname İhlali / Puan Etkisi: Yanlış sağlık yorumu, özellikle oturumun ilk bölümünde gereksiz görsel moda geçişe neden olursa Öklid hata metriğinde birikimli sapma üretebilir.
- Olasılık: Düşük-Orta
- Tetikleyici Senaryo: Sunucudan bozuk `gps_health_status` veya format sapması gelmesi.
- Tespit Yöntemi: Statik veri doğrulama akışı incelemesi.
- Mimari Düzeyde İyileştirme Yönü: Sağlık verisini çok-kaynaklı doğrulayan ve belirsizlik durumunda kontrollü karar veren durum yönetimi.

### Bulgu M5

- Dosya: `src/network.py`, `tests/test_all.py`
- Fonksiyon/Sınıf: `NetworkManager.send_result`, `TestIdempotencySubmit.test_second_submit_same_frame_is_blocked`
- Risk Tanımı: Test ismi “blocked” ifadesi taşısa da davranış fiilen tekrar gönderime izin veriyor; tasarım niyeti ile uygulama davranışı ayrışmış durumda.
- Şartname İhlali / Puan Etkisi: Bu tutarsızlık yarışma döngüsü kuralında yanlış güven duygusu doğurur; operasyon sırasında duplicate-submit riskinin gözden kaçmasına neden olabilir.
- Olasılık: Orta
- Tetikleyici Senaryo: Regresyon denetiminde testin adı nedeniyle davranışın yanlış yorumlanması.
- Tespit Yöntemi: Statik test-kod karşılaştırması.
- Mimari Düzeyde İyileştirme Yönü: Kural-kritik davranışlarda test niyeti, test adı ve gerçek beklentiyi tek doğruluk çizgisine bağlayan doğrulama mimarisi.

# 5. Performans ve Kaynak Değerlendirmesi

1. `main.py` yarışma akışı fetch/submit işlerini iki iş parçacığında yürütüp işlemeyi tek bir `pending_result` üzerinden sıraya alıyor; bu yaklaşım döngü kontrolünü sadeleştiriyor ancak yoğun sahnede işlem gecikmesi birikirse 1 FPS hedefini koruma garantisi vermiyor.
2. `src/detection.py` içinde SAHI dilimleme + birleşik NMS, `src/image_matcher.py` içinde referans başına eşleştirme, `src/utils.py` içinde periyodik JSON/disk loglama birlikte değerlendirildiğinde özellikle 4K ve yoğun obje senaryolarında hesaplama maliyeti artıyor.
3. `src/network.py` fallback/degrade yaklaşımı throughput’u korumaya yardımcı olurken doğruluk maliyeti yaratıyor; yarışma metrikleri açısından performans-doğruluk dengesi kırılgan.
4. 4 oturum ve nominal 2250 kare ölçeğinde, kısa süreli ağ dalgalanmalarının dahi oturumun ilerleyen kısmında birikimli etki üretmesi olasıdır.

# 6. Belirsizlikler ve Koşullu Riskler

1. **Varsayım:** Nihai JSON formatı şartnamede TBD olduğu için mevcut taslak alan adları üzerinden değerlendirme yapılmıştır; nihai format değişikliği bulguların bazılarını kritikleştirebilir.
2. **Varsayım:** Yarışma sunucusu duplicate frame, 4xx davranışı ve pacing tarafında mock/test ortamından farklı davranabilir; bu durumda döngü risklerinin gerçekleşme olasılığı değişir.
3. **Varsayım:** Model eğitimi/sınıf kapsaması (özellikle UAP/UAİ ve özel etiket kuralları) yalnız statik koddan tam doğrulanamaz; bu alanın gerçek mAP etkisi veri dağılımına bağlıdır.
4. **Varsayım:** İşletim sistemi veya yarışma ağı seviyesinde ayrı bir internet engeli olabilir; kod tabanında bunun garantisini veren bir mekanizma görünmemektedir.
5. **Varsayım:** Kamera parametreleri (TBD-010) güncellendiğinde Görev 2 hata profili değişebilir; mevcut değerlendirme kod içi varsayılanlarla sınırlıdır.

# 7. Genel Sağlık Skoru (0–10)

**4.9 / 10**

Gerekçe: Çekirdek görev boru hatları mevcut ve şartname taslak sözleşmesine belirli ölçüde uyum var; ancak kare başına tek sonuç kuralını zayıflatan duplicate-submit davranışı, fallback kaynaklı boş çıktı üretimi, görev pasifleştirme ve oturum erken sonlandırma riskleri yarışma puanı açısından yüksek etkili kalmaktadır.
