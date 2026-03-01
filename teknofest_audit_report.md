# TEKNOFEST 2026 Havacılıkta Yapay Zeka — Rekabet Sistemi Kod Denetim Raporu

## 1. Özet Bulgular

- **JSON Anahtar Uyuşmazlığı:** Şartnamede taşıtlar için zorunlu olan `movement_status` alanı kodda `motion_status` olarak kodlanmış. Yarışma sunucusunda JSON şema hatasına (HTTP 4xx) yol açarak doğrudan AP puanını düşürür veya kareyi geçersiz kılar.
- **Döngü Kilitlenmesi:** Ağ katmanında sunucudan alınacak kalıcı bir `HTTP 400 (Permanent Reject)` hatasında ana döngü o kareyi atlamadığı için sonsuz döngüye girme ve oturumu kilitleme riski mevcut.
- **Optik Akış İrtifa (Z) Kayması:** Görev 2'de ölçek (scale_ratio) değişimi üzerinden yapılan irtifa kestirimi son derece gürültüye açıktır. Ufak hatalar üstel birikir ve Öklid hatasından ciddi puan kaybı yaratır.
- **SAHI Performans Darboğazı:** 4K çözünürlüklü drone görüntülerinde $640\times640$ boyutunda SAHI çalıştırmak tek karede ~21 defa model tahmini (inference) demektir. Puanlama için gereken saniyede 1 kare (1 FPS) ve oturum başına toplam 60 dakika (kare başı ~1.6 sn) sınırı rahatlıkla aşılabilir.
- **Homografi Yanlış Pozitifleri:** Görev 3'te (Image Matching) yalnızca 4 ortak nokta ile hesaplanan homografi, dejenere algoritmik BBox'lar (kendisiyle kesişen veya tüm kadrajı kaplayan) çıkarırsa sisteme yanlış pozitif olarak sızar.
- **Aşırı İyimser Optik Akış Fiziği:** Piksel → Metre hesabında pinhole kamera formülü kullanılıyor, ancak drone'un eğimli (pitch/roll açısına sahip) aktığı varsayılmayıp doğrudan dikey (nadir) bakış açısı tahmini yapılıyor.
- **İniş Alanına Engel Sızması:** Görev 1 sınıf haritalamasında (COCO_TO_TEKNOFEST), haritalanamayan objelerin iniş alanında bulunması sistemden temizlenebilir (UNKNOWN_OBJECTS_AS_OBSTACLES=False durumu). Bu durum inişe uygun olmayan alanın '1' olarak bildirilmesine yol açarak doğrudan eksi puan yazdırabilir.

## 2. Şartname Uyumluluk Analizi

### Görev 1: Nesne Tespiti Şartname Aktarımı

- **Gereksinim:** Taşıt (0), İnsan (1), UAP (2), UAİ (3) tespiti zorunlu.
  - **Durum:** Sınıf indeksleri başarıyla eşleşmiş. Fakat `UNKNOWN_OBJECTS_AS_OBSTACLES` dinamik ayarlanabiliyor. İsteğe bağlı olarak bilinmeyen nesneler iniş uygunluğu için hesaba dahil ediliyor, bu başarılı bir koruma.
- **Gereksinim:** `landing_status` (UAP/UAİ için) ve `movement_status` (Taşıtlar için) verilmeli.
  - **Durum:** UAP/UAİ için `landing_status` değerleri doğru (`-1, 0, 1`) hesaplanıyor. İniş alanı dışı tespitte IoU = 0 olarak alınmış, kenar piksellere temas durumu gözetilmiş (uyumlu).
  - **Risk:** Şartnamedeki `movement_status` parametresi yerine Payload üreticisi `network.py` ve referansı `detection.py` dosyalarında `motion_status` kullanılmıştır. Kesin reddedilme.

### Görev 2: Konum Kestirimi (Visual Odometry)

- **Gereksinim:** GPS=1 ise referans değer, GPS=0 ise üretilen m cinsinden (x, y, z) gönderilmeli. Hata birikimi = mAP kaybı.
  - **Durum:** `localization.py` GPS koptuğunda tam olarak son alınan düzgün GPU noktasından Lucas-Kanade Optik Akış ile göreceli ilerlemeyi hesaplıyor. İstenen modül başarı ile dahil edilmiş.
  - **Risk:** Odak uzaklığı `FOCAL_LENGTH_PX = 800.0` sabit tanımlı. Yarışmada verilecek TBD-010 parametreleri entegre edilmezse m bazlı hesaplama tamamen kurgusal (hatalı) çıkacaktır.

### Görev 3: Eşleştirme (Image Matching)

- **Gereksinim:** Oturum başında verilen referans objelerin video akışında eşlenmesi; bulunmayanlar pas geçilmeli.
  - **Durum:** `image_matcher.py` başlangıçta görüntüleri çekiyor ve ORB/SIFT ile offline veri üretiyor (Uygun). Bulunmadığı zaman boş output geçiyor (Uygun).

### Genel Döngü ve Çevre Koşulları

- **Gereksinim:** İnternet erişimi yasak, lokal çalışmalı.
  - **Durum:** Model diskten yükleniyor, API sorguları yalnızca `BASE_URL`'e yapılıyor (Uygun).
- **Gereksinim:** Kareye tam olarak 1 sonuç düşürülmeli, asenkronluk kısıtlı.
  - **Durum:** `main.py` thread pool kullansa da `pending_result` muteksiyle kilitlenmesi tek işlem sırasını (sıralı kare işlemeyi) koruyor.

## 3. Kritik Riskler

### 1- JSON Şema Parametresi İhlali

- **Risk Tanımı:** Şartnamede zorunlu "movement_status" anahtar sözcüğü, kodun genelinde "motion_status" olarak tanımlanmış ve sunucuya bu formatta payload edilmektedir.
- **Şartname İhlali / Puan Etkisi:** Görev 1 AP puanı %0; sunucunun payload'ı `preflight` reddi vererek geçerli hiçbir nesnenin puanını yazmamasına sebebiyet verebilir.
- **Olasılık:** Yüksek (Taslak Formata %100 Uyumsuzluk)
- **Tetikleyici Senaryo:** İlk taşıt tespit edilen ve sunucuya iletilen kare.
- **Tespit Yöntemi:** `src/detection.py` (satır 658), `src/network.py` (satır 653) objelerine doğrudan "motion_status" yazılması.
- **Mimari Düzeyde İyileştirme Yönü:** Kod tabanındaki tüm "motion_status" atıfları, DTO nesneleri ve payload oluşturucu dictionary yapılarında "movement_status" olarak düzenlenmelidir.

### 2- Permanent Rejection Durumunda Döngü Kilitlenmesi (Deadlock)

- **Risk Tanımı:** `network.py` içerisindeki `send_result` metodu, 4xx Permanent Reject aldığında `SendResultStatus.PERMANENT_REJECTED` döner. Ancak `main.py`'deki `_submit_competition_step`, hata sınıflandırması ne olursa olsun iterasyonu bitirmek yerine `pending_result` nesnesini hafızada tutarak ("continue") aynı frame'i sisteme ısrarla itmeye çalışabilir.
- **Şartname İhlali / Puan Etkisi:** Süre aşımı ve rekabet oturumunun kitlenerek o anki kareden (Örn: 50. Kare) itibariyle tüm kareleri süreden (%0) kaybetme.
- **Olasılık:** Yüksek
- **Tetikleyici Senaryo:** Görev JSON formatının kabul edilmediği ilk anda sunucundan 400 Serisi hata geldiğinde, sistem "fallback" atsa da o da 4xx dönerse tetiklenir.
- **Tespit Yöntemi:** `main.py`'nin `action_result == "continue"` kısmında `pending_result` objesinin temizlenmemesi mantıksal inceleme ile görülmektedir.
- **Mimari Düzeyde İyileştirme Yönü:** `SendResultStatus.PERMANENT_REJECTED` state moduna girildiğinde ya o frame atılmalı (pas geçilmeli, `pending_result=None` olmalı) ya da sıfır değerli boş bir güvenli "boş frame" gönderilerek frame atlanmalıdır.

### 3- Optik Akış Odak ve Z-Ekseni Sapması

- **Risk Tanımı:** Lucas-Kanade (LK) tabanlı Optical Flow ile hesaplanan X/Y matris değerleri, "Pinhole Kamera" modeli (`dx = dx * altitude / f`) baz alınarak çıkarılmış. Drone'un eğimi (pitch) hesapta yok formül dikey projeksiyon varsayıyor. İrtifa (Z) kaybı ise `scale_ratio` (LK noktaları uzaklaşması) bazlı belirleniyor, küçük köşe dağılımlarında inanılmaz gürültülüdür.
- **Şartname İhlali / Puan Etkisi:** GPS'in kapatılacağı 450. kare sonrası aşamalı drift ile Öklid sapmasının katlanması. (Puan kaybı).
- **Olasılık:** Orta/Yüksek (Pürüzsüz uçuş dışındaki tüm sekanslar)
- **Tetikleyici Senaryo:** Drone kamerasının öne tilt edildiği, ya da rüzgar altında sürüklendiği hareketler.
- **Tespit Yöntemi:** `src/localization.py` -> `_pixel_to_meter` formül mekaniği; `scale_ratio` hesabı.
- **Mimari Düzeyde İyileştirme Yönü:** Sadece LK Optical Flow yerine Homografi ile global hareket matrisi hesaplanması, veya LK hatalarını pürüzsüzleştirmek için Kalman Filtresi tasarlanması en gerçekçi çözümdür.

## 4. Orta ve Düşük Öncelikli Riskler

### Görev 3 Homografi Dejenerasyonu BBox Hataları

- **Risk Tanımı:** SIFT/ORB minimum 4 eşleşen nokta bulduğunda hesaplanan RANSAC homografisi, fiziksel olarak imkansız veya tersyüz olmuş bounding box'lar üretebilir.
- **Şartname İhlali / Puan Etkisi:** Hatalı bounding box'lar Görev 3'te (Image Matching) yanlış pozitif (false positive) olarak algılanarak mAP hassasiyetini düşürecektir.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Güçlü perspektif distorsiyonu içeren bir açından referans objenin görülmesi veya arka plandaki hatalı feature eşleşmeleri.
- **Tespit Yöntemi:** `src/image_matcher.py` içerisinde BBox genişliği hesaplanırken yalnızca basit max/min hesaplaması yapılıyor olması; konveks poligon kontrolünün eksik olması.
- **Mimari Düzeyde İyileştirme Yönü:** Homografiden üretilen 4 köşe noktasının anlamlı bir dörtgen (konveks) oluşturup oluşturmadığı kontrol edilmeli, en-boy oranı aşırı sapan (örn. %50'den fazla) çıktılar geçersiz sayılmalıdır.

### Sınıflandırmada Bilinmeyen Cisimler İçin İniş Engeli

- **Risk Tanımı:** NMS veya sınıf eşleme aşamasında elenen veya model tarafından algılanıp "bilinmeyen (-1)" olarak değerlendirilen cisimler, `UNKNOWN_OBJECTS_AS_OBSTACLES` bayrağının dinamik olarak `False` yapılması durumunda iniş engeli sayılmayabilir.
- **Şartname İhlali / Puan Etkisi:** Tespiti yapılamayan herhangi bir aracın iniş alanı üzerinde olmasına rağmen UAP/UAİ alanının inişe uygun ("1") olarak işaretlenmesi, AP eksi puan yazdıracaktır.
- **Olasılık:** Düşük (Kodda varsayılan olarak `True` yapılandırılmış)
- **Tetikleyici Senaryo:** Bilinmeyen veya `rider_suppress` filtresine takılmış ancak aslında var olan bir nesnenin olması ve ayarların yarışma günü yanlışlıkla `False` çekilmesi.
- **Tespit Yöntemi:** `config/settings.py` içinde özelliğin bir konfigürasyon (bool) yapılması ve `src/detection.py`'da çalışma zamanında değerlendirilmesi.
- **Mimari Düzeyde İyileştirme Yönü:** Şartname kurallarını doğrudan ilgilendiren bu kritik kontrolün opsiyonel bir ayar olmaktan çıkarılıp, zorunlu (hardcoded) mantıksal doğrulama seviyesine (`True` olarak sabitlenmesi) getirilmesi.

## 5. Performans ve Kaynak Değerlendirmesi

- **SAHI Overhead:** `config/settings.py` içinde 640 parça + 0.35 örtüşme boyutu ile ayarlı SAHI, 4K orijinal görüntülerde tek kare için ortalamanın üzerinde (20 civarı) Inference penceresi acar. `CLAHE` ile ön işleme ve `cv2.absdiff` gibi frame maliyetini katlayan hesaplamalar mevcut.
- **Donanım Kısıtları:** HW-007 kısıtı minimum "1 FPS" işleyişin devamı için yarışma süresine sığışma limitidir (2250 Kare, 15 Dk. min opsiyon için ~2.5 FPS elzemdir). SAHI bu limitin altına düşmeyi tetikleyebilir.
- **Risk:** GPU Memory Limit, Tensor bellek taşması (CUDA OOM), CPU darboğazı (Multithreading kısıtlı).
- **Varsayım:** Yarışma için "FP16" açık ve Max Batch cap limitliyse sistem ayakta kalabilir; `AUGMENTED_INFERENCE` kapalı olmalı. SAHI penceresi opsiyonel 1280'e çekilmeli veya 1080p küçültülmüş kare üzerinden işlem değerlendirilmeli.

## 6. Belirsizlikler ve Koşullu Riskler

- **Varsayım:** Sunucudan TBD-001 gereği dönmesi koşullu olan `altitude`'un yerine `translation_z` verileceği tahmini yapılmış. Eğer sunucu `z` veya tam tersi `altitude` dönerse, JSON şema pars'ta hata çıkabilir (`network.py`).
- **Varsayım:** Yarışma öncesinde "Kamera Odak Uzaklığı" verilmezse Optik Flow için metre çevrimi 800px varsayımı patlar.
- **Varsayım:** Şartnamenin `TBD-001` taslak Payload gereği sunucunun 2250 Karede (End-of-Stream) döndüreceği formatın HTTP 204 olup olmaması `network.py` üzerinde varsayıma bağlı kodlanmış.

## 7. Genel Sağlık Skoru (0–10)

**Skor:** 6.5 / 10

**Gerekçeler:**

- Mimari tasarım son derece iyi yapılandırılmış; "Resilience (Circuit Breaker)" kullanılarak ağ direnci hat safhada artırılmıştı. (Pozitif)
- Frame Loader simülasyon mantığı ve Object Tracker mantığı doğru ve modüler yazılmıştır. (Pozitif)
- Şartnamedeki kritik `movement_status` parametresi yanlış isimlendirilmiş. (-2 puan)
- Sunucu Redlerinde Iteration takılmaları (Deadlock) var. (Kritik hata: -1.0 puan)
- Optik flow (Metrik kalibrasyon) 4K drone dinamikleri için fazla basite indirgenmiş (Drift riski yüksek: -0.5 puan)
- Toparlanacak birkaç teknik kelime ve Thread düzeltmesi dışarıda bırakıldığında yarışma sunucusunda kararlı performans sergileyebilecek mimari olgunluğa büyük oranda yakındır.

