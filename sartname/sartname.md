
=======================================================================
TEKNOFEST 2026 – HAVACILIKTAKİ YAPAY ZEKA YARIŞMASI
BİRLEŞİK TEKNİK ŞARTNAMESİ
(Genel Şartname + Teknik Şartname)
=======================================================================

Versiyon: V1.0 (Teknik Şartname: 21.02.2026)
Kaynak Belgeler:
  - TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması Genel Şartnamesi (V1.1 – 20.02.2026)
  - TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması Teknik Şartnamesi (V1.0 – 21.02.2026)

========================
TANIMLAR VE KISALTMALAR
========================
- KYS        : TEKNOFEST Kurumsal Yönetim Sistemi
- UAP        : Uçan Araba Park (Alanı) – 4,5 m çapında daire işareti
- UAİ        : Uçan Ambulans İniş (Alanı) – 4,5 m çapında daire işareti
- TEKNOFEST  : Havacılık, Uzay ve Teknoloji Festivali
- T3 Vakfı   : Türkiye Teknoloji Takımı Vakfı
- JSON       : JavaScript Object Notation (veri iletim formatı)
- GPS        : Global Positioning System (Uydu tabanlı konumlandırma)
- mAP        : Mean Average Precision (Ortalama Kesinlik Değerlerinin Ortalaması)
- AP         : Average Precision (Ortalama Kesinlik)
- IoU        : Intersection over Union (Kesişimin Birleşime Oranı)
- FPS        : Frames Per Second (Saniyedeki Kare Sayısı)
- API        : Application Programming Interface
- RGB        : Red-Green-Blue (standart renk kamera)
- Takım Kaptanı       : Takımın organizasyonundan sorumlu, liderlik görevini üstlenen kişi
- Takım Danışmanı     : Her takım için en fazla bir (1) öğretmen/eğitmen/akademisyen
- İletişim Sorumlusu  : Takım adına teknik sunum yapan takım üyesi
- Yarışma Süreci      : Başvuruların alınmaya başladığı tarih ile final sonuçlarının açıklandığı tarih arası


=====================
1. GİRİŞ VE AMAÇ
=====================

Bu doküman TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması kapsamında yayımlanan Genel
Şartname ve Teknik Şartname belgelerinin birleştirilmiş ve bütünleşik halidir. Yarışma
öncesinde ve yarışma süresince yarışmacıların bilgisi dahilinde olması gereken tüm teknik,
operasyonel ve idari bilgileri tek bir referans belgede toplamaktadır.

Yarışmanın temel hedefleri şunlardır:
- Havacılık sektöründe karşılaşılan gerçek dünya problemlerine yapay zeka tabanlı çözümler
  geliştirmek.
- Hava araçlarının kamera/sensör verilerini işleyerek çevresel farkındalık (situational awareness)
  kabiliyetini artırmak.
- GPS gibi uydu tabanlı konumlandırmanın devre dışı kaldığı durumlarda yalnızca görsel veri
  üzerinden pozisyon kestirimi yapabilmek.
- Daha önce tanımlanmamış nesnelere karşı anlık adaptasyon ve genel nesne tanıma yeteneğini
  test etmek.


======================
2. KAPSAM
======================

Dahil olanlar:
  - Görev 1 : Hava aracı kamera görüntülerinden nesne tespiti ve sınıflandırması
               (taşıt, insan, UAP alanı, UAİ alanı; iniş uygunluğu ve hareketlilik durumu dahil)
  - Görev 2 : Alt-görüş kamerası görüntülerinden GPS'siz görsel odometri tabanlı pozisyon
               kestirimi (X, Y, Z eksenleri, metre cinsinden)
  - Görev 3 : Yarışma oturumu başlangıcında anlık bildirilen referans objelerin kamera
               görüntülerinde tespit edilmesi ve koordinatlarının raporlanması
  - Tüm görev çıktılarının yerel ağ üzerinden yarışma sunucusuna JSON formatında iletilmesi

Hariçler:
  - İnternet bağlantısı ve çevrimiçi servis kullanımı (kesinlikle yasaktır)
  - Görev 3 puanlama detaylarının bir bölümü ilerleyen şartname revizyonlarında verilecektir
  - Yarışma oturum temaları yarışma gününe kadar saklı tutulacaktır


=============================
3. GÖREVLER
=============================

-----------------------------
3.1. BİRİNCİ GÖREV: NESNE TESPİTİ
-----------------------------

Yarışmacılar tarafından tespit edilmesi beklenen nesne türleri dört adettir:
  (1) Taşıt
  (2) İnsan
  (3) Uçan Araba Park Alanı (UAP)
  (4) Uçan Ambulans İniş Alanı (UAİ)

Taşıt sınıfı için nesnenin hareketli veya hareketsiz olma durumu ayrıca bildirilmelidir.
UAP ve UAİ sınıfları için iniş uygunluk durumu ayrıca bildirilmelidir.

3.1.1. VİDEO VERİSİ HAKKINDA TEKNİK BİLGİLER

- Videolar hava aracının kalkışını, inişini ve seyrüseferini içerebilir; hava aracının
  yerden yüksekliği değişkenlik gösterebilir.
- Her oturumda yarışmacılara verilecek videonun süresi 5 dakikadır, FPS değeri 7,5'tir.
  Bu nedenle her oturumda toplam 2250 görüntü karesi verilecek ve 2250 adet sonuç
  beklenmektedir. Uçuş süreleri ve kare sayısı değişkenlik gösterebilir.
- Videolar Full HD veya 4K çözünürlüğünde çekilmektedir.
- Video kareleri herhangi bir görüntü formatında olabilir (jpg, png vb.).
- Videolar tek tek karelere ayrılmış ve sıralı olarak yarışmacılara sunulacaktır.
- Videolar günün herhangi bir vaktinde kayda alınmış görüntülerden elde edilebilir
  (gece/gündüz/alacakaranlık).
- Hava aracı kar, yağmur ve benzeri olumsuz hava koşullarında uçabileceğinden bu
  koşullar altında da algoritmaların test edilebileceği göz önünde bulundurulmalıdır.
- Hava aracı şehir, orman, deniz gibi farklı coğrafyalar üzerinde uçabilmektedir.
- Kamera açısı hava aracının hareketine bağlı olarak 70–90 derece aralığında değişkendir.
  Dik açıdan kaynaklı insan tespiti güçlükleri ve 0–70 derece aralığında uzaktaki nesnelerin
  tespitindeki sorunları engellemek amacıyla veri seti belirli açı değerleri kullanılarak
  hazırlanmıştır.
- Alt-görüş kamerasında olabilecek olağan hatalar nedeniyle görüntü karelerinde bozulmalar
  bulunabilir. Örnek bozukluklar: bulanıklık ve ölü pikseller. Ayrıca tekrarlamalar/donmalar
  veya görüntünün tamamen kaybı gibi durumlar da oluşabilir.
- Hava aracından alınan görüntüler RGB veya termal kamera ile elde edilmiş olabilir.

3.1.2. TAŞIT VE İNSAN TESPİTİ

- Görüntü karesinin tamamında bulunan tüm taşıt ve insanlar tespit edilmelidir.
- Tespit edilen taşıtlar hareketli veya hareketsiz olma durumuna göre sınıflandırılmalıdır.

Taşıt Sınıfı Kapsamındaki Nesne Türleri:

| Kategori               | Alt Türler                                                    |
|------------------------|---------------------------------------------------------------|
| Motorlu karayolu       | Otomobil, motosiklet, otobüs, kamyon, traktör/ATV/kara aracı |
| Raylı taşıtlar         | Tren, lokomotif, vagon, tramvay, monoray, füniküler           |
| Deniz taşıtları        | Tüm deniz taşıtları                                           |

Sınıf ve Durum ID Tablosu:

| Sınıf     | Sınıf ID | İniş Durumu Değerleri | Hareket Durumu Değerleri | Notlar                                         |
|-----------|----------|-----------------------|--------------------------|------------------------------------------------|
| Taşıt     | 0        | -1                    | 0, 1                     | Hareket durumu zorunlu çıktıdır                |
| İnsan     | 1        | -1                    | -1                       | Ayakta veya oturan tüm insanlar dahil           |
| UAP Alanı | 2        | 0, 1                  | -1                       | İniş uygunluk durumu zorunlu çıktıdır          |
| UAİ Alanı | 3        | 0, 1                  | -1                       | İniş uygunluk durumu zorunlu çıktıdır          |

Hareket Durumu ID Tablosu:

| Hareket Durum ID | Hareket Durumu |
|------------------|----------------|
| 0                | Hareketsiz     |
| 1                | Hareketli      |

Özel Etiketleme Kuralları:
- Görüntü karesinde tren bulunması durumunda lokomotif ve her bir vagon ayrı birer obje
  olarak tanımlanmalıdır.
- Tamamı görünmeyen taşıt ve insanlar da tespit edilmelidir (karenin dışına taşan nesneler dahil).
- Başka bir objenin arkasında kısmen görünen insan ve taşıtların da tespit edilmesi beklenmektedir.
- Bisiklet ve motosiklet sürücüleri "insan" olarak etiketlenmemelidir; taşıt ve sürücüsü bir bütün
  olarak yalnızca "taşıt" etiketi ile etiketlenmelidir.
- Scooter: sürücüsü olmadığı zamanlarda "taşıt", sürücüsü olduğu zamanlarda "insan" olarak
  etiketlenmelidir.
- Hava aracı uçuş sırasında kamera sürekli hareket halinde olduğundan, sabit taşıt nesneleri de
  görüntü üzerinde hareketliymiş gibi algılanabilir. Yarışmacıların, bir taşıtın gerçekten mi
  hareket ettiğini yoksa yalnızca kameranın hareketinden dolayı mı yer değiştirdiğini ayırt
  edebilecek yöntemler geliştirmeleri gerekmektedir.

3.1.3. UAP VE UAİ TESPİTİ

- UAP ve UAİ alanları 4,5 metre çapında birer daire ile belirtilmektedir.
  - UAP: Uçan arabanın park edebileceğini gösteren işareti barındıran alan (mavi daire, "UAP" yazılı)
  - UAİ: Uçan ambulansın iniş yapabileceğini gösteren işareti barındıran alan (kırmızı daire, "UAİ" yazılı)
- Yarışmada kullanılacak UAP ve UAİ görüntüleri örnek veri seti olarak yarışmacılarla paylaşılacaktır.
- UAP ve UAİ tespit edildikten sonra iniş durumunun da bildirilmesi zorunludur.

İniş Uygunluğu Belirleme Kuralları:
- Alan boşsa (üzerinde herhangi bir nesne yoksa): İniş Durumu = 1 (Uygun)
- Alan üzerinde taşıt, insan veya başka herhangi bir nesne varsa (nesne tespiti yapılabilsin
  ya da yapılamasın): İniş Durumu = 0 (Uygun Değil)
- İniş alanı niteliği taşımıyorsa (taşıt ve insan için): İniş Durumu = -1

İniş Durumu ID Tablosu:

| İniş Durum ID | İniş Durumu     |
|---------------|-----------------|
| 0             | Uygun Değil     |
| 1             | Uygun           |
| -1            | İniş Alanı Değil|

Kısmi Görünürlük ve Çekim Açısı Kuralları:
- UAP ve UAİ alanlarının yalnızca bir kısmının görüntü karesinde olması tespit için yeterlidir.
- Ancak iniş durumunun "Uygun (1)" olabilmesi için UAP/UAİ alanının TAMAMI kare içinde
  bulunmalıdır; kısmen kare dışında kalan alan otomatik olarak "Uygun Değil (0)" sayılır.
- UAP ve UAİ alanları üzerindeki insan ve taşıt nesneleri de ayrıca tespit edilmelidir.
- Çekim açısına bağlı olarak alana yakın cisimler alanın üstünde olmasa bile öyleymiş gibi
  görülebilmektedir (perspektif yanılsaması). Bu yanıltıcı durumda da iniş durumu "Uygun Değil (0)"
  olarak bildirilmelidir.

3.1.4. GÖREV 1 ALGORİTMA ÇALIŞMA ŞARTLARI

- Yarışmacılar sunucu ile bağlantı kurup istek gönderdiklerinde bir adet görüntü karesi alacaklardır.
- Her görüntü karesinde tespit ettikleri nesnelerin bilgisi, istenen JSON formatında sunucuya
  yollanacaktır.
- Sırası ile gönderilen video görüntülerinden herhangi birine sonuç göndermeden, sıradaki karenin
  alınması için istek gönderilemez. Bu nedenle tüm görüntü karelerinin toplu olarak indirilmesi
  mümkün değildir.
- Her görüntü karesine 1 adet sonuç yollanmalıdır; aynı kare için birden çok sonuç yollanırsa
  ilk yollanan sonuç değerlendirmeye alınacaktır.
- Bir görüntü karesi için belirlenen limit değerden fazla sonuç yollayan takımların o oturum
  içindeki sonuç gönderme kabiliyetleri belirli bir süreliğine engellenebilir. Yarışmacıların
  her görüntü karesi için gönderdikleri tahmin sayısını takip etmeleri gerekmektedir.


-----------------------------
3.2. İKİNCİ GÖREV: POZİSYON TESPİTİ
-----------------------------

İkinci görevde hava aracının konumlandırma sisteminin kullanılamaz veya güvenilemez hale
geldiği durumlar simüle edilecek ve yalnızca görüntü verileri üzerinden pozisyon kestirimi
yapılması beklenecektir. Bu yetenek, GPS arızası gibi olumsuz durumlara karşın hava aracının
görev yapabilme kabiliyetini artırmayı hedeflemektedir.

3.2.1. POZİSYON TESPİTİ DETAYLARI

- Yarışmacılar, geliştirdikleri pozisyon kestirimi algoritmaları ile verilen kamera görüntülerini
  kullanarak hava aracının referans koordinat sistemindeki pozisyonunu kestireceklerdir.
- Her oturumda yarışmacılara verilecek videonun ilk karelerine ait yer değiştirme bilgisini
  kullanarak X, Y, Z eksenlerindeki hareket yönleri belirlenebilir.
- Referans koordinat sisteminde ilk pozisyon bilgisi: x₀=0,00 [m], y₀=0,00 [m], z₀=0,00 [m]
- Pozisyon kestirimi yapılacak oturumlarda kamera açısı yeryüzüne bakacak şekilde 70–90 derece
  aralığında olacaktır.
- Hava aracının kamera parametre bilgileri yarışmacılarla paylaşılacaktır.
- Algoritma konusunda herhangi bir kısıtlama bulunmamaktadır; öğrenen modeller veya matematiksel
  temellere dayanan sistemlerin her ikisi de kullanılabilir.

Hata Metriği:
- Referans ve kestirim pozisyon bilgileri karşılaştırılarak yarışmacının hatası hesaplanacaktır.
- Bu hata miktarının büyüklüğü, yarışmacıların bu görevden aldığı puan ile doğrudan ilgilidir.

3.2.2. GÖREV 2 ALGORİTMA ÇALIŞMA ŞARTLARI

- Yarışmacılar sunucuya istek gönderdiğinde video karesinin yanı sıra aşağıdaki bilgileri de alacaklardır:

| Başlık                  | Detay                                                                              |
|-------------------------|------------------------------------------------------------------------------------|
| Video Karesi Bilgisi    | Video karesi alınırken ve sonuçlar yollanırken kullanılacak benzersiz isim         |
| Pozisyon Bilgisi – X    | İlk görüntüye göre X eksenindeki metre cinsinden yer değiştirmesi                  |
| Pozisyon Bilgisi – Y    | İlk görüntüye göre Y eksenindeki metre cinsinden yer değiştirmesi                  |
| Pozisyon Bilgisi – Z    | İlk görüntüye göre Z eksenindeki metre cinsinden yer değiştirmesi                  |
| Pozisyon Bilgisi–Sağlık | Hava aracının pozisyon tespit sisteminin sağlıklı çalışıp çalışmadığını gösteren değer |

Sağlık Değerine Göre Davranış:
- Sağlık değeri = 1: Yarışmacı, kendi geliştirdiği algoritma ile kestirdiği pozisyon bilgisini
  gönderebileceği gibi sunucudan aldığı referans değeri de değiştirmeden gönderebilir. Karar
  yarışmacıya aittir.
- Sağlık değeri = 0: Yarışmacının kendi geliştirdiği algoritma ile kestirdiği pozisyon bilgisini
  sunucuya göndermesi zorunludur.

GPS Sağlık Senaryosu:
- Her oturumda, ilk 1 dakikada (450 kare) uçan arabanın referans koordinat sistemine göre
  pozisyon bilgisi sağlıklı olarak alınacaktır (bu kesindir).
- Oturumun son 4 dakikasında (1800 kare) pozisyon bilgisi sağlıksız durumuna geçebilir.
  Bu geçişin ne zaman başlayacağı ve ne kadar süreceği önceden bilinmeyecektir.
- Yukarıdaki süre ve kare sayıları değişkenlik gösterebilir.


-----------------------------
3.3. ÜÇÜNCÜ GÖREV: GÖRÜNTÜ EŞLEME (REFERANS OBJE TESPİTİ)
-----------------------------

Üçüncü görev, hava araçlarının daha önce tanımlanmamış nesneleri görsel veri üzerinden anlık
olarak tanıma ve takip etme yeteneğini test etmektedir. Temel amaç, sistemin daha önce
karşılaşmadığı yeni nesnelere karşı adaptasyon yeteneğini ve genel nesne tanıma kabiliyetini
ölçmektir.

Görev Akışı:
- Yarışma oturumu başlangıcında belirli sayıda ve farklı zorluk seviyelerinde referans nesne
  görüntüsü paylaşılacaktır.
- Yarışmacılar, görüntü akışı esnasında verilen referanslardan tespit ettikleri nesnelerin
  koordinatlarını sonuçları ile birlikte sunucuya göndereceklerdir.
- Oturum başında verilen referans nesnelerin tamamı oturum içindeki görüntülerde mevcut
  olmayabilir; bu senaryo göz önünde bulundurularak geliştirme yapılması gerekmektedir.

Oturum Esnasında Paylaşılan Görüntülerin Özellikleri:
- Farklı bir kameradan çekilmiş olabilir. Örneğin termal kameradan çekilen nesnenin RGB kamera
  görüntüleri üzerinde eşleştirilmesi istenebilir.
- Farklı bir açıdan veya irtifadan çekilmiş olabilir.
- Uydu görüntüsü üzerinden alınmış bir nesnenin hava aracı görüntüsü üzerinde eşlenmesi
  istenebilir.
- Yer yüzeyinden çekilmiş nesneler olabilir.
- Çeşitli görüntü işleme işlemlerinden geçmiş olabilir.

Bu nedenle yarışmacılardan çeşitli koşullara dayanıklı (robust) bir eşleştirme algoritması
geliştirmeleri beklenmektedir.

Puanlama: Birinci görevdeki puan hesaplama yöntemi (mAP) kullanılacaktır. Detaylı bilgilendirme
şartname revizyonlarında verilecektir (TBD).


=========================
4. YARIŞMA YAPISI
=========================

Genel Kurallar:
- Ön Tasarım Raporu'nu teslim etmiş ve Çevrimiçi Yarışma Simülasyonu'ndan yeterli puanı
  alan takımlar TEKNOFEST 2026'da yarışmak için hak kazanacaktır.
- Yarışma alanında hem videoları hem de ikinci görev için pozisyon verilerini çekebilecekleri
  bir sunucu ve yerel ağ kurulacaktır.
- Yarışmacılar bu ağa ethernet kablosu ile bağlanacak, test video görsellerini sunucudan
  alacak ve cevaplarını sunucuya yükleyeceklerdir.
- Yerel ağın internet bağlantısı olmayacak; yarışmacıların sistemlerinin internete bağlanmasına
  kesinlikle izin verilmeyecektir.
- Bağlantıların yapılması ile ilgili teknik detaylar yarışma esnasında belirtilecek ve teknik
  ekip tarafından yarışmacılara yardımcı olunacaktır.
- Yarışmacılar, yarışma alanında kullanacakları bilgisayarlardan kendileri sorumludur;
  organizasyon tarafından bilgisayar desteği verilmeyecektir.
- Bilgisayarlarda ethernet girişi ve ethernet bağlantı kabiliyeti bulunması zorunludur.
- Her oturumda, her takımdan aynı anda 3 kişinin yarışma alanına girişine izin verilecektir.
  Takım danışmanının bulunması halinde, alanda 2 yarışmacı öğrenci ve 1 danışman alınacaktır.
- Yarışma esnasında bir takımın başka bir takıma yardımcı olmasına kesinlikle izin verilmemektedir.

-----------------------------
4.1. TEST OTURUMU
-----------------------------

- Yarışma öncesinde gerekli tüm hazırlıkların yapılabilmesi için 75 dakikalık bir test oturumu
  yapılacaktır.
- Bu oturumun amacı yarışmacıların donanım kurulumlarını gerçekleştirmeleridir.
- Yarışma şartlarını en iyi şekilde test edebilmeleri için 2 dakikalık (900 video karesi) bir
  video sunucudan yayınlanacaktır.
- Yarışmacıların bu test videosunu uygun şekilde alıp sonuçlarını uygun şekilde yolladığı
  yarışmayı düzenleyen teknik ekip tarafından test edilecek ve geri bildirim verilecektir.
- Test oturumunda yollanan sonuçların puanlandırmada etkisi olmayacaktır.

-----------------------------
4.2. YARIŞMA OTURUMLARI
-----------------------------

- 4 yarışma oturumu yapılacaktır.
- Her oturumun toplam süresi 75 dakika olacaktır:
  - İlk 15 dakika: hazırlık süresi
  - Sonraki 60 dakika: yarışma süresi
- Her oturumda 2250 video karesi verilecek ve sonuçlar uygun formatta sunucuya gönderilecektir.
- Her bir oturumun bir teması bulunacaktır. Örnek temalar: "Güneşli", "Zorlu Hava Şartları",
  "Akşam", "Deniz Üstü". Oturum temaları yarışma gününe kadar saklı tutulacaktır.
- Oturum süreleri ve kullanılan video uzunlukları değişebilmektedir; değişiklik olması halinde
  yarışmacılar önceden bilgilendirilecektir.


=========================
5. TEKNİK SUNUM
=========================

- Yarışmacı takımlardan yarışma oturumları esnasında bir sunum yapmaları beklenmektedir.
- Her takımdan bir adet İletişim Sorumlusu sunumu yapmakla görevlendirilmelidir.
- Sunum süresi takım başına en fazla 5 dakikadır.
- Her yarışma oturumunun başında, o oturumda sunum yapacak takımlar duyurulacaktır.
- Sunumlar, 3 kişiden oluşan bir hakem heyetine sunulacaktır.
- Sunum sırasında veya sonrasında takım temsilcisine sorular yöneltilebilir.
- Hazırlanan sunumlar, TEKNOFEST Yarışmalar Komitesi tarafından iletilecek tarihe kadar
  t3kys.com adresine yollanarak teslim edilmelidir.
- Örnek sunum şablonu Haziran 2026 tarihine kadar katılımcı takımlarla paylaşılacaktır.
- Sunum şablonu bozulmamak kaydıyla sunum içeriği konusunda bir kısıtlama bulunmamaktadır.
- Sunumda değerlendirilebilecek örnek konu başlıkları (zorunlu değildir):
  - Ek Veri Toplama Süreci
  - Kullanılan Algoritma
  - Alternatif Algoritmalar
  - Test Sonuçları
  - Yenilikçi Yaklaşım
- Yarışma sunumu genel yarışma puanının %5'ini oluşturmaktadır.


=========================
6. RAPORLAMA
=========================

Yarışmacı takımlardan iki ayrı doküman yazmaları beklenmektedir. Her iki raporun da teslim
edilmesi zorunludur.

-----------------------------
6.1. ÖN TASARIM RAPORU
-----------------------------

- Rapor şablonu en geç 22/04/2026 tarihinde teknofest.org internet sitesinde paylaşılacaktır.
- Raporda konu ile ilgili yapılan araştırma ve problemlerin çözümüne yönelik çözüm önerileri
  yer alacaktır.
- Rapor şablonu bozulmamak kaydıyla içerik veya uzunluk konusunda bir kısıtlama yoktur.
- Yarışmaya katılım için Ön Tasarım Raporu'nun teslim edilmesi ZORUNLUDUR.
- Ön Tasarım Raporu'nun yarışma sonuçlarının belirlenmesinde ve ödüllendirmede herhangi bir
  etkisi bulunmamaktadır.
- Teslim tarihi: En geç 22/04/2026 – t3kys.com adresine yollanarak teslim edilmelidir.

-----------------------------
6.2. FİNAL TASARIM RAPORU
-----------------------------

- Rapor şablonu en geç Ağustos 2026 tarihinde teknofest.org internet sitesinde paylaşılacaktır.
- Raporda yarışmaya hazırlık sürecindeki literatür çalışmaları, yarışma esnasında kullanılan
  algoritmalar, yapılan testler ve diğer teknik bilgiler yer alacaktır.
- Raporun değerlendirmesinde rapor içeriği ve rapor formatı etkili olacaktır.
- Final Tasarım Raporu puanı genel yarışma puanının %5'ini oluşturmaktadır.
- Dereceye girebilmek için Final Tasarım Raporu'nun teslim edilmesi ZORUNLUDUR.
- Rapor şablonu bozulmamak kaydıyla içerik veya uzunluk konusunda bir kısıtlama yoktur.
- Teslim tarihi: TEKNOFEST Yarışmalar Komitesi tarafından iletilecek tarihte t3kys.com
  adresine yollanarak teslim edilmelidir.


=========================
7. ÇEVRİMİÇİ YARIŞMA SİMÜLASYONU
=========================

- Ön Tasarım Raporu değerlendirmelerinden sonra, yarışma alanına gelecek takımların
  belirlenebilmesi için bir ön eleme yarışması yapılacaktır.
- Çevrimiçi Yarışma Simülasyonu'nda yarışmacılardan, geliştirdikleri modeller ile çevrimiçi
  ortamda paylaşılacak karelerdeki nesneleri tespit etmeleri ve hava aracının pozisyonunu
  kestirmeleri beklenmektedir.
- Çevrimiçi Yarışma Simülasyonu, birinci yarışma oturumu ile aynı kurallar ve tema ile
  yapılacaktır.
- Mayıs 2026 ayı içerisinde Çevrimiçi Yarışma Simülasyonu ile ilgili ayrıntılı bilgi içeren
  doküman yarışmacılar ile paylaşılacaktır.
- Duyurulacak başarı kriterinin altında kalan ve sunucuya hiç bağlanmayan takımlar bir sonraki
  aşamaya geçemeyecektir.


=========================
8. TAKIM YAZILIM VE DONANIM ÖZELLİKLERİ
=========================

- Her takım kendi yazılım ve donanım sisteminden sorumludur; yarışma alanında herhangi bir
  yazılım ya da donanım (bilgisayar, mouse vb.) desteği sunulmayacaktır.
- İhtiyaç duyulacak her donanım (adaptör, mouse, klavye, ethernet kablosu vb.) ve yazılıma
  sahip olarak yarışmaya katılım sağlanmalıdır.
- Herhangi bir işletim sistemi kullanılabilir.
- Takımlar istedikleri platformda ve programlama dilinde geliştirme yapabilir.
- Yarışmacılardan saniyede en az 1 görüntü karesi işleyebilecek donanıma sahip olmaları
  yeterli olacaktır.
- Algoritmanın çalışma HIZI bir puanlandırma kriteri değildir; donanımların güçlü veya zayıf
  olması yarışmanın seyrine etki etmeyecek şekilde platform tasarlanacaktır.
- Bilgisayarlarda ethernet girişi ve ethernet bağlantı kabiliyeti bulunması ZORUNLUDUR.


=========================
9. YARIŞMA SIRASINDA SUNUCU İLE BAĞLANTI
=========================

- Yarışma sırasında takımlara, yarışma sunucusunun da içinde bulunduğu yerel ağa bağlanabilmeleri
  için bir ethernet kablosu sağlanacaktır.
- Her takım bu ethernet kablosu aracılığıyla yarışma ağına yalnızca tek bir IP adresi ile
  bağlanmalıdır.
- Yarışma sırasında takımlara birer IP adresi belirtilecek; sisteme yalnızca belirtilen IP adresleri
  üzerinden bağlantıya izin verilecektir.
- Yarışma sunucusunun adresi yarışma günü belirlenecektir. Format örneği: http://127.0.0.25:5000
- Sunucu ile yapılacak tüm haberleşmeler API mantığıyla JSON formatında gerçekleşecektir.
- Yarışma anında kullanılacak API adres bilgileri test oturumu öncesinde yarışmacılarla
  paylaşılacaktır.
- ÖNEMLİ KISIT: Geliştirilecek sistemin altyapısı, doğrudan IP ve portların (örn. "localhost") kod içine statik (hardcoded) gömülmesine bağımlı olmamalıdır. Sistem her zaman bir lokalde test (örn. localhost) bir de yarışma anında gerçek sunucu IP adresiyle (örn. 192.168.1.25) çalışacak şekilde dinamik konfigüre edilebilir olmalıdır.

-----------------------------
9.1. SUNUCUDAN ALINAN VERİ LİSTESİ (JSON FRAME LİSTESİ)
-----------------------------

Yarışmacılar, yarışma başladıktan sonra sunucudan 7,5 fps ile kaydedilmiş görüntü listesini
aşağıdaki JSON formatında alacaklardır:

Alan Adları:
- url           : Video karesi ID'sinin benzersiz URL'i
- image_url     : Video karesi görselinin bulunduğu URL
- video_name    : Video karesinin alındığı videonun adı ya da numarası
- session       : Oturumu belirten URL
- translation_x : Hava aracının ilk görüntüye göre X eksenindeki metre cinsinden yer değiştirmesi
- translation_y : Hava aracının ilk görüntüye göre Y eksenindeki metre cinsinden yer değiştirmesi
- translation_z : Hava aracının ilk görüntüye göre Z eksenindeki metre cinsinden yer değiştirmesi
- gps_health_status : Hava aracının pozisyon tespit sisteminin sağlıklı çalışıp çalışmadığını
                      gösteren değer (1 = sağlıklı, 0 = sağlıksız)

Örnek Frame Listesi JSON:
```json
[
  {
    "url": "http://localhost/frames/3598/",
    "image_url": "/ljfgpemcvkmuadhxabwn_V2_1/frame_000000.jpg",
    "video_name": "ljfgpemcvkmuadhxabwn_V2_1",
    "session": "http://localhost/session/2/",
    "translation_x": 0.02,
    "translation_y": 0.01,
    "translation_z": 0.03,
    "gps_health_status": 1
  },
  {
    "url": "http://localhost/frames/4787/",
    "image_url": "/ljfgpemcvkmuadhxabwn_V2_1/frame_000004.jpg",
    "video_name": "ljfgpemcvkmuadhxabwn_V2_1",
    "session": "http://localhost/session/2/",
    "translation_x": 0.01,
    "translation_y": 0.02,
    "translation_z": 0.01,
    "gps_health_status": 1
  },
  {
    "url": "http://localhost/frames/3916/",
    "image_url": "/ljfgpemcvkmuadhxabwn_V2_1/frame_000008.jpg",
    "video_name": "ljfgpemcvkmuadhxabwn_V2_1",
    "session": "http://localhost/session/2/",
    "translation_x": "NaN",
    "translation_y": "NaN",
    "translation_z": "NaN",
    "gps_health_status": 0
  }
]
```
NOT: translation değerlerinin "NaN" olması, GPS sağlığının 0 (sağlıksız) olduğu duruma örnektir.

-----------------------------
9.2. SUNUCUYA GÖNDERİLEN SONUÇ JSON YAPISI
-----------------------------

Takımlar, bir görüntüyü işlemeyi bitirdikten sonra tespit ettikleri nesneleri ve hava aracı
pozisyonunu sunucuya bildirmelidir. Sunucuya bildirilmeyen cevaplar geçersiz sayılacaktır.
Sonuçlar her bir görüntü karesi için AYRI AYRI gönderilmelidir.

Üst Düzey Alanlar:
- id     : Gönderilen tahminin ID'si
- user   : Kullanıcının bilgilerini içeren URL
- frame  : Video karesi ID'sinin benzersiz URL'i

Alt Diziler:

1) detected_objects (Tespit edilen bilinen nesneler dizisi):
   - cls            : Nesnenin sınıfı ("0"=Taşıt, "1"=İnsan, "2"=UAP, "3"=UAİ)
   - landing_status : İniş durumu ("-1", "0", "1")
   - motion_status  : Hareket durumu ("-1", "0", "1")
   - top_left_x     : Bounding box sol üst köşesinin resmin sol kenarına piksel uzaklığı
   - top_left_y     : Bounding box sol üst köşesinin resmin üst kenarına piksel uzaklığı
   - bottom_right_x : Bounding box sağ alt köşesinin resmin sol kenarına piksel uzaklığı
   - bottom_right_y : Bounding box sağ alt köşesinin resmin üst kenarına piksel uzaklığı

2) detected_translations (Pozisyon kestirimi dizisi):
   - translation_x  : İlk görüntüye göre X eksenindeki metre cinsinden yer değiştirme kestirimi
   - translation_y  : İlk görüntüye göre Y eksenindeki metre cinsinden yer değiştirme kestirimi
   - translation_z  : İlk görüntüye göre Z eksenindeki metre cinsinden yer değiştirme kestirimi

3) detected_undefined_objects (Tespit edilen tanımsız/referans nesneler dizisi – Görev 3):
   - object_id      : Tespit edilen nesnenin ID'si
   - top_left_x     : Bounding box sol üst köşesinin resmin sol kenarına piksel uzaklığı
   - top_left_y     : Bounding box sol üst köşesinin resmin üst kenarına piksel uzaklığı
   - bottom_right_x : Bounding box sağ alt köşesinin resmin sol kenarına piksel uzaklığı
   - bottom_right_y : Bounding box sağ alt köşesinin resmin üst kenarına piksel uzaklığı

Örnek Sonuç JSON:
```json
[
  {
    "id": 22246,
    "user": "http://localhost/users/4/",
    "frame": "http://localhost/frames/4000/",
    "detected_objects": [
      {
        "cls": "1",
        "landing_status": "-1",
        "motion_status": "-1",
        "top_left_x": 262.87,
        "top_left_y": 734.47,
        "bottom_right_x": 405.2,
        "bottom_right_y": 847.3
      }
    ],
    "detected_translations": [
      {
        "translation_x": 0.02,
        "translation_y": 0.01,
        "translation_z": 0.03
      }
    ],
    "detected_undefined_objects": [
      {
        "object_id": 1,
        "top_left_x": 262.87,
        "top_left_y": 734.47,
        "bottom_right_x": 405.2,
        "bottom_right_y": 847.3
      }
    ]
  }
]
```

ÖNEMLİ NOT: Şekil 16 ve Şekil 17'de ifade edilen JSON formatı yarışmada kullanılacak olan
formatın TASLAK halidir. Kesin ve güncel format yarışmacılarla ayrıca paylaşılacaktır.


=========================
10. PUANLAMA
=========================

-----------------------------
10.0. GENEL PUAN DAĞILIMI
-----------------------------

| Puan Türü            | Puan Oranı |
|----------------------|-----------|
| Birinci Görev        | %25       |
| İkinci Görev         | %40       |
| Üçüncü Görev         | %25       |
| Final Tasarım Raporu | %5        |
| Yarışma Sunumu       | %5        |
| TOPLAM               | %100      |

-----------------------------
10.1. BİRİNCİ GÖREV PUANLAMA KRİTERİ
-----------------------------

Metrik: mAP (Mean Average Precision – Ortalama Kesinlik Değerlerinin Ortalaması)

mAP, IoU (Intersection over Union – Kesişimin Birleşime Oranı) değeri üzerinden hesaplanır.

IoU Formülü (Denklem 1):
  IoU = (GerçekReferansDörtgen ∩ TahminEdilenDörtgen) / (GerçekReferansDörtgen ∪ TahminEdilenDörtgen)

IoU eşik değeri: 0,5
  - IoU ≥ 0,5 → Doğru tespit (True Positive) olarak sayılır; AP değerini artırır.
  - IoU < 0,5 → Hatalı tespit (False Positive) olarak sayılır; AP değerini düşürür.

Ek Etkiler:
  - İniş durumunun yanlış tespit edilmesi, ilgili sınıfın AP değerini olumsuz etkiler.
  - Hareket durumunun yanlış tespit edilmesi, ilgili sınıfın AP değerini olumsuz etkiler.
  - Aynı nesne için birden fazla bounding box gönderilmesi durumunda en yüksek IoU'ya sahip
    olan tespit doğru sayılırken diğerleri false positive olarak işlenir ve AP'yi düşürür.

10.1.1. BİRİNCİ GÖREV ÖRNEK PUANLAMA DURUMLARI

Örnek 1 – Doğru Sınıf, Yeterli IoU, Doğru İniş Değeri:
| Alan                              | Değer  |
|-----------------------------------|--------|
| Gerçek Sınıf                      | İnsan  |
| Tespit Edilen Sınıf               | İnsan  |
| Gönderilen Dörtgen Sayısı         | 1      |
| Tespit Edilen Alanların IoU Değeri| 0,63   |
| Tespit Edilen İniş Değeri         | -1     |
| Gerçek İniş Değeri                | -1     |
Sonuç: İnsan sınıfı için AP değerini ARTTIRACAK şekilde puan alır.

Örnek 2 – Yanlış Sınıf, Yeterli IoU:
| Alan                              | Değer  |
|-----------------------------------|--------|
| Gerçek Sınıf                      | İnsan  |
| Tespit Edilen Sınıf               | Taşıt  |
| Gönderilen Dörtgen Sayısı         | 1      |
| Tespit Edilen Alanların IoU Değeri| 0,66   |
| Tespit Edilen İniş Değeri         | -1     |
| Gerçek İniş Değeri                | -1     |
Sonuç: İnsan sınıfı için AP değerini DÜŞÜRECEK şekilde puan alır
(doğru sınıf olarak raporlanmadığından false positive).

Örnek 3 – Doğru Sınıf, Yetersiz IoU (< 0,5):
| Alan                              | Değer  |
|-----------------------------------|--------|
| Gerçek Sınıf                      | İnsan  |
| Tespit Edilen Sınıf               | İnsan  |
| Gönderilen Dörtgen Sayısı         | 1      |
| Tespit Edilen Alanların IoU Değeri| 0,42   |
| Tespit Edilen İniş Değeri         | -1     |
| Gerçek İniş Değeri                | -1     |
Sonuç: Sınıf ve iniş değerleri doğru olmasına rağmen IoU eşik değerinin (0,5) altında
olduğundan insan sınıfı için AP değerini DÜŞÜRECEK şekilde puan alır.

Örnek 4 – Birden Fazla Tespit Gönderilmesi:
| Alan                              | Değer            |
|-----------------------------------|------------------|
| Gerçek Sınıf                      | Taşıt            |
| Tespit Edilen Sınıf               | Taşıt            |
| Gönderilen Dörtgen Sayısı         | 3                |
| Tespit Edilen Alanların IoU Değeri| 0,85 / 0,61 / 0,54 |
| Tespit Edilen İniş Değerleri      | -1 / -1 / -1     |
| Gerçek İniş Değeri                | -1               |
Sonuç: En yüksek IoU (0,85) için AP artırıcı puan alınır; 0,61 ve 0,54 için AP düşürücü
puan alınır. Toplamda taşıt sınıfı için AP DÜŞECEK şekilde etkilenir.

Örnek 5 – UAP, Doğru Tespit Fakat Yanlış İniş Durumu:
| Alan                              | Değer       |
|-----------------------------------|-------------|
| Gerçek Sınıf                      | UAP         |
| Tespit Edilen Sınıf               | UAP         |
| Gönderilen Dörtgen Sayısı         | 1           |
| Tespit Edilen Alanların IoU Değeri| 0,91        |
| Tespit Edilen İniş Değeri         | 1 (Uygun)   |
| Gerçek İniş Değeri                | 0 (Uygun Değil) |
Sonuç: İnişe uygun olmayan alanı uygun olarak tespit ettiğinden UAP sınıfı için AP
değerini DÜŞÜRECEK şekilde puan alır.

Örnek 6 – Tespit Edilemeyen Nesne (Kaçırma):
| Alan                              | Değer      |
|-----------------------------------|------------|
| Gerçek Sınıf                      | Taşıt      |
| Tespit Edilen Sınıf               | Tespit yok |
| Gönderilen Dörtgen Sayısı         | 0          |
| Tespit Edilen Alanların IoU Değeri| –          |
| Tespit Edilen İniş Değerleri      | –          |
| Gerçek İniş Değeri                | -1         |
Sonuç: Tespit edilmesi gereken taşıt nesnesini tespit edemediğinden taşıt sınıfı için
AP değerini DÜŞÜRECEK şekilde puan alır (false negative).

-----------------------------
10.2. İKİNCİ GÖREV PUANLAMA KRİTERİ
-----------------------------

Hata Metriği: Ortalama 3D Öklid Hatası (Denklem 2)

  Ortalama Yarışmacı Hatası = E = (1/N) * Σᵢ √[(x̂ᵢ−xᵢ)² + (ŷᵢ−yᵢ)² + (ẑᵢ−zᵢ)²]

  Burada:
  - x̂ᵢ, ŷᵢ, ẑᵢ : Yarışmacının i. görsel için gönderdiği pozisyon kestirimi
  - xᵢ, yᵢ, zᵢ  : Hava aracının mutlak doğru pozisyon bilgisi
  - N           : Toplam kare sayısı

- Her oturum sonunda tüm takımların hata miktarları karşılaştırılarak yarışmacı takımın
  göreli puanı hesaplanmaktadır.

-----------------------------
10.3. ÜÇÜNCÜ GÖREV PUANLAMA KRİTERİ
-----------------------------

Puanlama metodu olarak Birinci Görev'deki mAP yöntemi kullanılacaktır.
Detaylı bilgilendirme şartname revizyonlarında verilecektir (TBD).

-----------------------------
10.4. OTURUM SONU PUANLAMA VE SIRALAMALAR
-----------------------------

- Her yarışma oturumu başlarken, yarışmacıların bir önceki oturumdan almış oldukları puana
  göre güncel sıralamaları bildirilecektir.
- Tüm oturumların tamamlanmasının ardından, Bölüm 10.0'da belirtilen oranlar kullanılarak
  nihai yarışma puanı hesaplanacaktır.
- Yarışmada dereceye giren takımlar TEKNOFEST'in son gününde kürsüye çıkarak ödüllerini
  alacaklardır.


=========================
11. DONANIM VE ORTAM KISITLARI
=========================

| Kısıt Kodu | Kısıt Tanımı                                                                      |
|------------|-----------------------------------------------------------------------------------|
| HW-001     | Takımlar yazılımlarını kendi bilgisayarlarında çalıştırır; organizasyon bilgisayar sağlamaz. |
| HW-002     | Bilgisayarda ethernet bağlantı girişi ve ethernet bağlantı kabiliyeti zorunludur. |
| HW-003     | Ekran, klavye, fare, adaptör gibi çevre birimleri takım tarafından temin edilir.  |
| HW-004     | Yarışma sırasında bilgisayarların internete bağlanması kesinlikle yasaktır.        |
| HW-005     | Tüm algoritmalar yalnızca yarışmacı donanımında çalışacak; çevrimiçi hizmet yasaktır. |
| HW-006     | Yerel ağın internet bağlantısı bulunmayacak; yarışmacı sistemlerinin internete bağlanmasına izin verilmeyecektir. |
| HW-007     | Minimum performans gereksinimi: saniyede 1 görüntü karesi işlenebilecek kapasitede donanım yeterlidir. |
| HW-008     | Herhangi bir işletim sistemi ve programlama dili kullanılabilir.                  |
| HW-009     | Yazılım kaynak kodu değiştirilmeden, konfigürasyon değişkenleri veya parametreleri aracılığıyla lokal test modundan yarışma moduna geçiş yapabilmelidir. |


=========================
12. FONKSİYONEL GEREKSİNİMLER (ÖZETLENMİŞ REFERANS LİSTESİ)
=========================

-- GÖREV 1 --

- [FR-001] Hava aracı kamera görüntülerinden Taşıt (ID:0), İnsan (ID:1), UAP Alanı (ID:2) ve
           UAİ Alanı (ID:3) sınıflarına ait nesneler tespit edilmelidir.
- [FR-002] Taşıt sınıfı; motorlu kara araçları, raylı taşıtlar ve deniz taşıtlarını kapsar.
           Tüm alt türler "Taşıt (0)" olarak etiketlenir.
- [FR-003] İnsan tespiti; ayakta veya oturan, kısmen görünen tüm insanları kapsar.
           Bisiklet/motosiklet sürücüsü "taşıt" kapsamındadır.
- [FR-004] UAP/UAİ alanlarında iniş uygunluğu bildirilmelidir; yabancı nesne veya kısmi
           görünürlük durumunda "Uygun Değil (0)" olarak işaretlenmelidir.
- [FR-005] Taşıtlar için hareket durumu (0=hareketsiz, 1=hareketli) bildirilmelidir.
           Kamera hareketi ile gerçek taşıt hareketi ayırt edilebilmelidir.
- [FR-006] Gece/gündüz, yağmur/kar/sis gibi farklı ortam koşullarında çalışabilirlik beklenir.
- [FR-007] Buğulanma, ölü piksel, donma gibi kamera kaynaklı bozulmalara toleranslı olunmalıdır.
- [FR-008] Her karede tespit sonuçları JSON formatında sunucuya iletilmelidir; kare atlanamaz.

-- GÖREV 2 --

- [FR-009] Alt-görüş kamerası görüntülerinden GPS bağımsız pozisyon kestirimi yapılmalıdır.
- [FR-010] Çıktı: video başlangıcına göre X, Y, Z eksenleri metre cinsinden yer değiştirme.
- [FR-011] GPS sağlık değeri 0 olduğunda kendi algoritması ile üretilen değer gönderilmelidir.
- [FR-012] GPS sağlık değeri 1 olduğunda algoritma çıktısı veya sunucu referans değeri
           gönderilebilir (yarışmacının kararına bırakılmıştır).
- [FR-013] Kamera parametreleri yarışma öncesinde paylaşılacaktır.

-- GÖREV 3 --

- [FR-014] Oturum başlangıcında anlık bildirilen referans obje, video karelerinde tespit
           edilmelidir; nesne tanımlanmamış ve yeni olabilir.
- [FR-015] Tespit edilen referans nesnenin bounding box koordinatları ve object_id değeri
           JSON formatında sunucuya bildirilmelidir.
- [FR-016] Referans obje, görüntü akışının tamamında mevcut olmayabilir; bu durum
           algoritmada göz önünde bulundurulmalıdır.

-- GENEL AKIŞ --

- [FR-017] Test oturumu: 75 dk; yarışma öncesi donanım kurulumu ve 900 kareli test videosu.
- [FR-018] Yarışma oturumları: 4 oturum × (15 dk hazırlık + 60 dk yarışma) = 4 × 75 dk.
- [FR-019] Her oturumda 2250 kare işlenecek ve sunucuya yüklenecektir.
- [FR-020] Süre dışında sunucuya yüklenemeyen sonuçlar geçersiz sayılacaktır.
- [FR-021] Teknik sunum: oturum sırasında İletişim Sorumlusu tarafından ≤5 dk yapılacaktır.


=========================
13. YARIŞMA GİTHUB VE GOOGLE GROUPS SAYFALARI
=========================

Yarışma kapsamında iki dijital destek platformu oluşturulmuştur:

GitHub Proje Deposu:
  - Yarışma boyunca kullanılacak kod blokları, örnek veri setleri ve teknik materyaller
    GitHub üzerinden paylaşılacaktır.
  - Katılımcılar bu platform üzerinden gerekli dokümanlara erişebilir, kodlar üzerinde
    inceleme yapabilir ve kendi çalışmalarına entegre edebilirler.
  - GitHub sayfası, teknik materyallere kolay erişim sağlamak amacıyla sürekli güncel
    tutulacaktır.
  - (URL: Şartname revizyonunda açıklanacaktır – TBD)

Google Groups Platformu:
  - Takımlar arasındaki bilgi alışverişini kolaylaştırmak ve organizasyon ekibine soru
    iletebilmek amacıyla bir Google Groups tartışma platformu oluşturulmuştur.
  - Bu grup üzerinden yarışmaya dair önemli duyurular yapılacak, sıkça sorulan sorular
    yanıtlanacak ve teknik destek sağlanacaktır.
  - (URL: Şartname revizyonunda açıklanacaktır – TBD)


=========================
14. ÖNEMLİ TARİHLER (REFERANS)
=========================

| Aşama                                     | Tarih / Dönem               |
|-------------------------------------------|-----------------------------|
| Teknik Şartname İlanı                     | 21.02.2026                  |
| Son Başvuru Tarihi                        | 28.02.2026                  |
| Örnek Video Teslimi (Görev 2 için)        | 10.03.2026 – 28.03.2026     |
| Ön Tasarım Raporu Şablonu Yayını          | En geç 22.04.2026           |
| Ön Tasarım Raporu Teslimi                 | En geç 22.04.2026 (t3kys.com) |
| ÇYS Detay Duyurusu                        | Mayıs 2026                  |
| Çevrimiçi Yarışma Simülasyonu             | 09.07.2026 (tahmini)        |
| Yarışma Sunumu Şablonu Paylaşımı          | Haziran 2026                |
| Final Tasarım Raporu Şablonu Yayını       | En geç Ağustos 2026         |
| Final Tasarım Raporu Teslimi              | TEKNOFEST Komitesi duyurusuna göre |
| Final (TEKNOFEST)                         | Ağustos–Eylül 2026          |
| TEKNOFEST 2026                            | 30 Eylül – 4 Ekim 2026      |


=========================
15. VERİ ŞEMALARI – HIZLI REFERANS
=========================

SINIF ID TABLOSU:
| Sınıf     | Sınıf ID | İniş Durumu (alabileceği değerler) | Hareket Durumu (alabileceği değerler) |
|-----------|----------|------------------------------------|---------------------------------------|
| Taşıt     | 0        | -1                                 | 0, 1                                  |
| İnsan     | 1        | -1                                 | -1                                    |
| UAP Alanı | 2        | 0, 1                               | -1                                    |
| UAİ Alanı | 3        | 0, 1                               | -1                                    |

İNİŞ DURUMU ID TABLOSU:
| İniş Durum ID | İniş Durumu      |
|---------------|------------------|
| 0             | Uygun Değil      |
| 1             | Uygun            |
| -1            | İniş Alanı Değil |

HAREKET DURUMU ID TABLOSU:
| Hareket Durum ID | Hareket Durumu |
|------------------|----------------|
| 0                | Hareketsiz     |
| 1                | Hareketli      |


=========================
16. ÇELİŞKİLER / TUTARSIZLIKLAR
=========================

- Teknik şartnamede Sınıf ID tablosunda (Tablo 4) UAP alanı için Şekil 6'ya referans verilmiş
  olup doğru referans Şekil 7 olmalıdır. UAİ alanı için ise Şekil 7'ye referans verilmiş olup
  doğru referans Şekil 8 olmalıdır. Bu görsel referanslar yanlış etiketlenmiştir; içerik tutarlıdır.
- Şekil 16 ve Şekil 17'deki JSON format örneklerinin yarışmada kullanılacak formatın taslak
  versiyonu olduğu belirtilmekte olup kesin format ayrıca paylaşılacaktır. Yarışmacılar bu
  formatlara bağlı kalmayıp güncel açıklamayı takip etmelidir.
- Oturum başına kare sayısı nominal 2250 olarak belirtilmekle birlikte "değişkenlik gösterebilir"
  ibaresi de eklenmiştir; algoritmalar sabit kare sayısına bağımlı olmamalıdır.


=========================
17. AÇIK SORULAR (TBD)
=========================

| Kod     | Açık Soru / Eksik Bilgi                                                                      |
|---------|----------------------------------------------------------------------------------------------|
| TBD-001 | Görev 1, 2 ve 3 için kesin JSON çıktı şemaları (alan adları, veri tipleri) ayrıca duyurulacaktır. |
| TBD-002 | Giriş video formatı (codec, çözünürlük, kesin FPS) ayrıca duyurulacaktır.                   |
| TBD-003 | Görev 3 puanlama detaylarının tamamı şartname revizyonlarında verilecektir.                  |
| TBD-004 | Çevrimiçi Yarışma Simülasyonu için geçiş notu / minimum başarı eşiği Mayıs 2026'da açıklanacaktır. |
| TBD-005 | İkinci görev için pozisyon kestirimi puanlandırma eşikleri (sıralama yöntemi) belirtilmemiştir. |
| TBD-006 | GitHub ve Google Groups URL bağlantıları henüz paylaşılmamıştır.                             |
| TBD-007 | Yarışma oturumlarının kesin tarih ve sıralaması ayrıca duyurulacaktır.                       |
| TBD-008 | Final Tasarım Raporu teslim tarihi TEKNOFEST Komitesi tarafından ayrıca bildirilecektir.     |
| TBD-009 | Yarışma sunumu için şablon Haziran 2026'da paylaşılacaktır.                                  |
| TBD-010 | Kamera parametre bilgileri (odak uzaklığı, lens bozulma katsayıları vb.) yarışma öncesinde paylaşılacaktır. |
