# LLM Alternative - Eger Okuyamazsan Sana Yaziyorum

- Source PDF: `2026_HAVACILIKTA_YAPAY_ZEKA_TEKNIK_SARTNAME_TR_v1_2026_02_21_W48vr.pdf`
- Purpose: This file is a plain-text fallback for LLM/text-based processing.
- Generated on: 2026-02-23

## Extracted Text

HAVACILIKTA YAPAY




       1
İÇİNDEKİLER
İÇİNDEKİLER................................................................................................................................................................ 2
TABLOLAR ................................................................................................................................................................... 3
ŞEKİLLER ..................................................................................................................................................................... 4
VERSİYONLAR ......................................................................................................................................................... 5
TANIMLAR VE KISALTMALAR DİZİNİ ........................................................................................................................... 6
1.     GİRİŞ .................................................................................................................................................................... 7
2.     GÖREVLER .......................................................................................................................................................... 7
2.1.        Birinci Görev: Nesne Tespiti ............................................................................................................................ 7
       2.1.1.       Taşıt ve İnsan Tespiti .............................................................................................................................. 9
       2.1.2.       UAP ve UAI Tespiti ................................................................................................................................ 13
       2.1.3.       Algoritma Çalışma Şartları ................................................................................................................... 15
2.2.        İkinci Görev: Pozisyon Tespiti ........................................................................................................................ 16
       2.2.1.       Pozisyon Tespiti ................................................................................................................................... 16
       2.2.2.       Algoritma Çalışma Şartları ................................................................................................................... 17
2.3.        Üçüncü Görev: Görüntü Eşleme ................................................................................................................... 18
3.     YARIŞMA ............................................................................................................................................................ 20
3.1.        Test Oturumu ................................................................................................................................................ 21
3.2.        Yarışma Oturumları ....................................................................................................................................... 21
4.     TEKNİK SUNUM ................................................................................................................................................. 21
5.     RAPORLAMA...................................................................................................................................................... 22
5.1.        Ön Tasarım Raporu ....................................................................................................................................... 22
5.2.        Final Tasarım Raporu..................................................................................................................................... 22
6.     ÇEVRİMİÇİ YARIŞMA SİMÜLASYONU ............................................................................................................... 23
7.     TAKIMLARIN YAZILIM VE DONANIM ÖZELLİKLERİ ........................................................................................... 23
8.     YARIŞMA SIRASINDA SUNUCU İLE BAĞLANTI ................................................................................................. 24
9.     PUANLAMA ........................................................................................................................................................ 28
9.1.        Birinci Görev Puanlama Kriteri ...................................................................................................................... 28
       9.1.1.       Birinci Görev Örnek Puanlama Durumları ............................................................................................ 29
9.2.        İkinci Görev Puanlama Kriteri ........................................................................................................................ 31
9.3.        Üçüncü Görev Puanlama Kriteri .................................................................................................................... 31
10.         YARIŞMA GİTHUB ve GOOGLE GROUPS SAYFALARI................................................................................... 32
11.         YARIŞMA SONUÇLARININ DUYURULMASI VE ÖDÜLLENDİRME ................................................................ 32




                                                                                      2
TABLOLAR
Tablo 1: Versiyonlar Tablosu............................................................................................... 5
Tablo 2: Taşıt ve İnsan Sınıflarını İçeren Tablo .................................................................... 10
Tablo 3: Hareket Durumu Değerleri ................................................................................... 10
Tablo 4: UAP ve UAİ Sınıf Bilgilerini İçeren Tablo ................................................................. 14
Tablo 5: İniş Durumu Değerleri ......................................................................................... 14
Tablo 6: Yarışmacıların Sunucu İsteği Sonrasında Alacağı Bilgiler ......................................... 18
Tablo 7: Genel Yarışma Puanlandırması ............................................................................ 28
Tablo 8: Örnek 1 Tablo .................................................................................................... 29
Tablo 9: Örnek 2 Tablo .................................................................................................... 29
Tablo 10: Örnek 3 Tablo .................................................................................................. 30
Tablo 11: Örnek 4 Tablo .................................................................................................. 30
Tablo 12: Örnek 5 Tablo .................................................................................................. 30
Tablo 13: Örnek 6 Tablo .................................................................................................. 31




                                                                  3
ŞEKİLLER
Şekil 1: Çekim Açısı Durumları (Uygun Olmayan Çekim Açısı) .......................................... 8
Şekil 2: Çekim Açısı Durumları (Uygun Olmayan Çekim Açısı) .......................................... 8
Şekil 3: Örnek Görüntü Bozulması ................................................................................. 9
Şekil 4: Nesnelerin Tamamının Görüntü Karesi İçinde Olmaması Durumunda Etiketleme
Yönergesi ................................................................................................................. 11
Şekil 5: Nesnelerin Görüntü Karesi İçinde Bulunma Durumları ........................................ 11
Şekil 6: Ardışık İki Karede Hareketli ve Hareketsiz Taşıt Tespiti ............................ 12
Şekil 7: UAP Alan Bilgileri ........................................................................................... 13
Şekil 8: UAİ Alan Bilgileri ............................................................................................ 13
Şekil 9: Nesnenin Tamamının Görüntü Karesi İçinde Olmaması Durumu.......................... 14
Şekil 10: UAP ve UAİ Alanlarının Üzerinde Herhangi Bir Cisim Olması Durumu ................. 15
Şekil 11: Alanların Yanında Cisim Bulunması Durumu ................................................... 15
Şekil 12: Referans ve Kestirim Pozisyon Bilgisi Örneği ................................................... 17
Şekil 13: Referans ve Kestirim Pozisyon Bilgileri Kullanılarak Hesaplanan Hata ................ 17
Şekil 14: Referans Obje Eşleme Örnek Görseli ............................................................. 19
Şekil 15: Sonuç Paketleri Oluşturma Diyagramı ............................................................ 20
Şekil 16: Görsel Bilgileri ............................................................................................. 25
Şekil 17: JSON Formatı .............................................................................................. 27
Şekil 18: IoU Formül Gösterimi ................................................................................... 28




                                                                   4
VERSİYONLAR
Tablo 1: Versiyonlar Tablosu

          VERSİYON               TARİH               Açıklama
             V1.0              21.02.2026    TEKNOFEST 2026 İlk Versiyon




                                         5
TANIMLAR VE KISALTMALAR DİZİNİ

İşbu şartnamede belirtilen;

KYS: TEKNOFEST Kurumsal Yönetim Sistemi’ni,
Takım Kaptanı: Takımın organizasyonundan sorumlu olan ve süreçlerde liderlik görevini üstlenen kişi,
Takım Danışmanı: Her takım için en fazla bir (1) öğretmen/eğitmen/akademisyen,
TEKNOFEST: Havacılık, Uzay ve Teknoloji Festivalini,
T3 Vakfı: Türkiye Teknoloji Takımı Vakfını,
Yarışma Süreci: Yarışma başvurularının alınmaya başladığı tarih ile final sonuçlarının açıklandığı tarih
arasında geçen süreyi tanımlamaktadır.




                                                  6
1. GİRİŞ
Bu doküman Havacılıkta Yapay Zekâ Yarışması öncesi ve yarışma sırasında yarışmacıların bilgisi
dahilinde olması gereken durumları içermektedir.


2. GÖREVLER
TEKNOFEST 2026 Havacılıkta Yapay Zekâ yarışması kapsamında yarışmacılar üç farklı görevi yerine
getirecek algoritmalar geliştirmelidir. Bu görevler nesne tespiti, pozisyon kestirimi ve görüntü
eşleştirmedir. Yarışmacılar, hava aracının alt-görüş kamerasından aldıkları görüntüleri kendi
geliştirdikleri algoritmalar ile işleyerek, ilk görev için görüntü karesindeki belirli nesneleri ve bu
nesnelere ait hareket durumlarını tespit etmeli, ikinci görev için hava aracının zamana bağlı olarak
pozisyon bilgisini kestirmeli ve üçüncü görev için ise, görev başlangıcında paylaşılan referans
nesneleri hava aracı görüntülerinden tespit etmelidir. Aşağıda görevlerle ilgili ayrıntılı bilgi verilmiştir.


2.1. Birinci Görev: Nesne Tespiti
Havacılıkta Yapay Zekâ yarışması kapsamında yarışmacılar tarafından tespit edilmesi beklenen
nesne türleri taşıt, insan, Uçan Araba Park (UAP) ve Uçan Ambulans İniş (UAİ) alanları olmak üzere
4 adettir. Taşıt sınıfı için ayrıca nesnenin hareketli veya hareketsiz olma durumunun, UAP ve UAİ
sınıfları için ise inilebilir olma durumu bilgisinin de tespit edilmesi gerekmektedir. Yarışmacılara
verilecek olan video kareleri ile ilgili teknik bilgiler aşağıdaki gibidir:
    ●   Yarışmacılara verilecek olan videolar, hava aracının kalkışını, inişini ve seyrüseferini
        içerebilir. Bu nedenle yarışmacılar uçuş videolarında hava aracının yerden yüksekliğinin
        değişebileceği durumlar için de hazırlık yapmalıdır.
    ●   Her bir oturumda, yarışmacılara işlemeleri için verilecek olan videonun süresi 5 dakikadır ve
        saniyedeki kare sayısı (FPS) 7.5 olacaktır. Bu sebeple her oturumda yarışmacılara toplam
        2250 adet görüntü karesi verilecektir ve takımlardan toplam 2250 adet sonuç
        beklenmektedir. Uçuş süreleri ve verilecek görüntü kare sayısı değişkenlik gösterebilir.
    ●   Videolar Full HD veya 4K çözünürlüğünde çekilmektedir.
    ●   Video kareleri herhangi bir görüntü formatında olabilir (jpg, png vs.).
    ● Videolar tek tek karelere ayrılacak ve sıralı olarak yarışmacılara sunulacaktır.
    ● Yarışmada kullanılacak videolar günün herhangi bir vaktinde kayda alınmış görüntülerden
      elde edilecektir.
    ● Hava aracının kar, yağmur vb. hava koşullarında da uçabilmesi sebebi ile yarışma esnasında
      bu şartlar altında da algoritmalarının test edilebileceği göz önünde bulundurulmalıdır.
    ●   Hava aracının şehir, orman ve deniz üzerinde uçabilmesi sebebi ile yarışma esnasında bu
        şartlar altında da algoritmalarının test edilebileceği göz önünde bulundurulmalıdır.
    ●   Kamera açısı, hava aracının hareketine bağlı olarak 70-90 derece aralığında değişken
        olacaktır. İnsan tespitinde dik açıdan kaynaklı problemleri ve kamera açısı 0-70 derece



                                                         7
    aralığındayken uzaktaki nesnelerin tespit edilememesi ( Şekil 1 ) gibi durumları engellemek
    için veri seti içeriği belirlenmiş açı değerleri kullanılarak hazırlanacaktır. Şekil 2’de uygun olan
    çekim açısı örneği ifade edilmiştir.




                            Şekil 1: Çekim Açısı Durumları (Uygun Olmayan Çekim Açısı)




                            Şekil 2: Çekim Açısı Durumları (Uygun Olmayan Çekim Açısı)



●   Hava aracının alt-görüş kamerasında olabilecek olağan hatalar sebebi ile dağıtılan görüntü
    karelerinde bozulmalar bulunabilir. Görülebilecek bozukluklara örnek olarak bulanıklık ve
    ölü pikseller verilebilir. Örnek bir bozulma Şekil 3’te gösterilmiştir.
● Hava aracının alt-görüş kamerasında olabilecek olağan hatalar sebebi ile dağıtılan görüntü
  karelerinde tekrarlamalar/donmalar veya verilen karedeki görüntünün tamamen kaybı gibi
  durumlar bulunabilir.
● Hava aracından alınan görüntüler rgb veya termal kamera ile elde edilmiş olabilir.




                                                        8
                                      Şekil 3: Örnek Görüntü Bozulması




2.1.1. Taşıt ve İnsan Tespiti
●   Taşıt ve insan tespiti yapılırken görüntü karesinin tamamında bulunan tüm taşıt ve insanlar
    dikkate alınmalıdır.
●   Tespit edilen taşıtlar, hareketli veya hareketsiz olma durumuna göre sınıflandırılmalıdır.
●   Taşıt listesi yarışma şartnamesinde de belirtildiği üzere Tablo 2’de gösterilmiştir.




                                                     9
Tablo 2: Taşıt ve İnsan Sınıflarını İçeren Tablo


    Sınıf      Sınıf       İniş Durumunun       Hareket                            Detay
                ID            Alabileceği      Durumunun
                               Değerler        Alabileceği
                                                Değerler
                                                                Aşağıda maddeler halinde verilen tüm nesne
                                                                  türleri taşıt olarak değerlendirilmelidir:

                                                                          Motorlu karayolu taşıtları
                                                                                 Otomobiller
                                                                                Motosikletler
                                                                                  Otobüsler
   Taşıt        0                -1                0,1                            Kamyonlar
                                                                         Traktör, atv vb. kara araçları
                                                                                 Raylı taşıtlar
                                                                                    Trenler
                                                                                Lokomotifler
                                                                                   Vagonlar
                                                                                  Tramvaylar
                                                                                  Monoraylar
                                                                                   Füniküler
                                                                             Tüm deniz taşıtları
   İnsan        1                -1                -1           Ayakta duran ya da oturan fark etmeksizin tüm
                                                                        insanlar değerlendirilmelidir.



   ●    Görüntü karesinde tren olması durumunda lokomotif ve vagonların her biri ayrı birer obje
        olarak tanımlanmalıdır.
   ●    Tamamı görünmeyen taşıt ve insan nesnelerinin de tespit edilmesi beklenmektedir. Örneğin
        Şekil 4’teki gibi bir kısmı görüntüden çıkmış araçlar da dahil karelerde bulunan tüm nesneler
        tespit edilmelidir.


Tablo 3: Hareket Durumu Değerleri
            Hareket Durum ID                Hareket Durumu

                       0                      Hareketsiz

                       1                       Hareketli




                                                           10
             Şekil 4: Nesnelerin Tamamının Görüntü Karesi İçinde Olmaması Durumunda Etiketleme Yönergesi



●   Şekil 5’te örneklendiği üzere başka bir objenin arkasında olması sebebi ile bir kısmı görünen
    insanlar ve taşıtların da tespit edilebilmesi beklenmektedir.




                              Şekil 5: Nesnelerin Görüntü Karesi İçinde Bulunma Durumları



●   Bisiklet ve motosiklet sürücüleri “insan” olarak etiketlenmemelidir. Taşıt ve sürücüsü bir
    bütün olarak sadece “taşıt” etiketi ile etiketlenmelidir.
●   Scooter, sürücüsü olmadığı zamanlarda taşıt, sürücüsü olduğu zamanlarda ise insan olarak
    etiketlenmelidir.
●   Hava aracının uçuşu sırasında kamera sürekli hareket halinde olduğundan, sabit taşıt
    nesneleri de görüntü üzerinde hareketliymiş gibi algılanabilir. Yarışmacıların, bir taşıtın
    gerçekten mi hareket ettiğini yoksa sadece kameranın hareketinden dolayı mı yer
    değiştirdiğini ayırt edebilecek yöntemler geliştirmeleri gerekmektedir.




                                                         11
●   Şekil 6’da hareketli ve hareketsiz taşıt nesnelerini örnek olarak gösterilmiştir.




                           Şekil 6: Ardışık İki Karede Hareketli ve Hareketsiz Taşıt Tespiti




                                                      12
2.1.2. UAP ve UAI Tespiti
●   Şekil 7 ve Şekil 8 ile görselleştirilen Uçan Araba Park (UAP) ve Uçan Ambulans İniş
    (UAİ) alanları 4,5 metre çapında birer daire ile belirtilmektedir.
●   Aşağıda yer alan ve önceki yıllarda kullanılan UAP ve UAİ görselleri temsilidir.
    Yarışmada kullanılacak olan UAP ve UAİ görüntüleri örnek veri seti olarak yarışmacılarla
    paylaşılacaktır.




                                      Şekil 7: UAP Alan Bilgileri




                                      Şekil 8: UAİ Alan Bilgileri



●   UAP ve UAİ tespit edilmesinin ardından iniş durumunun da bildirilmesi gerekmektedir.
    UAP ve UAİ alanlarının iniş durumunun uygun olup olmaması, bu alanlarının üzerinde
    herhangi bir cisim bulunup bulunmaması ile ilişkilidir. Alanların üzerinde taşıt ve insan
    gibi nesne tespiti yapılan veya nesne tespiti yapılamayan herhangi bir nesne bulunduğu
    takdirde bu alan iniş için uygun değildir (Şekil 10).
●   UAP ve UAİ sınıf numaralandırmaları ve iniş durumu bilgileri Tablo 4’de belirtilmiştir.
    İniş durumu ID bilgileri Tablo 5’te gösterilmiştir.




                                                 13
Tablo 4: UAP ve UAİ Sınıf Bilgilerini İçeren Tablo


     Sınıf         Sınıf        İniş             Hareket                                  Detay
                    ID       Durumunun          Durumunun
                             Alabileceği        Alabileceği
                              Değerler           Değerler

  Uçan Araba                                                         Uçan arabanın park edebileceğini gösteren
  Park (UAP)        2             0,1                 -1          işaretin bulunduğu alandır. Şekil 6’da uçan araba
    Alanı                                                              park alanı için temsili figür belirtilmiştir.
    Uçan                                                           Uçan ambulansın iniş yapabileceğini gösteren
  Ambulans          3             0,1                 -1            işaretin bulunduğu alandır. Şekil 7’te uçan
  İniş (UAİ)                                                      ambulans iniş alanı için temsili figür belirtilmiştir.
    Alanı



Tablo 5: İniş Durumu Değerleri

       İniş Durum                   İniş Durumu
           ID
               0                    Uygun Değil
               1                        Uygun
             -1                   İniş Alanı Değil


   ●    UAP ve UAİ alanları da tıpkı taşıt ve insan nesneleri gibi tespit edilirken alanların bir
        kısmının görüntü karesinde olması tespit için yeterlidir. Fakat iniş durumunun “uygun”
        olabilmesi için UAP ve UAİ alanlarının tamamının kare içinde bulunması gerekmektedir.
        Şekil 9’da örnek olarak verilen resimde UAİ alanı nesne olarak tespit edilmeli ve iniş
        durumu uygun değil olarak belirtilmelidir.




                           Şekil 9: Nesnenin Tamamının Görüntü Karesi İçinde Olmaması Durumu




                                                           14
                  Şekil 10: UAP ve UAİ Alanlarının Üzerinde Herhangi Bir Cisim Olması Durumu

          (Bu örnekte alan üzerinde iki insan ve yerde serili bir mont bulunmaktadır.)
●   UAP ve UAİ alanlarının üzerinde insan ve taşıt nesneleri var ise o nesneler de ayrıca
    tespit edilmelidir.
●   Çekim açısına bağlı olarak alana yakın cisimler alanın üstünde olmasa bile öyleymiş gibi
    görülebilmektedir (Şekil 11). Bu yanıltıcı durumda olması gereken iniş durumu, “inişe
    uygun değildir” olmalıdır.




                             Şekil 11: Alanların Yanında Cisim Bulunması Durumu


2.1.3. Algoritma Çalışma Şartları
●   Yarışmacılar sunucu ile bağlantı kurup istek gönderdiklerinde bir adet görüntü karesi
    alacaklardır.
●   Her görüntü karesinde tespit ettikleri nesnelerin bilgisini istenen formatta sunucuya
    yollayacaklardır.
●   Yarışmacılar, sırası ile gönderilen video görüntülerinden herhangi birine sonuç
    göndermeden sıradaki karenin alınması için istek gönderemeyeceklerdir. Bu sebeple
    tüm görüntü karelerinin toplu olarak indirilmesi mümkün değildir.
●   Her görüntü karesine 1 adet sonuç yollanmalıdır, aynı kare için birden çok sonuç yollanır
    ise ilk yollanan sonuç değerlendirmeye alınacaktır.
●   Bir görüntü karesi için belirlenen limit değerden fazla sonuç yollayan takımların
    bulunulan oturum içerisinde sonuç gönderme kabiliyetleri belirli bir süreliğine




                                                     15
       engellenebilir. Yarışmacıların her görüntü karesi için gönderdikleri tahmin sayısını takip
       etmeleri gerekmektedir.


2.2. İkinci Görev: Pozisyon Tespiti
 İkinci görevde hava aracının konumlandırma sisteminin kullanılamaz veya güvenilemez hale
 geldiği durumlar simüle edilecek ve sadece görüntü verileri üzerinden pozisyon kestirimi
 yapılması beklenecektir. Böylelikle yaşanabilecek herhangi bir olumsuz duruma karşın hava
 aracının görev yapabilme kabiliyeti arttırılmış olacaktır. Bu görevin hangi şartlarda ve
 yarışmanın hangi aşamalarında yapılacağı bu dokümanın ilerleyen bölümlerinde
 açıklanacaktır.

   2.2.1. Pozisyon Tespiti
  ●   Yarışmacılar geliştirdikleri pozisyon kestirimi algoritmaları ile verilen kamera
      görüntülerini kullanarak hava aracının referans koordinat sistemindeki pozisyonunu
      kestireceklerdir.
  ●   Her oturumda yarışmacılara verilecek videonun ilk karelerine ait yer değiştirme bilgisini
      kullanarak x, y, z eksenlerindeki hareket yönlerini belirleyebileceklerdir.
  ●   Referans koordinat sisteminde ilk pozisyon bilgisi x0=0.00 [m], y0=0.00 [m], z0=0.00 [m]
      şeklinde olacaktır.
  ●   Pozisyon kestirimi yapılacak oturumlarda kamera açısı yeryüzüne bakacak şekilde 70-
      90 derece aralığında olacaktır.
  ●   Hava aracının kamera parametre bilgileri yarışmacılarla paylaşılacaktır.
  ●   Yarışmacıların geliştirecekleri sistem konusunda herhangi bir kısıtlama
      bulunmamaktadır. Geliştirdikleri algoritmalar, öğrenen modelleri kullanabileceği gibi
      matematiksel temellere dayanan sistemlerden de faydalanabilir.
  ●   Şekil 13’de referans ve kestirim pozisyon bilgisinin referans koordinat sisteminde
      çizdirilmiş örneği bulunmaktadır. Referans ve kestirim pozisyon bilgileri kullanılarak
      yarışmacıların hatası hesaplanacaktır. Bu hata miktarının büyüklüğü yarışmacıların bu
      görevden aldığı puan ile ilişkili olacaktır. Başlık 9.2’de puanlama açıklanmaktadır.




                                                  16
                            Şekil 12: Referans ve Kestirim Pozisyon Bilgisi Örneği




                 Şekil 13: Referans ve Kestirim Pozisyon Bilgileri Kullanılarak Hesaplanan Hata



●   Şekil 13’te referans ve kestirim pozisyon bilgileri kullanılarak hesaplanan hatanın
    görselleştirilmiş hali bulunmaktadır. Şekilde gözüktüğü üzere x, y ve z eksenlerindeki
    hata miktarları görselleştirilmiştir.



2.2.2. Algoritma Çalışma Şartları
●   Yarışmacılar sunucuya istek gönderdiğinde video karesinin yanı sıra, bu kare ile ilgili
    Tablo 6’te verilen bilgileri de alacaklardır.




                                                      17
Tablo 6: Yarışmacıların Sunucu İsteği Sonrasında Alacağı Bilgiler

            Başlık                                            Detay

     Video Karesi Bilgisi                     Video Karesi alınırken ve sonuçlar yollanırken
                                                    kullanılacak benzersiz isim


     Pozisyon Bilgisi - X                      Hava Aracının referans koordinat sistemindeki
                                        ilk görüntüye göre X eksenindeki metre cinsinden yer
                                                             değiştirmesi
     Pozisyon Bilgisi - Y                      Hava Aracının referans koordinat sistemindeki
                                        ilk görüntüye göre Y eksenindeki metre cinsinden yer
                                                             değiştirmesi
       Pozisyon Bilgisi- Z                     Hava Aracının referans koordinat sistemindeki
                                        ilk görüntüye göre Z eksenindeki metre cinsinden yer
                                                             değiştirmesi
   Pozisyon Bilgisi- Sağlık                      Hava aracının pozisyon tespit sisteminin
                                            sağlıklı çalışıp çalışmadığını gösteren değer



   ●     Sağlık değeri 1 ise yarışmacının kendi geliştirdiği algoritma ile kestirdiği pozisyon
         bilgisini gönderebileceği gibi sunucudan aldığı referans değeri de değiştirmeden
         gönderebilir. Bu yarışmacının vereceği bir karardır.
   ●     Sağlık değeri 0 ise yarışmacının kendi geliştirdiği algoritma ile kestirdiği pozisyon
         bilgisini sunucuya göndermesi gerekmektedir.
   ●     Oturumlarda yarışmacılar sunucudan aldıkları toplam 5 dakikalık (toplam 2250 kare)
         videonun ilk 1 dakikasında (450 kare) uçan arabanın referans koordinat sistemine göre
         pozisyon bilgisini sağlıklı olarak alacakları kesindir.
   ●     Oturumun son 4 dakikasında (1800 kare) uçan arabanın pozisyon bilgisi sağlıksız
         durumuna geçebilir. Bu sağlıksız durumunun ne zaman başlayacağı ve ne kadar süre
         devam edeceği belirli olmayacaktır.
   ●     Yukarıda yer alan süre ve görüntü karesi sayıları değişebilir.
   ●     Şekil 15’te yarışmacılardan geliştirmesi beklenen sistem şeması görselleştirilmiştir.
   ●     ‘Nesne Görseli’, oturum esnasında yarışmacılarla paylaşılacak tanımsız nesneleri ifade
         etmektedir.


2.3. Üçüncü Görev: Görüntü Eşleme
       Üçüncü görev, hava araçlarının daha önce tanımlanmamış nesneleri görsel veri
 üzerinden anlık olarak tanıma ve takip etme yeteneğini test etmektedir. Yarışma oturumunun




                                                  18
başlangıcında paylaşılacak referans nesnelerin hava aracı görüntülerinden tespit edilmesi
beklenmektedir. Bu görevde temel amaç, sistemin daha önce karşılaşmadığı yeni nesnelere
karşı adaptasyon yeteneğini ve genel nesne tanıma kabiliyetini ölçmektir.




                                 Şekil 14: Referans Obje Eşleme Örnek Görseli



       Oturum başlangıcında belirli sayıda ve farklı zorluk seviyelerinde referans nesne
görüntüsü paylaşılacaktır. Yarışmacılar görüntü akışı esnasında verilen referanslardan tespit
ettikleri nesnelerin koordinatlarını sonuçları ile beraber sunucuya göndereceklerdir. Oturum
başında verilen referans nesnelerin tamamı oturum içerisindeki görüntülerde mevcut
olmayabilir. Yarışmacıların bu senaryoyu göz önünde bulundurarak geliştirme yapmaları
gerekmektedir.
      Oturum esnasında paylaşılan görüntüler;
          •   Farklı kameradan çekilmiş olabilir. Örneğin termal kameradan çekilen nesne
              görüntüsünün RGB kamera görüntüleri üzerinde eşleştirilmesi istenebilir.
          •   Farklı bir açıdan veya irtifadan çekilmiş olabilir.
          •   Uydu görüntüleri üzerinden alınmış bir nesnenin görüntü üzerinde eşlenmesi
              istenebilir.
          •   Yer yüzeyinden çekilmiş nesneler olabilir.
          •   Çeşitli görüntü işleme işlemlerinden geçmiş olabilir.
       Bu sebeplerden yarışmacılardan çeşitli koşullara dayanıklı bir eşleştirme algoritması
geliştirmeleri beklenmektedir.




                                                     19
                              Şekil 15: Sonuç Paketleri Oluşturma Diyagramı




3. YARIŞMA
 ●   Ön Tasarım Raporunu teslim etmiş ve Çevrimiçi Yarışma Simülasyonundan yeterli
     puanı alan takımlar TEKNOFEST 2026’da yarışmak için hak kazanacaklardır.
 ●   Yarışma alanında yarışmacıların istek atarak hem videoları hem de ikinci görev için
     pozisyon verilerini çekebilecekleri bir sunucu ve yerel ağ kurulacaktır.
 ●   Yarışmacılar bu ağa ethernet kablosu ile bağlanacaklar, test video görsellerini
     sunucudan alacaklar ve cevaplarını yine sunucuya yükleyeceklerdir.
 ●   Belirtilen yerel ağın internet bağlantısı olmayacak ve yarışmacıların sistemlerinin
     internete bağlanmasına kesinlikle izin verilmeyecektir.
 ●   Bağlantıların yapılması ile ilgili teknik detaylar yarışma esnasında belirtilecek ve
     yarışma teknik ekibi tarafından yarışmacılara sisteme bağlanmaları konusunda
     yardımcı olunacaktır.
 ●   Yarışmacılar yarışma alanında kullanacakları bilgisayarlardan sorumlu olacaklardır.
     Yarışma ekibi tarafından yarışmacılara herhangi bir bilgisayar desteği verilmeyecektir.
 ●   Yarışmacıların bilgisayarlarında ethernet girişi ve ethernet bağlantı kabiliyeti olması
     gerekmektedir.
 ●   Her oturumda, her takımdan aynı anda 3 yarışmacının yarışma alanına girişine izin
     verilecektir. Takımın danışmanın olması halinde, yarışma alanında 2 yarışmacı öğrenci
     ve 1 danışman alınacaktır.
 ●   Yarışma esnasında bir takımın, başka bir takıma yardımcı olmasına kesinlikle izin
     verilmemektedir.




                                                  20
3.1. Test Oturumu
  ●   Yarışma öncesinde gerekli tüm hazırlıkların yapılabilmesi için 75 dakikalık bir oturum
      yapılacaktır.
  ●   Bu oturumun amacı yarışmacıların donanım kurulumlarını yapmalarıdır.
  ●   Yarışma şartlarını en iyi şekilde test edebilmeleri için 2 dakikalık (900 video karesi) bir
      video sunucudan yayınlanacaktır.
  ●   Yarışmacıların bu test videosunu uygun şekilde aldığı ve sonuçlarını uygun şekilde
      yolladığı yarışmayı düzenleyen teknik ekip tarafından test edilecek ve geri bildirim
      verilecektir.
  ●   Yarışmacıların test oturumunda yollamış oldukları sonuçların puanlandırmada etkisi
      olmayacaktır.


3.2. Yarışma Oturumları
  ●   4 yarışma oturumu yapılacaktır.
  ●   Bu oturumların her birinin toplam süresi 75 dakika olacaktır.
  ●   Her oturumun ilk 15 dakikası yarışmacılara hazırlık için verilecektir.
  ●   Sonraki 60 dakikalık süre yarışma için ayrılacaktır.
  ●   Her oturumda 2250 video karesi verilecek ve bu karelerin işlenmesi sonucunda elde
      ettikleri sonuçları uygun formatta sunucuya göndermeleri istenecektir.
  ●   Her bir oturumda yayınlanacak olan videonun bir teması bulunacaktır. Bu temalara
      örnek olarak “Güneşli”, “Zorlu Hava Şartları”, “Akşam”, “Deniz Üstü” verilebilir.
      Oturum temaları ile ilgili bilgi yarışma gününe kadar saklı tutulacaktır.
  ●   Oturum süreleri ve kullanılan video uzunlukları değişebilmektedir. Bir değişiklik olması
      halinde yarışmacılar önceden bilgilendirilecektir.


4. TEKNİK SUNUM
  ●   Yarışmacı takımlardan, yarışma oturumları esnasında bir sunum yapmaları
      beklenmektedir.
  ●   Her takımdan bir adet İletişim Sorumlusu sunumu yapmak ile görevlendirilmelidir.
  ●   Sunum süresi, takım başına 5 dakikadan uzun olmayacaktır.
  ●   Her yarışma oturumunun başında, oturumda sunum yapacak takımlar duyurulacaktır.
  ●   Sunumlar, 3 kişiden oluşan bir hakem heyetine sunulacaktır.
  ●   Sunu sırasında veya sonrasında takım temsilcisine sorular yönetilebilir.
  ●   Hazırlanan sunumlar, yarışmacılar tarafından TEKNOFEST Yarışmalar komitesi
      tarafından iletilecek tarihe kadar t3kys.com adresine yollanarak teslim edilmelidir.



                                                  21
  ●   Örnek sunum şablonu Haziran 2026 tarihine kadar katılımcı takımlarla paylaşılacaktır.
  ●   Sunum şablonu bozulmamak kaydı ile sunum içeriği konusunda bir kısıtlama
      bulunmamaktadır.
  ●   Sunumda sunulabilecek konu başlıklarına örnek olarak “Ek Veri Toplama Süreci”,
      “Kullanılan Algoritma”, “Alternatif Algoritmalar”, “Test Sonuçları” ve “Yenilikçi
      Yaklaşım” verilebilir. Yarışmacıların sunumlarında bu başlıkları kullanmaları zorunlu
      değildir.


5. RAPORLAMA
  ●   Yarışmacı takımlardan iki ayrı doküman yazmaları beklenmektedir.
  ●   Ön Tasarım Raporu yarışma katılımı ve Final Tasarım Raporu puanlandırma sürecinde
      kullanılacağı için iki raporun da teslim edilmesi şarttır.


5.1. Ön Tasarım Raporu
  ●   Ön Tasarım Raporu şablonu, en geç 22/04/2026 tarihinde teknofest.org internet
      sitesinde paylaşılacaktır.
  ●   Rapor şablonunda ayrıntılı olarak açıklanacağı üzere, Ön Tasarım Raporunda konu ile
      ilgili yapılan araştırma ve problemlerin çözümüne yönelik olarak verilen çözüm önerileri
      bulunacaktır.
  ●   Rapor şablonu bozulmamak kaydı ile rapor içeriği veya uzunluğu konusunda bir
      kısıtlama yoktur.
  ●   Yarışmaya katılım için Ön Tasarım Raporunu teslim etmek zorunludur.
  ●   Yarışma sonuçlarının belirlenmesinde ve ödüllendirmede Ön Tasarım Raporunun bir
      etkisi bulunmamaktadır.
  ●   Ön Tasarım Raporu, en geç 22/04/2026 tarihinde t3kys.com adresine yollanarak
      teslim edilmelidir.


5.2. Final Tasarım Raporu
  ●   Final Tasarım Raporu şablonu, en geç Ağustos 2026 tarihinde teknofest.org internet
      sitesinde paylaşılacaktır.
  ●   Rapor şablonunda ayrıntılı olarak açıklanacağı üzere, Final Tasarım Raporunda,
      yarışmacıların yarışmaya hazırlandıkları süre boyunca yaptıkları literatür çalışmalarını,
      yarışma esnasında kullandıkları algoritmaları, yapmış oldukları testleri ve bunun gibi
      birçok teknik bilgiyi yarışma düzenleyicilerine raporlamaları beklenmektedir.
  ●   Raporun değerlendirmesinde rapor içeriği ve rapor formatı etkili olacaktır.
  ●   Final Tasarım Raporu puanı genel yarışma puanının %5’ini oluşturmaktadır.




                                                 22
  ●   Bir yarışma takımının yarışmada dereceye girebilmesi için Final Tasarım Raporu’nu
      teslim etmesi zorunludur.
  ●   Rapor şablonu bozulmamak kaydı ile rapor içeriği veya uzunluğu konusunda bir
      kısıtlama yoktur. Final Tasarım Raporu, TEKNOFEST Yarışmalar komitesi tarafından
      iletilecek tarihte t3kys.com adresine yollanarak teslim edilmelidir.


6. ÇEVRİMİÇİ YARIŞMA SİMÜLASYONU
  ●   Ön Tasarım Raporu değerlendirmelerinden sonra, yarışma alanına gelecek takımların
      belirlenebilmesi için bir ön eleme yarışması yapılacaktır.
  ●   Çevrimiçi Yarışma Simülasyonu’nda yarışmacılardan, geliştirdikleri modeller ile
      çevrimiçi ortamda paylaşılacak olan karelerdeki nesneleri tespit etmeleri ve hava
      aracının pozisyonunu kestirmeleri beklenmektedir.
  ●   Çevrimiçi Yarışma Simülasyonu, birinci yarışma oturumu ile aynı kurallar ve tema ile
      yapılacaktır.
  ●   Mayıs ayı içerisinde Çevrimiçi Yarışma Simülasyonu ile ilgili ayrıntılı bilgi içeren
      doküman yarışmacılar ile paylaşılacaktır.
  ●   Doküman ile duyurulacak olan başarı kriterinin altında kalan ve sunucuya hiç
      bağlanmayan takımlar bir sonraki aşamaya geçemeyecektir.



7. TAKIMLARIN YAZILIM VE DONANIM ÖZELLİKLERİ
  ●   Her takım kendi yazılım ve donanım sisteminden sorumludur. Yarışma alanında
      herhangi bir yazılım ya da donanım (bilgisayar, mouse vs.) desteği sunulmayacaktır.
  ●   İhtiyaç duyulacak her donanım (adaptör, mouse, klavye vs.) ve yazılıma sahip olarak
      yarışmaya katılım sağlanması beklenmektedir.
  ●   İstenilen işletim sistemi kullanılabilir.
  ●   Takımlar istedikleri platformda ve programlama dillerinde geliştirme yapabilir.
  ●   Yarışmacılardan saniyede 1 görüntü karesi işleyebilecek donanıma sahip olmaları
      yeterli olacaktır.
  ●   Algoritmanın çalışma hızı, bir puanlandırma kriteri değildir. Bu sebeple donanımların
      güçlü ve zayıf olması yarışmanın seyrine etki etmemektedir.
  ●   Yarışma platformu, yarışmacıların kullanacakları donanımların güçlü ya da zayıf
      olmasının yarışmanın seyrine etki etmeyecek şekilde hazırlanacaktır.




                                                  23
8. YARIŞMA SIRASINDA SUNUCU İLE BAĞLANTI
  ●   Yarışma sırasında takımlara, yarışma sunucunun da içinde bulunduğu yerel ağa
      bağlanabilmeleri için bir ethernet kablosu sağlanacaktır. Her takım bu ethernet kablosu
      aracılığı ile yarışma ağına yalnızca tek bir ip adresi ile bağlanmalıdır. Yarışma sırasında
      takımlara birer ip adresi belirtilecek ve sisteme yalnızca belirtilen ip adresleri üzerinden
      bağlantıya izin verilecektir.
  ●   Yarışma sunucusunun adresi yarışma günü belirlenecektir. Bu adres örnek olarak
      http://127.0.0.25:5000 formatında olacaktır. Sunucu ile yapılacak olan tüm
      haberleşmeler API mantığı ile JSON formatında olacaktır.
  ●   Yarışma anında kullanılacak API adres bilgileri yarışma ortamında test oturumu
      öncesinde yarışmacılar ile paylaşılacaktır.
  ●   Yarışma sırasında takımlar bir videoya ait görüntü karelerinin üzerinde şartnamede
      belirtilen nesnelerin tespitini yapacaklar, gerekli koşullar sağlandığında da pozisyon
      kestirimi yapacaklardır. Yarışmacılara videolar verilmeyecek, bu videolardan 7.5 fps ile
      kaydedilmiş görüntü listesi verilecektir. Bu liste Şekil 17 ile gösterilen JSON formatında
      olup içerisinde bulunacak bilgiler aşağıdaki gibi olacaktır:
          ○ url: Video karesi id’sinin benzersiz url’i
          ○ image_url: Video karesi görselinin bulunduğu url
          ○ video_name: Video karesinin alındığı videonun adı ya da numarası
          ○ session: Oturumu belirten url
          ○ translation_x: Hava Aracının referans koordinat sistemindeki ilk görüntüye göre
               X eksenindeki metre cinsinden yer değiştirmesi
          ○ translation_y: Hava Aracının referans koordinat sistemindeki ilk görüntüye göre
               Y eksenindeki metre cinsinden yer değiştirmesi
          ○ translation_z: Hava Aracının referans koordinat sistemindeki ilk görüntüye göre
               Z eksenindeki metre cinsinden yer değiştirmesi
          ○ gps_health_status: Hava aracının pozisyon tespit sisteminin sağlıklı çalışıp
               çalışmadığını gösteren değer.
  ●   Yarışmacılar, yarışma başladıktan sonra yarışma sunucusundan aşağıdaki örneğe
      benzer bir liste alacaklardır.




                                                  24
                                   Şekil 16: Görsel Bilgileri



● Takımlar bir resmi işlemeyi bitirdikten sonra bu resimde buldukları nesneleri ve hava
   aracı pozisyonunu sunucuya bildirmeleri gerekmektedir. Sunucuya bildirilmeyen
   cevaplar geçersiz sayılacaktır. Sonuçlar her bir resim için ayrı ayrı gönderilmelidir.
● Tespit edilen nesneleri sunucuya bildirmek için tespit edilen nesne konumları,



                                              25
   sınıflar ve hava aracının konumu haberleşme dokümanında belirtilecek olan adrese
   gönderilmelidir. Yarışmacıların gönderecekleri Şekil 17 ile formatı gösterilen JSON
   dosyasında bulunması gereken bilgiler şunlardır:
● id: Gönderilen tahminin id’si

● user: Kullanıcının bilgilerini içeren url

● frame: Video karesi id’sinin benzersiz url’i

    ○ detected_objects: Tespit edilen nesnelerin konumlarını içeren dizi.
          ●   cls: Tespit edilen nesnenin sınıfı (“0”, “1”, “2”, “3”)
          ●   landing_status: İniş durumunun uygun olup olmadığını içeren bilgi
              (“-1”, “0”, “1”)
          ●   motion_status: Hareket durumunu içeren bilgi (“-1”, “0”, “1”)
          ●   top_left_x: Tespit edilen nesneyi içine alan en küçük dörtgenin sol üst
              köşesinin resmin sol kenarına olan piksel cinsinden uzaklığı
          ●   top_left_y: Tespit edilen nesneyi içine alan en küçük dörtgenin sol üst
              köşesinin resmin üst kenarına olan piksel cinsinden uzaklığı
          ●   bottom_right_x: Tespit edilen nesneyi içine alan en küçük dörtgenin sağ
              alt köşesinin resmin sol kenarına olan piksel cinsinden uzaklığı
          ●   bottom_right_y: Tespit edilen nesneyi içine alan en küçük dörtgenin sağ
              alt köşesinin resmin üst kenarına olan piksel cinsinden uzaklığı


    ○ detected_translations: Tespit edilen yer değiştirme bilgisini içeren dizi.
          ●   translation_x: Hava Aracının referans koordinat sistemindeki ilk
              görüntüye göre X eksenindeki metre cinsinden yer değiştirmesi
          ●   translation_y: Hava Aracının referans koordinat sistemindeki ilk
              görüntüye göre Y eksenindeki metre cinsinden yer değiştirmesi
          ●   translation_z: Hava Aracının referans koordinat sistemindeki ilk
              görüntüye göre Z eksenindeki metre cinsinden yer değiştirmesi


    ○ detected_undefined_objects: Tespit edilen tanımsız nesnelerin konumlarını
      içeren dizi.


          ●   object_id: Tespit edilen nesnenin ID’si
          ●   top_left_x: Tespit edilen nesneyi içine alan en küçük dörtgenin sol üst
              köşesinin resmin sol kenarına olan piksel cinsinden uzaklığı
          ●   top_left_y: Tespit edilen nesneyi içine alan en küçük dörtgenin sol üst



                                              26
    köşesinin resmin üst kenarına olan piksel cinsinden uzaklığı
●   bottom_right_x: Tespit edilen nesneyi içine alan en küçük dörtgenin sağ
    alt köşesinin resmin sol kenarına olan piksel cinsinden uzaklığı
●   bottom_right_y: Tespit edilen nesneyi içine alan en küçük dörtgenin sağ
    alt köşesinin resmin üst kenarına olan piksel cinsinden uzaklığı




                         Şekil 17: JSON Formatı




                                  27
Şekil 16 ve Şekil 17’ de ifade edilen json formatı yarışmada kullanılacak olan formatın taslak
halidir. Yarışmada kullanılacak format yarışmacılar ile paylaşılacaktır.


9. PUANLAMA
Tablo 7: Genel Yarışma Puanlandırması

                    Puan Türü                             Puan Oranı

               Birinci Görev                                  %25

               İkinci Görev                                   %40

              Üçüncü Görev                                    %25

          Final Tasarım Raporu                                %5

            Yarışma Sunumu                                    %5

              Toplam Puan                                    %100


9.1. Birinci Görev Puanlama Kriteri
   ●   Nesne tespitinin çalışma performansı Ortalama Kesinlik Değerlerinin Ortalaması
       (mean Average Precision, mAP) değerine göre belirlenecektir. mAP, Kesişimin
       Birleşime Oranı (Intersection Over Union, IoU) değeri üzerinden hesaplanır. Bu oran,
       takımların bulduğu alan (𝑇𝑎ℎ𝑚𝑖𝑛 𝐸𝑑𝑖𝑙𝑒𝑛 𝐷ö𝑟𝑡𝑔𝑒𝑛) ile nesnenin gerçek alanını
       (𝐺𝑒𝑟ç𝑒𝑘 𝑅𝑒𝑓𝑒𝑟𝑎𝑛𝑠 𝐷ö𝑟𝑡𝑔𝑒𝑛) gösteren alan arasındaki eşleşme miktarını belirtir (Şekil
       18 ve Denklem 1).




                                       Şekil 18: IoU Formül Gösterimi




                                         Denklem 1: IoU Formülü

                       𝐺𝑒𝑟ç𝑒𝑘𝑅𝑒𝑓𝑒𝑟𝑎𝑛𝑠𝐷ö𝑟𝑡𝑔𝑒𝑛 ∩ 𝑇𝑎ℎ𝑚𝑖𝑛𝐸𝑑𝑖𝑙𝑒𝑛𝐷ö𝑟𝑡𝑔𝑒𝑛
               𝐼𝑜𝑈 =
                       𝐺𝑒𝑟ç𝑒𝑘𝑅𝑒𝑓𝑒𝑟𝑎𝑛𝑠𝐷ö𝑟𝑡𝑔𝑒𝑛 ∪ 𝑇𝑎ℎ𝑚𝑖𝑛𝐸𝑑𝑖𝑙𝑒𝑛𝐷ö𝑟𝑡𝑔𝑒𝑛




                                                   28
 Yöntemlerin değerlendirilmesinde kullanılacak mAP metriği klasik nesne tespiti yöntemlerinin
 değerlendirilmesinde kullanıldığı gibi olacaktır. Ek olarak, iniş durumunun doğru tespit
 edilemediği durumlarda ilgili sınıf için ortalama kesinlik (AP) değeri olumsuz etkilenecektir.
 Hareket durumunun doğru tespit edilemediği durumlarda ilgili sınıf için ortalama kesinlik (AP)
 değeri olumsuz etkilenecektir.
 mAP metriğinin hesaplanmasında kullanılan 𝑇𝑎ℎ𝑚𝑖𝑛 𝐸𝑑𝑖𝑙𝑒𝑛 𝐷ö𝑟𝑡𝑔𝑒𝑛 bölgesinin 𝐼𝑜𝑈 eşik
 değeri 0.5’tir.

 9.1.1.      Birinci Görev Örnek Puanlama Durumları

Örnek 1:
Tablo 8: Örnek 1 Tablo

                   Gerçek Sınıf                                       İnsan
                Tespit Edilen Sınıf                                   İnsan
      Tespit İçin Gönderilen Dörtgen Sayısı                                1
       Tespit Edilen Alanların Iou Değerleri                           0.63
            Tespit Edilen İniş Değerleri                                -1
                Gerçek İniş Değeri                                      -1
Açıklama: Tabloda gösterilen örnekte yarışmacı, insan sınıfı için AP değerini arttıracak şekilde
puan alır.


Örnek 2:
Tablo 9: Örnek 2 Tablo

                  Gerçek Sınıf                                       İnsan
               Tespit Edilen Sınıf                                   Taşıt
     Tespit İçin Gönderilen Dörtgen Sayısı                             1
      Tespit Edilen Alanların Iou Değerleri                          0.66
           Tespit Edilen İniş Değerleri                               -1
               Gerçek İniş Değeri                                     -1
 Açıklama: Tabloda gösterilen örnekte yarışmacı, insan sınıfı için AP değerini düşürecek
 şekilde puan alır.


Örnek 3:




                                                  29
Tablo 10: Örnek 3 Tablo

                   Gerçek Sınıf                                       İnsan
                Tespit Edilen Sınıf                                   İnsan
     Tespit İçin Gönderilen Dörtgen Sayısı                              1
      Tespit Edilen Alanların Iou Değerleri                           0.42
           Tespit Edilen İniş Değerleri                                 -1
                Gerçek İniş Değeri                                      -1
Açıklama: Tabloda gösterilen örnekte yarışmacı, sınıfı ve iniş değerlerini doğru tespit etmesine
rağmen tespit edilen alanın IoU değeri 0.5’ten küçük olduğundan bu örnekte insan sınıfı için AP
değerini düşürecek şekilde puan alır.


Örnek 4:
Tablo 11: Örnek 4 Tablo

                  Gerçek Sınıf                                         Taşıt
               Tespit Edilen Sınıf                                     Taşıt
    Tespit İçin Gönderilen Dörtgen Sayısı                                3
     Tespit Edilen Alanların Iou Değerleri                    0.85, 0.61, 0.54

           Tespit Edilen İniş Değerleri                              -1, -1, -1
               Gerçek İniş Değeri                                       -1
Açıklama: Tabloda gösterilen örnekte yarışmacı, bütün tespitler IoU eşik değerinden büyük
olmasına rağmen birden fazla tespit gönderdiğinden taşıt sınıfı için 1 kez (0.85 IoU değerli tespit
için) AP değerini arttıracak şekilde puan alırken 2 kez (0.61 ve 0.54 IoU değerli tespitler için) AP
değerini düşürecek şekilde puan alır. Toplamda yarışmacının taşıt sınıfı için bu örnekte AP
değeri düşecek şekilde etkilenir.
Örnek 5:
Tablo 12: Örnek 5 Tablo

                    Gerçek Sınıf                                         UAP
                 Tespit Edilen Sınıf                                     UAP
      Tespit İçin Gönderilen Dörtgen Sayısı                                  1
      Tespit Edilen Alanların Iou Değerleri                              0.91
            Tespit Edilen İniş Değerleri                                     1
                Gerçek İniş Değeri                                           0



                                                    30
Açıklama: Tabloda gösterilen örnekte yarışmacı, inişe uygun olmayan bir alanı inişe uygun
olarak tespit ettiğinden UAP sınıfı için AP değerini düşürecek şekilde puan alır.
Örnek 6:
Tablo 13: Örnek 6 Tablo

                    Gerçek Sınıf                                               Taşıt
                Tespit Edilen Sınıf                                         Tespit yok
       Tespit İçin Gönderilen Dörtgen Sayısı                                     0
       Tespit Edilen Alanların Iou Değerleri                                      -
            Tespit Edilen İniş Değerleri                                          -
                Gerçek İniş Değeri                                               -1
Açıklama: Tabloda gösterilen örnekte yarışmacı, tespit etmesi gereken taşıt nesnesini tespit
edilemediğinden bu örnekte taşıt sınıfı için AP değerini düşürecek şekilde puan alır.


9.2. İkinci Görev Puanlama Kriteri
   ●     Hava aracının referans pozisyon bilgisi ile yarışmacıların geliştirdiği algoritmaların
         kestirdiği pozisyon bilgisi arasındaki ortalama hata kullanılarak bir puanlandırma
         yapılacaktır.
   ●     Aşağıda, ikinci görev için kullanılacak hata hesaplama formülü                     Denklem 2’de
         gösterilmiştir.

                                Denklem 2: Görev için Ortalama Hata Hesaplama Formülü

                    𝑂𝑟𝑡𝑎𝑙𝑎𝑚𝑎 𝑌𝑎𝑟𝚤ş𝑚𝑎𝑐𝚤 𝐻𝑎𝑡𝑎 = 𝐸
                                             𝑁
                                        1
                                      =   ∑ √(𝑥̂𝑖 − 𝑥𝑖 )2 + (𝑦̂𝑖 − 𝑦𝑖 )2 + (𝑧̂𝑖 − 𝑧𝑖 )2
                                        𝑁
                                           𝑖=1
   ●     𝑥̂𝑖 , 𝑦̂𝑖 , 𝑧̂𝑖 yarışmacının i. görsel için yollamış olduğu pozisyon kestirimi bilgisini ifade et-
         mektedir. 𝑥𝑖 , 𝑦𝑖 ve 𝑧𝑖 ise hava aracının mutlak doğru pozisyon bilgisini ifade etmektedir.
   ●     Her oturum sonunda tüm takımların ikinci görevde yapmış oldukları hata miktarları ile
         yarışmacı takımın puanı hesaplanmaktadır.

9.3.     Üçüncü Görev Puanlama Kriteri
 Üçüncü görevde puanlama olarak birinci yöntemdeki puan hesaplama yöntemi (mAP)
 kullanılacaktır. Detaylı bilgilendirme şartname revizyonlarında verilecektir.




                                                        31
10. YARIŞMA GİTHUB ve GOOGLE GROUPS SAYFALARI
Yarışma kapsamında, katılımcıların süreç boyunca destek alabilmeleri, sorularını
paylaşabilmeleri ve ekipler arası iletişim kurabilmeleri amacıyla çeşitli dijital platformlar
oluşturulmuştur. Bu platformlar, teknik detayların paylaşımı, güncellemelerin duyurulması ve
topluluk içi etkileşimin artırılması için kritik bir rol oynamaktadır.
Github Proje Deposu: Yarışma boyunca kullanılacak kod blokları, örnek veri setleri ve diğer
teknik materyaller Github üzerinden paylaşılacaktır. Katılımcılar, bu platform üzerinden
gerekli dokümanlara erişebilir, kodlar üzerinde inceleme yapabilir ve kendi çalışmalarına
entegre edebilirler. Github sayfası, yarışmacıların teknik materyallere kolay erişimini
sağlamak amacıyla sürekli güncel tutulacaktır.
Google Groups Platformu: Takımlar arasındaki bilgi alışverişini kolaylaştırmak ve
organizasyon ekibine sorularını iletebilmeleri için bir Google Groups tartışma platformu
oluşturulmuştur. Bu grup üzerinden yarışmaya dair önemli duyurular yapılacak, sıkça sorulan
sorular yanıtlanacak ve yarışmacıların teknik destek alabilmesi sağlanacaktır.
Bu platformlar, yarışmacıların hazırlık süreçlerini kolaylaştırmayı ve organizasyonun tüm
katılımcılar için şeffaf ve erişilebilir olmasını sağlamayı hedeflemektedir. Tüm katılımcıların
bu platformları etkin bir şekilde kullanmaları önerilmektedir.

11. YARIŞMA SONUÇLARININ DUYURULMASI VE ÖDÜLLENDİRME
  ●   Her yarışma oturumu başlarken, yarışmacıların bir önceki almış oldukları puana göre
      sıralamaları bildirilecektir.
  ●   Tüm yarışma oturumlarının tamamlanmasının ardından, Başlık 9’da belirtilen oranlar
      kullanılarak yarışma puanı hesaplanacaktır. Bu puan değeri kullanılarak yarışmada
      dereceye giren takımlar belirlenecektir.
  ●   Yarışmada dereceye giren takımlar TEKNOFEST’in son gününde kürsüye çıkarak
      ödüllerini alacaklardır.
  ●   TEKNOFEST ve Havacılıkta Yapay Zekâ Yarışması takvimleri ile ilgili bilgiler genel
      şartnamede ve çeşitli medya kaynakların yarışmacılara duyurulacaktır.




                                                 32
33
