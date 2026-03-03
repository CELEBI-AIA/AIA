# 1. Özet Bulgular

- Sistem mimarisi, kare-bazlı fetch→işleme→submit sıralamasını zorlayacak şekilde kurgulanmış; bu yönüyle rekabet döngüsüyle kısmi uyum gösteriyor (özellikle pending-result disiplini ve idempotent gönderim korumaları). Ancak hata/degrade dallarında “her kare için geçerli sonuç üretimi” garantisi kırılabiliyor.
- Görev 1 için sınıf kümesi (0/1/2/3), `motion_status` ve `landing_status` alanları uçtan uca taşınıyor; fakat payload clipping (quota/cap) ve yoğun filtreleme nedeniyle recall kaybı riski yüksek.
- Görev 2’de `gps_health=0` durumunda sistem kendi kestirimini gönderiyor; fakat frame verisi bozulduğunda sağlık değerinin zorla `0`’a çekilmesi ve agresif fallback davranışları Öklid hata metriğinde hata sarmalı riski üretiyor.
- Görev 3’te referans bütünlüğü için güçlü doğrulama var; buna karşın kritik duplicate oranında modülün tamamen pasife alınması puan kaybını sınırlamak yerine doğrudan kalıcı kaçırma riski doğuruyor.
- İnternet yasağı bağlamında belirgin bir dış servis çağrısı görünmüyor; ancak `frame_url` mutlak URL kabulü (http ile başlayan URL’lerin doğrudan kullanılması) şartname güvenlik sınırını mimari seviyede esnetiyor.

# 2. Şartname Uyumluluk Analizi

## 2.1 Görev 1 (Nesne Tespiti) Uyum Durumu

- Sınıf kapsamı şartnamedeki 4 sınıfla hizalı (`0,1,2,3`) ve tespit sonrası yalnız bu sınıflar payload’a alınıyor.
- `landing_status` üretimi UAP/UAİ için zorunlu, taşıt/insan için `-1` olacak şekilde uygulanmış.
- `motion_status` alanı kanonik isimle normalize edilip zorunlu alana dönüştürülüyor.
- Aynı nesneye çoklu bbox riskini azaltmak için NMS + containment suppression mevcut.

Değerlendirme: Şema uyumu **yüksek**, metrik dayanıklılığı (özellikle recall) **orta**.

## 2.2 Görev 2 (GPS Kestirimi) Uyum Durumu

- `gps_health=1` iken sunucu translasyonunu kullanma davranışı şartnameyle uyumlu.
- `gps_health=0` iken optik akış tabanlı kendi pozisyonunu üretip gönderme davranışı mevcut.
- Ancak bozuk sağlık alanlarının “0’a zorlama” yaklaşımı, gerçek sağlık sinyalinin kaybında gereksiz vision-only moda düşmeye neden olabilir.

Değerlendirme: Fonksiyonel uyum **orta-yüksek**, hata birikimi riski **yüksek**.

## 2.3 Görev 3 (Referans Obje Eşleştirme) Uyum Durumu

- Oturum başı referans alma, doğrulama, ID normalizasyonu ve duplicate karantinası uygulanıyor.
- Çerçeve bazında tespitler `detected_undefined_objects` içinde raporlanıyor.
- Kritik duplicate oranında modülün tamamen pasife alınması şartname ihlali değil; ancak puanlama açısından yüksek fırsat kaybı riski.

Değerlendirme: Şema uyumu **orta-yüksek**, süreklilik/coverage **orta**.

## 2.4 Genel Rekabet Döngüsü Uyum Durumu

- Önce gönderim, sonra yeni kare fetch yaklaşımı korunuyor; aynı kare için çoklu gönderimde istemci-idempotency koruması var.
- Fallback payload ile “boş ama geçerli” JSON gönderme stratejisi döngü kırılmasını azaltıyor.
- Buna rağmen kalıcı reject, duplicate-frame fırtınası ve transient wall-time limitinde oturum erken sonlandırma var; bu durum 2250 kare kapsamını fiilen azaltabilir.

Değerlendirme: Döngü kontrolü **güçlü**, tam oturum tamamlama garantisi **orta**.

# 3. Kritik Riskler

## Bulgu K1 — Sonuç sayısı sınırlama mekanizması nedeniyle mAP/Recall sistematik düşüşü

- **Dosya/Fonksiyon:** `src/network.py` / `_apply_object_caps`, `_preflight_validate_and_normalize_payload`
- **Risk Tanımı:** Algılanan nesneler sınıf kotaları ve global cap ile kırpılıyor; yoğun sahnelerde gerçek tespitler rapora hiç girmeyebilir.
- **Şartname İhlali / Puan Etkisi:** Şartname kare başına “zorunlu sınıf raporlama” ister; cap nedeniyle eksik bbox artar, recall düşer, mAP geriler (Görev 1 ve Görev 3).
- **Olasılık:** Yüksek
- **Tetikleyici Senaryo:** Kalabalık şehir sahnesi, çoklu insan/taşıt, eşzamanlı UAP/UAİ görüntüsü; quota dolması.
- **Tespit Yöntemi:** Kod akışında normalize→quota→global cap sırasının statik izlenmesi; drop loglarının varlığı.
- **Mimari Düzeyde İyileştirme Yönü:** Kare başı rapor bütçesi yönetimini metrik-duyarlı hale getiren ve kritik sınıf kaybını önceliklendiren çıktı politikası.

## Bulgu K2 — Frame metadata bozulmasında `gps_health` zorla 0’a çekildiği için hata sarmalı

- **Dosya/Fonksiyon:** `src/network.py` / `_validate_frame_data`; `src/localization.py` / `update`
- **Risk Tanımı:** Sağlık alanı bozuk/eksik olduğunda değer 0’a set ediliyor; sistem vision-only moda düşüp drift üretebilir.
- **Şartname İhlali / Puan Etkisi:** Şartnamede ilk 450 karede sağlık=1 beklentisi var; yanlış 0 yorumlaması gereksiz kendi kestirime geçerek Öklid hata metriğini yükseltir (Görev 2).
- **Olasılık:** Orta-Yüksek
- **Tetikleyici Senaryo:** Sunucudan geçici şema sapması, null/unknown sağlık değeri.
- **Tespit Yöntemi:** Sağlık alanı dönüşüm/varsayılanlama kodunun statik incelenmesi; 0 fallback dalı.
- **Mimari Düzeyde İyileştirme Yönü:** Sağlık sinyalinde belirsizlikte kararı ayrıştıran güvenilirlik katmanı ve kaynak-doğruluk ayrımı.

## Bulgu K3 — Görev 3’ün kritik duplicate oranında tamamen kapanması

- **Dosya/Fonksiyon:** `main.py` / `_validate_task3_references`, `run_competition`
- **Risk Tanımı:** Duplicate oranı kritik eşik üstüne çıkınca `image_matcher=None` ile Görev 3 pasifleşiyor.
- **Şartname İhlali / Puan Etkisi:** Doğrudan ihlal olmayabilir; fakat oturum boyunca referans objeler hiç raporlanamayarak Görev 3 mAP’inde sert düşüş.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Sunucu referans paketinde tekrar eden object_id yoğunluğu.
- **Tespit Yöntemi:** Duplicate kritik koşulu ve pasif moda geçiş dalının statik analizi.
- **Mimari Düzeyde İyileştirme Yönü:** Tam kapatma yerine kapsama kaybını sınırlayan kademeli bütünlük modları.

## Bulgu K4 — Permanent reject ve transient limit sonrası oturum erken sonlandırma

- **Dosya/Fonksiyon:** `main.py` / `run_competition`, `_submit_competition_step`; `src/resilience.py` / `should_abort`
- **Risk Tanımı:** Art arda reject/failure durumlarında döngü kırılıp oturum erken bitebilir.
- **Şartname İhlali / Puan Etkisi:** 2250 kare hedef kapsamı düşer; sonuçsuz kareler toplam puanı düşürür.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** API tarafında ardışık 4xx/5xx, uzun transient dönem.
- **Tespit Yöntemi:** Abort eşikleri, breaker limitleri ve break path’lerinin statik incelenmesi.
- **Mimari Düzeyde İyileştirme Yönü:** Oturum sürekliliğini önceleyen, kritik-path’te kapsam kaybını minimize eden dayanıklılık politikası.

# 4. Orta ve Düşük Öncelikli Riskler

## Bulgu O1 — Agresif bbox filtreleme nedeniyle küçük/kısmi nesne kaçırma

- **Dosya/Fonksiyon:** `src/detection.py` / `_post_filter`
- **Risk Tanımı:** Min boyut, aspect ve üst sınır filtreleri küçük/uzak nesneleri dışarı atabilir.
- **Şartname İhlali / Puan Etkisi:** Kısmi görünür UAP/UAİ ve uzaktaki insan/taşıtta recall kaybı, mAP düşüşü.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Yüksek irtifa, düşük piksel alanlı hedefler.
- **Tespit Yöntemi:** Filtre koşullarının statik analizi ve şartnamedeki kısmi görünürlük maddesiyle karşılaştırma.
- **Mimari Düzeyde İyileştirme Yönü:** Filtrelerin sahne bağlamı ve sınıf kritikliğine göre uyarlanması.

## Bulgu O2 — UAP/UAİ kenar temasında otomatik uygunsuzluk yaklaşımının aşırı ceza üretme riski

- **Dosya/Fonksiyon:** `src/detection.py` / `_determine_landing_status`, `_is_touching_edge`
- **Risk Tanımı:** Kenar temasını katı uygunsuzluk sebebi saymak, bazı sınır durumlarda yanlış `landing_status` üretebilir.
- **Şartname İhlali / Puan Etkisi:** Yanlış iniş durumu AP’yi düşürür.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** UAP/UAİ neredeyse tam görünür ama piksel seviyesinde kenara temas eden kareler.
- **Tespit Yöntemi:** Landing karar ağacının statik incelemesi.
- **Mimari Düzeyde İyileştirme Yönü:** Karar güvenilirliğini geometrik kalite sinyalleriyle kademelendiren yaklaşım.

## Bulgu O3 — Referans eşleştirmede dönemsel düşük eşik (fallback interval) ile false positive artışı

- **Dosya/Fonksiyon:** `src/image_matcher.py` / `_match_reference`
- **Risk Tanımı:** Belirli aralıklarda eşik düşürülmesi yanlış eşleşme riskini artırır.
- **Şartname İhlali / Puan Etkisi:** Görev 3’te yanlış ID/bbox eşleşmesi mAP kaybına neden olur.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Tekrarlı desenli yüzeyler, düşük doku, benzer referanslar.
- **Tespit Yöntemi:** Eşik seçim mantığının statik analizi.
- **Mimari Düzeyde İyileştirme Yönü:** Eşleşme doğruluğunu çoklu kanıtla doğrulayan karar mimarisi.

## Bulgu O4 — `frame_url` için mutlak HTTP URL kabulü ile ağ sınırı gevşemesi

- **Dosya/Fonksiyon:** `src/network.py` / `download_image`
- **Risk Tanımı:** `http` ile başlayan URL’lerin doğrudan kullanılması, yerel ağ sınırı dışına çıkma riskini mimari olarak açık bırakır.
- **Şartname İhlali / Puan Etkisi:** İnternet yasağı (HW-004/HW-006) bağlamında güvenlik ve diskalifiye riski.
- **Olasılık:** Düşük-Orta
- **Tetikleyici Senaryo:** Yanlış yapılandırılmış veya kötü niyetli frame metadata.
- **Tespit Yöntemi:** URL birleştirme/dogrulama akışının statik incelenmesi.
- **Mimari Düzeyde İyileştirme Yönü:** Ağ hedeflerini yarışma yerel ağıyla sınırlayan açık güven sınırı.

## Bulgu O5 — Varsayılan çalışma modu ve debug ayarları yarışma modu ile çelişebilir

- **Dosya/Fonksiyon:** `config/settings.py` / `Settings` sınıfı
- **Risk Tanımı:** Varsayılan `SIMULATION_MODE=True`, `DEBUG=True` bırakılması operasyonel hata riski doğurur.
- **Şartname İhlali / Puan Etkisi:** Yanlış modla başlama halinde oturum çıktısı geçersizleşebilir veya hız düşebilir.
- **Olasılık:** Orta
- **Tetikleyici Senaryo:** Yarışma gününde config override unutulması.
- **Tespit Yöntemi:** Statik konfigürasyon incelemesi.
- **Mimari Düzeyde İyileştirme Yönü:** Çalıştırma profiline göre güvenli varsayılan ve açılışta zorlayıcı mod doğrulama.

# 5. Performans ve Kaynak Değerlendirmesi

- Mimari; thread pool ile fetch/submit ayrıştırarak gecikmeyi azaltmayı hedefliyor, 1 FPS asgari gereksinimine erişim açısından olumlu.
- Buna karşın SAHI + yüksek çözünürlük + ORB/SIFT + optik akış kombinasyonu kaynak baskısını artırır; yoğun sahnede FPS dalgalanması beklenir.
- Circuit breaker ve degrade stratejisi kilitlenmeyi azaltır; ancak fazla agresif abort koşulları kapsam kaybı yaratabilir.
- GPU yokluğunda CPU fallback mevcut, fakat uzun oturum (4x60 dk) boyunca süreklilik riski artar.

# 6. Belirsizlikler ve Koşullu Riskler

- **Varsayım:** Nihai yarışma JSON şeması taslaktan farklılaşırsa, mevcut kanonik alanlar (`motion_status`, `detected_translations` vb.) yeniden uyumsuzlaşabilir.
- **Varsayım:** Sunucu `frame_url`/`frame_id` sözleşmesini tutarlı korursa mevcut doğrulama yeterli kalır; aksi halde frame hizalaması bozulabilir.
- **Varsayım:** Kamera iç parametreleri yarışma günü farklıysa mevcut odometri ölçekleme hatası artabilir.
- **Varsayım:** Görev 3 referanslarının kalite/çeşitliliği düşük olursa feature tabanlı yaklaşımda mAP oynaklığı yükselir.

# 7. Genel Sağlık Skoru (0–10)

**6.8 / 10**

Gerekçe: Temel şartname maddeleri için işlevsel karşılıklar mevcut ve rekabet döngüsü bilinçli kurgulanmış. Ancak puan kırıcı risklerin önemli kısmı “güvenli düşüş” adı altında kapsama kaybı yaratıyor (özellikle obje clipping, task3 pasifleştirme, erken abort). Uzun oturum ve değişken kare koşullarında bu riskler toplam puanı anlamlı biçimde aşağı çekebilir.
