# Sprint Backlog - Kalan Maddeler (2026-03-03)

Bu dosya, [genel_saglik_skoru_maksimizasyon_todo.md](C:/Users/siimsek/Desktop/AIA/audit/genel_saglik_skoru_maksimizasyon_todo.md) icindeki acik (`[ ]`) maddeleri sprint planina cevirir.

Ozet:
- Kalan acik madde: `25`
- Tamamlanan madde: `78`
- Ortam notu: Windows, CUDA yok, dataset yok. Model kalite metrik sprintleri "dataset/gpu gerekli" olarak isaretlendi.

## Sprint 1 - Runtime Guvenligi ve Frame Akisi
Hedef: KR-01 + KR-03 cekirdek akisini tamamlamak.

1. `frame_state_machine` (`FETCHED -> PROCESSED -> SUBMITTING -> ACKED`) kod seviyesinde netlestir.
2. Idempotent gonderim + "frame kapanisi sadece ACK ile" kuralini testlerle zorunlu hale getir.
3. `ack_before_next_fetch` kosulunu tum hata yollarinda dogrula.
4. Duplicate storm icin deterministik terminal eylem + guvenli oturum kapanisi ekle.
5. Duplicate kararlarini KPI/telemetri loglarina sabit formatta yaz.

Definition of Done:
- Frame drop yolu yok.
- Duplicate storm senaryosu kilitlenmeden deterministik sonlaniyor.
- ACK gelmeden yeni frame fetch edilmiyor (test ile dogrulandi).

Durum (2026-03-03):
- Tamamlandi (Madde 1-5 kodlandi, testlerle dogrulandi).

## Sprint 2 - Resilience ve Degrade Karar Matrisi
Hedef: KR-04 kalan maddeleri kapatmak.

1. Degrade modunda "akisi surdur / oturumu sonlandir / fallback agirligini artir" karar matrisini netlestir.
2. Uzun kosu debug testini (`DEBUG=True`) ekleyip dongu durmasi olmadigini kanitla.
3. Transient ag bozulmasi senaryosu icin kontrolsuz kesilmeme regression testi ekle.

Definition of Done:
- Degrade karar matrisi kodda tek noktada.
- DEBUG uzun kosu testleri geciyor.
- Transient agda erken abort yok.

Durum (2026-03-03):
- Tamamlandi (Madde 1-3 kodlandi, uzun-kosu debug + transient regression testleri eklendi).

## Sprint 3 - Gorev 2 Kalibrasyon ve 3B Dogruluk (Dataset/GPU Gerekli)
Hedef: KR-05 metrik hedeflerini olculebilir hale getirmek.

1. `FOCAL_LENGTH_PX`, `CAMERA_CX`, `CAMERA_CY` icin startup kalibrasyon guard ekle.
2. GPS kesinti senaryolari icin olcum pipeline'i kur (ortalama 3D hata).
3. `%20` hata iyilesmesi hedefini benchmark raporuyla dogrula.

Definition of Done:
- Kalibrasyon guard active.
- 3D hata benchmark raporu var.
- `%20` iyilesme metrikle kanitlandi.

Durum (2026-03-03):
- Kismi tamamlandi (Madde 1 tamam; metrik benchmark maddeleri dataset/gpu bagimli).

## Sprint 4 - Gorev 1 mAP/Recall ve Motion Dogrulugu (Dataset/GPU Gerekli)
Hedef: KR-06 kalan maddeleri tamamlamak.

1. Class-aware/hibrit NMS davranisini netlestir ve test et.
2. Boyut/aspect filtrelerini sinif bazli adaptif yap.
3. `MOVEMENT_MIN_HISTORY` oncesi erken hareket sinyalini guclendir.
4. Yeni track'lerde histerezis + guven skoru ile motion kararini stabilize et.
5. mAP@0.5 tuning pipeline calistir ve raporla.

Definition of Done:
- mAP@0.5 artisi `>= %8`
- motion_status dogrulugu artisi `>= %10`

Durum (2026-03-03):
- Kismi tamamlandi (Madde 1-4 tamam; Madde 5 tuning pipeline dataset/gpu bagimli).

## Sprint 5 - Fallback Skor Koruma + Payload/Task3
Hedef: KR-07, KR-PAYLOAD, KR-09 kalanlari tamamlamak.

1. Degrade modunda bos tespit gonderimini son care politikasi ile sinirla.
2. Hafif inference profili ile algi surekliligini koru.
3. Versioned payload adapter katmanini tamamla.
4. Schema degisim testlerinde reject oranini hedef altina indir.
5. Task3 referans dogrulamayi tek merkezde birlestir.
6. `TASK3_MAX_REFERENCES` asiminda onceliklendirme + raporlama + batch stratejisi ekle.
7. `detected_undefined_objects` icin quality/confidence sinyali ekle.
8. Termal/uydu benzeri domain farklari icin fallback descriptor stratejisi ekle.

Definition of Done:
- Submit reject hedef altinda.
- Task3 referans yogunluk senaryolarinda AP dususu azaltiyor.

Durum (2026-03-03):
- Kismi tamamlandi (Madde 1-3/5-8 tamam; Madde 4 icin reject-orani metrik prove acik).

## Sprint 6 - Performans ve Go/No-Go
Hedef: KR-11 + KR-12 ve operasyonel kapanis.

1. `FPS < 1` otomatik koruma profili.
2. SAHI/full-frame inference tutarlilik hizalama.
3. Uzun kosuda periyodik GPU bellek/telemetri bakimi.
4. JSON log frekansini performans profiline gore dinamiklestir.
5. 4 prova oturumu (60 dk) tamamla.
6. Son surum freeze: config hash, model surumu, payload profil surumu, metrik ozet.

Definition of Done:
- 60 dk uzun kosu stabil.
- Go kriterleri saglanmadan competition'a gecis yok.

Durum (2026-03-03):
- Kismi tamamlandi (Madde 1-4 tamam; Madde 5-6 operasyonel prova/freeze acik).

## Hemen Baslanacak Isler (Oncelik Sirasi)
1. Sprint 5 / Madde 4 (schema degisim testlerinde reject oranini hedef altina indirme)
2. Sprint 3-4 / Dataset-GPU metrik sprintleri (mAP@0.5, 3D hata benchmark, AP raporu)
3. Sprint 6 / Madde 5-6 (4 prova oturumu + surum freeze/go-no-go kapanisi)
