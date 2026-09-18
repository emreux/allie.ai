# live-assistant

Canlı konuş-konuş modelleri üzerinde çalışan, kendi API anahtarınla ve kendi
dilinde konuşan bir Windows masaüstü sesli asistanı.

**Durum: henüz kullanılabilir değil.** Bu depo,
[windows-voice-assistant](https://github.com/emreux/windows-voice-assistant)
projesinin devamıdır (kapsamı için tamamlandı, v0.4.0'da rafa kaldırıldı). Sesin
modele nasıl ulaştığından bağımsız her şey oradan aktarıldı - tek izin
kapısının arkasındaki araçlar, notlar, hatırlatıcılar, posta, mesajlaşma, müzik,
tepsi simgesi, `doctor`, `purge`, `autostart`, dil paketleri - ve üç aşamalı
boru hattı (yerel tanıyıcı, metin modeli, yerel ses) tek bir canlı oturumla
değiştiriliyor: mikrofon modele akar, modelin sesi geri akar, araya girip
konuşarak kesebilirsin.

Bugün olan o temel: `live-assistant setup`, `doctor`, `cost`, `purge --all`,
`autostart` ve girişler çalışıyor; `live-assistant run` canlı döngünün henüz
bağlanmadığını söyleyip çıkıyor. İlk canlı adaptör, ücretsiz bir AI Studio
anahtarıyla Google'ın Gemini Live API'si; OpenAI adaptörü sonra gelecek.

Bu README bir yer tutucudur; canlı döngü çalıştığında yeniden yazılacak.
