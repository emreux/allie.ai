# live-assistant

Canlı konuş-konuş modelleri üzerinde çalışan, kendi API anahtarınla ve kendi
dilinde konuşan bir Windows masaüstü sesli asistanı.

Sen konuşursun, o kendi sesiyle cevap verir, üstüne konuşarak kesebilirsin.
Arada tanıyıcı da metin modeli de yok: mikrofon tek bir oturum üzerinden modele
akar, modelin sesi de geri akar.

**Durum: çalışıyor.** Bu depo,
[windows-voice-assistant](https://github.com/emreux/windows-voice-assistant)
projesinin devamıdır (kapsamı için tamamlandı, v0.4.0'da rafa kaldırıldı). Sesin
modele nasıl ulaştığından bağımsız her şey oradan aktarıldı - tek izin
kapısının arkasındaki araçlar, notlar, hatırlatıcılar, posta, mesajlaşma, müzik,
tepsi simgesi, `doctor`, `purge`, `autostart`, dil paketleri - ve üç aşamalı
boru hattının yerini Google'ın Gemini Live API'si üzerindeki tek canlı oturum
aldı. Bu README hâlâ kısa: programın bugün ne yaptığını anlatır, yapabileceği
her şeyi değil.

## Başlarken

```
uv sync
uv run live-assistant setup     # sağlayıcı, anahtar, model, dil, ses, mikrofon
uv run live-assistant run       # pencere; --terminal durum satırı, --tray simge
uv run live-assistant doctor    # kim cevaplıyor, bu makineden ne çıkıyor, dosyalar nerede
```

Ctrl+Alt+H dinlemeyi durdurur ve başlatır, ayrıca süren cevabı keser. Ctrl+C çıkar.

## Ne yapar

Bir sohbet, ve modelin çağırabildiği yirmi altı araç: saat, hava durumu,
aramalı notlar, turlar arasında sesle okunan hatırlatıcılar, posta okuma,
WhatsApp ve Telegram mesajları, müzik ve video, sistem sesi, web sayfaları,
pano, uygulama ve ayar sayfası açma, Store'dan kurulum, ve senin hakkında
hatırladıkları. Önemi olabilecek her şey sana geri okunur ve sözlü bir evet
bekler.

Oturum sen konuşmaya başlayınca açılır ve bir dakikalık sessizlikten sonra
kapanır: canlı model açık kaldığı sürece dakika başına ücretlendirir.
`live-assistant cost` turların ne harcadığını gösterir.

## Bu makineden ne çıkıyor

Oturum açıkken mikrofon Google'a akar, modelin cevabı da öyle. Canlı model bu
demektir. Yerel tanıyıcı - CPU üzerinde Whisper - tek bir iş için kaldı: bir
araç izin istediğinde evet mi hayır mı dediğini duymak. API anahtarın Windows
Kimlik Bilgisi Yöneticisi'nde durur; hiçbir dosyaya ve hiçbir günlüğe yazılmaz.

## Henüz yok

- **OpenAI.** İkinci adaptör ödemeli bir anahtar bekliyor; bu sürümün
  açabildiği tek sağlayıcı Gemini Live.
- **Uyandırma sözcüğü.** Algılayıcı yazıldı ve kapalı (`[wake] enabled`): ifadeyi
  tanıyan model henüz depoda değil.
- **Web arama.** Modele Google'ın kendi aramasını verebilirsin
  (`[live] web_search`); ücretsiz bir AI Studio anahtarı bunu reddediyor, bu
  yüzden kapalı geliyor.
- **Canlı modelin fiyatı.** Jetonun yanı sıra sesle de ücretlendirdiği için
  `pricing.toml` ona fiyat vermiyor; `cost` faturanın yalnızca bir parçası olan
  bir sayıyı doğruymuş gibi göstermek yerine bilinmediğini söylüyor.

Posta yalnızca sahte bir IMAP sunucusuna karşı çalıştırıldı.

## Geliştirici notu

`docs/` planı, kararları ve ölçümleri tutar; git'te değildir. Parçaların
şartnamesi testlerdir: `uv run pytest`. Lint, biçim ve tipler
`uv run ruff check .`, `uv run ruff format .`, `uv run mypy src --strict`;
dördü her commit'ten önce yeşildir.
