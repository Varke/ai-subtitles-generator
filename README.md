# subtitles-generator

Скрипт автоматически создаёт субтитры для видео: 
- распознаёт речь (Whisper),
- формирует `.srt`,
- при необходимости озвучивает перевод (Piper TTS),
- рендерит финальный ролик с наложенным текстом.

Поддерживается:
- выбор устройства (`cpu`, `cuda` и т.д.) и точности вычислений (`int8`, `float16`),
- кастомные шрифты и оформление субтитров,
- генерация дубляжа на основе TTS (с настройкой голоса и параметров речи),
- очистка текста от мусора (таймкоды, цифры, ссылки).

## Пример запуска
Субтитры без озвучки:
```bash
python autosubs.py "202505151721.mp4" --device cpu --compute int8 --font "GT Eesti Pro Display Medium"
```

С озвучкой перевода (английский голос Ryan):
```bash
python autosubs.py "202505151721.mp4" --device cpu --compute int8 \
  --font "Arial Unicode MS" \
  --piper-voice voices/en_US-ryan-high.onnx \
  --piper-voice-config voices/en_US-ryan-high.onnx.json \
  --piper-length-scale 0.95
```

## Результат
- `.srt` файл с субтитрами,
- озвучка (если включена) в `result/<имя>_voiceover_en.wav`,
- итоговое видео с наложенным текстом в папке `result/`.
