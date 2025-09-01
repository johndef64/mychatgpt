# Audio2Text - Trascrizione Audio Avanzata

Questo repository contiene strumenti per la trascrizione di file audio e video in testo utilizzando API di intelligenza artificiale.

## Scripts Disponibili

### 1. Audio2Text2Clipboard.py
Script originale che utilizza l'API OpenAI Whisper per trascrivere file audio brevi (fino a 25MB).

**Caratteristiche:**
- Supporta file fino a 25MB
- Copia automaticamente il testo negli appunti
- Salva la trascrizione in file
- Supporta molteplici formati audio/video

### 2. Audio2Text-long-groq.py ⭐ NUOVO
Script avanzato per file audio lunghi (fino a 1 ora) utilizzando l'API Groq.

**Caratteristiche:**
- ✅ Supporta file audio lunghi (fino a 1 ora)
- ✅ Divisione automatica in chunks di 10 minuti
- ✅ Pulizia e formattazione del testo con AI
- ✅ Output in formato Markdown
- ✅ Utilizza FFmpeg per la divisione audio
- ✅ Salvataggio automatico in cartella dedicata

## Requisiti

### Software Necessario
1. **Python 3.7+**
2. **FFmpeg** (necessario per Audio2Text-long-groq.py)
   - Windows: Scarica da [ffmpeg.org](https://ffmpeg.org/download.html)
   - Assicurati che ffmpeg sia nel PATH di sistema

### Dipendenze Python
```bash
pip install -r requirements.txt
```

### File di Configurazione
- `openai_api_key.txt` - Chiave API OpenAI (per Audio2Text2Clipboard.py)
- `groq_api.txt` - Chiave API Groq (per Audio2Text-long-groq.py)

## Utilizzo

### Script per Audio Lunghi (Consigliato)

```bash
# Trascrizione completa con pulizia AI
python Audio2Text-long-groq.py audio_lungo.mp3

# Solo trascrizione senza pulizia
python Audio2Text-long-groq.py audio_lungo.mp3 --no-cleanup

# Personalizza durata chunks (5 minuti)
python Audio2Text-long-groq.py audio_lungo.mp3 --chunk-duration 300
```

**Utilizzo con file batch:**
```batch
audio2text-long.bat "percorso\al\file\audio.mp3"
```

### Script per Audio Brevi

```bash
# Trascrizione con copia negli appunti
python Audio2Text2Clipboard.py audio_breve.mp3

# Con lingua specifica
python Audio2Text2Clipboard.py audio_breve.mp3 --language it

# Solo stampa, senza appunti
python Audio2Text2Clipboard.py audio_breve.mp3 --no-clipboard
```

## Formati Supportati

- **Audio:** MP3, WAV, FLAC, OGG, M4A, WEBM
- **Video:** MP4, AVI, MOV, WMV, MKV, MPEG

## Output

### Audio2Text-long-groq.py
Lo script crea una cartella `Audio2Text_Output` con:
- `filename_timestamp_raw.txt` - Trascrizione grezza
- `filename_timestamp_cleaned.md` - Versione pulita e formattata in Markdown

### Audio2Text2Clipboard.py
- Testo copiato negli appunti
- File salvato in cartella `Audio2Text`

## Caratteristiche Avanzate

### Gestione File Lunghi
- Divisione automatica in chunks di 10 minuti
- Trascrizione parallela dei chunks
- Ricomposizione intelligente del testo

### Pulizia AI del Testo
Il modello `openai/gpt-oss-20b` di Groq viene utilizzato per:
- Correggere errori di ortografia e grammatica
- Migliorare la punteggiatura
- Organizzare il testo in paragrafi
- Aggiungere titoli e sottotitoli
- Formattare in Markdown

### Gestione Errori
- Controllo automatico della presenza di FFmpeg
- Gestione di file troppo grandi
- Recovery da errori di trascrizione di singoli chunks
- Pulizia automatica dei file temporanei

## Configurazione API

### Groq API
1. Registrati su [console.groq.com](https://console.groq.com)
2. Ottieni la tua API key
3. Salvala nel file `groq_api.txt`

### OpenAI API
1. Registrati su [platform.openai.com](https://platform.openai.com)
2. Ottieni la tua API key
3. Salvala nel file `openai_api_key.txt`

## Esempi di Utilizzo

### Trascrizione di una lezione universitaria (1 ora)
```bash
python Audio2Text-long-groq.py lezione_fisica.mp3
```
Output: Trascrizione organizzata con titoli, paragrafi e formattazione Markdown.

### Trascrizione di un meeting aziendale
```bash
python Audio2Text-long-groq.py meeting_team.mp4 --chunk-duration 600
```

### Trascrizione rapida di una nota vocale
```bash
python Audio2Text2Clipboard.py nota_vocale.m4a --language it
```

## Troubleshooting

### Errore "FFmpeg not found"
- Installa FFmpeg e aggiungilo al PATH di sistema
- Su Windows, riavvia il prompt dei comandi dopo l'installazione

### Errore "API key not found"
- Verifica che i file `groq_api.txt` o `openai_api_key.txt` esistano
- Controlla che le API keys siano valide

### File troppo grande
- Usa Audio2Text-long-groq.py per file lunghi
- Riduci la qualità audio se necessario

## Prestazioni

- **Audio2Text2Clipboard.py:** File fino a 25MB in ~30 secondi
- **Audio2Text-long-groq.py:** File di 1 ora in ~10-15 minuti (dipende dalla velocità di rete)

## Licenza

Questo progetto è rilasciato sotto licenza MIT.
