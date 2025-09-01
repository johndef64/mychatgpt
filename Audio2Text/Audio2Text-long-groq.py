
#%%
#!/usr/bin/env python3
"""
Audio to Text Converter with Chunking Support for Long Audio Files
Uses Groq API for transcription and text cleaning/formatting
"""

import os
import sys
import argparse
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from groq import Groq

# Models configuration
WHISPER_MODEL = "whisper-large-v3-turbo"
CLEANUP_MODEL = "openai/gpt-oss-20b"

# Audio chunk settings (in seconds)
CHUNK_DURATION = 600  # 10 minutes per chunk
MAX_FILE_SIZE_MB = 25  # Groq file size limit

# Common language codes for quick reference
COMMON_LANGUAGES = {
    'en': 'English',
    'it': 'Italian', 
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'pl': 'Polish'
}

def show_language_codes():
    """Display common language codes"""
    print("\n=== COMMON LANGUAGE CODES ===")
    for code, name in COMMON_LANGUAGES.items():
        print(f"  {code:2} - {name}")
    print("\nFor complete list, use: python Audio2Text-long-groq.py --help")
    print("="*30)

def load_groq_api_key():
    """Load Groq API key from file"""
    try:
        api_key_file = Path(__file__).parent / "groq_api.txt"
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
            if not api_key:
                raise ValueError("API key file is empty")
            return api_key
    except FileNotFoundError:
        raise ValueError("groq_api.txt file not found. Please create it with your Groq API key.")
    except Exception as e:
        raise ValueError(f"Failed to load API key: {e}")

def check_ffmpeg():
    """Check if ffmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_audio_duration(file_path):
    """Get audio duration in seconds using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'csv=p=0', str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Audio duration: {result.stdout.strip()} seconds")
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        return None

def split_audio_into_chunks(file_path, chunk_duration):
    """Split audio file into chunks using ffmpeg"""
    chunks = []
    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    
    try:
        print(f"Splitting audio into {chunk_duration//60}-minute chunks...")
        
        # Get audio duration
        duration = get_audio_duration(file_path)
        if duration:
            num_chunks = int(duration // chunk_duration) + 1
            print(f"Audio duration: {duration/60:.1f} minutes, creating {num_chunks} chunks")
        
        chunk_index = 0
        start_time = 0
        
        while True:
            chunk_file = Path(temp_dir) / f"chunk_{chunk_index:03d}.wav"
            
            cmd = [
                'ffmpeg', '-i', str(file_path),
                '-ss', str(start_time),
                '-t', str(chunk_duration),
                '-acodec', 'pcm_s16le',  # Use uncompressed format
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                '-y',  # Overwrite output files
                str(chunk_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                if chunk_index == 0:
                    raise Exception(f"FFmpeg failed: {result.stderr}")
                else:
                    # No more chunks to extract
                    break
            
            # Check if chunk file was created and has content
            if chunk_file.exists() and chunk_file.stat().st_size > 1000:  # At least 1KB
                chunks.append(str(chunk_file))
                print(f"Created chunk {chunk_index + 1}: {chunk_file.name}")
                chunk_index += 1
                start_time += chunk_duration
            else:
                break
        
        if not chunks:
            raise Exception("No valid chunks were created")
            
        return chunks, temp_dir
        
    except Exception as e:
        # Clean up on error
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise Exception(f"Failed to split audio: {e}")

def transcribe_chunk(client, chunk_path, chunk_number, language="en"):
    """Transcribe a single audio chunk"""
    try:
        print(f"Transcribing chunk {chunk_number}...")
        
        # Check file size
        file_size_mb = Path(chunk_path).stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise Exception(f"Chunk {chunk_number} is too large ({file_size_mb:.1f}MB)")
        
        with open(chunk_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(Path(chunk_path).name, file.read()),
                model=WHISPER_MODEL,
                response_format="text",
                language=language
            )
            
        return transcription.strip()
        
    except Exception as e:
        print(f"Error transcribing chunk {chunk_number}: {e}")
        return f"[TRANSCRIPTION ERROR CHUNK {chunk_number}]"

def transcribe_long_audio(file_path, chunk_duration=None, language="en"):
    """Transcribe long audio file by splitting into chunks"""
    if chunk_duration is None:
        chunk_duration = CHUNK_DURATION
        
    # Initialize Groq client
    api_key = load_groq_api_key()
    client = Groq(api_key=api_key)
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        raise Exception("FFmpeg not found. Please install FFmpeg to process long audio files.")
    
    # Get file info
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"Processing file: {file_path.name} ({file_size_mb:.1f}MB)")
    print(f"Using language: {language}")
    
    # Check if we need to split the file
    if file_size_mb <= MAX_FILE_SIZE_MB:
        duration = get_audio_duration(file_path)
        if duration and duration <= chunk_duration:
            print("File is small enough, transcribing directly...")
            with open(file_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(file_path.name, file.read()),
                    model=WHISPER_MODEL,
                    response_format="text",
                    language=language
                )
            return transcription.strip()
    
    # Split into chunks and transcribe
    chunks, temp_dir = split_audio_into_chunks(file_path, chunk_duration)
    all_transcriptions = []
    
    try:
        for i, chunk_path in enumerate(chunks, 1):
            transcription = transcribe_chunk(client, chunk_path, i, language)
            if transcription and transcription != f"[TRANSCRIPTION ERROR CHUNK {i}]":
                all_transcriptions.append(transcription)
        
        # Combine all transcriptions
        combined_text = "\n\n".join(all_transcriptions)
        
        print(f"✓ Transcription completed! Total chunks: {len(chunks)}")
        return combined_text
        
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def clean_and_format_text(text, client):
    """Clean and format transcribed text using GPT model"""
    print("Cleaning and formatting text with AI...")
    
    prompt = f"""Sei un esperto editor di testi. Il tuo compito è pulire e formattare la seguente trascrizione audio in italiano.

Segui queste istruzioni:
1. Correggi errori grammaticali e di ortografia
2. Migliora la punteggiatura e la struttura delle frasi
3. Organizza il testo in paragrafi logici
4. Aggiungi titoli e sottotitoli appropriati dove necessario
5. Formatta il risultato finale in Markdown
6. Mantieni il significato originale del contenuto
7. Se ci sono ripetizioni o errori di trascrizione, correggili
8. Se il testo sembra essere una conversazione, organizzalo di conseguenza

Ecco la trascrizione da pulire e formattare:

{text}

Fornisci SOLO il testo pulito e formattato in Markdown, senza commenti aggiuntivi."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=CLEANUP_MODEL,
            max_tokens=4000,
            temperature=0.1
        )
        
        cleaned_text = chat_completion.choices[0].message.content
        print("✓ Text cleaning and formatting completed!")
        return cleaned_text
        
    except Exception as e:
        print(f"Warning: Text cleaning failed: {e}")
        print("Returning original transcription...")
        return text

def save_transcription(text, original_file_path, cleaned_text=None):
    """Save transcription to files"""
    try:
        original_path = Path(original_file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = original_path.parent / "Audio2Text_Output"
        output_dir.mkdir(exist_ok=True)
        
        base_name = f"{original_path.stem}_{timestamp}"
        
        # Save raw transcription
        raw_file = output_dir / f"{base_name}_raw.txt"
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✓ Raw transcription saved: {raw_file}")
        
        # Save cleaned version if available
        if cleaned_text:
            cleaned_file = output_dir / f"{base_name}_cleaned.md"
            with open(cleaned_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"✓ Cleaned transcription saved: {cleaned_file}")
            return str(cleaned_file)
        
        return str(raw_file)
        
    except Exception as e:
        print(f"Error saving transcription: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Transcribe long audio files using Groq API with chunking support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Audio2Text-long-groq.py audio.mp3
  python Audio2Text-long-groq.py long_audio.wav --no-cleanup --language it
  python Audio2Text-long-groq.py video.mp4 --chunk-duration 300 --language fr

Supported Language Codes:
  en - English (default)
  it - Italian
  es - Spanish
  fr - French
  de - German
  pt - Portuguese
  ru - Russian
  ja - Japanese
  ko - Korean
  zh - Chinese
  ar - Arabic
  hi - Hindi
  nl - Dutch
  sv - Swedish
  da - Danish
  no - Norwegian
  fi - Finnish
  pl - Polish
  tr - Turkish
  he - Hebrew
  th - Thai
  vi - Vietnamese
  uk - Ukrainian
  cs - Czech
  hu - Hungarian
  ro - Romanian
  bg - Bulgarian
  hr - Croatian
  sk - Slovak
  sl - Slovenian
  et - Estonian
  lv - Latvian
  lt - Lithuanian
  mt - Maltese
  ga - Irish
  cy - Welsh
  is - Icelandic
  mk - Macedonian
  sq - Albanian
  eu - Basque
  ca - Catalan
  gl - Galician
  be - Belarusian
  az - Azerbaijani
  kk - Kazakh
  ky - Kyrgyz
  uz - Uzbek
  mn - Mongolian
  hy - Armenian
  ka - Georgian
  am - Amharic
  ne - Nepali
  si - Sinhala
  km - Khmer
  lo - Lao
  my - Myanmar
  tl - Filipino
  ms - Malay
  id - Indonesian
  ta - Tamil
  te - Telugu
  ml - Malayalam
  kn - Kannada
  gu - Gujarati
  pa - Punjabi
  bn - Bengali
  ur - Urdu
  fa - Persian
  ps - Pashto
        """
    )
    
    parser.add_argument("file", help="Path to audio or video file")
    parser.add_argument("--no-cleanup", action="store_true", 
                       help="Skip AI text cleaning and formatting")
    parser.add_argument("--chunk-duration", type=int, default=CHUNK_DURATION,
                       help=f"Chunk duration in seconds (default: {CHUNK_DURATION})")
    parser.add_argument("--language", "-l", default="en",
                       help="Language code for transcription (default: en). Use --help to see all supported codes")
    
    args = parser.parse_args()
    
    try:
        # Use the chunk duration from args
        chunk_duration = args.chunk_duration
        
        # Transcribe the audio
        print("Starting transcription process...")
        transcription = transcribe_long_audio(args.file, chunk_duration, args.language)
        
        if not transcription or not transcription.strip():
            print("Warning: No text was transcribed from the file")
            return
        
        print(f"\nRaw transcription completed! ({len(transcription)} characters)")
        
        # Clean and format text if requested
        cleaned_text = None
        if not args.no_cleanup:
            try:
                api_key = load_groq_api_key()
                client = Groq(api_key=api_key)
                cleaned_text = clean_and_format_text(transcription, client)
            except Exception as e:
                print(f"Text cleaning failed: {e}")
                print("Proceeding with raw transcription only...")
        
        # Save transcriptions
        saved_file = save_transcription(transcription, args.file, cleaned_text)
        
        if cleaned_text:
            print(f"\n✓ Process completed! Final output: {saved_file}")
            print(f"Cleaned text length: {len(cleaned_text)} characters")
        else:
            print(f"\n✓ Transcription completed! Output: {saved_file}")
            print(f"Raw text length: {len(transcription)} characters")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#%% 
# NOTEBOOK VERSION - For interactive use in Jupyter/VSCode
def transcribe_audio_notebook(file_path, language="en", chunk_duration=600, skip_cleanup=False):
    """
    Notebook-friendly version for interactive transcription
    
    Args:
        file_path (str): Path to audio file
        language (str): Language code (default: "en")
        chunk_duration (int): Chunk duration in seconds (default: 600)
        skip_cleanup (bool): Skip AI cleanup (default: False)
    
    Returns:
        tuple: (raw_transcription, cleaned_transcription, saved_file_path)
    """
    try:
        print(f"🎵 Transcribing: {Path(file_path).name}")
        print(f"🌍 Language: {language} ({COMMON_LANGUAGES.get(language, 'Unknown')})")
        print(f"⏱️ Chunk duration: {chunk_duration//60} minutes")
        print("-" * 50)
        
        # Transcribe the audio
        transcription = transcribe_long_audio(file_path, chunk_duration, language)
        
        if not transcription or not transcription.strip():
            print("⚠️ Warning: No text was transcribed from the file")
            return None, None, None
        
        print(f"\n✅ Raw transcription completed! ({len(transcription)} characters)")
        
        # Clean and format text if requested
        cleaned_text = None
        if not skip_cleanup:
            try:
                api_key = load_groq_api_key()
                client = Groq(api_key=api_key)
                cleaned_text = clean_and_format_text(transcription, client)
            except Exception as e:
                print(f"⚠️ Text cleaning failed: {e}")
                print("📝 Proceeding with raw transcription only...")
        
        # Save transcriptions
        saved_file = save_transcription(transcription, file_path, cleaned_text)
        
        if cleaned_text:
            print(f"\n🎉 Process completed! Final output: {saved_file}")
            print(f"📄 Cleaned text length: {len(cleaned_text)} characters")
        else:
            print(f"\n✅ Transcription completed! Output: {saved_file}")
            print(f"📄 Raw text length: {len(transcription)} characters")
        
        return transcription, cleaned_text, saved_file
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None

# %%
# Quick test function
def test_with_sample():
    """Test function with sample file"""
    print("🔍 Looking for audio files in current directory...")
    current_dir = Path(".")
    audio_files = []
    
    # Common audio extensions
    audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.mp4', '.avi', '.mov']
    
    for ext in audio_extensions:
        files = list(current_dir.glob(f"*{ext}"))
        audio_files.extend(files)
    
    if audio_files:
        print(f"📁 Found {len(audio_files)} audio files:")
        for i, file in enumerate(audio_files[:5], 1):  # Show max 5 files
            print(f"  {i}. {file.name}")
        
        if len(audio_files) > 5:
            print(f"  ... and {len(audio_files) - 5} more files")
        
        print(f"\n💡 To transcribe a file, use:")
        print(f"   transcribe_audio_notebook('{audio_files[0].name}')")
        print(f"   # or with Italian:")
        print(f"   transcribe_audio_notebook('{audio_files[0].name}', language='it')")
    else:
        print("❌ No audio files found in current directory")
        print("💡 Place an audio file in this directory and try again")
    
    # Show language help
    show_language_codes()

# %%
