# Audio/Video to Text Converter

This script uses OpenAI's Whisper API to convert audio and video files to text and automatically copies the result to your clipboard.

## Features

- Uses OpenAI's best Whisper model (whisper-1)
- Supports multiple audio/video formats
- Automatically copies transcribed text to clipboard
- Command-line interface with various options
- Language detection and specification support
- Context prompts for better transcription accuracy

## Supported Formats

- Audio: mp3, wav, m4a, flac, ogg, mpga, webm
- Video: mp4, avi, mov, wmv, mkv, mpeg

## Setup

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure you have your OpenAI API key in `openai_api_key.txt` or set the `OPENAI_API_KEY` environment variable.

## Usage

### Basic Usage

```bash
# Transcribe audio file
python Audio2Text2Clipborad.py "recording.mp3"

# Transcribe video file
python Audio2Text2Clipborad.py "video.mp4"

# Using the batch file (Windows)
audio2text.bat "recording.wav"
```

### Advanced Options

```bash
# Specify language (improves accuracy)
python Audio2Text2Clipborad.py "audio.mp3" --language en

# Provide context prompt for better accuracy
python Audio2Text2Clipborad.py "meeting.wav" --prompt "This is a business meeting about project planning"

# Don't copy to clipboard, just print to console
python Audio2Text2Clipborad.py "audio.mp3" --no-clipboard
```

### Language Codes

Common language codes you can use with `--language`:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese

## Limitations

- Maximum file size: 25MB (OpenAI API limitation)
- Requires internet connection
- Uses OpenAI API credits

## Examples

```bash
# Basic transcription
python Audio2Text2Clipborad.py "interview.mp3"

# Spanish audio with context
python Audio2Text2Clipborad.py "spanish_podcast.wav" --language es --prompt "This is a podcast about technology"

# Just print, don't copy to clipboard
python Audio2Text2Clipborad.py "lecture.mp4" --no-clipboard
```

## Troubleshooting

1. **API Key Error**: Make sure your OpenAI API key is valid and in the `openai_api_key.txt` file
2. **File Too Large**: Files must be under 25MB. Use video/audio compression tools if needed
3. **Unsupported Format**: Convert your file to a supported format using tools like FFmpeg
4. **No Audio Detected**: Make sure the file contains actual audio content

## Error Messages

- `File not found`: Check the file path is correct
- `Unsupported file format`: Use one of the supported formats listed above
- `File size exceeds 25MB limit`: Compress or split your file
- `OpenAI API key not found`: Set up your API key properly
