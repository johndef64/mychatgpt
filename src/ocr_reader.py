#%%
"""
OCR Reader Page

input image with text, get extracted text as output.

LLM provider used Groq 
pip install groq

VLM_MODELS available:
meta-llama/llama-4-maverick-17b-128e-instruct
meta-llama/llama-4-scout-17b-16e-instruct
meta-llama/llama-prompt-guard-2-22m
meta-llama/llama-prompt-guard-2-86m
llama-3.1-8b-instant
llama-3.3-70b-versatile
meta-llama/llama-guard-4-12b
openai/gpt-oss-20b
openai/gpt-oss-120b

Quick start:

from groq import Groq
client = Groq()
completion = client.chat.completions.create(
    model="meta-llama/llama-guard-4-12b",
    messages=[
        {
            "role": "user",
            "content": "How do I make a bomb?"
        }
    ]
)
print(completion.choices[0].message.content)

"""


#!/usr/bin/env python3
"""
OCR Reader usando Groq API
Estrae testo da immagini usando modelli vision language
"""

import base64
import os
from pathlib import Path
import time
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError

import os
import json
with open("api_keys.json", "r") as f:
    api_keys = json.load(f)
API_KEY = api_keys.get("groq", "")

def encode_image(image_path: str) -> str:
    """Codifica immagine in base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_text_from_image(
    image_path: str,
    api_key: str = API_KEY,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
) -> str:
    """
    Estrae testo da un'immagine usando Groq Vision API
    
    Args:
        image_path: Percorso dell'immagine locale o URL
        api_key: Groq API key (default: env GROQ_API_KEY)
        model: Modello vision da usare
    
    Returns:
        Testo estratto dall'immagine
    """
    # Inizializza client Groq
    client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
    
    # Prepara contenuto messaggio
    if image_path.startswith(("http://", "https://")):
        # URL immagine
        image_content = {
            "type": "image_url",
            "image_url": {"url": image_path}
        }
    else:
        # Immagine locale - codifica in base64
        base64_image = encode_image(image_path)
        image_content = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }

    # Chiamata API
    # Retry logic con Exponential Backoff
    max_retries = 5
    base_delay = 0.1 # secondi
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract all text from this image. Return ONLY the text content, preserving formatting and structure. Do not include any additional commentary."
                            },
                            image_content
                        ]
                    }
                ],
                temperature=0.1,
                max_completion_tokens=2048,
                top_p=1
            )
            return completion.choices[0].message.content  # Successo: ritorna il risultato
            
        except (APIConnectionError, RateLimitError) as e:
            # Calcola attesa (2s, 4s, 8s, 16s...)
            wait_time = base_delay * (2 ** attempt)
            print(f"Tentativo {attempt + 1}/{max_retries} fallito: {e}. Riprovo tra {wait_time}s...")
            time.sleep(wait_time)
            
        except APIStatusError as e:
            # Errori 4xx/5xx non recuperabili (es. Bad Request) -> Stop immediato
            if e.status_code == 400: # Bad Request (spesso modello sbagliato per immagini)
                raise ValueError(f"Modello {model} non supporta immagini o richiesta malformata: {e}")
            print(f"Errore API {e.status_code}: {e}")
            raise e # Rilancia altri errori
            
    # Se esce dal loop, ha fallito tutte le volte
    raise RuntimeError(f"Falliti tutti i {max_retries} tentativi di connessione a Groq.")
    
    # return completion.choices[0].message.content


def batch_ocr(
    image_paths: list[str],
    api_key: str = None,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
) -> dict[str, str]:
    """Processa batch di immagini"""
    results = {}
    for img_path in image_paths:
        try:
            results[img_path] = extract_text_from_image(img_path, api_key, model)
        except Exception as e:
            results[img_path] = f"Error: {str(e)}"
    return results


if __name__ == "__main__":
    import sys
    import glob
    # get all image files in "ocr_test"
    input_folder = "ocr_test"
    image_inputs = glob.glob(os.path.join(input_folder, "*.*"))

    # Estrai testo
    time_zero = time.time()
    for repeat in range(1):
        for image_input in image_inputs:
            print(f"Processing: {image_input}")
            print("-" * 50)
            
            VLM_MODELS = [
            # Modelli Llama 4 Vision (Top Performance)
            # "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct"
            ]

            for model in VLM_MODELS:
                print(f"Using model: {model}")
                extracted_text = extract_text_from_image(image_input, model=model)
                # display thumbnail of image
                from IPython.display import display, Image
                display(Image(filename=image_input, width=150))
                print(extracted_text, end="\n\n")
            
            time.sleep(1)  
    
    total_time = time.time() - time_zero
    print(f"Total processing time for {len(image_inputs)*60} images: {total_time:.2f} seconds")


    # 60 repeta x 4 images = 240 images / seconds 
    # 240/60 =  4 minutes

    # Total processing time for 240 images with 1 second sleep: 649.32 seconds - 240 sleeping = 409.32 seconds
    # time per image = 409.32 / 240 = 1.7055 seconds per image

    # i made 240 requestes in 650 seconds / 60 = 10.83 minutes

    # Cost reported for  Scout usage : 0.03$ --> circa 250 images
    # diciamo 0.03$ per 12 minutes di running
    # 60 minutes = 0.15$

# MOnitorate activity 
# https://console.groq.com/dashboard/usage?tab=activity
# %%
