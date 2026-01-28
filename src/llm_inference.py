#%%

# A generic function script to call inference from Groq and OpenRouter models
# using OpenAI library as base for compatibility

import os
from typing import Optional, List, Dict, Any, Union
from openai import OpenAI
import json
import pandas as pd
import io

import base64
import requests
from PIL import Image
from io import BytesIO
from typing import Optional, Union

# Load API keys
with open("api_keys.json") as f:
    api_keys = json.load(f)

os.environ["GROQ_API_KEY"] = api_keys.get("groq")
os.environ["OPENROUTER_API_KEY"] = api_keys.get("openrouter")

# Model dictionaries for each provider
GROQ_MODELS = {
    # Modelli VLM (Vision & Text)
    1:"meta-llama/llama-4-maverick-17b-128e-instruct", 
    2:"meta-llama/llama-4-scout-17b-16e-instruct", 

    # Modelli Text-Only
    3:"meta-llama/llama-prompt-guard-2-22m", 
    4:"meta-llama/llama-prompt-guard-2-86m", 
    5:"llama-3.1-8b-instant", 
    6:"llama-3.3-70b-versatile", 
    7:"meta-llama/llama-guard-4-12b", 
    8:"openai/gpt-oss-20b", 
    9:"openai/gpt-oss-120b", 
}

OPENROUTER_MODELS = {
    # === ANTHROPIC (Vision & Text) ===
    # Modelli multimodali avanzati
    1: "anthropic/claude-opus-4.5",
    2: "anthropic/claude-sonnet-4.5",
    3: "anthropic/claude-haiku-4.5",
    4: "anthropic/claude-opus-4",
    5: "anthropic/claude-sonnet-4",

    # === OPENAI (Vision & Text) ===
    # Modelli general purpose multimodali
    6: "openai/gpt-5.2-pro",
    7: "openai/gpt-5.2",
    8: "openai/gpt-5.2-chat",
    
    # === OPENAI (Text & Code Only) ===
    # Modelli specializzati per coding e testo veloce
    9: "openai/gpt-5.2-codex",
    10: "openai/gpt-5.1-codex-max",

    # === GOOGLE (Vision & Text) ===
    11: "google/gemini-3-flash-preview",

    # === X.AI / GROK (Vision & Text) ===
    12: "x-ai/grok-4",
    13: "x-ai/grok-4.1-fast",
    14: "x-ai/grok-4-fast",
    15: "x-ai/grok-3",
    16: "x-ai/grok-3-mini",
    
    # === X.AI / GROK (Code Only) ===
    17: "x-ai/grok-code-fast-1",

    # === MISTRAL AI (Text Only) ===
    # Ottimizzati per efficienza e creatività testuale
    18: "mistralai/mistral-small-creative",
    19: "mistralai/ministral-14b-2512",
    20: "mistralai/ministral-8b-2512",
    21: "mistralai/ministral-3b-2512",
    22: "mistralai/devstral-2512",

    # === ALLEN AI (Text Only) ===
    # Modelli open science
    23: "allenai/olmo-3.1-32b-instruct",
    24: "allenai/olmo-3.1-32b-think",

    # === ALTRI PROVIDER (Text Only) ===
    # Deepseek, Moonshot, Minimax, ecc.
    25: "deepseek/deepseek-v3.2",
    26: "moonshotai/kimi-k2.5",
    27: "writer/palmyra-x5",
    28: "minimax/minimax-m2-her",
    29: "minimax/minimax-m2.1",
    30: "prime-intellect/intellect-3",
    31: "deepcogito/cogito-v2.1-671b",
}

def get_model(query: str, models_dict = GROQ_MODELS) -> str:
    """Simple function to select a model based on query content from GROQ_MODELS or OPENROUTER_MODELS 
    queries are splitted by " " while models are splitted by "-" and "/"
    
    Args:
        query: Search query with space-separated terms (e.g., "llama 3.3", "gpt 5", "claude opus")
        provider: "groq" or "openrouter" - which provider's models to search
    
    Returns:
        Model string from the selected provider, or the first model if no match found
    """
    import re
    
    # Normalize and split query
    query_parts = [part.lower().strip() for part in query.split() if part.strip()]
    
    
    if not query_parts:
        # Return first model from the selected provider
        return list(models_dict.values())[0]
    
    # Score each model based on query matches
    scores = {}
    for key, model_name in models_dict.items():
        # Split model name by "-" and "/" but keep dots in version numbers
        # This preserves "3.3", "5.2", etc.
        model_parts = [part.lower() for part in re.split(r'[-/]', model_name) if part]
        
        # Count matches between query parts and model parts
        score = 0
        matched_parts = set()
        
        for query_part in query_parts:
            best_part_score = 0
            for i, model_part in enumerate(model_parts):
                part_score = 0
                
                # Exact match (highest priority)
                if query_part == model_part:
                    part_score = 20
                # Version number: check if query contains the version
                elif '.' in query_part and query_part in model_part:
                    part_score = 18
                # Model part contains version that matches query
                elif '.' in model_part and query_part in model_part:
                    part_score = 15
                # Partial match
                elif query_part in model_part:
                    part_score = 8
                elif model_part in query_part:
                    part_score = 6
                
                if part_score > best_part_score:
                    best_part_score = part_score
            
            score += best_part_score
            if best_part_score > 0:
                matched_parts.add(query_part)
        
        # Bonus if all query parts matched
        if len(matched_parts) == len(query_parts) and len(query_parts) > 1:
            score += 10
        
        # Penalty for wrong version numbers - check if query has a number that conflicts
        query_numbers = [p for p in query_parts if any(c.isdigit() for c in p)]
        model_numbers = [p for p in model_parts if any(c.isdigit() for c in p)]
        
        for query_num in query_numbers:
            # Extract numeric part
            query_num_clean = re.sub(r'[^0-9.]', '', query_num)
            conflict = False
            for model_num in model_numbers:
                model_num_clean = re.sub(r'[^0-9.]', '', model_num)
                # Check if they're different version numbers
                if (query_num_clean and model_num_clean and 
                    query_num_clean != model_num_clean and
                    not model_num_clean.startswith(query_num_clean) and
                    not query_num_clean.startswith(model_num_clean)):
                    conflict = True
                    break
            if conflict:
                score -= 15  # Heavy penalty for conflicting versions
        
        if score > 0:
            scores[model_name] = score
    
    # Return model with highest score, or first model if no matches
    if scores:
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        return best_model
    else:
        return list(models_dict.values())[0]


# Example usage
if __name__ == "__main__":
    print("\n[Testing get_model function]")
    print(f"Query: 'llama 3.3', provider: 'groq' -> {get_model('llama 3.3', GROQ_MODELS)}")
    print(f"Query: 'gpt 5', provider: 'openrouter' -> {get_model('gpt 5', OPENROUTER_MODELS)}")
    print(f"Query: 'claude opus', provider: 'openrouter' -> {get_model('claude opus', OPENROUTER_MODELS)}")
    print(f"Query: 'llama 4 scout', provider: 'groq' -> {get_model('llama 4 scout', GROQ_MODELS)}")
    print(f"Query: 'gemini', provider: 'openrouter' -> {get_model('gemini', OPENROUTER_MODELS)}")



# Default model selection
DEFAULT_MODEL = GROQ_MODELS[2]  

# System prompts dictionary
SYSTEM_PROMPTS: Dict[str, str] = {
    "default": "You are a helpful AI assistant.",
    "summarize_food": "You are a nutrition expert. Provide concise, accurate nutritional information.",
    "extract_nutrients": "Extract nutritional information and return it as valid JSON.",
    "format_recipe": "Format recipes in a clear, structured way.",
    "extract_food_names": "Extract food items from text and list them clearly.",
    "translate_to_english": "Translate the given text to English accurately.",
    "unit_converter": "Convert measurement units accurately, maintaining the same quantities.",
    "extract_food_names_grams": "Extract food names and quantities in grams, return as CSV with columns: food_name,quantity_grams",
    "vision_analysis": "You are a vision expert. Analyze images carefully and provide detailed, accurate descriptions.",
}


def get_client_for_model(model: str, groq_api_key: Optional[str] = None, 
                         openrouter_api_key: Optional[str] = None) -> tuple[OpenAI, str]:
    """
    Determine which client to use based on the model name.
    
    Args:
        model: Model identifier
        groq_api_key: Optional Groq API key
        openrouter_api_key: Optional OpenRouter API key
        
    Returns:
        Tuple of (OpenAI client, provider name)
    """
    # Check if model is in Groq models
    if model in GROQ_MODELS.values():
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No Groq API key provided (pass groq_api_key or set GROQ_API_KEY).")
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        return client, "groq"
    
    # Check if model is in OpenRouter models
    elif model in OPENROUTER_MODELS.values():
        api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No OpenRouter API key provided (pass openrouter_api_key or set OPENROUTER_API_KEY).")
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        return client, "openrouter"
    
    else:
        raise ValueError(f"Model '{model}' not found in GROQ_MODELS or OPENROUTER_MODELS dictionaries.")


# =============================================================================
# IMAGE ENCODING FUNCTIONS
# =============================================================================

def encode_image_from_url(image_url: str, max_size: tuple = (1024, 1024)) -> str:
    """
    Scarica un'immagine da URL e la converte in base64 data URI.
    
    Args:
        image_url: URL dell'immagine da scaricare
        max_size: Dimensione massima (width, height) per ridimensionamento
        
    Returns:
        Data URI in formato base64 (JPEG)
    """
    # Headers per evitare il blocco 403 da parte dei server
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.google.com/',
    }
    
    response = requests.get(image_url, headers=headers, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image: HTTP {response.status_code}")
    
    img = Image.open(BytesIO(response.content))
    
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Resize if too large
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/jpeg;base64,{base64_image}"



def encode_image_from_path(image_path: str, max_size: tuple = (1024, 1024)) -> str:
    """
    Legge un'immagine da file locale e la converte in base64 data URI.
    
    Args:
        image_path: Path del file immagine
        max_size: Dimensione massima (width, height) per ridimensionamento
        
    Returns:
        Data URI in formato base64 (JPEG)
    """
    img = Image.open(image_path)
    
    # Convert RGBA to RGB if needed
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    # Resize if too large
    if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    # Convert to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_bytes = buffered.getvalue()
    base64_image = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/jpeg;base64,{base64_image}"


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================

def llm_inference(
    user_prompt: Union[str, List[Dict[str, Any]]],
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    groq_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    max_image_size: tuple = (1024, 1024),
) -> Dict[str, Any]:
    """
    Run LLM inference using either Groq or OpenRouter based on the model.
    Supports both text-only and vision (VLM) capabilities.
    
    Args:
        user_prompt: The user-facing prompt (string) or list of message content parts
        system_prompt: Optional system prompt; if None uses SYSTEM_PROMPTS['default']
        model: Model id to use for inference
        groq_api_key: Groq API key string
        openrouter_api_key: OpenRouter API key string
        temperature: Optional temperature to pass to the model
        max_tokens: Optional max tokens for completion
        image_url: Optional image URL (will be downloaded and converted to base64)
        image_path: Optional local image file path (will be converted to base64)
        max_image_size: Maximum (width, height) for image resizing
    
    Returns:
        A dict with keys: success (bool), content (str|None), raw (original response or None),
        error (str|None), provider (str)
    """
    chosen_system = system_prompt or SYSTEM_PROMPTS.get("default")
    
    try:
        client, provider = get_client_for_model(model, groq_api_key, openrouter_api_key)
    except ValueError as e:
        return {
            "success": False,
            "content": None,
            "raw": None,
            "error": str(e),
            "provider": None,
        }
    
    # Handle image encoding to base64 if provided
    base64_image_uri = None
    if image_url:
        try:
            # Check if already base64
            if image_url.startswith("data:image"):
                base64_image_uri = image_url
            else:
                base64_image_uri = encode_image_from_url(image_url, max_size=max_image_size)
        except Exception as e:
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": f"Error encoding image from URL: {str(e)}",
                "provider": provider,
            }
    
    elif image_path:
        try:
            base64_image_uri = encode_image_from_path(image_path, max_size=max_image_size)
        except Exception as e:
            return {
                "success": False,
                "content": None,
                "raw": None,
                "error": f"Error encoding image from path: {str(e)}",
                "provider": provider,
            }
    
    # Build messages
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": chosen_system},
    ]
    
    # Handle vision input (image + text)
    if base64_image_uri:
        if isinstance(user_prompt, str):
            text_content = user_prompt
        elif isinstance(user_prompt, list):
            # Extract text from list of content parts if present
            text_content = next((item.get("text", "") for item in user_prompt if item.get("type") == "text"), "")
        else:
            text_content = str(user_prompt)
        
        user_content = [
            {"type": "text", "text": text_content},
            {"type": "image_url", "image_url": {"url": base64_image_uri}},
        ]
        messages.append({"role": "user", "content": user_content})
    else:
        # Text-only input
        if isinstance(user_prompt, str):
            messages.append({"role": "user", "content": user_prompt})
        else:
            # user_prompt is already a list of content parts
            messages.append({"role": "user", "content": user_prompt})
    
    # Build kwargs for the SDK call
    call_kwargs: Dict[str, Any] = {"messages": messages, "model": model}
    if temperature is not None:
        call_kwargs["temperature"] = temperature
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    
    try:
        completion = client.chat.completions.create(**call_kwargs)
    except Exception as e:
        return {
            "success": False,
            "content": None,
            "raw": None,
            "error": str(e),
            "provider": provider,
        }
    
    # Extract content
    try:
        content = completion.choices[0].message.content
    except Exception:
        content = str(completion)
    
    return {
        "success": True,
        "content": content,
        "raw": completion,
        "error": None,
        "provider": provider,
    }



#%%
# =============================================================================
# TEST EXAMPLES - Text-only inference
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TEXT-ONLY INFERENCE TESTS")
    print("=" * 80)
    
    # Test 1: Groq model - nutritional summarization
    print("\n[TEST 1] Groq Model - Nutritional Summary")
    print("-" * 80)
    response = llm_inference(
        user_prompt="Describe the nutritional highlights of 100g of boiled lentils.",
        system_prompt=SYSTEM_PROMPTS["summarize_food"],
        model=GROQ_MODELS[3],
        temperature=0.7,
    )
    if response["success"]:
        print(f"Provider: {response['provider']}")
        print(f"Response:\n{response['content']}\n")
    else:
        print(f"Error: {response['error']}\n")
    
    # Test 2: OpenRouter model - nutrient extraction
    print("\n[TEST 2] OpenRouter Model - Nutrient Extraction (JSON)")
    print("-" * 80)
    response = llm_inference(
        user_prompt="Lentils per 100g: Protein 9g, Carbohydrates 20g, Fiber 8g, Iron 3.3mg. Extract as JSON.",
        system_prompt=SYSTEM_PROMPTS["extract_nutrients"],
        model=OPENROUTER_MODELS[3],
        temperature=0.3,
    )
    if response["success"]:
        print(f"Provider: {response['provider']}")
        print(f"Response:\n{response['content']}\n")
    else:
        print(f"Error: {response['error']}\n")
    
    # Test 3: Groq model - food name extraction
    print("\n[TEST 3] Groq Model - Extract Food Names")
    print("-" * 80)
    response = llm_inference(
        user_prompt="I had 2 apples, a banana, 150g of Greek yogurt, and some almonds for breakfast.",
        system_prompt=SYSTEM_PROMPTS["extract_food_names"],
        model=GROQ_MODELS[2],
    )
    if response["success"]:
        print(f"Provider: {response['provider']}")
        print(f"Response:\n{response['content']}\n")
    else:
        print(f"Error: {response['error']}\n")
    
    # Test 4: OpenRouter model - CSV extraction
    print("\n[TEST 4] OpenRouter Model - CSV Format Extraction")
    print("-" * 80)
    response = llm_inference(
        user_prompt="Extract food and grams: 200g chicken breast, 150g brown rice, 100g broccoli, 1 tbsp olive oil (14g)",
        system_prompt=SYSTEM_PROMPTS["extract_food_names_grams"],
        model=OPENROUTER_MODELS[2],
    )
    if response["success"]:
        print(f"Provider: {response['provider']}")
        print(f"Response:\n{response['content']}")
        try:
            df = pd.read_csv(io.StringIO(response["content"]), sep=",")
            print("\nParsed DataFrame:")
            print(df)
        except Exception as e:
            print(f"Could not parse as CSV: {e}")
        print()
    else:
        print(f"Error: {response['error']}\n")


#%%
# =============================================================================
# TEST EXAMPLES - Vision (VLM) inference
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VISION MODEL (VLM) INFERENCE TESTS")
    print("=" * 80)
    
    # Test 5: Groq VLM - analyze food image
    print("\n[TEST 5] Groq VLM - Food Image Analysis")
    print("-" * 80)
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/640px-Good_Food_Display_-_NCI_Visuals_Online.jpg"
    test_image_path = "test_food_image.jpg"
    
    response = llm_inference(
        user_prompt="What foods can you see in this image? List them with approximate quantities if possible.",
        system_prompt=SYSTEM_PROMPTS["vision_analysis"],
        model=GROQ_MODELS[2], 
        image_url=test_image_url,
        # image_path=test_image_path,
        temperature=0.5,
    )
    if response["success"]:
        print(f"Provider: {response['provider']}")
        print(f"Image URL: {test_image_url}")
        print(f"Response:\n{response['content']}\n")
    else:
        print(f"Error: {response['error']}\n")
    
    # Test 6: OpenRouter VLM - detailed image description
    print("\n[TEST 6] OpenRouter VLM - Detailed Image Description")
    print("-" * 80)
    test_image_url_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Spaghetti_Bolognese_mit_Parmesan_oder_Grana_Padano.jpg/640px-Spaghetti_Bolognese_mit_Parmesan_oder_Grana_Padano.jpg"
    
    response = llm_inference(
        user_prompt="Describe this dish in detail. What are the main ingredients visible?",
        system_prompt=SYSTEM_PROMPTS["vision_analysis"],
        model=OPENROUTER_MODELS[7],  # gpt-4o
        image_url=test_image_url_2,
        temperature=0.7,
    )
    if response["success"]:
        print(f"Provider: {response['provider']}")
        print(f"Image URL: {test_image_url_2}")
        print(f"Response:\n{response['content']}\n")
    else:
        print(f"Error: {response['error']}\n")
    
    # Test 7: OpenRouter VLM (Claude) - nutritional estimation from image
    print("\n[TEST 7] OpenRouter VLM (Claude) - Nutritional Estimation")
    print("-" * 80)
    test_image_url_3 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Brekkies.jpg/640px-Brekkies.jpg"
    
    response = llm_inference(
        user_prompt="Analyze this meal and provide an estimated nutritional breakdown (calories, protein, carbs, fats).",
        system_prompt=SYSTEM_PROMPTS["vision_analysis"],
        model=OPENROUTER_MODELS[8],  # claude-3.5-sonnet
        image_url=test_image_url_3,
        temperature=0.5,
        max_tokens=1000,
    )
    if response["success"]:
        print(f"Provider: {response['provider']}")
        print(f"Image URL: {test_image_url_3}")
        print(f"Response:\n{response['content']}\n")
    else:
        print(f"Error: {response['error']}\n")


#%%
# =============================================================================
# BATCH TESTING - Multiple providers and models
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BATCH TESTING - Comparing Providers")
    print("=" * 80)
    
    test_prompt = "What are the top 3 health benefits of consuming omega-3 fatty acids?"
    
    test_models = [
        ("Groq Llama 3.3 70B", GROQ_MODELS[3]),
        ("OpenRouter Gemini Pro 1.5", OPENROUTER_MODELS[2]),
        ("OpenRouter Qwen 2.5 72B", OPENROUTER_MODELS[4]),
    ]
    
    for model_name, model_id in test_models:
        print(f"\n[{model_name}]")
        print("-" * 80)
        response = llm_inference(
            user_prompt=test_prompt,
            system_prompt=SYSTEM_PROMPTS["summarize_food"],
            model=model_id,
            temperature=0.7,
            max_tokens=300,
        )
        if response["success"]:
            print(f"Provider: {response['provider']}")
            print(f"Response:\n{response['content']}\n")
        else:
            print(f"Error: {response['error']}\n")

# %%

if __name__ == "__main__":
    image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQEXIx54-KnuM6W4J3Cj7uxY4wwaxj9vHH-IA&s"
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Spaghetti_Bolognese_mit_Parmesan_oder_Grana_Padano.jpg/640px-Spaghetti_Bolognese_mit_Parmesan_oder_Grana_Padano.jpg"
    response = llm_inference("cosa vedi?", image_url=image_url, 
    model=GROQ_MODELS[1])
    print(response["content"])
    response = llm_inference("cosa vedi?", system_prompt=SYSTEM_PROMPTS["vision_analysis"], image_url=image_url, 
    model=get_model("grok 4 fast", OPENROUTER_MODELS))
    print(response["content"])
