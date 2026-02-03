#%%
"""
llm_lite.py - Lightweight text-only LLM inference module

A simplified version of llm_inference.py that focuses on text-only processing
and minimizes complexity while supporting both Groq and OpenRouter models.
"""

import os
import json
from typing import Optional, Dict, Any, Union
from openai import OpenAI

# Load API keys
# se non trova il file "api_keys.json" lo cerca nella directory superiore (utile per evitare di commettere chiavi nel repo)
try:
    with open("api_keys.json") as f:
        api_keys = json.load(f)
except FileNotFoundError:
     try: 
         with open("../api_keys.json") as f:
            api_keys = json.load(f)
     except FileNotFoundError:
         api_keys = {}

os.environ["GROQ_API_KEY"] = api_keys.get("groq")
os.environ["OPENROUTER_API_KEY"] = api_keys.get("openrouter")

# Model dictionaries
GROQ_MODELS = [
  "meta-llama/llama-4-maverick-17b-128e-instruct",
  "meta-llama/llama-4-scout-17b-16e-instruct",
  "llama-3.1-8b-instant",
  "llama-3.3-70b-versatile",
  "openai/gpt-oss-20b",
  "openai/gpt-oss-120b",
]

OPENROUTER_MODELS = [
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",
    "openai/gpt-5.2-pro",
    "openai/gpt-5.2",
    "openai/gpt-5.2-chat",
    "openai/gpt-5.2-codex",
     "openai/gpt-5.1-codex-max",
     "google/gemini-3-flash-preview",
     "x-ai/grok-4",
     "x-ai/grok-4.1-fast",
     "x-ai/grok-4-fast",
     "x-ai/grok-3",
     "x-ai/grok-3-mini",
     "x-ai/grok-code-fast-1",
     "mistralai/mistral-small-creative",
     "deepseek/deepseek-v3.2",
     "mistralai/ministral-14b-2512",
     "mistralai/ministral-8b-2512",
     "mistralai/ministral-3b-2512",
     "mistralai/devstral-2512",
     "allenai/olmo-3.1-32b-instruct",
     "allenai/olmo-3.1-32b-think",
     "moonshotai/kimi-k2.5",
     "writer/palmyra-x5",
     "minimax/minimax-m2-her",
     "minimax/minimax-m2.1",
     "prime-intellect/intellect-3",
     "deepcogito/cogito-v2.1-671b",
]

OPENROUTER_EMBEDDING_MODELS = [
    "google/gemini-embedding-001",
    "qwen/qwen3-embedding-8b",
    "qwen/qwen3-embedding-4b",
    "qwen/qwen3-embedding-0.6b",
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "mistralai/codestral-embed-2505",
    "openai/text-embedding-ada-002"
    ]

# Default model
DEFAULT_MODEL = GROQ_MODELS[4]

# System prompts
SYSTEM_PROMPTS = {
    "default": "You are a helpful AI assistant.",
    "summarize": "You are an expert at summarizing information concisely and accurately.",
    "extract_json": "Extract information and return it as valid JSON.",
    "analyze": "You are an analytical assistant. Provide clear, structured analysis.",
}


def get_model(query: str, models_list: Optional[list[str]] = None) -> str:
    """
    Select a model based on query content.
    
    Args:
        query: Search query with space-separated terms (e.g., "llama 3.3", "gpt 5", "claude")
        models_list: List of models to search (default: GROQ_MODELS)
    
    Returns:
        Model string, or the first model if no match found
    """
    import re
    
    query_parts = [part.lower().strip() for part in query.split() if part.strip()]
    
    if not query_parts:
        return models_list[0] if models_list else DEFAULT_MODEL
    
    scores = {}
    for model_name in models_list or GROQ_MODELS:
        model_parts = [part.lower() for part in re.split(r'[-/]', model_name) if part]
        score = 0
        
        for query_part in query_parts:
            for model_part in model_parts:
                if query_part == model_part:
                    score += 20
                elif '.' in query_part and query_part in model_part:
                    score += 18
                elif query_part in model_part:
                    score += 8
                elif model_part in query_part:
                    score += 6
        
        if score > 0:
            scores[model_name] = score
    
    return max(scores.items(), key=lambda x: x[1])[0] if scores else (models_list[0] if models_list else DEFAULT_MODEL)


def get_client(model: str) -> tuple[OpenAI, str]:
    """
    Get the appropriate OpenAI client for the model.
    
    Args:
        model: Model identifier
        
    Returns:
        Tuple of (OpenAI client, provider name)
    """
    if model in GROQ_MODELS:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("No Groq API key found in environment.")
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1"), "groq"
    
    elif model in OPENROUTER_MODELS:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("No OpenRouter API key found in environment.")
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"), "openrouter"
    
    else:
        raise ValueError(f"Model '{model}' not found in available models.")


def llm_inference(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Simple text-only LLM inference.
    
    Args:
        prompt: User prompt
        system: System prompt (default: "You are a helpful AI assistant.")
        model: Model identifier
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response
    
    Returns:
        Model response as string
    
    Raises:
        Exception: If inference fails
    """
    system_prompt = system or SYSTEM_PROMPTS["default"]
    client, provider = get_client(model)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    kwargs = {"messages": messages, "model": model, "temperature": temperature}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def _normalize_format(format_value: Union[str, type]) -> str:
    allowed = {
        "dict": {dict, "dict", "dizionario", "json", "object"},
        "list": {list, "list", "lista", "array"},
        "string": {str, "string", "str", "testo", "stringa"},
        "int": {int, "int", "integer", "intero"},
        "float": {float, "float", "numero", "decimale"},
        "bool": {bool, "bool", "boolean", "booleano"},
    }
    if isinstance(format_value, str):
        normalized = format_value.strip().lower()
        for key, values in allowed.items():
            if normalized in {val for val in values if isinstance(val, str)}:
                return key
    else:
        for key, values in allowed.items():
            if format_value in values:
                return key
    raise ValueError("format must be one of Dict, List, string, int, float, bool")

def _parse_response(value: str, target: str) -> Any:
    text = value.strip()
    if target == "dict":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON object: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("response is not a JSON object")
        return parsed
    if target == "list":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON array: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError("response is not a JSON array")
        return parsed
    if target == "string":
        if (text.startswith("\"") and text.endswith("\"")) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        return text
    if target == "int":
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError("response is not an integer literal") from exc
    if target == "float":
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError("response is not a float literal") from exc
    if target == "bool":
        lowered = text.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        raise ValueError("response is not a boolean literal (true/false)")
    raise ValueError(f"Unsupported format '{target}'")

FORMAT_INSTRUCTIONS: Dict[str, str] = {
    "dict": "Return only a valid JSON object (dictionary) with no additional text.",
    "list": "Return only a valid JSON array with no additional text.",
    "string": "Return only the requested string without surrounding quotes or extra text.",
    "int": "Return only an integer number with no extra text.",
    "float": "Return only a valid decimal number with no extra text.",
    "bool": "Return only the boolean literal true or false with no extra text.",
}

def llm_agentic_inference(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    format: Union[str, type] = "string",
) -> Any:
    """
    Variant of llm_inference che impone un formato di uscita.

    Oltre agli argomenti di llm_inference accetta "format" che deve essere uno fra
    Dict, List, string, int, float, bool (anche come tipo Python). Il modello riceve
    istruzioni aggiuntive per rispettare il formato e, in caso di errore di parsing,
    viene effettuato un tentativo di correzione basato sull'errore rilevato.
    """
    target = _normalize_format(format)
    system_base = system or SYSTEM_PROMPTS["default"]
    system_prompt = f"{system_base.rstrip()}\n\n{FORMAT_INSTRUCTIONS[target]}"
    attempts = 0
    max_attempts = 2  # prima risposta + un tentativo di correzione
    current_prompt = prompt
    last_error: Optional[Exception] = None
    last_response: Optional[str] = None

    while attempts < max_attempts:
        attempts += 1
        response = llm_inference(
            current_prompt,
            system=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        last_response = response
        try:
            return _parse_response(response, target)
        except ValueError as exc:
            last_error = exc
            if attempts >= max_attempts:
                break
            correction_instruction = (
                f"WARNING: the previous response is not a valid {target}."
                f"\nResponse: {response}"
                f"\nParsing error: {exc}."
                f"\nCorrect it by returning only a valid {target} with no extra text."
            )
            current_prompt = f"{prompt}\n\n{correction_instruction}"

    raise ValueError(
        "Impossibile ottenere un output nel formato richiesto: "
        f"{target}. Ultimo errore: {last_error}. Ultima risposta: {last_response}"
    )


def llm_safe(
    prompt: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Safe text-only LLM inference with error handling.
    
    Args:
        prompt: User prompt
        system: System prompt
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
    
    Returns:
        Dict with keys: success (bool), content (str or None), error (str or None), provider (str)
    """
    try:
        content = llm_inference(prompt, system, model, temperature, max_tokens)
        client, provider = get_client(model)
        return {
            "success": True,
            "content": content,
            "error": None,
            "provider": provider,
        }
    except Exception as e:
        return {
            "success": False,
            "content": None,
            "error": str(e),
            "provider": None,
        }


# Example usage
if __name__ == "__main__":
    # Test 1: Simple query
    print("Test 1: Simple query")
    response = llm_inference("What is the capital of France?")
    print(f"Response: {response}\n")
    
    # Test 2: With custom system prompt
    print("Test 2: Custom system prompt")
    response = llm_inference(
        "Explain quantum computing in one sentence.",
        system="You are a physics teacher. Explain concepts simply.",
        temperature=0.5
    )
    print(f"Response: {response}\n")
    
    # Test 3: Safe inference with error handling
    print("Test 3: Safe inference")
    result = llm_safe("What are the benefits of exercise?", model=GROQ_MODELS[4])
    if result["success"]:
        print(f"Provider: {result['provider']}")
        print(f"Response: {result['content']}\n")
    else:
        print(f"Error: {result['error']}\n")
    
    # Test 4: Model selection
    print("Test 4: Model selection")
    model = get_model("claude", OPENROUTER_MODELS)
    print(f"Selected model: {model}")
    response = llm_inference("Hello!", model=model, max_tokens=50)
    print(f"Response: {response}\n")


# %%
