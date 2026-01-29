#%%
import os
from litellm import openrouter_models
import requests
import load_api_keys
import tiktoken

def token_counter(text: str, model: str, fallback_encoding: str = "cl100k_base") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding(fallback_encoding)
    return len(enc.encode(text))


def get_openrouter_models(api_key: str) -> list[dict]:
    """
    Recupera la lista completa dei modelli OpenRouter disponibili.
    
    Returns:
        Lista di dizionari con struttura completa OpenRouter:
        {
            'id': 'anthropic/claude-3.5-sonnet',
            'name': 'Anthropic: Claude 3.5 Sonnet',
            'pricing': {'prompt': '0.000003', 'completion': '0.000015'},
            'context_length': 200000,
            'description': '...',
            'top_provider': {...},
            'per_request_limits': {...},
            ...
        }
    """
    resp = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    resp.raise_for_status()
    return resp.json()["data"]

OPENROUTER_MODELS = get_openrouter_models(os.environ["OPENROUTER_API_KEY"])
#%%

if __name__ == "__main__":
    models = []
    for m in OPENROUTER_MODELS:
        models.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "prompt_cost": m.get("pricing", {}).get("prompt"),  
            "completion_cost": m.get("pricing", {}).get("completion"),
            "context_length": m.get("context_length"),
            "description": m.get("description"),
        })

        # cost are strings like "0.00000005" (USD per token)

    # esempio: stampa le prime 5 righe
    for row in models[:5]:
        print(row)
#%%
from typing import List, Dict, Optional
import re

def get_model_from_query(
    query: str,
    models_list: List[Dict],
    top_n: int = 5
) -> List[Dict]:
    """
    Trova i modelli che meglio corrispondono alla query data.
    
    Args:
        query: La stringa di ricerca (es. "gpt-4", "claude", "llama 70b")
        models_list: Lista di dizionari (dati grezzi da get_openrouter_models) con struttura:
            {
                'id': 'anthropic/claude-3.5-sonnet',
                'name': 'Anthropic: Claude 3.5 Sonnet',
                'pricing': {'prompt': '0.000003', 'completion': '0.000015'},
                'context_length': 200000,
                'description': '...',
                ... altri campi OpenRouter
            }
        top_n: Numero massimo di risultati da restituire (default: 5)
    
    Returns:
        Lista dei modelli completi ordinati per rilevanza (score decrescente).
        Ogni modello contiene tutti i campi originali OpenRouter.
    
    Example:
        >>> data = get_openrouter_models(API_KEY)
        >>> results = get_model_from_query("claude sonnet", data)
        >>> print(results[0]['id'])
        'anthropic/claude-3.5-sonnet'
        >>> print(results[0]['pricing'])
        {'prompt': '0.000003', 'completion': '0.000015'}
    """
    # Varianti comuni che indicano modelli non-base
    VARIANT_SUFFIXES = {
        "mini", "micro", "nano", "tiny", "small", "lite", "light",
        "fast", "turbo", "instant", "flash", "quick", "speed",
        "preview", "beta", "experimental", "exp", "test",
        "free", "nitro", "extended", "thinking",
        "online", "search", "vision", "multimodal",
    }
    
    query_lower = query.lower()
    query_parts = query_lower.split()
    
    # Controlla se la query specifica una variante
    query_specifies_variant = any(part in VARIANT_SUFFIXES for part in query_parts)
    
    scored_models = []
    
    for model in models_list:
        score = 0
        model_id = model.get("id", "").lower()
        model_name = model.get("name", "").lower()
        description = model.get("description", "").lower() if model.get("description") else ""
        
        # Estrai la parte dopo "/" nell'id (es. "openai/gpt-4o" -> "gpt-4o")
        model_id_short = model_id.split("/")[-1] if "/" in model_id else model_id
        
        # Match esatto dell'id corto (massima priorità)
        if query_lower == model_id_short:
            score += 500
        
        # Match esatto dell'intera query nell'id
        if query_lower in model_id:
            score += 100
            # Bonus extra se la query corrisponde quasi esattamente all'id corto
            if model_id_short.startswith(query_lower):
                score += 50
        
        # Match esatto dell'intera query nel nome
        if query_lower in model_name:
            score += 80
        
        # Match di ogni parte della query
        for part in query_parts:
            # Match nell'id (alta priorità)
            if part in model_id:
                score += 30
                # Bonus se è all'inizio o dopo il "/"
                if model_id.startswith(part) or f"/{part}" in model_id:
                    score += 20
            
            # Match nel nome
            if part in model_name:
                score += 20
            
            # Match nella descrizione (bassa priorità)
            if part in description:
                score += 5
        
        # Bonus per match di pattern comuni (es. "70b", "8b", "3.5")
        version_patterns = re.findall(r'\d+\.?\d*b?', query_lower)
        for pattern in version_patterns:
            if pattern in model_id or pattern in model_name:
                score += 25
        
        # PENALITÀ per varianti se la query NON specifica una variante
        # Questo favorisce il modello "base" quando la query è generica
        if not query_specifies_variant and score > 0:
            model_id_parts = set(re.split(r'[-_/:\s]', model_id_short))
            variant_matches = model_id_parts.intersection(VARIANT_SUFFIXES)
            if variant_matches:
                # Penalizza i modelli con suffissi variante
                score -= 40 * len(variant_matches)
        
        # Penalità per lunghezza eccessiva dell'id rispetto alla query
        # (modelli con nomi più lunghi/specifici sono meno probabili se la query è corta)
        if score > 0:
            length_diff = len(model_id_short) - len(query_lower.replace(" ", "-"))
            if length_diff > 5:
                score -= min(length_diff, 20)
        
        if score > 0:
            scored_models.append({**model, "_score": score})
    
    # Ordina per score decrescente
    scored_models.sort(key=lambda x: x["_score"], reverse=True)
    
    # Rimuovi il campo _score e restituisci i top_n risultati
    results = []
    for m in scored_models[:top_n]:
        result = {k: v for k, v in m.items() if k != "_score"}
        results.append(result)
    
    return results


def get_best_model(query: str, models_list: List[Dict]) -> Optional[Dict]:
    """
    Restituisce il singolo modello che meglio corrisponde alla query.
    
    Args:
        query: La stringa di ricerca
        models_list: Lista di dizionari modello
    
    Returns:
        Il modello con lo score più alto, o None se nessun match
    """
    results = get_model_from_query(query, models_list, top_n=1)
    return results[0] if results else None


if __name__ == "__main__":
    get_model_from_query("gpt-4", OPENROUTER_MODELS)[:3]
    get_best_model("claude 4", OPENROUTER_MODELS)

#%%
import os, requests


def get_GROQ_MODELS(api_key: str) -> list[dict]:
    # 1) list models
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    resp.raise_for_status()
    GROQ_MODELS_base = resp.json()["data"]

    # 2) prezzi: esempio parziale (USD per 1M token) preso dalla pagina pricing
    # NOTA: devi mappare gli ID reali restituiti da /models ai nomi nella tabella pricing.
    pricing_per_million = {
        # "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        # "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    }

    pricing_per_million = [
        {"id": "llama-3.1-8b-instant", "pricing": {"prompt": "0.00000005", "completion": "0.00000008"}},  # $0.05 / 1M  # $0.08 / 1M
        {"id": "llama-3.3-70b-versatile", "pricing": {"prompt": "0.00000059", "completion": "0.00000079"}},
        {"id": "meta-llama/llama-guard-4-12b", "pricing": {"prompt": "0.00000020", "completion": "0.00000020"}},
        {"id": "openai/gpt-oss-120b", "pricing": {"prompt": "0.00000015", "completion": "0.00000060"}},
        {"id": "openai/gpt-oss-20b", "pricing": {"prompt": "0.000000075", "completion": "0.00000030"}},
        # Audio pricing: non token-based, quindi non mappabile 1:1 al pricing OpenRouter per token.
        {"id": "whisper-large-v3", "pricing": {"per_hour_usd": "0.111"}},
        {"id": "whisper-large-v3-turbo", "pricing": {"per_hour_usd": "0.04"}},
        {"id": "meta-llama/llama-4-maverick-17b-128e-instruct", "pricing": {"prompt": "0.00000020", "completion": "0.00000060"}},
        {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "pricing": {"prompt": "0.00000011", "completion": "0.00000034"}},
        {"id": "meta-llama/llama-prompt-guard-2-22m", "pricing": {"prompt": "0.00000003", "completion": "0.00000003"}},
        {"id": "meta-llama/llama-prompt-guard-2-86m", "pricing": {"prompt": "0.00000004", "completion": "0.00000004"}},
        {"id": "moonshotai/kimi-k2-instruct-0905", "pricing": {"prompt": "0.00000100", "completion": "0.00000300"}},
        {"id": "openai/gpt-oss-safeguard-20b", "pricing": {"prompt": "0.000000075", "completion": "0.00000030"}},
        {"id": "qwen/qwen3-32b", "pricing": {"prompt": "0.00000029", "completion": "0.00000059"}},
    ]


    GROQ_MODELS = []
    for m in GROQ_MODELS_base:
        model_id = m.get("id")
        pricing_info = next((item for item in pricing_per_million if item["id"] == model_id), None)
        if pricing_info:
            m["pricing"] = pricing_info["pricing"]
        GROQ_MODELS.append(m)
    return GROQ_MODELS

GROQ_MODELS = get_GROQ_MODELS(os.environ["GROQ_API_KEY"])


#%%

from typing import Dict, Optional, Union

def estimate_text_cost_usd(
    text: str,
    model_id: str,
    models_list: List[Dict],
    *,
    token_counter=None,
    completion_tokens: int = 0,
    tokenizer_model: Optional[str] = None,
) -> float:
    """
    Stima costo USD di una richiesta: costo_prompt + costo_completion.

    Args:
        text: Il testo del prompt da valutare.
        model_id: L'ID del modello (es. "llama-3.1-8b-instant").
        models_list: Lista di dizionari modello con struttura:
            [{"id": "model-id", "pricing": {"prompt": "0.00000005", "completion": "0.00000008"}}, ...]
        token_counter: Funzione (text, model) -> int che conta i token.
        completion_tokens: Stima dei token di output (default 0).
        tokenizer_model: Modello alternativo per il tokenizer (fallback: model_id).

    Returns:
        Costo stimato in USD.
    """
    if token_counter is None:
        raise ValueError("Passa token_counter=token_counter (o equivalente).")

    # Cerca il modello nella lista per id
    model_entry = None
    for m in models_list:
        if m.get("id") == model_id:
            model_entry = m
            break
    
    if not model_entry:
        raise KeyError(f"Prezzi non trovati per model_id='{model_id}'")

    pricing = model_entry.get("pricing", {})
    
    if "per_hour_usd" in pricing:
        raise ValueError(f"Modello '{model_id}' ha pricing orario (audio), non token-based.")

    prompt_cost_per_token = float(pricing["prompt"], 0)
    completion_cost_per_token = float(pricing["completion"], 0)
    audio_cost_per_hour = float(pricing.get("audio", 0))
    image_cost_per_image = float(pricing.get("image", 0))
    internal_reasoning_cost_per_token = float(pricing.get("internal_reasoning", 0))

    enc_model = tokenizer_model or model_id
    prompt_tokens = int(token_counter(text, enc_model))

    total_cost = (prompt_tokens * prompt_cost_per_token) + (completion_tokens * completion_cost_per_token)
    
    print(f"Model: {model_id}")
    print(f"  Prompt tokens: {prompt_tokens:,} × ${prompt_cost_per_token:.10f} = ${prompt_tokens * prompt_cost_per_token:.8f}")
    print(f"  Completion tokens: {completion_tokens:,} × ${completion_cost_per_token:.10f} = ${completion_tokens * completion_cost_per_token:.8f}")
    print(f"  Total: ${total_cost:.8f}")

    return total_cost

def get_model_cost_data(
    model_id: str,
    models_list: List[Dict],
    *,
    token_counter=None,
) -> Dict[str, float]:
    if token_counter is None:
        raise ValueError("Passa token_counter=token_counter (o equivalente).")

    # Cerca il modello nella lista per id
    model_entry = None
    for m in models_list:
        if m.get("id") == model_id:
            model_entry = m
            break
    
    if not model_entry:
        raise KeyError(f"Prezzi non trovati per model_id='{model_id}'")

    pricing = model_entry.get("pricing", {})
    
    if "per_hour_usd" in pricing:
        raise ValueError(f"Modello '{model_id}' ha pricing orario (audio), non token-based.")

    prompt_cost_per_token = float(pricing["prompt"], 0)
    completion_cost_per_token = float(pricing["completion"], 0)
    audio_cost_per_hour = float(pricing.get("audio", 0))
    image_cost_per_image = float(pricing.get("image", 0))
    internal_reasoning_cost_per_token = float(pricing.get("internal_reasoning", 0))


    cost_data = {
        "prompt_cost_per_token": prompt_cost_per_token,
        "completion_cost_per_token": completion_cost_per_token,
        "audio_cost_per_hour": audio_cost_per_hour,
        "image_cost_per_image": image_cost_per_image,
        "internal_reasoning_cost_per_token": internal_reasoning_cost_per_token,
    }
    
    return cost_data


def import_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    # get with glob every .txt .tex .md in uppper folder 
    import glob
    import os
    file_paths = glob.glob(os.path.join("..", "*.*"))
    file_paths = [fp for fp in file_paths if fp.endswith((".txt", ".tex", ".md"))]

    text = import_text(file_paths[0])
    cost = estimate_text_cost_usd(
        text,
        model_id="llama-3.1-8b-instant",
        models_list=GROQ_MODELS,
        token_counter=token_counter,
        completion_tokens=200,  # stima output
    )
    # print(cost)


get_best_model("image gemini", OPENROUTER_MODELS)
# %%
if __name__ == "__main__":
    model_dict = get_best_model("gpt-4", OPENROUTER_MODELS)

    cost = estimate_text_cost_usd(
        text,
        model_id=model_dict['id'],
        models_list=OPENROUTER_MODELS,
        token_counter=token_counter,
        completion_tokens=200,  # stima output
    )

#%%

# simulate 1000 requests to the model and estimate total cost
if __name__ == "__main__":
    open_models = [
        get_best_model("gpt-4", OPENROUTER_MODELS),
        get_best_model("claude 4", OPENROUTER_MODELS)
    ]
    groq_models = [
        get_best_model("llama-3.1-8b-instant", GROQ_MODELS),
        get_best_model("meta-llama/llama-4-scout", GROQ_MODELS),
        # get_best_model("kimi", GROQ_MODELS)
        ]

    total_requests = 1000
    for model_dict in open_models + groq_models:
        if model_dict in open_models:
            model_list = OPENROUTER_MODELS
        else:
            model_list = GROQ_MODELS
        print(f"Estimating cost for model: {model_dict['id']}")
        single_cost = estimate_text_cost_usd(
            text,
            model_id=model_dict['id'],
            models_list=model_list,
            token_counter=token_counter,
            completion_tokens=500,  # stima output
        )

        total_cost = single_cost * total_requests
        print(f"Estimated total cost for {total_requests} requests: ${total_cost:.2f}")
        print("=" * 40)

#%% plot cost per all OPENROUTER models and GROQ models in the same plot 
import matplotlib.pyplot as plt
import numpy as np

def extract_pricing_data(models_list: List[Dict], source_name: str) -> List[Dict]:
    """Estrae i dati di pricing da una lista di modelli."""
    data = []
    for m in models_list:
        pricing = m.get("pricing", {})
        # Salta modelli audio con pricing orario
        if "per_hour_usd" in pricing or not pricing:
            continue
        try:
            prompt_cost = float(pricing.get("prompt", 0))
            completion_cost = float(pricing.get("completion", 0))
            if prompt_cost > 0 or completion_cost > 0:
                data.append({
                    "id": m.get("id", "unknown"),
                    "name": m.get("name", m.get("id", "unknown")),
                    "prompt_cost_per_1M": prompt_cost * 1_000_000,  # Converti a $ per 1M token
                    "completion_cost_per_1M": completion_cost * 1_000_000,
                    "source": source_name,
                })
        except (ValueError, TypeError):
            continue
    return data

if __name__ == "__main__":
    # Estrai dati
    openrouter_data = extract_pricing_data(OPENROUTER_MODELS, "OpenRouter")
    groq_data = extract_pricing_data(GROQ_MODELS, "Groq")

    all_data = openrouter_data + groq_data

    # Ordina per costo prompt
    all_data.sort(key=lambda x: x["prompt_cost_per_1M"])

    # Filtra solo modelli con costo ragionevole per visualizzazione (< $50/1M token)
    filtered_data = [d for d in all_data if d["prompt_cost_per_1M"] < 50]

    print(f"OpenRouter models con pricing: {len(openrouter_data)}")
    print(f"Groq models con pricing: {len(groq_data)}")
    print(f"Totale dopo filtro (< $50/1M): {len(filtered_data)}")

    # Scatter plot: prompt vs completion cost
    fig, ax = plt.subplots(figsize=(12, 8))

    # Separa per source
    for source, color, marker in [("OpenRouter", "blue", "o"), ("Groq", "red", "^")]:
        source_data = [d for d in filtered_data if d["source"] == source]
        if source_data:
            x = [d["prompt_cost_per_1M"] for d in source_data]
            y = [d["completion_cost_per_1M"] for d in source_data]
            ax.scatter(x, y, c=color, marker=marker, alpha=0.6, label=f"{source} ({len(source_data)} models)", s=50)

    ax.set_xlabel("Prompt Cost ($/1M tokens)", fontsize=12)
    ax.set_ylabel("Completion Cost ($/1M tokens)", fontsize=12)
    ax.set_title("Model Pricing Comparison: OpenRouter vs Groq", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

#%% Bar chart: Top 20 cheapest models (prompt cost)

if __name__ == "__main__":
    def plot_bar_chart_cheapest_models(filtered_data: List[Dict], bottom_n: int, top_n: int):
        cheapest = filtered_data[bottom_n:top_n]

        fig, ax = plt.subplots(figsize=(14, 8))

        model_names = [d["id"].split("/")[-1][:25] for d in cheapest]  # Tronca nomi lunghi
        prompt_costs = [d["prompt_cost_per_1M"] for d in cheapest]
        completion_costs = [d["completion_cost_per_1M"] for d in cheapest]
        colors = ["red" if d["source"] == "Groq" else "blue" for d in cheapest]

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, prompt_costs, width, label="Prompt", alpha=0.8)
        bars2 = ax.bar(x + width/2, completion_costs, width, label="Completion", alpha=0.8)

        # Colora le barre in base alla source
        for i, (b1, b2) in enumerate(zip(bars1, bars2)):
            color = colors[i]
            b1.set_color(color)
            b2.set_color(color)
            b2.set_alpha(0.5)

        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel("Cost ($/1M tokens)", fontsize=12)
        ax.set_title(f"Top {top_n} Cheapest Models (Blue=OpenRouter, Red=Groq)", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()

    plot_bar_chart_cheapest_models(filtered_data, bottom_n=0, top_n=50)

    # plot_bar_chart_cheapest_models(filtered_data, bottom_n=50, top_n=100)

    # plot_bar_chart_cheapest_models(filtered_data, bottom_n=100, top_n=150)

    # plot_bar_chart_cheapest_models(filtered_data, bottom_n=150, top_n=200)

    # plot_bar_chart_cheapest_models(filtered_data, bottom_n=200, top_n=250)

    # plot_bar_chart_cheapest_models(filtered_data, bottom_n=250, top_n=300)

#%% Histogram: distribuzione dei costi
if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prompt cost distribution
    ax1 = axes[0]
    openrouter_prompt = [d["prompt_cost_per_1M"] for d in filtered_data if d["source"] == "OpenRouter"]
    groq_prompt = [d["prompt_cost_per_1M"] for d in filtered_data if d["source"] == "Groq"]

    ax1.hist(openrouter_prompt, bins=30, alpha=0.6, label=f"OpenRouter (n={len(openrouter_prompt)})", color="blue")
    ax1.hist(groq_prompt, bins=30, alpha=0.6, label=f"Groq (n={len(groq_prompt)})", color="red")
    ax1.set_xlabel("Prompt Cost ($/1M tokens)")
    ax1.set_ylabel("Count")
    ax1.set_title("Prompt Cost Distribution")
    ax1.legend()
    ax1.set_xscale('log')

    # Completion cost distribution
    ax2 = axes[1]
    openrouter_completion = [d["completion_cost_per_1M"] for d in filtered_data if d["source"] == "OpenRouter"]
    groq_completion = [d["completion_cost_per_1M"] for d in filtered_data if d["source"] == "Groq"]

    ax2.hist(openrouter_completion, bins=30, alpha=0.6, label=f"OpenRouter (n={len(openrouter_completion)})", color="blue")
    ax2.hist(groq_completion, bins=30, alpha=0.6, label=f"Groq (n={len(groq_completion)})", color="red")
    ax2.set_xlabel("Completion Cost ($/1M tokens)")
    ax2.set_ylabel("Count")
    ax2.set_title("Completion Cost Distribution")
    ax2.legend()
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.show()

# %%
