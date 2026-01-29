#%%
import os
import json
import string
from typing import Optional, Dict, Any
from openai import OpenAI

# Load API keys
with open("api_keys.json") as f:
    api_keys = json.load(f)

api_keys.keys()

for key in api_keys:
    if key is string:
        os.environ[key.upper() + "_API_KEY"] = api_keys[key]
        
# print all os environment variables that end with _API_KEY
for key, value in os.environ.items():
    if key.endswith("_API_KEY"):
        # print(f"{key}: {value[:4]}...{value[-4:]}")
        pass
