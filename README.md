# MyChatGPT: Python Module
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johndef64/mychatgpt/blob/main/notebooks/mychatgpt_trial.ipynb) [![PyPI Latest Release](https://img.shields.io/pypi/v/mychatgpt.svg)](https://pypi.org/project/mychatgpt/) 

`mychatgpt` is a small and useful Python module that provides functions for interacting with OpenAI's GPT models to create conversational agents. This module allows users to have interactive conversations with the GPT models and keeps track of the conversation history in your Python Projects and Jupyter Notebooks.

The package includes vision, hearing and drawing functions for openai models.

Now implemented with **ollama**. Run 🤗 open-source models locally and interact through in-build functions.

## Installation

Install package with pip: 
```bash
pip install mychatgpt
```
Install latest version through git
```bash
pip install git+https://github.com/johndef64/mychatgpt.git
```

To use this module, you need an **OpenAI API key**. You have to provide your API key when requested once and it will be stored as `openai_api_key.txt` in your working directory.

## Usage

The module provides the following main class:

- `GPT()`


- Core Functions:
  - `GPT().ask_gpt(prompt, *args)`:  
  This basic function takes a prompt as input and generates a single response from the GPT chosen model. It returns the generated response and update `chat_log.json`.
  You can simply use `op.ask_gpt(prompt)` and keep the default parameters.

  - `GPT().send_message(message,*args)`:  
  This core function allows for a more interactive conversation with the GPT chosen model. It takes a message as input, generates a response from the model, and updates the conversation history. It also logs the conversation in the `chat_log.json` file.  
  This function is implemented with GPT vision, Text2Speech and Dall-E functionalities.
  

- Main Function:
  - `GPT().chat(message,*args)`:
  Main chat function tuned with class parameters.
```python
from mychatgpt import GPT

op = GPT(assistant='',                   # in-build assistant name 
         persona='',                     # any known character
         format=None,                    # output format (latex,python,markdown)
         translate=False,                # translate outputs
         save_log=True,                  # save log file
         to_clip=True,                   # send reply t clipboard
         print_token=True,               # print token count
         model='gpt-4o-mini',            # set openai main model
         talk_model='gpt-4o-2024-08-06', # set openai speak model
         dalle="dall-e-2",               # set dall-e model
         image_size='512x512',           # set generated image size
         memory=False,                   # enable memory function   
         ollama_server=None,             # pass an ollama server address         
         my_key=None                     # default api key
         )

op.chat('Your message goes here', 
        max=1000,          # max tokens in reply
        image=None,        # insert an image path to activate gpt vision
        paste=False,       # append clipboard to message
        create=False       # create an image
        )

# Usage Example
op.add_persona('Elon Musk') # add custom personality to your agent
op.chat("""What do you think about OpenAI?""")
```
- Initialize a conversational agent: `GPT('Character')`
```python
elon = GPT('Elon Musk') 
elon.chat("""What do you think about OpenAI?""")
```

- Activate GPT Vision: `op.chat("What’s in this image?", image="url/path",*parameters*)` insert in your chat context gpt-vision, activate  a multimodal chat
```python
vincent = GPT('Vincent Van Gogh')
vincent.chat("""Tell me what you see.""", 
             image=vincent.dummy_img)
```
        

- Image Generation in chat: `op.chat("prompt", create=Ture, *parameters*)`
ask your conversational agent to create an image 
```python
vincent = GPT('Vincent Van Gogh')
vincent.dalle = "dall-e-3"  # change dall-e model
vincent.chat("""Tell me what you see. Can you paint it?""", 
             image=vincent.dummy_img, 
             create = True)
```
- Image Generation main function
`op.create_image(prompt,*parameters*)`
```python
op.create_image(prompt= "a cute kitten",
                model="dall-e-2",
                size='512x512',
                response_format='b64_json',
                quality="standard",
                time_flag=True, show_image=True)
```
5. Whisper and Text-to-Speech
```python
op.whisper(filepath, # audio.mp3, audio.wav
           translate = False,
           response_format = "text",
           print_transcriprion = True)

op.text2speech(text,
               voice: str = "alloy",
               model: str = "tts-1",
               stream:bool = True,
               save_audio: bool = False,
               response_format: str = "mp3",
               filename: str = "speech",
               speed: int = 1)

voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
response_formats = ["mp3", "flac", "aac", "opus"]
```

6. Speak With...
```python

# Use an in-build assistant or any character of your choice, example:
socrates = GPT('Socrates')
socrates.chat('Tell me about the Truth.', speak=True, voice='onyx')
```
```python
# Endless chat, keyboard controlled
socrates.speak_loop(system=None,
                    voice='nova', tts= 'tts-1', max=1000, language='eng', printall=False, exit_chat='stop')
```
7. Talk With...
```python
dua = GPT('Dua Lipa')
dua.talk(voice='nova', language='eng', gpt='gpt-4o', tts= 'tts-1', max=1000, printall=False)
```
```python
# Endless talk, keyboard controlled
dua.talk_loop(voice='nova', language='eng', gpt='gpt-4o', tts= 'tts-1', max=1000, printall=False, chat='alt' , exit='shift')
```
```python
nietzsche = GPT('Friedrich Nietzsche')
nietzsche.talk_loop('onyx')
```

The module also provides additional utility functions for managing the conversation, such as clearing the chat history, setting a persona, and setting system instructions, save/load chats.

1. `choose_model()`
2. `clear_chat()`
3. `expand_chat()`
4. `save_chat()`
5. `load_chat()`
6. `load_file()`

To set-up multiple conversations or change the API-key, follow the example proposed in [mychatgpt_trial.ipynb](https://github.com/johndef64/mychatgpt/blob/main/mychatgpt_trial.ipynb)

## In-Build Assistants
```python
from mychatgpt import display_assistants
display_assistants()
```
```python
from mychatgpt import agent

# select assistant from the list above
delamain = agent("delamain")

# Call an assistant simply by name
delamain.chat('your message',
              gpt='gpt-4o', 
              max = 1000, 
              clip=True)  

#n.b. assistants sends reply to clipboard by default
```
```python
from mychatgpt import agent

# select assistant from the list above
julia = agent("julia")

# Call an assistant simply by name
julia.chat('Hi Julia. Are you ready for another day of work?',
            gpt='gpt-4o', 
            max = 1000, 
            clip=True)  

#n.b. assistants sends reply to clipboard by default
```

## Other models

Remember to set up your api keys before
```python
from mychatgpt import load_api_keys
load_api_keys(True)
```
**DeepSeek**
```python
from mychatgpt import agent
julia = agent("julia")

julia.c('@Hi Julia, how are you today?', "deepseek-chat")
```

```python
from mychatgpt import chatgpt

chatgpt.c('Reason about human existence as if you were an alien', "deepseek-reasoner")
```

**Grok**
```python
from mychatgpt import chatgpt

chatgpt.c('Reason about human existence as if you were an alien', "grok-2-latest")
```

**Gemini**
```python
from mychatgpt.gemini import Gemini

Ge = Gemini(system="you are fixed with candies, you can't talk about anything else")
Ge.send("Hi how are you?")
```

## Setup Ollama (linux)
guide: https://github.com/RamiKrispin/ollama-poc/blob/main/ollama-poc.ipynb
```bash
# Install ollama in Linux System
curl -fsSL https://ollama.com/install.sh | sh
```
```bash
# Pull desired models from 🤗
ollama pull mistral
```
```bash
# Start ollama server
ollama serve
```
```python
from mychatgpt import GPT

# simply pass to chat functions
mylama = GPT(model = "mistral")
mylama.chat("Hi, what LLM model are you?", gpt=model)
```
### Use Ollama with server client
```python
from mychatgpt import GPT

# set your server. If localhost, leave empty
mylama = GPT(ollama_server="http://0.0.0.0:7869")
model = "gemma2:2b"

# Pull your model
mylama.ollama_client.pull(model)

# Chat
mylama.chat("Hi, what LLM model are you?", gpt=model)
```


## Notes
The code in this module assumes that the conversation history is stored in a global variable named `chat_thread`. Use `print(op.chat_thread)` to show conversation history and `op.chat_thread.pop()` to remove last interacition. `op.send_message('clearchat')` to start a new conversation.

Using `op.send_message()`, the code checks if the total number of tokens exceeds the model's maximum context length (gpt 3.5 turbo-16k: 16,384 tokens). If it does, a warning message indicates that the token limit is being reached and then then the first part of the conversation will automatically be deleted to make room for the next interaction.

## 



## Openai-based applications 
Some other python applications executable in Terminal that take advantage of openai modulo features:
- auto-gpt
- gpt-cli 
- rec-whisper-paste 


## Author
Written by: JohnDef64 

## Acknowledgment
This module only facilitates interaction with the OpenAI API and keeps track of it. OpenAI holds all rights to ChatGPT.
