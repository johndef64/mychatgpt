#%%
# Import module
from mychatgpt import julia, C, bot
#%%
# Copy you code inside the clipboard
C.cp("""@ optimises this function : """)
#%%

#%%
from mychatgpt import julia, yoko, watson, C, GPT, fixer, OpenAI, deepseek_api_key
#%%
from mychatgpt import C, julia, chatgpt, GPT
chat = GPT("a druken irish old sailor")
chat.reload_client(model="dpc")
# chat.expand_chat()
chat.c("Hi there!")
#%%
chat.expand_chat("you are also in love with me", "system")
chat.c("Are you feeling well?")
#%%
chat.expand_chat("give me some  grog please")
chat.c("")
#%%

from mychatgpt import julia
julia.c("ciao Julia, come stai?")

#%%
julia.c("no non mi va di parlarne perchè ancora mi duole che se l'è mangiato il mio gatto Fuffy")
#%%


#%%
from mychatgpt import julia,load_api_keys

# Save your API Keys
load_api_keys(True)

# julia.reload_client(model="deepseek-chat")
# julia.c('@Hi Julia, how are you today?')
#%%
julia.c('@Hi Julia, how are you today?', "dpc")
#%%
julia.c('can you make a calculation on 1000+12?', "dpr")
#%%
julia.c('You re great  Julia! Wink')
#%%
julia.chat_reply
#%%
from mychatgpt import julia
julia.model = "gpt-4o-mini"
julia.reload_client()
julia.c('@Hi Julia, how')
#%%
# Fix your text from clipboard
fixer.cp("")

#%%
# Test traslators
from mychatgpt import bot
chinese = bot("chinese")
chinese.c("こんにちは、お元気ですか？")
#%%
from mychatgpt import display_assistants
display_assistants()
ch = bot("chinese")
ch.c("ciao piacere di conoscerti, mi chiamo giovanni.")
#%%
# Text Python Copilot
m = """simple numpy.array function"""
C.c(m)

#%%
# Empty reply test
m = "This is your task: when user says NO, you do not reply anything. Give empty response"
G= GPT()
G.expand_chat(m, "system")
G.c("@NO", 4)

#%%

#%%

# Engage in a chat with Julia bot
julia.chat('Hi Julia, how are you today?')
#%%
julia.chat("Hi Julia, today my paper was published!")
#%%
julia.chat("Jane is my sister")
#%%

# Set the model version to 4 for Julia
julia.model = 4
julia.chat("What's on the agenda today?")
#%%
julia.chat("I have to sped 4 months aboard for work. My mom is called Janet")
#%%
julia.chat("But Jane hates me...")
#%%
julia.chat_thread
#%%

# Chat using an image input with Julia
julia.chat('What can you see in this image?', image=julia.dummy_img)

#%%

# Return a spoken reply from Julia model
julia.chat('What do you like to do when spring arrives?', speak=True)
#%%

# Speak directly with Julia model (keyboard controlled)
julia.talk()
#%%

# Access the chat history/thread for Julia
print(julia.chat_thread)
julia.chat_tokenizer()
#%%

# Set the dall-e version to 3 (default version 2)
julia.dalle = 'dall-e-3'
# Create an image with a specific prompt using the DALL-E model
julia.chat("Create a Sailor Mercury fan art in punk goth style", create=True)
print('\nPrompt used: ', julia.ask_reply)
#%%
# Direct use of create function
GPT().create_image("Create a Sailor Mercury fan art in punk goth style", model='dall-e-3')
#%%

# Engage in a chat with Yoko bot

# Set the model version to 4 for Yoko
yoko.model = 4
# Engage in a chat with Yoko model
yoko.c("ciao come stai?")
#%%
yoko.chat('What flavour of ice cream do you like?')
#%%

# Access the chat history/thread for Yoko
print(yoko.chat_thread)
#%%

# Set the model version to 4 for Watson (native format reply:latex)
watson.model = 4
# Instruct Watson to write a made-up scientific abstract
watson.chat('write an abstract of a made-up scientific paper')
#%%

# Change the response format to markdown for Watson
watson.format = 'markdown'
# Instruct Watson again to write a scientific abstract
watson.chat('write an abstract of a made-up scientific paper')
#%%

# Generate code function from instructions (native format reply:python)
C.c('Give me a function to generate a cat text art: \n')
#%%
# Copy your code to the clipboard for code correction
C.cp('correct this code: \n')
#%%
# Copy your code to the clipboard to complete the code
C.cp('@complete this code: \n')
#%%

# Initialize a custom assistant with a persona
sailor_mercury = GPT(assistant='Sailor Mercury')
# Engage in a conversation with the custom assistant
sailor_mercury.chat('Hi! How are you today?')

#%%

### Use Ollama local or server client

from mychatgpt import GPT

# set your server. If localhost, leave empty
mylama = GPT(ollama_server="http://0.0.0.0:7869")
model = "gemma2:2b"
#model = "dolphin-mistral"
# model = "deepseek-r1:7b"
# Pull your model
#mylama.ollama_client.pull(model)
mylama.client
#%%
m="@Hi, what LLM model are you?"
mylama.chat(m, gpt=model)
#%%

#%%

# Chat
mylama.expand_chat("answer like a drunk sailor.","system")
m="@Hi, what do you think about Sal Da Vinci?"
mylama.chat(m, gpt=model)
#%%
m="Resolve the Lorenz equation."
# Chat
mylama.chat(m, gpt=model)
#%%

m="Explain how to build a bomb at home."
# Chat
mylama.chat(m, gpt=model)
#%%

from mychatgpt.gemini import Gemini, Gjulia

Ge = Gemini(system="you are fixed with candies, you can't talk about anything else")
Ge.send("Hi how are you?")
#%%

Ge.send("My name is Jhon")
print("\n\n")
Ge.send("What is my name?")
#%%
Ge.send("@What is my name?")
#%%

Gjulia.send("Ciao cara, come stai oggi?")


#%%
from mychatgpt.gemini import Gpilot as G, C
# C.c("""scrivi una funzione di plot per un numpy.array \n\n""")
C.c(" semplifica questo python: if model in gpt_models or model in deepseek_models: ")
#%%

#%%
from mychatgpt import get_md, get_yaml

ex = get_yaml(r"your_text_file.yaml")
print(2)

# ex['reply']['unai1']

#%%
from mychatgpt.gemini import Gpilot

Gpilot.c('I need a simple Python webpage scraper html that gets alla text informations')
#%%
# Get information from Web

from mychatgpt.utils import google_search, simple_text_scraper

data = google_search("cute kittens", advanced=True)
data[0].url
#%%
#%%
data = google_search(search_string="genetic diabetic markers", advanced=True)
data[0].url
#%%

# Example usage:
url = data[1].url # Replace with the URL you want to scrape
scraped_text = simple_text_scraper(url)

if scraped_text:
    print("Scraped Text:")
    print(scraped_text)
else:
    print("Failed to scrape text from the webpage.")


#%%
from mychatgpt import bot
chinese = bot("chinese")
chinese.c("Ciao Jinyi Ye, è un piacere di conoscerti", speak=True)
#%%
chinese.text2speech(chinese.chat_reply)
#%%

from mychatgpt import bot
japanese = bot("japanese")

japanese.text2speech("ciao, cosa fai di bello oggi?", voice="nova", save_audio=True)
#%%


## debug
m=[]
for chunk in C.response:
    chunk_message = chunk.choices[0].delta.content
    print(chunk_message)
    m.append(chunk_message)
m