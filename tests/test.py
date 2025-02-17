# Import module
from mychatgpt import julia, yoko, chinese, watson, C, GPT, fixer

#%%
julia.chat('Hi Julia, how are you today?')
#%%
# Fix your text from clipboard
fixer.cp("")

#%%
# Test traslators

chinese.c("こんにちは、お元気ですか？")
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

# Engage in a chat with Julia agent
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

# Engage in a chat with Yoko agent

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

from mychatgpt import GPT, creator

# set your server. If localhost, leave empty
mylama = GPT(ollama_server="http://0.0.0.0:7869")
model = "gemma2:2b"
#model = "dolphin-mistral"
#model = "deepseek-r1:7b"
# Pull your model
#mylama.ollama_client.pull(model)
#%%
m="@Hi, what LLM model are you?"
mylama.chat(m, gpt=model)
#%%

#%%

# Chat
mylama.expand_chat("answer like a drunk sailor.","system")
m="@Hi, what do you think about Sal Da Vinci???"
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
C.c("""scrivi una funzione di plot per un numpy.array \n\n""")
#%%

#%%
from mychatgpt import get_md, get_yaml

ex = get_yaml(r"your_text_file.yaml")
print(2)

# ex['reply']['unai1']

#%%
from mychatgpt.gemini import Gpilot

Gpilot.c('I need a simpile Python webpage scraper html that gets alla text informations')
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
from mychatgpt import chinese

chinese.c("Ciao Jinyi Ye, è un piacere di conoscerti", speak=True)
#%%
chinese.text2speech(chinese.chat_reply)
#%%

from mychatgpt import japanese

japanese.text2speech("ciao, cosa fai di bello oggi?", voice="nova", save_audio=True)
#%%
