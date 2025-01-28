# Import module
from mychatgpt import julia, yoko, watson, C, GPT, Julia

#%%
# Engage in a chat with Julia model
julia.chat('Hi Julia, how are you today?')
#%%

# Set the model version to 4 for Julia
julia.model = 4
julia.chat("What's on the agenda today?")
#%%

# Chat using an image input with Julia
julia.chat('What can you see in this image?', image=julia.dummy_img)

#%%

# Return a spoken reply from Julia model
julia.speak('What do you like to do when spring arrives?')
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

# Set the model version to 4 for Yoko
yoko.model = 4
# Engage in a chat with Yoko model
yoko.chat('What flavour of ice cream do you like?')
#%%

# Access the chat history/thread for Yoko
yoko.chat_thread
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
model = "dolphin-mistral"
model = "deepseek-r1:7b"
# Pull your model
#mylama.ollama_client.pull(model)
#%%
m="@Hi, what LLM model are you?"
mylama.chat(m, gpt=model)
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


















#%%
