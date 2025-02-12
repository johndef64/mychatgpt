from .main import *

# in pyhton
# check se il Python version >= 3.9 :
#     chek if  "google-genai" is installed
#     if not install it
#     else:
#         stop this scripts here

#%%
import sys
import subprocess

# Check Python version
if sys.version_info >= (3, 9):
    try:
        # Try to import the module
        from google import genai
        # Stop script if module is already installed
        print("google-genai is already installed.")
    except ImportError:
        # Install the module if it is not installed
        print("google-genai is not installed. Installing now.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai"])
else:
    print("Python version is less than 3.9.")
    sys.exit()
#%%

from google import genai
from google.genai import types
if gemini_api_key != "missing":
    #print("Initializing Gemini Client...")
    gemini_client = genai.Client(api_key=gemini_api_key)


######## Gemini API ########
if gemini_api_key != "missing":
    class Gemini:

        def __init__(self,
                     client = gemini_client,
                     system="you are a helpful assistant",
                     GEMINI_MODEL = "gemini-2.0-flash-exp"):
            # Inizializza la classe con un'istruzione di sistema
            self.system = system
            self.client = client
            self.GEMINI_MODEL = GEMINI_MODEL
            self.chat_reply = None

            #def instruct(self, client, types):
            # Crea una chat usando il client e le configurazioni specificate
            self.chat = self.client.chats.create(
                model=self.GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=self.system,  # Usa l'istruzione della classe
                    temperature=0.5,
                ),
            )

        def send(self, message, paste=False, to_clip=True):
            if message.startswith("@"):
                self.clear_chat()
                message = message.lstrip("@")

            p = pc.paste() if paste else ''

            response = self.chat.send_message(message+p)
            self.chat_reply = response.text
            print(self.chat_reply)

            if to_clip and has_copy_paste:
                clip_reply = self.chat_reply.replace('```', '###')
                pc.copy(clip_reply)

        c = send

        def cp(self, *args, **kwargs):
            kwargs['paste'] = True  # Ensure paste is always set to True
            self.send(*args, **kwargs)


        def clear_chat(self, _print=True):
            if _print: print("<chat cleared>")
            self.chat._curated_history = []

        # usage Gemini.chat.send_message
        # Gemini().send("Hi how are you")

    gemini = Gemini(system=assistants['base'])
    Gpilot = Gemini(system=assistants['delamain'])
    Rpilot = Gemini(system=assistants['roger'])
    Gleonardo = Gemini(system=assistants['leonardo'])
    Gpenrose = Gemini(system=assistants['penrose'])
    Gjulia = Gemini(system=assistants['julia'])
