import os, sys, contextlib
from openai import OpenAI
import streamlit as st
import base64
#from assistants import *
from mychatgpt import GPT, play_audio
from mychatgpt.utils import *
from mychatgpt.assistants import *
from mychatgpt import rileva_lingua, update_log

save_log=True
ss = st.session_state
### session states name definition ####
chat_num = '2'
assistant_name = f"assistant_{chat_num}"
format_name    = f"format_{chat_num}"
chat_n         = f"chat_{chat_num}"
sys_addings    = f"sys_add_{chat_num}"
model_name = f"model_{chat_num}"


import os
import pickle

def generate_chat_name(path, assistant_name):
    # Counter starts at 1
    index = 1
    # Continuously checking if the file with the current index exists
    while os.path.exists(os.path.join(path, f"{assistant_name}_{index}.pkl")):
        index += 1
    # Return the chat name with the next available index
    return f"{assistant_name}_{index}"

def save_chat_as_pickle(path='chats/'):
    if not os.path.exists(path):
        os.mkdir(path)

    chat_name = generate_chat_name(path, ss[assistant_name])
    # Save chat content in pickle format
    with open(os.path.join(path, chat_name + '.pkl'), 'wb') as file:
        pickle.dump(ss[chat_n], file)
    return chat_name

def load_chat_from_pickle(file_path):
    # Load chat content from pickle file
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    else:
        return False

#%%
# General parameters
api_models = ['gpt-4o-mini', 'gpt-4o',
              # "o1-mini",
              "deepseek-chat", "deepseek-reasoner",
              "grok-2-latest",

              "gemma2-9b-it",
              #"llama-3.3-70b-versatile",
              #"llama-3.1-8b-instant",
              #"llama-guard-3-8b",
              #"llama3-70b-8192",
              #"llama3-8b-8192",
              #"allam-2-7b",
              "deepseek-r1-distill-llama-70b",
              "meta-llama/llama-4-maverick-17b-128e-instruct",
              "meta-llama/llama-4-scout-17b-16e-instruct",
              #"mistral-saba-24b",
              #"playai-tts",
              "qwen-qwq-32b"
              
              ]


# Function to be executed on button click
def clearchat():
    ss[chat_n] = [{"role": "system", "content": ss['assistant']}]
    st.write("Chat cleared!")
def clearsys():
    ss[chat_n] = [entry for entry in ss[chat_n] if entry['role'] != 'system']
    st.write("System cleared!")

def remove_system_entries(input_list):
    return [entry for entry in input_list if entry.get('role') != 'system']
def update_assistant(input_list):
    updated_list = remove_system_entries(input_list)
    updated_list.append({"role": "system", "content": ss.assistant })
    for add in  ss[sys_addings]:
        updated_list.append({"role": "system", "content": add })
    return updated_list

def remove_last_non_system(input_list):
    # Iterate backwards to find and remove the last non-system entry
    for i in range(len(input_list) - 1, -1, -1):
        if input_list[i].get('role') != 'system':
            del input_list[i]  # Remove the entry
            break  # Exit the loop once the entry is removed
    return input_list

# assistant_list = list(assistants.keys())
assistant_list = [
    'none', 'base', 'creator', 'fixer', 'novelist', 'delamain',  'oracle', 'snake', 'roger', #'robert',
    'leonardo', 'galileo', 'newton',
    'mendel', 'watson', 'crick', 'venter',
    'collins', 'elsevier', 'springer',
    'darwin', 'dawkins',
    'penrose', 'turing', 'marker',
    'mike', 'michael', 'julia', 'jane', 'yoko', 'asuka', 'misa', 'hero', 'xiao', 'peng', 'miguel', 'francois', 'luca',
    'english', 'spanish', 'french', 'italian', 'portuguese', 'korean', 'chinese', 'japanese', 'japanese_teacher', 'portuguese_teacher'
]

# Try to import 'extra' from 'extra_assistant' if it's available
try:
    from extra_assistants import extra
except ImportError:
    # If the import fails, initialize 'extra' as an empty dictionary
    extra = {}
# Add values from 'extra' to 'assistants'
assistants.update(extra)
# Add keys from 'extra' to 'assistant_list'
assistant_list.extend(extra.keys())


if assistant_name not in ss:
    ss[assistant_name] = 'none'

format_list = list(features['reply_style'].keys())
if format_name not in ss:
    ss[format_name] = 'base'

if "assistant" not in ss:
    ss['assistant'] = assistants[ss[assistant_name]]

# Build assistant
ss['assistant'] = assistants[ss[assistant_name]] + features['reply_style'][ss[format_name]]

# Initialize Chat Thread
if chat_n not in ss:
    #ss[chat_n] = [{"role": "assistant", "content": "How can I help you?"}]
    ss[chat_n] = [{"role": "system", "content": ss["assistant"]}]

if sys_addings not in ss:
    ss[sys_addings] = []

if model_name not in ss:
    ss[model_name] = "gpt-4o-mini"

# Update assistant in chat thread
ss[chat_n] = update_assistant(ss[chat_n])

# Immagazzina il valore corrente in session_state e imposta il valore predefinito se non esiste
# if 'assistant_index' not in ss:
#     ss.assistant_index = 0

#%%

# Check if the file exists
# if os.path.exists('openai_api_key.txt'):
if len(list(load_api_keys().keys() )) > 0:
    api_keys = load_api_keys()
    ss.openai_api_key   = api_keys.get("openai", "missing")
    ss.gemini_api_key   = api_keys.get("gemini", "missing")
    ss.deepseek_api_key = api_keys.get("deepseek", "missing")
    ss.x_api_key        = api_keys.get("grok", "missing")
    ss.groq_api_key     = api_keys.get("groq", "missing")

    # with open('openai_api_key.txt', 'r') as file:
    #     ss.openai_api_key = file.read().strip()
    #     #ss.openai_api_key = str(open('openai_api_key.txt', 'r').read())
else:
    ss.openai_api_key   = None
    ss.gemini_api_key   = None
    ss.deepseek_api_key = None
    ss.x_api_key        = None
    ss.groq_api_key    = None

print("App Ready!")

# <<<<<<<<<<<<Sidebar code>>>>>>>>>>>>>
with st.sidebar:

    if not ss.openai_api_key:
        ss.openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    if not ss.deepseek_api_key:
        #ss.gemini_api_key   = st.text_input("Gemini API Key", type="password")
        ss.deepseek_api_key = st.text_input("Deepseek API Key",  type="password")
    if not ss.x_api_key:
        ss.x_api_key  = st.text_input("Xai API Key",  type="password")
    if not ss.x_api_key:
        ss.groq_api_key  = st.text_input("Groq API Key",  type="password")

    # if ss.openai_api_key and ss.deepseek_api_key and ss.x_api_key:
    #     st.markdown("[API key provided]")
    keys = {
        "OpenAI": ss.openai_api_key,
        "DeepSeek": ss.deepseek_api_key,
        "X": ss.x_api_key,
        "Groq": ss.groq_api_key
    }
    provided_keys = [name for name, key in keys.items() if key]
    st.markdown(", ".join(provided_keys) + " API key(s) provided" if provided_keys else "No API keys provided")

    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    st.markdown("[View the source code](https://github.com/johndef64/mychatgpt/tree/main/mychatbot)")
    #st.markdown("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

    # Button to show markdown text
    Info = False
    if st.button("Show Info"):
        Info = True


    ### select values ###
    model = st.radio('Choose a model:', api_models, index=api_models.index(ss[model_name]))

    #assistant = st.selectbox("Assistant", ['none', 'penrose', 'leonardo', 'mendel', 'darwin','delamain'])
    #language = st.selectbox("Language", ['none', 'Japanese','French','English', 'Portuguese', 'Italian', 'Chinese', 'Spanish'])
    get_assistant = st.selectbox("**Assistant**", assistant_list, index=assistant_list.index(ss[assistant_name]))
    get_format = st.selectbox("**Reply Format**", format_list, index=format_list.index(ss[format_name]))

    translate_in = st.selectbox("**Translate Reply in**", ["none", "English", "French", "Japanese", "Italian", "Spanish"])

    instructions = st.text_input("Add Instructions")

    #play_audio_ = st.checkbox('Play Audio?')
    col1, col2 = st.columns(2)
    play_audio_ = col1.checkbox('Play Audio', value=False)
    copy_reply_ = col2.checkbox('Copy Reply', value=False)
    run_code = col1.checkbox('Run Code', value=False)

    # Update session state with the selected value
    ss[assistant_name] = get_assistant
    ss[format_name] = get_format
    ss[model_name] = model
    ### UPDATE HERE CHAT THERD WITH NEW ASSISTANT (Replace)



    # Add a button in the sidebar and assign the function to be executed on click
    #st.markdown("Press Clearchat after Assistant selection")
    col12, col22 = st.columns(2)
    if col12.button("Clear chat"):
        clearchat()
    if col22.button("Clear system"):
        clearsys()

    # Uploaders

    image_path = st.text_input("Load Image (path or url)")
    #image_file = st.file_uploader("Upload an image file", type=("jpg", "png"))
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    uploaded_file = st.file_uploader("Upload an text file", type=("txt", "md"))

    user_avi = st.selectbox('Change your avatar', ['🧑🏻', '🧔🏻', '👩🏻', '👧🏻', '👸🏻','👱🏻‍♂️','🧑🏼','👸🏼','🧒🏽','👳🏽','👴🏼', '🎅🏻', ])

    # Additional button
    if st.button("Save Chat"):
        chat_name = save_chat_as_pickle()
        st.write(f"Chat Saved as {chat_name}!")

    # List files in the 'chats/' directory
    files_in_chats = os.listdir('chats/') if os.path.exists('chats/') else (os.makedirs('chats'), [])[1]

    # Implement a select box to choose a file
    chat_path = st.selectbox("Choose a file to load", files_in_chats)
    full_path = os.path.join('chats/', chat_path)

    col1, col2 = st.columns(2)
    if col1.button("Load Chat"):
        ss[chat_n] = load_chat_from_pickle(full_path)
        st.write("Chat Loaded!")

    if col2.button("Delete Chat"):
        delete_file(full_path)
        st.write("Chat Deleted!")



############################################################################################
############################################################################################

from mychatgpt import gpt_models, deepseek_models, x_models, groq_models, Groq
# selct client
def select_client(model):
    if model in gpt_models:
        client = OpenAI(api_key=load_api_keys()["openai"])
    elif model in deepseek_models:
        print("using DeepSeek model")
        client = OpenAI(api_key=load_api_keys()["deepseek"], base_url="https://api.deepseek.com")
    elif model in x_models:
        print("using Xai model")
        client = OpenAI(api_key=load_api_keys()["grok"], base_url="https://api.x.ai/v1")
    elif model in groq_models:
        print("using Groq models")
        client = Groq(api_key=load_api_keys()["groq"])
    return client


# <<<<<<<<<<<< >>>>>>>>>>>>>

def add_instructions(instructions):
    if not any(entry.get("role") == "system" and instructions in entry.get("content", "") 
           for entry in ss[chat_n]):
        ss[chat_n].append({"role": "system", "content": instructions})

### Add Context to system
if instructions:
   #add_instructions(instructions)
   ss[sys_addings].append(instructions)

if uploaded_file:
    text = uploaded_file.read().decode()
    ss[chat_n].append({"role": "system", "content": "Read the text below and add it's content to your knowledge:\n\n"+text})

#if uploaded_file:
#image = uploaded_image.read().decode()
#ss[chat_n] = ...

# # Update session state with the selected value
# ss[assistant_name] = get_assistant
# ss["format_name"] = get_format



st.title("💬 Ask Assistant")
st.caption("🚀 Your GPT Assistant powered by OpenAI")

# Draft information formatted within an info box
info = """
#### Quick Commands:
- Start message with "+" to add message without getting a reply
- Start message with "++" to add additional system instructions
- Enter ":" to pop out last iteration
- Enter '-' to clear chat
"""
# Display the info box using Streamlit's 'info' function
# st.info(info)

AssistantInfo = """
#### Copilots 💻
- **Base**: Assists with basic tasks and functionalities.
- **Novelist**: Specializes in creative writing assistance.
- **Creator**: Aids in content creation and ideation.
- **Fixer**: Can fix any text based on the context.
- **Delamain**: Coding copilot for every purpose.
- **Oracle**: Coding copilot for every purpose.
- **Snake**: Python coding copilot for every purpose.
- **Roger**: R coding copilot for every purpose.

#### Scientific 🔬
- **Leonardo**: Supports scientific research activities.
- **Newton**: Aids in Python-based scientific computations (Python).
- **Galileo**: Specializes in scientific documentation (Markdown).
- **Mendel**: Assists with data-related scientific tasks.
- **Watson**: Focuses on typesetting scientific documents (LaTeX).
- **Venter**: Supports bioinformatics and coding tasks (Python).
- **Crick**: Specializes in structuring scientific content (Markdown).
- **Darwin**: Aids in evolutionary biology research tasks.
- **Dawkins**: Supports documentation and writing tasks (Markdown).
- **Penrose**: Assists in theoretical research fields.
- **Turing**: Focuses on computational and AI tasks (Python).
- **Marker**: Specializes in scientific documentation (Markdown).
- **Collins**: Aids in collaborative scientific projects.
- **Elsevier**: Focuses on publication-ready document creation (LaTeX).
- **Springer**: Specializes in academic content formatting (Markdown).

#### Characters 🎭
- **Julia**: Provides character-based creative support.
- **Mike**: Provides character-based interactive chat.
- **Michael**: Provides character-based interactive chat (English).
- **Miguel**: Provides character-based interactive chat (Portuguese).
- **Francois**: Provides character-based interactive chat (French).
- **Luca**: Provides character-based interactive chat (Italian).
- **Hero**: Provides character-based interactive chat (Japanese).
- **Yoko**: Provides character-based creative support (Japanese).
- **Xiao**: Provides character-based creative support (Chinese).
- **Peng**: Provides character-based interactive chat (Chinese).

#### Languages 🌐
- **English, French, Italian, Portuguese**
- **Chinese**: Facilitates Chinese language learning.
- **Japanese**: Aids in Japanese language learning and translation.
- **Japanese Teacher**: Specializes in teaching Japanese.
- **Portuguese Teacher**: Provides assistance with learning Portuguese.

"""

if Info:
    st.info(f"{AssistantInfo}")
    st.info(f"{info}")



# Update Language Automatically
#if ss.persona not in ss[chat_n]:
#    ss[chat_n] = update_assistant(ss[chat_n])

voice_dict = {
    'none':'echo','luca':'onyx',
    'hero':'echo', 'peng':'echo',
    'yoko':'nova', 'xiao':'nova',
    'miguel':'echo', 'francois':'onyx', 'michael':'onyx',
    'julia':'shimmer', 'mike':'onyx',
    'penrose':'onyx', 'leonardo':'onyx', 'mendel':'onyx', 'darwin':'onyx','delamain':'onyx'
}

avatar_dict = {
    'none':"🤖",
    'base':"🤖",
    'hero':"👦🏻", 'yoko':"👧🏻", 'peng':"👦🏻", 'xiao':"👧🏻",
    'miguel':"🧑🏼", 'francois':"🧑🏻",
    'luca':"🧔🏻", 'michael':"🧔🏻",
    'julia':"👱🏻‍♀️", 'mike':"👱🏻‍♂️",
    'penrose':"👨🏻‍🏫", 'leonardo':"👨🏻‍🔬", 'mendel':"👨🏻‍⚕️",
    'darwin':"👴🏻", 'dawkins':"👴🏻",
    'delamain':"👨🏻‍💻",'snake':"👨🏻‍💻",'roger':"👨🏻‍💻",
    'alfred':"🤵🏻",
    'laura':"👩🏻",
}
voice = voice_dict.get(get_assistant, "echo")
chatbot_avi = avatar_dict.get(get_assistant, "🤖")

print("Voice:", voice)



# Trigger the specific function based on the selection
#if assistant and not ss[chat_n] == [{"role": "system", "content": assistants[assistant]}]:
#    ss[chat_n] = [{"role": "system", "content": assistants[assistant]}]
#    #st.write('assistant changed')


# <<<<<<<<<<<<Display chat>>>>>>>>>>>>>
def display_chat():
    for msg in ss[chat_n]:
        if msg['role'] != 'system':
            if not isinstance(msg["content"], list):
                # Avatar
                if msg["role"] == 'user':
                    avatar = user_avi
                else:
                    avatar = chatbot_avi
                st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

display_chat()


# <<<<<<<<<<<<Engage chat>>>>>>>>>>>>>

if prompt := st.chat_input():
    # Quick commands:
    if prompt in ["-", "@"]:
        clearchat()
        time.sleep(0.7)
        st.rerun()

    elif prompt.startswith("+"):
        prompt = prompt[1:]
        role = "user"
        if prompt.startswith("+"):
            prompt = prompt[1:]
            role = "system"

        if role == "system":
            ss[sys_addings].append(prompt)
            st.write("Instrucition updated!")
        else:
            ss[chat_n].append({"role": role, "content": prompt})
            st.chat_message(role, avatar=user_avi).write(prompt)


    
    elif prompt in ["."]:
        remove_last_non_system(ss[chat_n])
        #ss[chat_n].pop()
        #ss[chat_n].pop()
        st.rerun()

    else:
        if not ss.openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        if image_path:
            if image_path.startswith('http'):
                print('<Image path:',image_path, '>')
                pass
            else:
                print('<Enconding Image...>')
                base64_image = encode_image(image_path)
                image_path = f"data:image/jpeg;base64,{base64_image}"

            image_add = {"role": 'user',
                        "content": [{"type": "image_url", "image_url": {"url": image_path} }] }
            if image_add not in ss[chat_n]:
                ss[chat_n].append(image_add)

        #client = OpenAI(api_key=ss.openai_api_key)
        client = select_client(model)
        
        # Get User Prompt:
        ss[chat_n].append({"role": "user", "content": prompt})
        st.chat_message('user', avatar=user_avi).write(prompt)
        
        # Build Chat Thread
        chat_thread = []
        for msg in ss[chat_n]:
            if isinstance(msg["content"], list):
                chat_thread.append(msg)
            elif not msg["content"].startswith('<<'):
                chat_thread.append(msg)

        # Generate Reply        
        response = client.chat.completions.create(model=model,
                                                messages=chat_thread,
                                                stream=False,
                                                top_p=1,
                                                frequency_penalty=0,
                                                presence_penalty=0
                                                )

        reply = response.choices[0].message.content

        # Append Reply
        ss[chat_n].append({"role": "assistant", "content": reply})
        st.chat_message('assistant', avatar=chatbot_avi).write(reply)
        if check_copy_paste() and copy_reply_:
            pc.copy(reply)
        if save_log:
            update_log(ss[chat_n][-2])
            update_log(ss[chat_n][-1])

        if translate_in != 'none':
            language = translate_in
            reply_language = rileva_lingua(reply)
            if reply_language == 'Japanese':
                translator = create_jap_translator(language)
            elif 'Chinese' in reply_language.split(" "):
                translator = create_chinese_translator(language)
            else:
                translator = create_translator(language)
            response_ = client.chat.completions.create(model=model,
                                                    messages=[{"role": "system", "content": translator},
                                                                {"role": "user", "content": reply}])
            translation = "<<"+response_.choices[0].message.content+">>"
            ss[chat_n].append({"role": "assistant", "content": translation})
            st.chat_message('assistant').write(translation)


        if play_audio_:
            Text2Speech(reply, voice=voice)

        if run_code:
            from ExecuteCode import ExecuteCode
            ExecuteCode(reply)



    #with col2:
    #    if st.button("New Chat"):
    #    clearchat()






#%%


