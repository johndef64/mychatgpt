import os
from openai import OpenAI
import streamlit as st
import base64
#from assistants import *
from mychatgpt import GPT, play_audio
from mychatgpt.utils import *
from mychatgpt.assistants import *
from mychatgpt import rileva_lingua, update_log

save_log=True
#%%
# General parameters
api_models = ['gpt-4o-mini', 'gpt-4o',
              # "o1-mini",
              "deepseek-chat", "deepseek-reasoner",
              "grok-2-latest"]


# Function to be executed on button click
def clearchat():
    st.session_state["chat_thread"] = [{"role": "system", "content": st.session_state['assistant']}]
    st.write("Chat cleared!")

def remove_system_entries(input_list):
    return [entry for entry in input_list if entry.get('role') != 'system']
def update_assistant(input_list):
    updated_list = remove_system_entries(input_list)
    updated_list.append({"role": "system", "content": st.session_state.assistant})
    return updated_list

# assistant_list = list(assistants.keys())
assistant_list = ['none', 'base', 'creator', 'fixer', 'novelist', 'delamain',  'oracle', 'roger', 'robert', 'galileo', 'newton', 'leonardo', 'mendel', 'watson', 'crick', 'venter', 'collins', 'elsevier', 'springer', 'darwin', 'dawkins', 'turing', 'marker', 'penrose', 'mike', 'michael', 'julia', 'jane', 'yoko', 'asuka', 'misa', 'hero', 'xiao', 'peng', 'miguel', 'francois', 'luca', 'english', 'spanish', 'french', 'italian', 'portuguese', 'korean', 'chinese', 'japanese', 'japanese_teacher', 'portuguese_teacher' ]
# print(assistant_list)
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

if "assistant_name" not in st.session_state:
    st.session_state['assistant_name'] = 'none'

format_list = list(features['reply_style'].keys())
if "format_name" not in st.session_state:
    st.session_state["format_name"] = 'base'

if "assistant" not in st.session_state:
    st.session_state['assistant'] = assistants[st.session_state["assistant_name"]]

# Build assistant
st.session_state['assistant'] = assistants[st.session_state["assistant_name"]] + features['reply_style'][st.session_state["format_name"]]

# Initialize Chat Thread
if "chat_thread" not in st.session_state:
    #st.session_state["chat_thread"] = [{"role": "assistant", "content": "How can I help you?"}]
    st.session_state["chat_thread"] = [{"role": "system", "content": st.session_state["assistant"]}]

# Update assistant in chat thread
st.session_state["chat_thread"] = update_assistant(st.session_state["chat_thread"])

#%%

# Check if the file exists
# if os.path.exists('openai_api_key.txt'):
if len(list(load_api_keys().keys() ))> 0:
    api_keys = load_api_keys()
    st.session_state.openai_api_key   = api_keys.get("openai", "missing")
    st.session_state.gemini_api_key   = api_keys.get("gemini", "missing")
    st.session_state.deepseek_api_key = api_keys.get("deepseek", "missing")
    st.session_state.x_api_key        = api_keys.get("grok", "missing")

    # with open('openai_api_key.txt', 'r') as file:
    #     st.session_state.openai_api_key = file.read().strip()
    #     #st.session_state.openai_api_key = str(open('openai_api_key.txt', 'r').read())
else:
    st.session_state.openai_api_key = None
    st.session_state.gemini_api_key   = None
    st.session_state.deepseek_api_key = None
    st.session_state.x_api_key        = None

print("App Ready!")

# <<<<<<<<<<<<Sidebar code>>>>>>>>>>>>>
with st.sidebar:

    if not st.session_state.openai_api_key:
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        #st.session_state.gemini_api_key   = st.text_input("Gemini API Key", type="password")
        st.session_state.deepseek_api_key = st.text_input("Deepseek API Key",  type="password")
        st.session_state.x_api_key        = st.text_input("Xai API Key",  type="password")

    else:
        st.markdown("[API key provided]")

    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")
    st.markdown("[View the source code](https://github.com/johndef64/mychatgpt/tree/main/mychatbot)")
    #st.markdown("[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)")

    # Button to show markdown text
    Info = False
    if st.button("Assistant Info"):
        Info = True



    #model = st.selectbox("GPT model", ['gpt-3.5-turbo', 'gpt-4o'])
    model = st.radio('Choose a model:', api_models)

    #assistant = st.selectbox("Assistant", ['none', 'penrose', 'leonardo', 'mendel', 'darwin','delamain'])
    #language = st.selectbox("Language", ['none', 'Japanese','French','English', 'Portuguese', 'Italian', 'Chinese', 'Spanish'])
    get_assistant = st.selectbox("**Assistant**", assistant_list)#, index=assistant_list.index(st.session_state["assistant_name"]))

    get_format = st.selectbox("**Format**", format_list     )#, index=format_list.index(st.session_state["format_name"]))

    translate_in = st.selectbox("**Translate in**", ["none", "English", "French", "Japanese", "Italian", "Spanish"])

    play_audio_ = st.checkbox('Play Audio?')

    # Update session state with the selected value
    st.session_state["assistant_name"] = get_assistant
    st.session_state["format_name"] = get_format
    ### UPDATE HERE CHAT THERD WITH NEW ASSISTANT (Replace)

    # Add a button in the sidebar and assign the function to be executed on click
    #st.markdown("Press Clearchat after Assistant selection")
    if st.button("Clear chat"):
        clearchat()

    # Uploaders
    uploaded_file = st.file_uploader("Upload an text file", type=("txt", "md"))

    image_path = st.text_input("Load Image (path or url)")
    #image_file = st.file_uploader("Upload an image file", type=("jpg", "png"))
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    instructions = st.text_input("Add Instructions")

    user_avi = st.selectbox('Change your avatar', ['🧑🏻', '🧔🏻', '👩🏻', '👧🏻', '👸🏻','👱🏻‍♂️','🧑🏼','👸🏼','🧒🏽','👳🏽','👴🏼', '🎅🏻', ])

############################################################################################
############################################################################################
#

from mychatgpt import gpt_models, deepseek_models,x_models
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
    return client


# <<<<<<<<<<<< >>>>>>>>>>>>>

### Add Context to system
if instructions:
    st.session_state["chat_thread"].append({"role": "system", "content": instructions})

if uploaded_file:
    text = uploaded_file.read().decode()
    st.session_state["chat_thread"].append({"role": "system", "content": "Read the text below and add it's content to your knowledge:\n\n"+text})

#if uploaded_file:
#image = uploaded_image.read().decode()
#st.session_state["chat_thread"] = ...

# # Update session state with the selected value
# st.session_state[assistant_name] = get_assistant
# st.session_state["format_name"] = get_format



st.title("💬 Ask Assistant")
st.caption("🚀 Your GPT Assistant powered by OpenAI")

AssistantInfo = """
#### Copilots 💻
- **Base**: Assists with basic tasks and functionalities.
- **Novelist**: Specializes in creative writing assistance.
- **Creator**: Aids in content creation and ideation.
- **Fixer**: Can fix any text based on the context.
- **Delamain**: Coding copilot for every purpose.
- **Oracle**: Coding copilot for every purpose.
- **Roger**: R coding copilot for every purpose.
- **Robert**: R coding copilot for every purpose.
- **Copilot**: Python coding copilot for every purpose.

#### Scientific 🔬
- **Leonardo**: Supports scientific research activities.
- **Newton**: Aids in Python-based scientific computations.
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



# Update Language Automatically
#if st.session_state.persona not in st.session_state["chat_thread"]:
#    st.session_state["chat_thread"] = update_assistant(st.session_state["chat_thread"])

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
    'delamain':"👨🏻‍💻",
    'alfred':"🤵🏻"
}
voice = voice_dict.get(get_assistant, "echo")
chatbot_avi = avatar_dict.get(get_assistant, "🤖")

print("Voice:", voice)



# Trigger the specific function based on the selection
#if assistant and not st.session_state["chat_thread"] == [{"role": "system", "content": assistants[assistant]}]:
#    st.session_state["chat_thread"] = [{"role": "system", "content": assistants[assistant]}]
#    #st.write('assistant changed')


# <<<<<<<<<<<<Display chat>>>>>>>>>>>>>
for msg in st.session_state["chat_thread"]:
    if msg['role'] != 'system':
        if not isinstance(msg["content"], list):
            # Avatar
            if msg["role"] == 'user':
                avatar = user_avi
            else:
                avatar = chatbot_avi
            st.chat_message(msg["role"], avatar=avatar).write(msg["content"])


# Create a container with input and button
#with st.container():
#    col1, col2 = st.columns([4, 1]) # create two columns within the container
#    with col1:

if prompt := st.chat_input():
    if prompt.startswith('@'):
        prompt = prompt[1:]
        clearchat()

    if not st.session_state.openai_api_key:
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
        if image_add not in st.session_state["chat_thread"]:
            st.session_state["chat_thread"].append(image_add)

    #client = OpenAI(api_key=st.session_state.openai_api_key)
    client = select_client(model)
    
    # Get User Prompt:
    st.session_state["chat_thread"].append({"role": "user", "content": prompt})
    st.chat_message('user', avatar=user_avi).write(prompt)
    
    # Generate Reply
    chat_thread = []
    for msg in st.session_state["chat_thread"]:
        if not msg["content"].startswith('<<'):
            chat_thread.append(msg)
    response = client.chat.completions.create(model=model,
                                            messages=chat_thread,
                                            stream=False,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0
                                            )

    reply = response.choices[0].message.content

    # Append Reply
    st.session_state["chat_thread"].append({"role": "assistant", "content": reply})
    st.chat_message('assistant', avatar=chatbot_avi).write(reply)
    if check_copy_paste():
        pc.copy(reply)
    if save_log:
        update_log(st.session_state["chat_thread"][-2])
        update_log(st.session_state["chat_thread"][-1])

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
        st.session_state["chat_thread"].append({"role": "assistant", "content": translation})
        st.chat_message('assistant').write(translation)


    if play_audio_:
        Text2Speech(reply, voice=voice)

    #with col2:
    #    if st.button("New Chat"):
    #    clearchat()



# # Additional button
# if st.button("Additional Action"):
#     st.write("Button pressed!")



#%%


