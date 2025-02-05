from cryptography.fernet import Fernet
import matplotlib.pyplot as plt
import pyperclip as pc
import keyboard as kb
import pandas as pd
import importlib
import subprocess
import platform
import requests
import PyPDF2
import base64
import time
import glob
import os
import re
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import markdown
from bs4 import BeautifulSoup


def is_package_installed(package_name):
    try:
        output = subprocess.check_output("dpkg -l | grep " + package_name, shell=True)
        return bool(output)
    except subprocess.CalledProcessError:
        return False

if platform.system() == "Linux":
    if not is_package_installed("libportaudio2"):
        subprocess.check_call(["sudo","apt-get", "update"])
        subprocess.check_call(["sudo","apt-get", "install", "libportaudio2"])
    else:
        pass

import sounddevice as sd
import soundfile as sf

# try:
#     import PIL
# except ImportError:
#     subprocess.check_call(['pip', 'install', 'pillow'])
from PIL import Image

# Simple functions
def simple_bool(message, y='y', n ='n'):
    choose = input(message+" ("+y+"/"+n+"): ").lower()
    your_bool = choose in [y]
    return your_bool

# Function to check whether pyperclip works in the system
def check_copy_paste():
    try:
        pc.copy("test")
        test_text = pc.paste()
        if test_text == "test":
            return True
    except pc.PyperclipException:
        return False

def is_base64_image(data):
    # Regex pattern to check for base64 image prefix
    pattern = r'^data:image\/[a-zA-Z]+;base64,'

    # Check if data matches pattern and if it can be decoded
    if re.match(pattern, data):
        base64_str = data.split(',')[1] # Extract the base64 string
        try:
            base64.b64decode(base64_str) # Attempt to decode base64 string
            return True
        except base64.binascii.Error:
            return False
    return False

def display_files_as_pd(path=os.getcwd(), ext='',  contains=''):
    file_pattern = os.path.join(path, "*." + ext) if ext else os.path.join(path, "*")
    files = glob.glob(file_pattern)
    files_name = []
    for file in files:
        file_name = os.path.basename(file)
        files_name.append(file_name)

    files_df = pd.Series(files_name)
    file = files_df[files_df.str.contains(contains)]
    return file


############## Install Requirements ###################

def check_and_install_requirements(requirements: list):
    missing_requirements = []
    for module in requirements:
        try:
            # Check if the module is already installed
            importlib.import_module(module)
        except ImportError:
            missing_requirements.append(module)
    if len(missing_requirements) == 0:
        pass
    else:
        x = simple_bool(str(missing_requirements)+" are missing.\nWould you like to install them all?")
        if x:
            for module in missing_requirements:
                subprocess.check_call(["pip", "install", module])
                print(f"{module}' was installed correctly.")
        else:
            print("Waring: missing modules")#exit()


def get_gitfile(url, flag='', dir=os.getcwd()):
    url = url.replace('blob','raw')
    response = requests.get(url)
    file_name = flag + url.rsplit('/',1)[1]
    file_path = os.path.join(dir, file_name)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully. Saved as {file_name}")
    else:
        print("Unable to download the file.")


def get_chat():
    handle = "https://github.com/johndef64/pychatgpt/blob/main/chats/"
    response = requests.get(handle)
    data = response.json()
    files = [item['name'] for item in data['payload']['tree']['items']]
    path = os.getcwd() + '/chats'
    if not os.path.exists(path):
        os.mkdir(path)

    file = files[int(input('select chat:\n'+str(pd.Series(files))))]
    url = handle + file
    get_gitfile(url, dir=os.getcwd()+'/chats')

###### Encrypters ######

def key_gen(input_value, random_key=False):
    input_string = str(input_value)
    # Create an initial key by multiplying
    key = (input_string * 32)[:32]
    # Ensure the exact length of 32
    key_bytes = key.encode('utf-8')[:32]
    # Base64 encode the byte array to create a Fernet key
    key = base64.urlsafe_b64encode(key_bytes)
    if random_key:
        key = Fernet.generate_key()
    return key

def simple_encrypter(password: str or int = 0, txt_to_encrypt: str = "Hello World"):
    key = key_gen(password)
    cipher = Fernet(key)
    # Encrypt the string
    encrypted_text = cipher.encrypt(txt_to_encrypt.encode('utf-8'))
    return encrypted_text

def simple_decrypter(password: str or int =  0, encrypted_text: str = "Hello World"):
    key = key_gen(password)
    cipher = Fernet(key)
    try:
        # Decrypt the string
        decrypted_string = cipher.decrypt(encrypted_text).decode('utf-8')
        return decrypted_string
    except Exception as e:
        print(f"Wrong password.")
        return None

###### file manager ######

def load_file(file='', path=os.getcwd()):
    with open(os.path.join(path, file),'r', encoding='utf-8') as file:
        my_file = file.read()#ast.literal_eval(file.read())
        file.close()
    return my_file

def load_choosen_file(path=os.getcwd(), ext='', contains=''):
    files_df = display_files_as_pd(path, ext=ext, contains=contains)
    filename = str(files_df[int(input('Choose file:\n'+str(files_df)))])
    my_file = load_file(filename, path)
    return my_file

def load_multiple_files(file_list):
    loaded_files = {}
    for file_name in file_list:
        loaded_files[os.path.basename(file_name).split('.')[0]] = load_file(file=file_name)
    print('Loaded Files:', list(loaded_files.keys()))
    return loaded_files


def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

####### text parsers #####

# to get contents use
#load_file()

# extract sections from latex and markdown files
def load_and_extract_sections(file_path, clean_text=True):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()  # Read file content

    # Determine file type based on extension
    if file_path.endswith('.tex'):
        section_pattern = r'\\section\{(.+?)\}(.*?)(?=\\section|\Z)'  # LaTeX section pattern
    elif file_path.endswith('.md'):
        section_pattern = r'^(#{1,6})\s(.+)$'  # Markdown header pattern
    else:
        raise ValueError("Unsupported file type. Only '.tex' and '.md' are supported.")


    # Find sections in the content based on the determined pattern
    sections = re.findall(section_pattern, content, re.DOTALL if file_path.endswith('.tex') else re.MULTILINE)
    #content = clean_tex(content) if clean_text else content

    sections_dict = {}
    for match in sections:
        if file_path.endswith('.tex'):
            title, body = match  # Unpack LaTeX sections
        else:
            header, title = match  # Unpack Markdown headers
            body = header  # Use header as body for Markdown

        key = title.lower().replace(' ', '_')  # Convert title to lowercase and replace spaces with underscores
        sections_dict[key] = clean_tex(body.strip()) if clean_text else body.strip() # Store section body

    #df = pd.Series(sections_dict, index=sections_dict.keys())  # Create a series from the dictionary
    #display(df)
    return sections_dict #, content

def clip_tex(tex_content):
    # Keep content between \section{introduction} and \section{conclusion}
    match = re.search(r'\\section{Introduction}(.*?)\\section{Conclusion}', tex_content, flags=re.DOTALL)
    if match:
        # Return only the text between introduction and conclusion
        return match.group(1)
    else:
        # Return the original content if sections are not found
        return tex_content

def clean_tex(tex_content):
    # Remove content in \begin{figure} ... \end{figure}
    tex_content = re.sub(r'\\begin{figure.*?\\end{figure', '', tex_content, flags=re.DOTALL)
    # Remove content in \begin{table} ... \end{table}
    tex_content = re.sub(r'\\begin{table.*?\\end{table', '', tex_content, flags=re.DOTALL)
    # Remove content in \begin{comment} ... \end{comment}
    tex_content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', tex_content, flags=re.DOTALL)

    tex_content = re.sub(r'\\begin{tcolorbox}.*?\\end{tcolorbox}', '', tex_content, flags=re.DOTALL)

    # Remove lines starting with '%'
    #tex_content = '\n'.join([line for line in tex_content.split('\n') if not line.strip().startswith('%')])
    #tex_content = re.sub(r'^%.*$', '', tex_content, flags=re.MULTILINE)
    tex_content = re.sub(r'^\s*%.*$', '', tex_content, flags=re.MULTILINE)
    return tex_content

def clean_markdown(md_content):
    # Remove images ![alt text](image_url)
    md_content = re.sub(r'!\[.*?\]\(.*?\)', '', md_content)
    # Remove tables | Header | Header |
    md_content = re.sub(r'(\|.*?\|)+', '', md_content)
    # Remove code blocks ```
    md_content = re.sub(r'```.*?```', '', md_content, flags=re.DOTALL)
    # Remove HTML comments <!-- comment -->
    md_content = re.sub(r'<!\-\-.*?\-\->', '', md_content, flags=re.DOTALL)
    return md_content

def markdown_to_dict(file_path, html=None):

    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    if not html:
        html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")

    result_dict = {}
    current_h1 = None
    current_h2 = None

    for element in soup.find_all(['h1', 'h2', 'p']):
        if element.name == 'h1':
            current_h1 = element.get_text().strip()
            result_dict[current_h1] = {}
        elif element.name == 'h2':
            current_h2 = element.get_text().strip()
            result_dict[current_h1][current_h2] = ''
        # elif element.name == 'p' and current_h1 and current_h2:
        #     result_dict[current_h1][current_h2] += element.get_text() + '\n\n'
        elif element.name == 'p':
            if current_h2:  # Appendi al corrente h2 se esiste
                result_dict[current_h1][current_h2] += element.get_text() + '\n\n'
            elif current_h1:  # Fall-back su current_h1 se current_h2 non esiste
                if '' not in result_dict[current_h1]:
                    result_dict[current_h1][''] = ''
                result_dict[current_h1][''] += element.get_text() + '\n\n'

    # Trim trailing newlines from text
    for h1 in result_dict:
        for h2 in result_dict[h1]:
            result_dict[h1][h2] = result_dict[h1][h2].rstrip('\n')

    return result_dict

def reload_paper(file_path):
    global sections_dict, full_paper
    sections_dict, full_paper = load_and_extract_sections(file_path)

#expand_context_by_sections
def expand_context_by_section(gpt,
                              section_list: list=None,
                              sections_dict: dict=None,
                              clear:bool=True,
                              property:str= "of my work",
                              format='tex'):
    if clear: gpt.clear_chat()

    if section_list:
        for section in section_list:
            section_clean = clean_tex(sections_dict[section]) if format == 'tex' else clean_markdown(sections_dict[section])
            gpt.expand_chat(f'\nThis is the {section} section {property}:\n{section_clean}', 'user')

"""
# Usage
gpt = GPT()
my_sections = load_and_extract_sections('my_text_file.tex')
expand_context_by_section(gpt, ["introduction", "conclusion"], my_sections)
gpt.chat("What are the drawbacks of my work?")
"""


####### image functions #######

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def display_image(filename, jupyter = False, plotlib=True, dpi=200):
    if jupyter:
        image = Image.open(filename)
        display(image)
    elif plotlib:
        image = Image.open(filename)
        plt.figure(dpi=dpi)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    else:
        image = Image.open(filename)
        image.show()


def image_encoder(image_path: str = None):

    if image_path.startswith('http'):
        print('Image path:',image_path)
        pass
    elif is_base64_image(image_path):
        base64_image = image_path
        image_path = f"data:image/jpeg;base64,{base64_image}"
    else:
        base64_image = encode_image(image_path)
        print('<Enconding Image...>', type(base64_image))
        image_path = f"data:image/jpeg;base64,{base64_image}"
    return image_path



###### audio functions  #####
def play_audio(file_name):
    pygame.mixer.init()
    pygame.mixer.music.load(file_name)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.quit() # Close the file after music play ends

def audio_loop(audio_file="speech.mp3", repeat='alt' , exit='shift'):
    print('Press '+repeat+' to repeat aloud, '+exit+' to exit.')
    while True:
        if kb.is_pressed(repeat):
            play_audio(audio_file)
            #print('Press '+repeat+' to repeat aloud, '+exit+' to exit.')
        elif kb.is_pressed(exit):
            print('Chat Closed')
            break


def record_audio(duration=5, filename="recorded_audio.mp3"): # duration: in seconds
    print('start recording for',str(duration),'seconds')
    sample_rate = 44100
    channels = 2
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait() # wait until recording is finished
    print('recording ended')
    sf.write(filename, recording, sample_rate) #save audio file


def record_audio_press(filename='recorded_audio.wav',
                       channels=1,
                       rate=44100,
                       subtype='PCM_16',
                       stop= 'ctrl'):
    # start recording with the given sample rate and channels
    print("Recording... Press "+stop+" to stop")
    myrecording = sd.rec(int(rate * 10), samplerate=rate, channels=channels)
    while True:
        # If  'Key'  is pressed stop the recording and break the loop
        if kb.is_pressed(stop):
            print("Recording Stopped.")
            break

    sd.wait()  # wait until recording is finished
    sf.write(filename, myrecording, rate, subtype)


def loop_audio(start='alt', stop='ctrl',exit='shift', filename='recorded_audio.wav', printinfo=True):
    if printinfo: print("Press "+start+" to start recording, "+exit+" to exit")
    while True:
        # If 'Key' is pressed start the recording
        if kb.is_pressed(start):
            record_audio_press(filename, stop=stop)
            break
        elif kb.is_pressed(exit):
            break


def while_kb_press(start='alt',stop='ctrl'):
    while True:
        if kb.is_pressed(start):
            print("Press "+stop+" to stop")
            while True:
                if kb.is_pressed(stop):  # if key 'ctrl + c' is pressed
                    break  # finish the loop
                else:
                    print('while...')
                    time.sleep(2)
            print("Finished loop.")