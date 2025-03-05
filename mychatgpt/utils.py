from cryptography.fernet import Fernet
import matplotlib.pyplot as plt
from googlesearch import search
from bs4 import BeautifulSoup
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
import yaml
import ast
import os
import re
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import markdown


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
        pc.paste()
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

def find_files(filename):
    # Ottieni il percorso della directory corrente
    root_dir = os.getcwd()

    found_files = []
    # Cammina ricorsivamente nella directory corrente
    for root, dirs, files in os.walk(root_dir):
        if filename in files:
            # Se il file è trovato, aggiungi il percorso completo alla lista
            found_files.append(os.path.join(root, filename))

    if len(found_files) > 1:
        print("<WARNING: Multiple files>")
    return found_files

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

###### Manage Json #####
import json

def save_json_in_lib(dati, nome_file):
    # Usa __file__ per ottenere il percorso della directory del file corrente
    file_path = os.path.join(os.path.dirname(__file__), nome_file)
    with open(file_path, 'w') as file:
        json.dump(dati, file, indent=4)

def load_json_from_lib(nome_file, local = False):
    # Usa __file__ per ottenere il percorso della directory del file corrente
    if not local:
        file_path = os.path.join(os.path.dirname(__file__), nome_file)
    else:
        file_path = nome_file
    with open(file_path, 'r') as file:
        return json.load(file)

###### API KEYS Management ######
openai_api_hash = b'gAAAAABnQFa7PhJzvEZmrHIbqIbXY67FYM0IhBaw8XOgnDurF5ij1oFYvNMikCpe8ebpqlRYYYOEDGuxuWdOkGPO74ljkWO07DVGCqW7KlzT6AJ0yv-0-5qTNeXTVzhorMP4RA5D8H2P73cmgwFr2Hlv6askLQjWGg=='

def pass_api_keys():
    #if simple_bool('Do you have an openai key? '):
    openai_api_key = input('Provide here your OpenAI api key, if not leave blank:')
    if openai_api_key == "":
        print('\nPlease, get your API-key at https://platform.openai.com/api-keys')
        openai_api_key = "missing"

    deepseek_api_key = input('Provide here your DeepSeek api key, if not leave blank:')
    if deepseek_api_key == "":
        print('\nPlease, get your DeepSeek API-key')
        deepseek_api_key = "missing"


    x_api_key = input('Provide here your Grok api key, if not leave blank:')
    if x_api_key == "":
        print('\nPlease, get your DeepSeek API-key')
        x_api_key = "missing"

    #if simple_bool('Do you have an openai key? '):
    gemini_api_key = input('Provide here your Gemini api key, if not leave blank:')
    if gemini_api_key == "":
        print('\nPlease, get your Gemini API-key')
        gemini_api_key = "missing"

    api_keys = {
        "openai": openai_api_key,
        "deepseek": deepseek_api_key,
        "grok": x_api_key,
        "gemini": gemini_api_key,
    }
    save_json_in_lib(api_keys, "api_keys.json")


def load_api_keys():
    # if not api_keys.json in cwd, save it in pkg dir
    if not os.path.exists("api_keys.json"):
        file_path = os.path.join(os.path.dirname(__file__), "api_keys.json")
        if os.path.exists(file_path):
            # load api keys from pkg
            api_keys = load_json_from_lib("api_keys.json")
        else:
            api_keys = {}
    else:
        # if api_keys.json in cwd, take them from here
        api_keys = load_json_from_lib("api_keys.json", local=True)

    return api_keys

def save_api_keys():
    file_path = os.path.join(os.path.dirname(__file__), "api_keys.json")
    if not os.path.exists(file_path):
        pass_api_keys()


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

def get_raw(file_path, separator="\n", print_=True):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        content.replace("\n\n","\n")

    my_data = [line.strip() for line in content.split(separator)]
    if print_:
        df = pd.Series(my_data)
        display(df)
    return my_data

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

# def concat_paragraphs_(soup, separator= "\n\n"):
#     # Trova tutti i tag <p> nel documento
#     paragraphs = soup.find_all('p')
#
#     for i in range(len(paragraphs) - 1):
#         # Controlla se il prossimo elemento è anch'esso un paragrafo
#         if paragraphs[i].find_next_sibling() == paragraphs[i + 1]:
#             # Concatena il testo del paragrafo successivo a quello attuale
#             paragraphs[i].string = paragraphs[i].text + separator + paragraphs[i + 1].text
#             # Rimuove il paragrafo successivo
#             paragraphs[i + 1].decompose()
#     return soup

def concat_paragraphs(soup, SEP ="\n\n"):
    # Find all headers and paragraphs
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    all_paragraphs = soup.find_all('p')

    # Container for concatenated content
    concatenated_paragraphs = []

    # Track the current header and paragraphs to be concatenated
    current_header = None
    current_paragraph_group = []

    for element in soup.descendants:
        if element in headers:
            # If there is an existing header and paragraphs
            if current_header and current_paragraph_group:
                # Concatenate all current paragraphs with line breaks
                concatenated_text = SEP.join(p.get_text() for p in current_paragraph_group)
                new_paragraph = soup.new_tag('p')
                new_paragraph.string = concatenated_text

                # Replace the first paragraph with the concatenated one
                current_paragraph_group[0].replace_with(new_paragraph)

                # Remove the other paragraphs
                for p in current_paragraph_group[1:]:
                    p.extract()

            # Reset for next group
            current_header = element
            current_paragraph_group = []

        elif element in all_paragraphs and current_header:
            # Collect paragraphs between headers
            current_paragraph_group.append(element)

    # Handle the last set of paragraphs
    if current_header and current_paragraph_group:
        concatenated_text = SEP.join(p.get_text() for p in current_paragraph_group)
        new_paragraph = soup.new_tag('p')
        new_paragraph.string = concatenated_text

        current_paragraph_group[0].replace_with(new_paragraph)
        for p in current_paragraph_group[1:]:
            p.extract()

    return soup


def markdown_to_dict(md_content, html=None):
    """
    Based on HTML BS4
    """
    if not html:
        html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")
    soup = concat_paragraphs(soup)

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
                current_h2 = None
            elif current_h1:  # Fall-back su current_h1 se current_h2 non esiste
                if '' not in result_dict[current_h1]:
                    result_dict[current_h1]['p'] = ''
                result_dict[current_h1]['p'] += element.get_text() + '\n\n'

    # Trim trailing newlines from text
    for h1 in result_dict:
        for h2 in result_dict[h1]:
            result_dict[h1][h2] = result_dict[h1][h2].rstrip('\n')

    return result_dict


def get_md(file_path=None, print_=True, indent=0):
    def print_dict_structure(d, indent=0):
        # iterating through dictionary keys
        for key in d:
            # printing the key with indent
            print('    ' * indent + '- ' + str(key))
            # if the value is a dictionary, recursively print its structure
            if isinstance(d[key], dict):
                print_dict_structure(d[key], indent+1)
        # print("\n")

    with open(file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
        md_content = '\n'.join(line for line in md_content.splitlines() if not line.startswith('[//]'))

    d = markdown_to_dict(md_content)
    if print_:
        print_dict_structure(d, indent=indent)
    return d


def yaml2dict(yaml_string):
    dati = yaml.safe_load(yaml_string)
    return dati

# Function for importing YAML data into a Python dictionary
def get_yaml(file_yaml, encoding='utf-8'):
    # Open the YAML file
    with open(file_yaml, 'r', encoding=encoding) as file:
        # Read file content
        raw_content = file.read()
        # Clean the YAML content
        cleaned_content = re.sub(r'/', r'_slash_', raw_content)
        cleaned_content = re.sub(r'\\', r'_slash2_', cleaned_content)
        # Load and return the file content as dictionary
        data = yaml.safe_load(cleaned_content)

    # Convert '_slash_' back to '/' in each element
    def replace_slash(element):
        # If the element is a string, replace '_slash_' with '/'
        if isinstance(element, str):
            return element.replace('_slash_', '/').replace('_slash2_', '\\')
        # If the element is a dictionary, apply the function to each key-value pair
        elif isinstance(element, dict):
            return {k: replace_slash(v) for k, v in element.items()}
        # If the element is a list, apply the function to each item
        elif isinstance(element, list):
            return [replace_slash(item) for item in element]
        # Return the element directly if it doesn't match any case above
        else:
            return element

    # Apply replacement function to the entire data structure
    data = replace_slash(data)

    return data


# extract sections from latex and markdown files
def get_dict_from_text(file_path, clean_text=True):
    """
    Regex Based
    """
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



#%%
###  Web Scraper ###



def google_search(search_string="cute kittens",
                  num_results=100,
                  lang='en',
                  region="us",
                  sleep_interval = 0,
                  advanced=False,
                  safe=False):

    # country codes : https://developers.google.com/custom-search/docs/json_api_reference?hl=it#countryCodes
    if num_results > 100:
        sleep_interval=5
    results = search(search_string, num_results, lang=lang, region=region, safe=safe, sleep_interval=sleep_interval,  advanced=advanced)
    links = []
    for link in results:
        links.append(link)

    print(f"Got {len(links)} results")

    # Advanced
    # Returns a list of SearchResult
    # Properties:
    # - title
    # - url
    # - description

    return links



def simple_text_scraper(url ,sep ='\n'):
    """
    Scrapes all text from a webpage.

    Args:
        url (str): The URL of the webpage to scrape.

    Returns:
        str: A string containing all the text from the webpage.
             Returns None if an error occurs.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract all text from the parsed HTML
        text = soup.get_text(separator=sep, strip=True)

        return text

    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def extract_and_convert_to(text, enclosure = "[]"):
    # Find the first occurrence of '{' and the last occurrence of '}'
    start = text.find(enclosure.split()[0][0])
    end = text.rfind(enclosure.split()[0][1])

    # Extract the substring between the braces
    substr = text[start:end+1]

    # Convert the substring into a dictionary
    try:
        result_value = ast.literal_eval(substr)
    except (SyntaxError, ValueError):
        result_value = ast.literal_eval(enclosure)

    return result_value