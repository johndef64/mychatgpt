from mychatgpt import *
import re, ast
import time

base_agent = GPT()

def BooleanAgent(question, bias="", gpt=base_agent, print_=True):
    """

    :param gpt: class GPT() from main.py
    :param question:
    :return:
    """
    n= time.time()
    prompt="You are a Boolean agent. You answer to any question with a True or False."+bias
    gpt.ask(question, system=prompt, print_reply=False)
    # parse the reply
    boolean = True if "true".lower() in gpt.ask_reply.lower() else False

    if print_: print(f"output: {str(boolean)} runtime: {time.time() - n}")
    return boolean


def parse_code_blobs(code_blob: str) -> str:
    """Parses the LLM's output to get any code blob inside. Will return the code directly if it's code."""
    pattern = r"```(?:py|python)?\n(.*?)\n```"
    matches = re.findall(pattern, code_blob, re.DOTALL)
    if len(matches) == 0:
        try:  # Maybe the LLM outputted a code blob directly
            ast.parse(code_blob)
            return code_blob
        except SyntaxError:
            pass

        if "final" in code_blob and "answer" in code_blob:
            raise ValueError(
                f"""
Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
Here is your code snippet:
{code_blob}
It seems like you're trying to return the final answer, you can do it as follows:
Code:
```py
final_answer("YOUR FINAL ANSWER HERE")
```<end_code>""".strip()
            )
        raise ValueError(
            f"""
Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
Here is your code snippet:
{code_blob}
Make sure to include code with the correct pattern, for instance:
Thoughts: Your thoughts
Code:
```py
# Your python code here
```<end_code>""".strip()
        )
    return "\n\n".join(match.strip() for match in matches)

def CodeAgent(command, gpt=base_agent, format ='python', add=""):
    """

    :param gpt: class GPT() from main.py
    :param instruction:
    :return:
    """
    print("Generating code...")
    # Specify prompt to guide model to generate executable Python code
    # system = "\nYou are a Code agent. You provide executable Python code for any task or question."
    system = instructions['delamain'] + features['reply_style'][format]+f"\n{add}"

    # The agent asks for the task or question
    gpt.ask(command, system=system, print_reply=False)

    # Parses the reply to extract code
    generated_code = parse_code_blobs(gpt.ask_reply)

    # Executes the generated code
    # Using `exec()` to execute the string as Python code
    print("Executing code...")
    try:
        exec(generated_code)
    except Exception as e:
        print(f"Error in executing code: {str(e)}")

# CodeAgent(C, "calcola la funzione di Lorentz")


def AssignTags(query,
               TAGS=[],
               data="",
               model="gpt-4o",
               print_=False):
    instructions=f""" Act as an Expert Tagger from textual data into structured list format. 
    text classification or text categorization machine. You use a corpus of predefined tags and labels to properly label and tag unstructured input text of the user."""
    corpus= f"""This below is the corpus of tags you are trained to indentify in the text provided: {TAGS}"""
    task=f"""Use the textual input of the user and assign tags in list format relative to this Google Search query:  "{query}".
                 
        Reply only updating this form:
        ["tag", "tag", "tag","tag", "..."]
        """
    reply_example=""" """  # aggiungi qui un Data Dictionary come esempio
    # contents, categories, appearance, features, orientation, style, passions, activities, attitudes, practices, kinks

    extractor = GPT(model=model)
    extractor.clear_chat()
    extractor.expand_chat(instructions, "system")
    extractor.expand_chat(corpus, "system")
    extractor.expand_chat(task, "system")
    #extractor.expand_chat("This is the data dictionary you should follow:\n\n"+json_data_dict, "system")
    #print(data)
    extractor.c(data)

    ###############################

    # Print the resulting dictionary
    assigned_tags = extract_and_convert_to(extractor.chat_reply, "[]")

    while len(assigned_tags) == 0:
          C.c("@ correct the following python list sintax and return the corrected list:\n\n"+extractor.chat_reply)
          assigned_tags = extract_and_convert_to(C.chat_reply, "[]")
    ###############################
    return assigned_tags


#### Web Data Extractors ###
def GsearchAssigTags(query,
                     TAGS=[],
                     num_results = 20,
                     model="gpt-4o",
                     print_=False,):
    data=""
    links = google_search(query, num_results, advanced=True)
    for n in range(len(links)):
        data += links[n].url+"\n"+    links[n].title+"\n"+    links[n].description+"\n\n"
        if print_: print(links[n].url)
        if print_: print(links[n].title)
        if print_: print(links[n].description)
    print(f"len data:{len(data)}")

    assigned_tags = AssignTags(query, TAGS, data, model=model, print_=print_)

    return assigned_tags


# tags = gsearch_assig_tags("jojo bizzarre avneture", ["kitten", "cute", "animal", "spoon", "monster", "person", "Jojo", "stand power", "Jotaro"])
#
# tags
#%%


def GsearchExtractMetadata(query,
                           json_form = None,
                           data_dictionary = None,
                           num_results = 20,
                           model="gpt-4o",
                           tags=None,
                           print_=False,
                           ):
    data=""
    links = google_search(query, num_results, advanced=True)
    for n in range(len(links)):
        data += links[n].url+"\n"+    links[n].title+"\n"+    links[n].description+"\n\n"
        if print_: print(links[n].url)
        if print_: print(links[n].title)
        if print_: print(links[n].description)


    instructions=f""" Act as a Information Extractor from textual data into structured JSON format. Use the textual input of the user and extract relevant data in json format about " {query}.
         
    """
    if json_form:
        instructions = instructions+f"""
            Reply only filling this JSON form:
            {str(json_form)}
            """

    extractor = GPT(model=model)
    extractor.clear_chat()
    extractor.expand_chat(instructions, "system")
    if data_dictionary:
        extractor.expand_chat("This is the data dictionary you should follow:\n\n"+data_dictionary, "system")
    # print(data)
    extractor.c(data)

    ###############################

    # Print the resulting dictionary
    data_dict = extract_and_convert_to(extractor.chat_reply, enclosure="{}")

    if len(data_dict) == 0:
        C.c("@ correct the following python dict synthax and return the corrected dict:\n\n"+extractor.chat_reply)
        data_dict = extract_and_convert_to(C.chat_reply, enclosure="{}")
    ###############################

    if tags:
        assigned_tags = AssignTags(query, tags, data, model=model, print_=print_)
        data_dict["tags"] = assigned_tags

    return data_dict



###json
dummy_form =  {
    "personalInformation": {
        "name": "John Doe",
        "age": 30,
        "gender": "male"
    },
    "physicalCharacteristics": {
        "height": "180 cm",
        "weight": "75 kg",
        "eyeColor": "brown",
        "hairColor": "black"
    },
    "personalityTraits": {
        "introvert": True,
        "extrovert": False,
        "optimistic": True,
        "pessimistic": False
    },
    "occupation": {
        "jobTitle": "Software Engineer",
        "company": "Tech Corp",
        "yearsOfExperience": 5
    },
    "hobbies": [
        "reading",
        "traveling",
        "cooking"
    ],
    "socialMedia": {
        "twitter": "@johndoe",
        "linkedIn": "linkedin.com/in/johndoe"
    },
    "relationships": {
        "maritalStatus": "single",
        "friends": ["Jane Smith", "Robert Brown"],
        "family": {
            "parents": ["Mary Doe", "Michael Doe"],
            "siblings": ["Anna Doe"]
        }
    }
}

dummy_tags = [
    # News related tags
    "news", "journalism", "reporting", "headline", "article", "media", "broadcast", "coverage", 
    "interview", "exclusive", "investigation", "press", "update", "bulletin", "source", "editorial", 
    "column", "correspondent", "subscription", "newsfeed", "publisher", "agency", "periodical", "tabloid",

    # Politics related tags
    "politics", "election", "campaign", "democracy", "government", "senate", "congress", "cabinet", 
    "policy", "legislation", "bill", "diplomacy", "debate", "reform", "law", "rights", "constitution", 
    "justice", "lobby", "parliament", "ministry", "vote", "poll", "party", "representative", 

    # Animal related tags
    "animals", "wildlife", "zoology", "habitat", "species", "endangered", "biodiversity", "ecosystem", 
    "fauna", "flora", "mammal", "reptile", "amphibian", "avian", "insect", "aquatic", "conservation", 
    "sanctuary", "safari", "pet", "veterinary", "animalcare", "domestic", "exotic", "trainer",

    # Travel related tags
    "travel", "vacation", "holiday", "destination", "tourism", "journey", "adventure", "itinerary", 
    "landscape", "culture", "heritage", "exploration", "backpacking", "flight", "cruise", "resort", 
    "hiking", "sightseeing", "luxury", "escape", "beach", "island", "city", "guide", "backpacking",

    # Food related tags
    "food", "cuisine", "recipe", "cooking", "baking", "meal", "dessert", "flavor", "ingredient", 
    "kitchen", "restaurant", "dining", "gourmet", "appetizer", "maincourse", "snack", "beverage", 
    "breakfast", "lunch", "dinner", "brunch", "buffet", "barbecue", "vegetarian", "vegan",

    # General content tags
    "content", "blog", "post", "article", "creator", "writer", "editor", "publication", "story", 
    "narrative", "creative", "platform", "media", "communication", "network", "engagement", "audience", 
    "analytics", "trending", "feature", "multimedia", "information", "data", "digital", "strategy"
]

###
#
# data = gsearch_extract_metadata("Dua Lipa Singer", json_form=dummy_form)
#
# data
#%%





#%%
