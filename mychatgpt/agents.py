from mychatgpt import *

def BooleanAgent(gpt, question):
    prompt="You are a Boolean agent. You answer to any question with a True or False."
    gpt.ask(question, system=prompt, print_reply=False)
    # parse the reply
    boolean = True if "true".lower() in gpt.ask_reply.lower() else False
    return boolean


#### Web Data Extractors ###
def GsearchAssigTags(query,
                     TAGS=[],
                     print_=False,
                     num_results = 20):
    data=""
    links = google_search(query, num_results, advanced=True)
    for n in range(len(links)):
        data += links[n].url+"\n"+    links[n].title+"\n"+    links[n].description+"\n\n"
        if print_: print(links[n].url)
        if print_: print(links[n].title)
        if print_: print(links[n].description)
    print(f"len data:{len(data)}")


    instructions=f""" Act as an Expert Tagger from textual data into structured list format. 
    text classification or text categorization machine. You use a corpus of predefined tags and labels to properly label and tag unstructured input text of the user."""
    corpus= f"""This below is the corpus of tags you are trained to indentify in the text provided: {TAGS}"""
    task=f"""Use the textual input of the user and assign tags in list format relative to this Google Search query:  "{query}".
                 
        Reply only updating this form:
        ["tag", "tag", "tag","tag", "..."]
        """
    reply_example=""" """  # aggiungi qui un Data Dictionary come esempio
    # contents, categories, appearance, features, orientation, style, passions, activities, attitudes, practices, kinks

    extractor = GPT(model="gpt-4o")
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

    if len(assigned_tags) == 0:
        C.c("@ correct the following python list sintax and return the corrected list:\n\n"+extractor.chat_reply)
        assigned_tags = extract_and_convert_to(C.chat_reply, "[]")
    ###############################
    return assigned_tags


# tags = gsearch_assig_tags("jojo bizzarre avneture", ["kitten", "cute", "animal", "spoon", "monster", "person", "Jojo", "stand power", "Jotaro"])
#
# tags
#%%


def GsearchExtractMetadata(query,
                           json_form = None,
                           data_dictionary = None,
                           print_=False,
                           num_results = 20):
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

    extractor = GPT(model="gpt-4o")
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
###
###
#
# data = gsearch_extract_metadata("Dua Lipa Singer", json_form=dummy_form)
#
# data
#%%





#%%
