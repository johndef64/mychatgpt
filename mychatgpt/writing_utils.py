from collections import Counter
import re
import pandas as pd
import pyperclip as pc


def count_words(text):
    # Normalize text to lowercase and find all words
    words = re.findall(r'\w+', text.lower())
    # Count words using Counter, which is efficient for object counting
    word_count = Counter(words)
    return word_count
def count_words_sections(text):
    # This function takes a dictionary with section names as keys and text as values
    # It returns a dictionary with section names as keys and word count as values
    word_counts = {section: len(content.split()) for section, content in text.items()}
    return word_counts
def count_words_in_string(text):
    # Split by spaces to count words
    return len(text.split())

##########

# def extract_sections(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#     content = clean_tex(content)
#
#     # Use non-greedy matching to avoid issues with nested sections
#     section_pattern = r'\\section\{(.+?)\}(.*?)(?=\\section|\Z)'
#     sections = re.findall(section_pattern, content, re.DOTALL)
#
#     sections_dict = {}
#     for title, body in sections:
#         # Create a key by converting title to lowercase and replacing spaces with underscores
#         key = title.lower().replace(' ', '_')
#         sections_dict[key] = body.strip()
#     df = pd.Series(sections_dict, index=sections_dict.keys())
#     #display(df)
#     return sections_dict, content

def extract_sections(file_path):
    # Determine file type based on extension
    if file_path.endswith('.tex'):
        section_pattern = r'\\section\{(.+?)\}(.*?)(?=\\section|\Z)'  # LaTeX section pattern
    elif file_path.endswith('.md'):
        section_pattern = r'^(#{1,6})\s(.+)$'  # Markdown header pattern
    else:
        raise ValueError("Unsupported file type. Only '.tex' and '.md' are supported.")

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()  # Read file content
    content = clean_tex(content)

    # Find sections in the content based on the determined pattern
    sections = re.findall(section_pattern, content, re.DOTALL if file_path.endswith('.tex') else re.MULTILINE)

    sections_dict = {}
    for match in sections:
        if file_path.endswith('.tex'):
            title, body = match  # Unpack LaTeX sections
        else:
            header, title = match  # Unpack Markdown headers
            body = header  # Use header as body for Markdown

        key = title.lower().replace(' ', '_')  # Convert title to lowercase and replace spaces with underscores
        sections_dict[key] = body.strip()  # Store section body

    df = pd.Series(sections_dict, index=sections_dict.keys())  # Create a series from the dictionary
    # display(df)
    return sections_dict, content

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


def reload_paper(file_path):
    global sections_dict, full_paper
    sections_dict, full_paper = extract_sections(file_path)


def add_info_set(gpt, sections = None, sections_dict = {}, clear = True):
    if clear: gpt.clear_chat()

    if sections:
        for section in sections:
            section_clean = clean_tex(sections_dict[section])
            gpt.expand_chat('\nThis is the '+section+' section of my new paper:\n'+section_clean, 'user')




from mychatgpt import GPT
###
class Writers:
    """
    Writers class initialize a GPT agent with a latex file slitted
    into modular sections
    """
    def __init__(self,
                 gpt = GPT(),
                 tex_file = None,
                 context = None,
                 format = 'latex',
                 model = GPT().model):
        self.gpt = gpt
        self.model = model
        self.context = context
        self.tex_file = tex_file
        self.gpt.format = format

        if self.context:
            gpt.add_system(f"\n\nUnderstand the context: {self.context}")
        if tex_file:
            self.sections_dict, self.content = extract_sections(self.tex_file)
            display(pd.Series(self.sections_dict, index=self.sections_dict.keys()))

            self.sections = list(self.sections_dict.keys())

        self.table_template = table_template

    def reload_paper(self):
        self.sections_dict, self.content = extract_sections(self.tex_file)


    # def add_info_set(self, sections = None, clear = True):
    #     if clear: self.gpt.clear_chat()
    #
    #     if sections:
    #         for section in sections:
    #             section_clean = clean_tex(self.sections_dict[section])
    #             self.gpt.expand_chat('\nThis is the '+section+' section of my new paper:\n'+section_clean, 'user')



    ### request functions ###
    def enhancer(self, text = None,
                 instructions = None,
                 task = None):
        # automatic paste
        if not text:
            text = pc.paste()

        # default instruction
        if not instructions:
            instructions = (
                "The paper I have written needs to be revised. "
                "It must use current and correct terminology of the 'Informatics Engineering' domain. "
                "Methods should be described accurately, consistently, and precisely with the correct current computer science terminology. "
                "Make the text more fluent."
            )
            #if self.context:
            #    instructions += f"\n\nUnderstand the context: {self.context}"

        # default task
        if not task:
            task = f"""Apply this criteria to this section below (in english) more concise and clear:
            write better this section 
            
            """

        self.gpt.c(instructions+task+"["+text+"]", gpt=self.model)

    def table_maker(self,
                    template= None,
                    task= None,
                    data = None,
                    instructions = None
                    ):
        if not template:
            template = self.table_template
        if not instructions:
            instructions = ''
        if not task:
            task = """
            Using this table provided as a template, make a table out of this data:
            
            """
        # automatic paste
        if not data:
            data = pc.paste()

        self.gpt.expand_chat(template)
        self.gpt.c(instructions+task+"\n\nData:"+data, gpt=self.model)


    def ask_paper(self,
                  sections: list = None,
                  table_template = False,
                  task= None,
                  data = None,
                  instructions = None,
                  clear = True
                  ):
        if table_template:
            template = self.table_template
        elif isinstance(table_template, str):
            template = table_template

        if not instructions:
            instructions = ''
        if not task:
            task = ""
        if data:
            data = "\n\nData:"+data
        else:
            data = ''

        # add sections
        #self.add_info_set(sections, clear=clear)
        add_info_set(self.gpt, sections=sections, sections_dict=self.sections_dict, clear = False)

        # add template
        if table_template:
            self.gpt.expand_chat(template)

        # request
        self.gpt.c(f"{instructions}{task}{data}", gpt=self.model)




#### TEMPLATES ###
table_template = r"""template table :
\begin{table*}[htbp]
    \centering
    \caption{Data Sources for Interaction Maps}
    \label{tab:data_sources}
    \small
    %\resizebox{\textwidth}{!}{ 
    \begin{tabular}{ll}
        \toprule
        
        \textbf{Category} & \textbf{Data Sources} \\
        \hline
        Protein-Protein Interaction (PPI) & Biogrid, STRING \\
        Transcription Factors (TF) & TF2DNA, TRRUST v2 \\
        MicroRNA & mirdb, miRBase, miRnet \\
        RNA Binding Proteins (RNAbp) & RBP2GO \\
        Biological Process, Molecular Function & Gene Ontology (GO) \\
        Metabolomics & KEGG, Reactome, CHEBI \\
        Drugs & DrugBank \\
        \bottomrule
    \end{tabular}
    %}
\end{table*}
"""