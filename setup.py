from setuptools import setup, find_packages


setup(
    name='mychatgpt',
    version='0.4.5',
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[         # Optional: specify dependencies
        "openai", "tiktoken", "langdetect", "pandas", "pyperclip", "gdown", "pyyaml","scipy", "nltk", "PyPDF2", 'cryptography', 'matplotlib', "pygame", "sounddevice", "soundfile", "keyboard", "markdown", "pillow", "ollama", "beautifulsoup4", "googlesearch-python"
    ],
    #entry_points={             # Optional: command-line scripts
    #    'console_scripts': [
    #        'your-command=your_package.module:function',
    #    ],
    #},
    author='JohnDef64',
    #author_email='youremail@example.com',
    description="""mychatgpt is a small and useful Python module that provides functions for interacting with OpenAI's GPT models to create conversational agents. This module allows users to have interactive conversations with the GPT models and keeps track of the conversation history in your Python Projects and Jupyter Notebooks.""",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/johndef64/mychatgpt',
    classifiers=[  # Optional: supply classifiers for search indexing
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify required Python versions
)