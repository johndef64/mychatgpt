from setuptools import setup, find_packages


setup(
    name='pychatgpt',
    version='0.1',
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[         # Optional: specify dependencies
        # 'dependency1',
        # 'dependency2',
    ],
    #entry_points={             # Optional: command-line scripts
    #    'console_scripts': [
    #        'your-command=your_package.module:function',
    #    ],
    #},
    author='JohnDef64',
    author_email='youremail@example.com',
    description='A short description of the package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/johndef64/pychatgpt',
    classifiers=[  # Optional: supply classifiers for search indexing
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify required Python versions
)