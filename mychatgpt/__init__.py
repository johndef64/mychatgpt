# pygpt64/__init__.py

# Definisci la versione del pacchetto
__version__ = '0.4.6'
print(f'mychatgpt version {__version__}')
print('Loading package...')

# Importa i simboli necessari dai moduli interni
from .utils import *
from .main import *

#from .assistants import *

# Se desideri esporre specifiche classi o funzioni da altri moduli, puoi farlo qui.
# Ad esempio:
# from .module1 import Class1, function1
# from .module2 import Class2, function2

print('mychatgpt is ready.')
#%%
