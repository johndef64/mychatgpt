"""
Docstring for auto-gpt.auto-ocr

questo è uno scrip cli python che fa l'auto OCR del testo presente in una box sullo schermo selzionata con il mouse una sola volta. Fa l'OCR del testo in automatico ogni secondo e lo manda nella clipboard

"""

# sudo apt-get install tesseract-ocr tesseract-ocr-ita xclip

# pip install pytesseract pyperclip Pillow

# Scarica e installa Tesseract da UB-Mannheim/tesseract. 
# https://github.com/UB-Mannheim/tesseract/wiki
# https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe

# Aggiungi il percorso di installazione (es. C:\Program Files\Tesseract-OCR) alle variabili d'ambiente PATH, oppure configuralo nello script.

#  C:\Program Files\Tesseract-OCR


"""
Docstring for auto-gpt.auto-ocr

CLI python che esegue l'auto OCR di una regione dello schermo.
- L'utente seleziona l'area una sola volta col mouse.
- Lo script cattura quell'area ogni secondo.
- Estrae il testo e lo copia nella clipboard se è cambiato.
"""

import time
import tkinter as tk
from PIL import ImageGrab
import pytesseract
import pyperclip
import threading
import sys
import os

# --- CONFIGURAZIONE ---
# Se sei su Windows e tesseract non è nel PATH, decommenta e adatta:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ScreenRegionSelector:
    """Classe per gestire la selezione visuale dell'area con il mouse."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-alpha', 0.3) # Semitrasparente
        self.root.attributes('-fullscreen', True)
        self.root.attributes("-topmost", True)
        self.root.configure(background='grey')
        
        self.canvas = tk.Canvas(self.root, cursor="cross", bg="grey")
        self.canvas.pack(fill="both", expand=True)

        self.start_x = None
        self.start_y = None
        self.rect = None
        self.selection = None # (x1, y1, x2, y2)

        # Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        # Crea rettangolo vuoto
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, 1, 1, outline='red', width=2)

    def on_move_press(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        # Salva coordinate finali normalizzate (per gestire selezioni dal basso in alto)
        x1 = min(self.start_x, event.x)
        y1 = min(self.start_y, event.y)
        x2 = max(self.start_x, event.x)
        y2 = max(self.start_y, event.y)
        
        self.selection = (x1, y1, x2, y2)
        self.root.destroy()

    def get_selection(self):
        self.root.mainloop()
        return self.selection

def ocr_loop(region):
    """Loop infinito che esegue OCR sulla regione specificata."""
    print(f"Monitoraggio attivo sulla regione: {region}")
    print("Premi Ctrl+C nel terminale per fermare.")
    
    last_text = ""
    
    try:
        while True:
            # 1. Cattura screenshot della regione
            # bbox = (left, top, right, bottom)
            screenshot = ImageGrab.grab(bbox=region)
            
            # 2. Esegui OCR (lang='ita+eng' per supportare italiano e inglese)
            # --psm 6 assume che sia un blocco di testo uniforme
            text = pytesseract.image_to_string(screenshot, lang='ita+eng', config='--psm 6')
            
            clean_text = text.strip()
            
            # 3. Se il testo è valido e diverso dall'ultimo, copia
            if clean_text and clean_text != last_text:
                pyperclip.copy(clean_text)
                print("-" * 20)
                print(f"OCR Catturato:\n{clean_text}")
                last_text = clean_text
            
            time.sleep(1.0) # Attendi 1 secondo
            
    except KeyboardInterrupt:
        print("\nMonitoraggio fermato dall'utente.")
        sys.exit()

def main():
    print("Avvio selezione area... (Lo schermo diventerà grigio)")
    print("Disegna un rettangolo con il mouse sull'area da monitorare.")
    time.sleep(1) # Un attimo per leggere il messaggio
    
    selector = ScreenRegionSelector()
    region = selector.get_selection()
    
    if region:
        # Verifica che l'area non sia nulla (clic a vuoto)
        if region[2] - region[0] < 10 or region[3] - region[1] < 10:
            print("Area troppo piccola o non selezionata. Riprova.")
            return

        # Avvia il loop OCR
        ocr_loop(region)
    else:
        print("Selezione annullata.")

if __name__ == "__main__":
    main()
