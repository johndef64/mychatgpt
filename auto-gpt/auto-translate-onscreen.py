# pip install groq pyperclip pynput pytesseract Pillow
# Su Linux ricorda: sudo apt install xclip tesseract-ocr

import os
import time
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
import pyperclip
from groq import Groq
from PIL import ImageGrab
import pytesseract
import json
import queue

# --- CONFIGURAZIONE ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# API KEY
try:
    with open("api_keys.json", "r") as f:
        config = json.load(f)
    os.environ["GROQ_API_KEY"] = config["groq"]
except Exception:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct" 

# Configura Tesseract se necessario (Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MIN_CHARS_TO_TRANSLATE = 30
gui_queue = queue.Queue()
ocr_region = None # Sarà settata all'avvio

# --- SELETTORE AREA INIZIALE ---
class AreaSelector:
    def __init__(self):
        self.selection = None
        self.root = tk.Tk()
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-fullscreen', True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg='black')
        self.root.config(cursor="cross")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # Istruzioni a schermo
        self.canvas.create_text(
            self.root.winfo_screenwidth()//2, 
            self.root.winfo_screenheight()//2, 
            text="SELEZIONA L'AREA DA MONITORARE PER OCR\n(Disegna un rettangolo col mouse)", 
            fill="white", font=("Arial", 20, "bold")
        )

        self.start_x = self.start_y = 0
        self.rect = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def on_press(self, e):
        self.start_x, self.start_y = e.x, e.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='#00ff00', width=2)

    def on_drag(self, e):
        self.canvas.coords(self.rect, self.start_x, self.start_y, e.x, e.y)

    def on_release(self, e):
        x1, y1 = min(self.start_x, e.x), min(self.start_y, e.y)
        x2, y2 = max(self.start_x, e.x), max(self.start_y, e.y)
        if x2 - x1 > 10 and y2 - y1 > 10:
            self.selection = (x1, y1, x2, y2)
        self.root.destroy()

    def get_selection(self):
        self.root.mainloop()
        return self.selection

# --- LOGICA DI BACKGROUND ---
def translate_text(text):
    if not text or len(text) < MIN_CHARS_TO_TRANSLATE:
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Sei un traduttore esperto verso l'italiano. Solo output tradotto."},
                {"role": "user", "content": text}
            ],
            model=MODEL_ID,
            temperature=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Errore API: {e}"

def ocr_worker():
    """Monitora l'area selezionata e aggiorna la clipboard se trova nuovo testo."""
    last_ocr_text = ""
    print(f"OCR attivo su regione: {ocr_region}")
    
    while True:
        try:
            img = ImageGrab.grab(bbox=ocr_region)
            # Ottimizzazione: OCR veloce
            text = pytesseract.image_to_string(img, lang='ita+eng', config='--psm 6')
            clean_text = text.strip()
            
            if clean_text and clean_text != last_ocr_text:
                # Filtraggio rumore (almeno 5 char e lettere)
                if len(clean_text) > 5 and any(c.isalpha() for c in clean_text):
                    print(f"OCR Rilevato: {clean_text[:20]}...")
                    last_ocr_text = clean_text
                    pyperclip.copy(clean_text) # Questo attiverà il clipboard_monitor
            
        except Exception as e:
            print(f"Errore OCR Loop: {e}")
        
        time.sleep(1.0) # Check ogni secondo

def clipboard_monitor():
    """Ascolta cambiamenti clipboard e traduce."""
    last_clip = pyperclip.paste()
    while True:
        time.sleep(0.5)
        try:
            curr = pyperclip.paste()
            if curr != last_clip:
                last_clip = curr
                if len(curr) >= MIN_CHARS_TO_TRANSLATE:
                    print("Testo in clipboard per traduzione...")
                    trans = translate_text(curr)
                    if trans:
                        gui_queue.put(trans)
        except Exception:
            pass

# --- GUI PRINCIPALE ---
class ResizableWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Groq Translate")
        self.overrideredirect(True) # No bordi OS
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.90)
        self.configure(bg="#2b2b2b")
        
        # Geometria Iniziale
        self.geometry("500x300+100+100")
        
        # Variabili resize
        self._drag_data = {"x": 0, "y": 0}
        
        self.setup_ui()
        self.setup_bindings()
        
        self.after(100, self.process_queue)

    # def setup_ui_deprecated(self):
    #     # 1. Barra Titolo Custom (per muovere)
    #     self.title_bar = tk.Frame(self, bg="#1f1f1f", height=30)
    #     self.title_bar.pack(fill="x", side="top")
    #     self.title_bar.pack_propagate(False) # Ferma resize automatico
        
    #     lbl = tk.Label(self.title_bar, text="Groq Auto-Translate", bg="#1f1f1f", fg="#aaaaaa", font=("Segoe UI", 9))
    #     lbl.pack(side="left", padx=10)
        
    #     btn_x = tk.Button(self.title_bar, text="✕", command=self.hide_window, 
    #                      bg="#1f1f1f", fg="#ff5555", bd=0, width=4)
    #     btn_x.pack(side="right", fill="y")

    #     # 2. Area Testo
    #     # self.text_area = scrolledtext.ScrolledText(
    #     #     self, wrap=tk.WORD, font=("Segoe UI", 11), 
    #     #     bg="#2b2b2b", fg="#e0e0e0", relief="flat",
    #     #     padx=10, pady=10
    #     # )
    #     # 2. Area Testo (Sostituisci ScrolledText con Text standard)
    #     # Nota: Usiamo 'Text' invece di 'scrolledtext.ScrolledText'
    #     self.text_area = tk.Text(
    #         self, 
    #         wrap=tk.WORD, 
    #         font=("Segoe UI", 11), 
    #         bg="#2b2b2b", 
    #         fg="#e0e0e0", 
    #         relief="flat",
    #         padx=10, 
    #         pady=10,
    #         highlightthickness=0, # Rimuove il bordo di focus
    #         bd=0 # Rimuove bordi 3D
    #     )
    #     self.text_area.pack(fill="both", expand=True, padx=2, pady=2)
        
    #     # 3. Grip per Ridimensionare (Angolo basso-destra)
    #     self.grip = tk.Label(self, text="◢", bg="#2b2b2b", fg="#555555", cursor="size_nw_se")
    #     self.grip.place(relx=1.0, rely=1.0, anchor="se")

    # def setup_bindings_deprecated(self):
    #     # Muovere finestra
    #     self.title_bar.bind("<ButtonPress-1>", self.start_move)
    #     self.title_bar.bind("<B1-Motion>", self.do_move)
        
    #     # Ridimensionare finestra
    #     self.grip.bind("<ButtonPress-1>", self.start_resize)
    #     self.grip.bind("<B1-Motion>", self.do_resize)
        
    #     # Hotkeys
    #     self.bind("<Escape>", lambda e: self.hide_window())

    def setup_ui(self):
        # 1. Rimuoviamo completamente la self.title_bar
        
        # 2. Area Testo (Pura, senza scrollbar visibile)
        self.text_area = tk.Text(
            self, 
            wrap=tk.WORD, 
            font=("Segoe UI", 11), 
            bg="#2b2b2b", 
            fg="#e0e0e0", 
            relief="flat",
            padx=15, pady=15, 
            highlightthickness=0, 
            bd=0,
            cursor="arrow"
        )
        self.text_area.pack(fill="both", expand=True)
        
        # 3. Grip per Ridimensionare (Angolo basso-destra)
        self.grip = tk.Label(self, text="◢", bg="#2b2b2b", fg="#555555", cursor="size_nw_se")
        self.grip.place(relx=1.0, rely=1.0, anchor="se")

        # 4. Menu Contestuale (Tasto Destro per sopperire alla mancanza della X)
        self.context_menu = tk.Menu(self, tearoff=0, bg="#2b2b2b", fg="white")
        self.context_menu.add_command(label="Nascondi (Esc)", command=self.hide_window)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Chiudi App", command=lambda: os._exit(0))

    def setup_bindings(self):
        # SPOSTAMENTO: Usa CTRL + Click Sinistro per trascinare la finestra
        # (Usiamo Ctrl per non rompere la selezione del testo normale)
        self.text_area.bind("<Control-Button-1>", self.start_move)
        self.text_area.bind("<Control-B1-Motion>", self.do_move)

        # MENU: Tasto Destro
        self.text_area.bind("<Button-3>", self.show_context_menu)
        
        # RIDIMENSIONAMENTO
        self.grip.bind("<ButtonPress-1>", self.start_resize)
        self.grip.bind("<B1-Motion>", self.do_resize)
        
        # HOTKEYS
        self.bind("<Escape>", lambda e: self.hide_window())

    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)


    def start_move(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def do_move(self, event):
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        x = self.winfo_x() + dx
        y = self.winfo_y() + dy
        self.geometry(f"+{x}+{y}")

    def start_resize(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def do_resize(self, event):
        # Calcola nuova larghezza/altezza basata sul movimento del mouse relativo al grip
        # Il grip è ancorato a bottom-right, quindi event.x/y sono relativi al grip
        # Ma per il resize serve calcolare la differenza globale o usare coordinate root
        
        # Approccio semplificato: usiamo pointerx/y
        x1 = self.winfo_pointerx()
        y1 = self.winfo_pointery()
        x0 = self.winfo_rootx()
        y0 = self.winfo_rooty()
        
        new_w = x1 - x0
        new_h = y1 - y0
        
        if new_w > 100 and new_h > 50: # Limiti minimi
            self.geometry(f"{new_w}x{new_h}")

    def hide_window(self):
        self.withdraw()

    def show_window_with_text(self, text):
        self.text_area.configure(state='normal')
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, text)
        self.text_area.configure(state='disabled')
        
        self.deiconify()
        self.attributes("-topmost", True)

    def process_queue(self):
        try:
            msg = gui_queue.get_nowait()
            self.show_window_with_text(msg)
        except queue.Empty:
            pass
        self.after(200, self.process_queue)

def main():
    global ocr_region

    # prionts sull utlizzo dell app
    print("Auto-GPT Groq Translate On-Screen")
    print("1. Seleziona l'area dello schermo da monitorare per OCR.")
    print("2. Quando viene rilevato testo, verrà tradotto e mostrato in una finestra ridimensionabile.")
    print("3. Usa CTRL + Click Sinistro per spostare la finestra.")
    print("4. Usa il tasto destro per aprire il menu (Nascondi/Chiudi).")
    print("5. Premi ESC per nascondere la finestra.")
    
    # 1. Selezione Area (Bloccante all'inizio)
    print("Seleziona l'area da monitorare...")
    selector = AreaSelector()
    ocr_region = selector.get_selection()
    
    if not ocr_region:
        print("Nessuna area selezionata. Esco.")
        return

    print("Area memorizzata. Avvio threads...")

    # 2. Avvia Thread OCR e Clipboard
    threading.Thread(target=ocr_worker, daemon=True).start()
    threading.Thread(target=clipboard_monitor, daemon=True).start()

    # 3. Avvia GUI
    app = ResizableWindow()
    app.mainloop()

if __name__ == "__main__":
    main()
