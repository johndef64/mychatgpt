import os
import time
import threading
import tkinter as tk
import queue
import json
import re
from tkinter import scrolledtext
from PIL import ImageGrab
import pytesseract
from groq import Groq
from langdetect import detect, LangDetectException

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

MIN_CHARS_TO_TRANSLATE = 5
gui_queue = queue.Queue()
ocr_region = None 

# --- SELETTORE AREA ---
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
        self.canvas.create_text(
            self.root.winfo_screenwidth()//2,
            self.root.winfo_screenheight()//2,
            text="SELEZIONA L'AREA DA MONITORARE (Tesseract OCR)\n(Disegna un rettangolo col mouse)",
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

# --- LOGICA TRADUZIONE ---

translation_memory = []
last_translation_time = 0
MEMORY_TIMEOUT = 30.0

def translate_text(text):
    global translation_memory, last_translation_time
    
    if not text:
        return None

    current_time = time.time()
    if current_time - last_translation_time > MEMORY_TIMEOUT:
        translation_memory = [] # Reset memoria

    system_prompt = (
        "Sei un sistema di traduzione professionale EN->IT in tempo reale.\n"
        "REGOLE:\n"
        "1. Traduci SOLO il testo fornito dall'utente in Italiano.\n"
        "2. NON aggiungere commenti.\n"
        "3. Mantieni lo stile originale."
    )

    if translation_memory:
        context_str = "\n".join(translation_memory[-3:])
        system_prompt += f"\n\n### CONTESTO RECENTE (Solo riferimento):\n{context_str}\n### FINE CONTESTO"

    try:
        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model=MODEL_ID,
            temperature=0.3,
        )
        translated_text = completion.choices[0].message.content
        
        if translated_text:
            translation_memory.append(translated_text)
            last_translation_time = current_time
            return translated_text
            
    except Exception as e:
        return f"Errore API: {e}"
    return None

def is_valid_text(text):
    """Verifica se il testo è valido (non rumore)."""
    if not text or len(text.strip()) < MIN_CHARS_TO_TRANSLATE:
        return False
    
    # Filtro base caratteri
    if not any(c.isalpha() for c in text):
        return False
        
    try:
        # Langdetect è ottimo per scartare simboli casuali "!!_@"
        lang = detect(text)
        return True
    except LangDetectException:
        return False
    except Exception:
        return False

# --- WORKER OCR DIRETTO (NO CLIPBOARD) ---

def ocr_worker_direct():
    """
    Legge lo schermo -> Tesseract -> Traduce direttamente.
    Senza passare per la clipboard.
    """
    last_processed_text = ""
    print(f"OCR Direct Worker avviato su regione: {ocr_region}")
    
    while True:
        try:
            if ocr_region:
                # 1. Cattura immagine
                img = ImageGrab.grab(bbox=ocr_region)
                
                # 2. Tesseract OCR (Veloce)
                # 'eng' o 'ita+eng' a seconda dei casi
                raw_text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
                clean_text = raw_text.strip()
                
                # 3. Controllo Validità e Novità
                if clean_text != last_processed_text:
                    if is_valid_text(clean_text):
                        print(f"[OCR] Nuovo testo rilevato: {clean_text[:20]}...")
                        
                        last_processed_text = clean_text # Aggiorna subito
                        
                        # 4. TRADUZIONE DIRETTA (No Clipboard)
                        translated = translate_text(clean_text)
                        
                        if translated:
                            gui_queue.put(translated)
                    else:
                        # Se è rumore ma diverso dal precedente, potremmo voler aggiornare 
                        # comunque last_processed_text per evitare di riprocessare lo stesso rumore?
                        # Spesso è meglio di no, ma dipende dal tipo di rumore.
                        pass
                        
            time.sleep(1.0) # Polling ogni secondo
            
        except Exception as e:
            print(f"Errore OCR Loop: {e}")
            time.sleep(1.0)

# --- GUI ---

class ResizableWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Groq Translate (Tesseract Direct)")
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.attributes("-alpha", 0.90)
        self.configure(bg="#2b2b2b")
        self.geometry("500x200+100+100")
        self._drag_data = {"x": 0, "y": 0}
        self.setup_ui()
        self.setup_bindings()
        self.after(100, self.process_queue)

    def setup_ui(self):
        self.text_area = tk.Text(
            self, wrap=tk.WORD, font=("Segoe UI", 12, "bold"),
            bg="#2b2b2b", fg="#e0e0e0", relief="flat",
            padx=15, pady=15, highlightthickness=0, bd=0, cursor="arrow"
        )
        self.text_area.pack(fill="both", expand=True)
        self.grip = tk.Label(self, text="◢", bg="#2b2b2b", fg="#555555", cursor="size_nw_se")
        self.grip.place(relx=1.0, rely=1.0, anchor="se")
        
        self.context_menu = tk.Menu(self, tearoff=0, bg="#2b2b2b", fg="white")
        self.context_menu.add_command(label="Reimposta Area", command=self.reset_ocr_selection)
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Chiudi", command=lambda: os._exit(0))

    def setup_bindings(self):
        self.text_area.bind("<Control-Button-1>", self.start_move)
        self.text_area.bind("<B1-Motion>", self.do_move)
        self.text_area.bind("<Button-3>", self.show_context_menu)
        self.grip.bind("<Button-1>", self.start_resize)
        self.grip.bind("<B1-Motion>", self.do_resize)

    def show_context_menu(self, event):
        self.context_menu.post(event.x_root, event.y_root)

    def start_move(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def do_move(self, event):
        dx = event.x - self._drag_data["x"]
        dy = event.y - self._drag_data["y"]
        self.geometry(f"+{self.winfo_x() + dx}+{self.winfo_y() + dy}")

    def start_resize(self, event):
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def do_resize(self, event):
        new_w = self.winfo_pointerx() - self.winfo_rootx()
        new_h = self.winfo_pointery() - self.winfo_rooty()
        if new_w > 100 and new_h > 50:
            self.geometry(f"{new_w}x{new_h}")
            
    def reset_ocr_selection(self):
        self.withdraw()
        sel = AreaSelector().get_selection()
        if sel:
            global ocr_region
            ocr_region = sel
        self.deiconify()

    def show_window_with_text(self, text):
        self.text_area.configure(state='normal')
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, text)
        self.text_area.configure(state='disabled')
        self.deiconify()

    def process_queue(self):
        try:
            msg = gui_queue.get_nowait()
            self.show_window_with_text(msg)
        except queue.Empty:
            pass
        self.after(200, self.process_queue)

def main():
    global ocr_region
    print("--- Groq Translate (Fast Tesseract Direct) ---")
    
    # 1. Selezione Area
    selector = AreaSelector()
    ocr_region = selector.get_selection()
    
    if not ocr_region:
        print("Nessuna area selezionata.")
        return

    # 2. Avvia Worker OCR (che fa anche da traduttore)
    t = threading.Thread(target=ocr_worker_direct, daemon=True)
    t.start()

    # 3. GUI
    app = ResizableWindow()
    app.mainloop()

if __name__ == "__main__":
    main()
