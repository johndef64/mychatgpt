"""
Docstring for auto-transalte-onscreen-vlm

il tuo compito p riscriver lapp auto-traslate-onscreen:

Tessercat va usato solo per rilevare se nell immagine c'è del testo VALIDO per poter passare oltre al vero OCR . Il Vero OCR è bsato su vision language models e api di Groq come implementato in ocr_reader.py

il tuo compito è quinid implentare il nuovo ocr  nell app ed usare tersact e validate text solo per il passaggio al vero OCR.

Se il testo esaratto dal vero ocr è troppo simile al testo dell immaigne precedente, questo non deve passa al Traduttore
---------------------------------------------------

The main changes I made are:

Integrated the VLM/Groq OCR logic directly into the ocr_worker thread.

Downgraded Tesseract to a simple "trigger": it now only checks if something text-like exists to justify calling the expensive/slower VLM API.

Added the similarity check using difflib.SequenceMatcher. If the text extracted by the VLM is >85% similar to the previous one, it is ignored.

Removed the Clipboard dependency: The app now passes text directly from the OCR worker to the Translator, avoiding conflicts with your system clipboard.

You can save this code as auto-translate-vlm.py.

"""


import os
import time
import threading
import tkinter as tk
import queue
import json
import base64
import io
import difflib
from tkinter import scrolledtext, ttk
from PIL import ImageGrab
import pytesseract
from groq import Groq, APIConnectionError, RateLimitError, APIStatusError


# --- CONFIGURAZIONE ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Caricamento API KEY
try:
    with open("api_keys.json", "r") as f:
        config = json.load(f)
        os.environ["GROQ_API_KEY"] = config["groq"]
except Exception:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Modelli
MODEL_TRANSLATION = "meta-llama/llama-4-maverick-17b-128e-instruct"
MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"  # Modello VLM da ocr_reader.py

# Config Tesseract (se necessario su Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MIN_CHARS_TO_TRIGGER = 5
SIMILARITY_THRESHOLD = 0.85  # Se il testo è simile all'85%, lo ignora

gui_queue = queue.Queue()
ocr_region = None 

# --- HELPER VLM / OCR ---

def encode_image_from_pil(pil_image):
    """Converte un'immagine PIL direttamente in base64 per l'API."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_true_ocr_text_old(pil_image):
    """
    Esegue il 'Vero OCR' usando il modello Vision di Groq.
    Preso dalla logica di ocr_reader.py.
    """
    if not GROQ_API_KEY:
        return "Error: Missing API Key"

    client = Groq(api_key=GROQ_API_KEY)
    base64_image = encode_image_from_pil(pil_image)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Extract all text from this image. Return ONLY the text content, preserving formatting. Do not include any additional commentary."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_completion_tokens=1024,
            top_p=1
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in VLM OCR: {e}")
        return None


def get_true_ocr_text(pil_image):
    """
    Esegue il 'Vero OCR' usando il modello Vision di Groq.
    Riprova all'infinito in caso di errori di connessione.
    """
    if not GROQ_API_KEY:
        print("Error: Missing GROQ_API_KEY")
        return None

    client = Groq(api_key=GROQ_API_KEY)
    base64_image = encode_image_from_pil(pil_image)
    
    # Loop infinito per garantire il retry
    while True:
        try:
            completion = client.chat.completions.create(
                model=MODEL_VISION,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": "Extract all text from this image. Return ONLY the text content, preserving formatting. Do not include any additional commentary."
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_completion_tokens=1024,
                top_p=1
            )
            # Se ha successo, esce dal loop e ritorna il testo
            return completion.choices[0].message.content.strip()

        except (APIConnectionError, RateLimitError) as e:
            # Gestisce specificamente errori di connessione e rate limit
            print(f"⚠️ Error in VLM OCR (Connection/RateLimit): {e}")
            print("🔄 Retrying in 0.2 seconds...")
            time.sleep(0.2)
            continue  # Torna all'inizio del while e riprova
            
        except Exception as e:
            # Per altri errori (es. 400 Bad Request, API Key invalida) meglio non insistere
            print(f"❌ Fatal Error in VLM OCR: {e}")
            return None


def is_too_similar(text1, text2):
    """Controlla se due testi sono sostanzialmente uguali."""
    if not text1 or not text2:
        return False
    return difflib.SequenceMatcher(None, text1, text2).ratio() > SIMILARITY_THRESHOLD

# --- LOGICA DI TRADUZIONE ---

translation_memory = []
last_translation_time = 0
MEMORY_TIMEOUT = 30.0

def translate_text(text):
    global translation_memory, last_translation_time
    
    if not text:
        return None

    current_time = time.time()
    if current_time - last_translation_time > MEMORY_TIMEOUT:
        translation_memory = [] # Reset contesto se passa troppo tempo

    # Prompt System
    system_prompt = (
        "Sei un sistema di traduzione professionale EN->IT in tempo reale per sottotitoli e giochi.\n"
        "REGOLE:\n"
        "1. Traduci SOLO il testo fornito dall'utente in Italiano.\n"
        "2. NON aggiungere commenti, note o premesse.\n"
        "3. Mantieni lo stile e il tono originale."
    )

    # Aggiunta contesto (opzionale)
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
            model=MODEL_TRANSLATION,
            temperature=0.3,
        )
        translated_text = completion.choices[0].message.content
        
        # Aggiorna memoria
        if translated_text:
            translation_memory.append(translated_text)
            last_translation_time = current_time
            
        return translated_text
    except Exception as e:
        return f"Errore Traduzione: {e}"

# --- WORKER PRINCIPALE ---

def ocr_worker_pipeline():
    """
    Pipeline modificata:
    1. Screenshot area
    2. Tesseract Check (veloce, locale) -> C'è testo?
    3. Se SI -> Chiama Groq Vision (Vero OCR)
    4. Check Similarità -> È diverso dal precedente?
    5. Se SI -> Traduci -> Mostra a schermo
    """
    last_processed_text = ""
    print(f"OCR Pipeline avviata su regione: {ocr_region}")
    
    while True:
        try:
            if ocr_region:
                # 1. Grab Image
                img = ImageGrab.grab(bbox=ocr_region)
                
                # 2. Pre-Check con Tesseract (bassa qualità ma veloce)
                # config='--psm 6' assume un blocco di testo unico
                tess_text = pytesseract.image_to_string(img, config='--psm 6').strip()
                
                # Criterio minimo per attivare il VLM: ci sono almeno N caratteri validi?
                if len(tess_text) > MIN_CHARS_TO_TRIGGER and any(c.isalpha() for c in tess_text):
                    # print(f"[DEBUG] Tesseract ha rilevato potenziale testo: {tess_text[:15]}...")
                    
                    # 3. Vero OCR con Groq Vision
                    true_text = get_true_ocr_text(img)
                    
                    if true_text:
                        # 4. Check Similarità con l'ultimo testo processato VERAMENTE
                        if not is_too_similar(true_text, last_processed_text):
                            print(f"[NEW TEXT] Rilevato: {true_text[:30]}...")
                            
                            last_processed_text = true_text # Aggiorna subito per evitare loop
                            
                            # 5. Traduzione
                            translated = translate_text(true_text)
                            if translated:
                                gui_queue.put(translated)
                        else:
                            # print("[SKIP] Testo troppo simile al precedente.")
                            pass
                
            time.sleep(1.0) # Polling rate
            
        except Exception as e:
            print(f"Errore Pipeline Loop: {e}")
            time.sleep(1.0)

# --- GUI & UTILS ---
# (Quasi identica all'originale, solo adattamenti minori)

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

class ResizableWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Groq VLM Translate")
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
        self.context_menu.add_command(label="Reimposta Area OCR", command=self.reset_ocr_selection)
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
        # Logica semplificata per reset: riapre selettore
        # (In produzione si potrebbe modularizzare meglio, ma qui replichiamo la logica esistente)
        sel = AreaSelector().get_selection()
        if sel:
            global ocr_region
            ocr_region = sel
            print(f"Area aggiornata: {ocr_region}")
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
    print("--- Groq VLM Translator ---")
    
    # Selezione Area Iniziale
    selector = AreaSelector()
    ocr_region = selector.get_selection()
    
    if not ocr_region:
        print("Nessuna area selezionata. Uscita.")
        return

    # Avvio Thread Pipeline
    t = threading.Thread(target=ocr_worker_pipeline, daemon=True)
    t.start()

    # Avvio GUI
    app = ResizableWindow()
    app.mainloop()

if __name__ == "__main__":
    main()
