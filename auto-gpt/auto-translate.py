import os
import time
import threading
import tkinter as tk
from tkinter import scrolledtext
import pyperclip
from pynput import keyboard
from groq import Groq
import queue
import json


# --- CONFIGURAZIONE ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Gestione caricamento API KEY (Mantenuta la tua logica)
try:
    with open("api_keys.json", "r") as f:
        config = json.load(f)
    os.environ["GROQ_API_KEY"] = config["groq"]
except Exception as e:
    print(f"Nota: Impossibile caricare api_keys.json ({e}). Assicurati che GROQ_API_KEY sia impostata nell'ambiente.")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY") 
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct" 

DOUBLE_PRESS_THRESHOLD = 1
last_hotkey_time = 0

# --- CODA PER COMUNICAZIONE TRA THREAD ---
# Serve per passare il testo dal listener alla GUI in modo thread-safe
gui_queue = queue.Queue()

def translate_text(text):
    """Chiama API Groq per tradurre il testo."""
    if not text or not text.strip():
        return None
    
    print(f"Traduco: {text[:20]}...")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Sei un traduttore esperto. Traduci il seguente testo in italiano. Restituisci SOLO il testo tradotto."
                },
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model=MODEL_ID,
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Errore API: {e}")
        return f"Errore: {e}"

def start_persistent_gui():
    """Avvia la GUI in un thread e la mantiene viva."""
    def run_gui():
        root = tk.Tk()
        root.title("Groq Translate")
        
        w, h = 400, 300
        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = int((ws - w) / 2)
        y = int((hs - h) / 2)
        root.geometry(f"{w}x{h}+{x}+{y}")
        root.attributes("-topmost", True)
        
        # Text widget
        txt = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10))
        txt.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Funzione per nascondere invece di chiudere
        def hide_window(event=None):
            root.withdraw() # Nasconde la finestra
        
        root.protocol("WM_DELETE_WINDOW", hide_window)
        root.bind("<Escape>", hide_window)
        root.bind("<Control-c>", hide_window) # Chiude/Nasconde anche con Ctrl+C sulla finestra
        
        btn = tk.Button(root, text="Chiudi (Esc)", command=hide_window)
        btn.pack(pady=2)

        # Nascondiamo all'avvio finché non c'è una traduzione
        root.withdraw()

        # Loop di controllo della coda
        def check_queue():
            try:
                # Se c'è un nuovo testo nella coda
                new_text = gui_queue.get_nowait()
                
                # Aggiorna il contenuto
                txt.configure(state='normal')
                txt.delete(1.0, tk.END)
                txt.insert(tk.END, new_text)
                txt.configure(state='disabled')
                
                # Mostra la finestra e portala in primo piano
                root.deiconify()
                root.attributes("-topmost", True)
                # Forza il focus (opzionale, utile per chiudere subito con Esc)
                root.focus_force() 
                
            except queue.Empty:
                pass
            
            # Ricontrolla tra 100ms
            root.after(100, check_queue)

        # Avvia il polling
        check_queue()
        root.mainloop()

    t = threading.Thread(target=run_gui)
    t.daemon = True
    t.start()

def on_ctrl_c():
    """Callback attivata da pynput."""
    global last_hotkey_time
    
    now = time.time()
    if now - last_hotkey_time < DOUBLE_PRESS_THRESHOLD:
        print(">> DOPPIO Ctrl+C Rilevato!")
        
        time.sleep(0.1)
        copied_text = pyperclip.paste()
        
        if copied_text:
            translated = translate_text(copied_text)
            if translated:
                pyperclip.copy(translated)
                # INVECE DI APRIRE UNA NUOVA GUI, METTIAMO IL TESTO IN CODA
                gui_queue.put(translated)
        
        last_hotkey_time = 0 
    else:
        last_hotkey_time = now

def main():
    print(f"Monitoraggio attivo su modello {MODEL_ID}...")
    
    # 1. Avvia la GUI persistente in background
    start_persistent_gui()
    
    # 2. Avvia il listener tastiera (bloccante per il main thread)
    with keyboard.GlobalHotKeys({'<ctrl>+c': on_ctrl_c}) as h:
        h.join()

if __name__ == "__main__":
    main()
