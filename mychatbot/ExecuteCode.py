import io, sys, contextlib
import streamlit as st  # Dependency for creating web apps
import matplotlib.pyplot as plt  # For plotting graphs
import pandas as pd  # For handling data frames
import numpy as np  # For numerical operations
import seaborn as sns  # For statistical data visualization
import plotly  # For interactive plots


def ExecuteCode(reply):
    try:
        if '```' not in reply:
            raise ValueError("Nessun blocco codice delimitato da ```")

        code_block = reply.split('```')[1]
        code_to_execute = code_block.split('```')[0].strip()

        if code_to_execute.startswith('python'):
            code_to_execute = code_to_execute[6:].lstrip('\n')

        # Nuovo ambiente di esecuzione con accesso a st
        exec_globals = {
            'st': st,
            'plt': plt,
            'pd': pd,
            'np': np,
            'sns': sns,
            'plotly': plotly
        }

        # Esegue il codice nel contesto modificato
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            exec(code_to_execute, exec_globals)

        # Mostra output testuale
        captured_output = output.getvalue()
        if captured_output:
            st.code(captured_output)

        # Gestione esplicita delle figure Matplotlib
        if 'plt' in exec_globals:
            fig = exec_globals['plt'].gcf()
            if fig.get_axes():
                st.pyplot(fig)
                exec_globals['plt'].close()

        # Gestione esplicita di altri tipi di output
        for var in exec_globals.values():
            if isinstance(var, (pd.DataFrame, pd.Series)):
                st.dataframe(var)
            elif isinstance(var, plotly.graph_objs.Figure):
                st.plotly_chart(var)

    except Exception as e:
        st.error(f"Errore esecuzione codice: {str(e)}")




def ExecuteCodeSimple(reply):
    try:
        # Estrazione codice con controllo delimitatori
        if '```' not in reply:
            raise ValueError("Nessun blocco codice delimitato da ```")

        # Isola il primo blocco di codice
        code_block = reply.split('```')[1]
        code_to_execute = code_block.split('```')[0].strip()

        # Rimuove eventuale specifica linguaggio (es. 'python')
        if code_to_execute.startswith('python'):
            code_to_execute = code_to_execute[6:].lstrip('\n')

        # Redirect stdout per catturare print
        output_buffer = io.StringIO()
        sys.stdout = output_buffer

        # Esecuzione codice in ambiente isolato
        exec_locals = {}
        exec(code_to_execute, {'st': st}, exec_locals)  # Passa st come variabile globale

        # Ripristino stdout
        sys.stdout = sys.__stdout__

        # Mostra output catturato
        captured_output = output_buffer.getvalue()
        if captured_output:
            st.code(captured_output)

        # Mostra oggetti ritornati (se presenti)
        if 'result' in exec_locals:
            st.write(exec_locals['result'])

    except Exception as e:
        st.error(f"Errore esecuzione codice: {str(e)}")