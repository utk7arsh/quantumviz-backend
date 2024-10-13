import re
import os
import json
import openai
import requests
from circuit_gen import *
from flask_cors import CORS
from speech_to_text import *
from viz_code import viz_code
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from prompt import chatbot_system_prompt, rag_system_prompt


import uuid
from flask import Flask, request, jsonify
from typing import List, Optional
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.document import Document
from phi.llm.groq import Groq
from phi.embedder.openai import OpenAIEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.document.reader.website import WebsiteReader

app = Flask(__name__)
CORS(app)

global_assistant: Optional[Assistant] = None
global_run_id: Optional[str] = None
# Database configuration
db_url = "postgresql+psycopg2://ai:ai@localhost:5532/ai"
GLOBAL_BUSINESS_ID = "global_knowledge_base"
CHATBOT_BUSINESS_ID = "chatbot_knowledge_base"

# Qiskit links (truncated for brevity)
Qiskit_links = ["https://docs.quantum.ibm.com/guides", "https://docs.quantum.ibm.com/guides/install-qiskit", "https://docs.quantum.ibm.com/guides/setup-channel", "https://docs.quantum.ibm.com/guides/online-lab-environments", "https://docs.quantum.ibm.com/guides/install-qiskit-source", "https://docs.quantum.ibm.com/guides/configure-qiskit-local", "https://docs.quantum.ibm.com/guides/hello-world", "https://docs.quantum.ibm.com/guides/intro-to-patterns", "https://docs.quantum.ibm.com/guides/map-problem-to-circuits", "https://docs.quantum.ibm.com/guides/optimize-for-hardware", "https://docs.quantum.ibm.com/guides/execute-on-hardware", "https://docs.quantum.ibm.com/guides/post-process-results", "https://docs.quantum.ibm.com/guides/latest-updates", "https://docs.quantum.ibm.com/guides/functions", "https://docs.quantum.ibm.com/guides/ibm-circuit-function", "https://docs.quantum.ibm.com/guides/algorithmiq-tem", "https://docs.quantum.ibm.com/guides/q-ctrl-performance-management", "https://docs.quantum.ibm.com/guides/qedma-qesem", "https://docs.quantum.ibm.com/guides/q-ctrl-optimization-solver", "https://docs.quantum.ibm.com/guides/circuit-library", "https://docs.quantum.ibm.com/guides/construct-circuits", "https://docs.quantum.ibm.com/guides/classical-feedforward-and-control-flow", "https://docs.quantum.ibm.com/guides/measure-qubits", "https://docs.quantum.ibm.com/guides/synthesize-unitary-operators", "https://docs.quantum.ibm.com/guides/bit-ordering", "https://docs.quantum.ibm.com/guides/save-circuits", "https://docs.quantum.ibm.com/guides/operators-overview", "https://docs.quantum.ibm.com/guides/specify-observables-pauli", "https://docs.quantum.ibm.com/guides/operator-class", "https://docs.quantum.ibm.com/guides/pulse", "https://docs.quantum.ibm.com/guides/introduction-to-qasm", "https://docs.quantum.ibm.com/guides/interoperate-qiskit-qasm2", "https://docs.quantum.ibm.com/guides/interoperate-qiskit-qasm3", "https://docs.quantum.ibm.com/guides/qasm-feature-table", "https://docs.quantum.ibm.com/guides/transpile", "https://docs.quantum.ibm.com/guides/transpiler-stages", "https://docs.quantum.ibm.com/guides/transpile-with-pass-managers", "https://docs.quantum.ibm.com/guides/dynamical-decoupling-pass-manager", "https://docs.quantum.ibm.com/guides/defaults-and-configuration-options", "https://docs.quantum.ibm.com/guides/set-optimization", "https://docs.quantum.ibm.com/guides/common-parameters", "https://docs.quantum.ibm.com/guides/represent-quantum-computers", "https://docs.quantum.ibm.com/guides/qiskit-transpiler-service", "https://docs.quantum.ibm.com/guides/ai-transpiler-passes", "https://docs.quantum.ibm.com/guides/transpile-rest-api", "https://docs.quantum.ibm.com/guides/custom-transpiler-pass", "https://docs.quantum.ibm.com/guides/custom-backend", "https://docs.quantum.ibm.com/guides/transpiler-plugins", "https://docs.quantum.ibm.com/guides/create-transpiler-plugin", "https://docs.quantum.ibm.com/guides/debugging-tools", "https://docs.quantum.ibm.com/guides/simulate-with-qiskit-sdk-primitives", "https://docs.quantum.ibm.com/guides/simulate-with-qiskit-aer", "https://docs.quantum.ibm.com/guides/local-testing-mode", "https://docs.quantum.ibm.com/guides/build-noise-models", "https://docs.quantum.ibm.com/guides/simulate-stabilizer-circuits", "https://docs.quantum.ibm.com/guides/primitives", "https://docs.quantum.ibm.com/guides/get-started-with-primitives", "https://docs.quantum.ibm.com/guides/primitive-input-output", "https://docs.quantum.ibm.com/guides/primitives-examples", "https://docs.quantum.ibm.com/guides/primitives-rest-api", "https://docs.quantum.ibm.com/guides/noise-learning", "https://docs.quantum.ibm.com/guides/runtime-options-overview", "https://docs.quantum.ibm.com/guides/specify-runtime-options", "https://docs.quantum.ibm.com/guides/error-mitigation-and-suppression-techniques", "https://docs.quantum.ibm.com/guides/configure-error-mitigation", "https://docs.quantum.ibm.com/guides/configure-error-suppression", "https://docs.quantum.ibm.com/guides/execution-modes", "https://docs.quantum.ibm.com/guides/sessions", "https://docs.quantum.ibm.com/guides/run-jobs-session", "https://docs.quantum.ibm.com/guides/run-jobs-batch", "https://docs.quantum.ibm.com/guides/repetition-rate-execution", "https://docs.quantum.ibm.com/guides/execution-modes-rest-api", "https://docs.quantum.ibm.com/guides/execution-modes-faq", "https://docs.quantum.ibm.com/guides/monitor-job", "https://docs.quantum.ibm.com/guides/estimate-job-run-time", "https://docs.quantum.ibm.com/guides/minimize-time", "https://docs.quantum.ibm.com/guides/minimize-time", "https://docs.quantum.ibm.com/guides/job-limits", "https://docs.quantum.ibm.com/guides/save-jobs", "https://docs.quantum.ibm.com/guides/processor-types", "https://docs.quantum.ibm.com/guides/qpu-information", "https://docs.quantum.ibm.com/guides/get-qpu-information", "https://docs.quantum.ibm.com/guides/native-gates", "https://docs.quantum.ibm.com/guides/retired-qpus", "https://docs.quantum.ibm.com/guides/dynamic-circuits-considerations", "https://docs.quantum.ibm.com/guides/instances", "https://docs.quantum.ibm.com/guides/fair-share-scheduler", "https://docs.quantum.ibm.com/guides/manage-cost", "https://docs.quantum.ibm.com/guides/visualize-circuits", "https://docs.quantum.ibm.com/guides/plot-quantum-states", "https://docs.quantum.ibm.com/guides/visualize-results", "https://docs.quantum.ibm.com/guides/serverless", "https://docs.quantum.ibm.com/guides/serverless-first-program", "https://docs.quantum.ibm.com/guides/serverless-run-first-workload", "https://docs.quantum.ibm.com/guides/serverless-manage-resources", "https://docs.quantum.ibm.com/guides/serverless-port-code", "https://docs.quantum.ibm.com/guides/addons", "https://docs.quantum.ibm.com/guides/qiskit-code-assistant", "https://docs.quantum.ibm.com/guides/qiskit-code-assistant-jupyterlab", "https://docs.quantum.ibm.com/guides/qiskit-code-assistant-vscode"]


# Chatbot helper functions
def get_chatbot_assistant(collection_name: str) -> Assistant:
    embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536)
    return Assistant(
        name="groq_chatbot_assistant",
        llm=Groq(model="llama-3.1-70b-versatile"),
        storage=PgAssistantStorage(table_name="groq_chatbot_assistant", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=collection_name,
                embedder=embedder,
            ),
            num_documents=2,
        ),
        description=chatbot_system_prompt,
        add_references_to_prompt=True,
        markdown=True,
        add_chat_history_to_messages=True,
        num_history_messages=5,
    )

def initialize_chatbot_knowledge_base():
    assistant = get_chatbot_assistant(collection_name=CHATBOT_BUSINESS_ID)
    return assistant

##########################
#### Chatbot endpoints ###
##########################

@app.route('/chatbot_upload', methods=['GET'])
def chatbot_upload():
    try:
        assistant = initialize_chatbot_knowledge_base()
        scraper = WebsiteReader(max_links=2, max_depth=1)
        for url in Qiskit_links:
            web_documents: List[Document] = scraper.read(url)
            if web_documents:
                assistant.knowledge_base.load_documents(web_documents, upsert=True)
        return jsonify({"message": "Files uploaded and processed successfully for chatbot", "count": len(Qiskit_links)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        data = request.json
        question = data.get("question")

        if not question:
            return jsonify({"error": "Missing question"}), 400

        assistant = get_chatbot_assistant(collection_name=CHATBOT_BUSINESS_ID)
        
        response = ""
        for delta in assistant.run(question):
            response += delta

        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# RAG helper functions
def get_groq_assistant(collection_name: str) -> Assistant:
    embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536)
    return Assistant(
        name="groq_rag_assistant",
        llm=Groq(model="llama-3.1-70b-versatile"),
        storage=PgAssistantStorage(table_name="groq_rag_assistant", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=collection_name,
                embedder=embedder,
            ),
            num_documents=2,
        ),
        description=rag_system_prompt,
        add_references_to_prompt=True,
        markdown=True,
        add_chat_history_to_messages=True,
        num_history_messages=3,
    )

def initialize_knowledge_base():
    assistant = get_groq_assistant(collection_name=GLOBAL_BUSINESS_ID)
    return assistant

def rag_chat_utkarsh(question):
    try:
        if not question:
            raise ValueError("Missing question")

        assistant = get_groq_assistant(collection_name=GLOBAL_BUSINESS_ID)
        
        response = ""
        for delta in assistant.run(question):
            response += delta

        return response
    
    except Exception as e:
        raise Exception(f"Error in rag_chat_utkarsh: {str(e)}")

##########################
#### RAG endpoints ######
##########################

@app.route('/rag_upload', methods=['GET'])
def rag_upload():
    try:
        assistant = initialize_knowledge_base()
        scraper = WebsiteReader(max_links=2, max_depth=1)
        for url in Qiskit_links:
            web_documents: List[Document] = scraper.read(url)
            if web_documents:
                assistant.knowledge_base.load_documents(web_documents, upsert=True)
        return jsonify({"message": "Files uploaded and processed successfully", "count": len(Qiskit_links)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/rag_chat', methods=['POST'])
def rag_chat():
    try:
        data = request.json
        question = data.get("question")

        if not question:
            return jsonify({"error": "Missing question"}), 400

        assistant = get_groq_assistant(collection_name=GLOBAL_BUSINESS_ID)
        
        response = ""
        for delta in assistant.run(question):
            response += delta

        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##########################
####  Qiskit code ######
##########################

@app.route('/get_qiskit_code', methods=['POST'])
def get_code_utkarsh():
    data = request.get_json()
    user_input = data["user_input"]

    try:
        qiskit_code = rag_chat_utkarsh(user_input)
        pattern = re.compile(r'-----FORMAT-----(.*?)-----FORMAT-----', re.DOTALL)
        matches = pattern.findall(qiskit_code)
        
        if not matches:
            raise ValueError("No formatted text found in response")
        
        qiskit_code = matches[0].strip()
        
    except Exception as e:
        return jsonify({"error": f"Failed to get response from rag_chat_utkarsh: {str(e)}"}), 500

    # Assuming viz_code is defined elsewhere
    print("quiskit code")
    print("--------------------------------")
    print(qiskit_code)
    print("--------------------------------")
    print("viz code")
    print("--------------------------------")
    print(viz_code)
    import_section = '''
import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import partial_trace
from qiskit.visualization import plot_bloch_multivector
import numpy as np
import plotly.graph_objects as go

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
    '''
    print("import section")
    print("--------------------------------")
    print(import_section)
    print("--------------------------------")
        
    final_exec_code = import_section + '\n' + qiskit_code + '\n' + viz_code
    
    print("final exec code")
    print("--------------------------------")
    print(final_exec_code)
    print("--------------------------------")
    
    html_output_folder = './quantum_plots'
    try:
        exec(final_exec_code, globals())
        generated_html_files = []
        
        for filename in os.listdir(html_output_folder):
            if filename.endswith('.html'):
                generated_html_files.append(os.path.join(html_output_folder, filename))
        
        return jsonify({
            "message": "code executed successfully",
            "html_files": generated_html_files,
            "code": final_exec_code
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##########################
#### Process prompt ######
##########################

@app.route('/process-prompt', methods=['POST'])
def process_prompt():
    error_chain = []
    try:
        data = request.json
        user_input = data.get('user_input')

        if not user_input:
            error_chain.append("user_prompt is required")
            return jsonify({"errors": error_chain}), 400

        try:
            response = process_user_prompt(user_input)
        except Exception as e:
            error_chain.append(f"Error in process_user_prompt: {str(e)}")

        return jsonify(response), 200
    except Exception as e:
        error_chain.append(f"Error in process_prompt: {str(e)}")

    return jsonify({"errors": error_chain}), 500


@app.route('/transcribe-audio', methods=['POST'])
def process_prompt():
    error_chain = []
    try:
        data = request.get_json()
        audio_file = data.get("audio_file")
        
        # Check if the audio file is in the request
        if 'audio_file' not in request.files:
            error_chain.append("Audio file is required")
            return jsonify({"errors": error_chain}), 400

        audio_filename = "./audio_files/recorded_audio.wav"
        audio_file.save(audio_filename)  # Save the uploaded audio file
        user_input = transcribe_audio(audio_filename)

        try:
            response = process_user_prompt(user_input)
        except Exception as e:
            error_chain.append(f"Error in process_user_prompt: {str(e)}")

        return jsonify(response), 200
    except Exception as e:
        error_chain.append(f"Error in transcribe_audio: {str(e)}")

    return jsonify({"errors": error_chain}), 500


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = '8080', debug=True)