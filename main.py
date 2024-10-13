from flask import Flask, request, jsonify
import os
import json
import openai
from prompt import *
from viz_code import viz_code
from circuit_gen import *
from dotenv import load_dotenv
import requests

app = Flask(__name__)

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
import uuid
import os

app = Flask(__name__)

# Database configuration
db_url = "postgresql+psycopg2://ai:ai@localhost:5532/ai"
GLOBAL_BUSINESS_ID = "global_knowledge_base"

# Qiskit links (truncated for brevity)
Qiskit_links = ["https://docs.quantum.ibm.com/guides", "https://docs.quantum.ibm.com/guides/install-qiskit", "https://docs.quantum.ibm.com/guides/setup-channel", "https://docs.quantum.ibm.com/guides/online-lab-environments", "https://docs.quantum.ibm.com/guides/install-qiskit-source", "https://docs.quantum.ibm.com/guides/configure-qiskit-local", "https://docs.quantum.ibm.com/guides/hello-world", "https://docs.quantum.ibm.com/guides/intro-to-patterns", "https://docs.quantum.ibm.com/guides/map-problem-to-circuits", "https://docs.quantum.ibm.com/guides/optimize-for-hardware", "https://docs.quantum.ibm.com/guides/execute-on-hardware", "https://docs.quantum.ibm.com/guides/post-process-results", "https://docs.quantum.ibm.com/guides/latest-updates", "https://docs.quantum.ibm.com/guides/functions", "https://docs.quantum.ibm.com/guides/ibm-circuit-function", "https://docs.quantum.ibm.com/guides/algorithmiq-tem", "https://docs.quantum.ibm.com/guides/q-ctrl-performance-management", "https://docs.quantum.ibm.com/guides/qedma-qesem", "https://docs.quantum.ibm.com/guides/q-ctrl-optimization-solver", "https://docs.quantum.ibm.com/guides/circuit-library", "https://docs.quantum.ibm.com/guides/construct-circuits", "https://docs.quantum.ibm.com/guides/classical-feedforward-and-control-flow", "https://docs.quantum.ibm.com/guides/measure-qubits", "https://docs.quantum.ibm.com/guides/synthesize-unitary-operators", "https://docs.quantum.ibm.com/guides/bit-ordering", "https://docs.quantum.ibm.com/guides/save-circuits", "https://docs.quantum.ibm.com/guides/operators-overview", "https://docs.quantum.ibm.com/guides/specify-observables-pauli", "https://docs.quantum.ibm.com/guides/operator-class", "https://docs.quantum.ibm.com/guides/pulse", "https://docs.quantum.ibm.com/guides/introduction-to-qasm", "https://docs.quantum.ibm.com/guides/interoperate-qiskit-qasm2", "https://docs.quantum.ibm.com/guides/interoperate-qiskit-qasm3", "https://docs.quantum.ibm.com/guides/qasm-feature-table", "https://docs.quantum.ibm.com/guides/transpile", "https://docs.quantum.ibm.com/guides/transpiler-stages", "https://docs.quantum.ibm.com/guides/transpile-with-pass-managers", "https://docs.quantum.ibm.com/guides/dynamical-decoupling-pass-manager", "https://docs.quantum.ibm.com/guides/defaults-and-configuration-options", "https://docs.quantum.ibm.com/guides/set-optimization", "https://docs.quantum.ibm.com/guides/common-parameters", "https://docs.quantum.ibm.com/guides/represent-quantum-computers", "https://docs.quantum.ibm.com/guides/qiskit-transpiler-service", "https://docs.quantum.ibm.com/guides/ai-transpiler-passes", "https://docs.quantum.ibm.com/guides/transpile-rest-api", "https://docs.quantum.ibm.com/guides/custom-transpiler-pass", "https://docs.quantum.ibm.com/guides/custom-backend", "https://docs.quantum.ibm.com/guides/transpiler-plugins", "https://docs.quantum.ibm.com/guides/create-transpiler-plugin", "https://docs.quantum.ibm.com/guides/debugging-tools", "https://docs.quantum.ibm.com/guides/simulate-with-qiskit-sdk-primitives", "https://docs.quantum.ibm.com/guides/simulate-with-qiskit-aer", "https://docs.quantum.ibm.com/guides/local-testing-mode", "https://docs.quantum.ibm.com/guides/build-noise-models", "https://docs.quantum.ibm.com/guides/simulate-stabilizer-circuits", "https://docs.quantum.ibm.com/guides/primitives", "https://docs.quantum.ibm.com/guides/get-started-with-primitives", "https://docs.quantum.ibm.com/guides/primitive-input-output", "https://docs.quantum.ibm.com/guides/primitives-examples", "https://docs.quantum.ibm.com/guides/primitives-rest-api", "https://docs.quantum.ibm.com/guides/noise-learning", "https://docs.quantum.ibm.com/guides/runtime-options-overview", "https://docs.quantum.ibm.com/guides/specify-runtime-options", "https://docs.quantum.ibm.com/guides/error-mitigation-and-suppression-techniques", "https://docs.quantum.ibm.com/guides/configure-error-mitigation", "https://docs.quantum.ibm.com/guides/configure-error-suppression", "https://docs.quantum.ibm.com/guides/execution-modes", "https://docs.quantum.ibm.com/guides/sessions", "https://docs.quantum.ibm.com/guides/run-jobs-session", "https://docs.quantum.ibm.com/guides/run-jobs-batch", "https://docs.quantum.ibm.com/guides/repetition-rate-execution", "https://docs.quantum.ibm.com/guides/execution-modes-rest-api", "https://docs.quantum.ibm.com/guides/execution-modes-faq", "https://docs.quantum.ibm.com/guides/monitor-job", "https://docs.quantum.ibm.com/guides/estimate-job-run-time", "https://docs.quantum.ibm.com/guides/minimize-time", "https://docs.quantum.ibm.com/guides/minimize-time", "https://docs.quantum.ibm.com/guides/job-limits", "https://docs.quantum.ibm.com/guides/save-jobs", "https://docs.quantum.ibm.com/guides/processor-types", "https://docs.quantum.ibm.com/guides/qpu-information", "https://docs.quantum.ibm.com/guides/get-qpu-information", "https://docs.quantum.ibm.com/guides/native-gates", "https://docs.quantum.ibm.com/guides/retired-qpus", "https://docs.quantum.ibm.com/guides/dynamic-circuits-considerations", "https://docs.quantum.ibm.com/guides/instances", "https://docs.quantum.ibm.com/guides/fair-share-scheduler", "https://docs.quantum.ibm.com/guides/manage-cost", "https://docs.quantum.ibm.com/guides/visualize-circuits", "https://docs.quantum.ibm.com/guides/plot-quantum-states", "https://docs.quantum.ibm.com/guides/visualize-results", "https://docs.quantum.ibm.com/guides/serverless", "https://docs.quantum.ibm.com/guides/serverless-first-program", "https://docs.quantum.ibm.com/guides/serverless-run-first-workload", "https://docs.quantum.ibm.com/guides/serverless-manage-resources", "https://docs.quantum.ibm.com/guides/serverless-port-code", "https://docs.quantum.ibm.com/guides/addons", "https://docs.quantum.ibm.com/guides/qiskit-code-assistant", "https://docs.quantum.ibm.com/guides/qiskit-code-assistant-jupyterlab", "https://docs.quantum.ibm.com/guides/qiskit-code-assistant-vscode"]

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
        description="You are an AI chatbot called 'QuantumViz' and your task is to take ideas, and turn them into quantum circuits. Your job is always to only create python code for quantum circuits and make sure that the quantum circuit is always labelled as qc variable name",
        add_references_to_prompt=True,
        markdown=True,
        add_chat_history_to_messages=True,
        num_history_messages=3,
    )

def initialize_knowledge_base():
    assistant = get_groq_assistant(collection_name=GLOBAL_BUSINESS_ID)
    return assistant

def load_documents_from_links(links: List[str]) -> List[Document]:
    reader = WebsiteReader()
    documents = []
    for link in links:
        try:
            doc = reader.load(link)
            documents.extend(doc)
        except Exception as e:
            print(f"Error loading {link}: {str(e)}")
    return documents

@app.route('/rag_upload', methods=['POST'])
def rag_upload():
    try:
        assistant = initialize_knowledge_base()
        
        # Check if documents are already loaded
        if assistant.knowledge_base.vector_db.count() > 0:
            return jsonify({"message": "Knowledge base already populated", "count": assistant.knowledge_base.vector_db.count()}), 200
        
        # Load documents from Qiskit links
        all_rag_documents = load_documents_from_links(Qiskit_links)
        
        if all_rag_documents:
            assistant.knowledge_base.load_documents(all_rag_documents, upsert=True)
            return jsonify({"message": "Files uploaded and processed successfully", "count": len(all_rag_documents)}), 200
        else:
            return jsonify({"error": "No documents to upload"}), 500
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

def rag_chat_utkarsh(question):
    global global_assistant, global_run_id
    
    if global_assistant is None:
        global_assistant, global_run_id = initialize_knowledge_base()
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Missing question"}), 400
    
    response = ""
    for delta in global_assistant.run(question):
        response += delta
    
    return jsonify({"response": response, "run_id": global_run_id}), 200

@app.route('/get_qiskit_code', methods=['POST'])
def get_code_utkarsh():
    data = request.get_json()
    user_input = data["user_input"]

    try:
        qiskit_code = rag_chat_utkarsh(user_input)
    except Exception as e:
        return jsonify({"error": f"Failed to get response from rag_chat: {str(e)}"}), 500

    # Assuming viz_code is defined elsewhere
    print("quiskit code", qiskit_code)
    final_exec_code = qiskit_code + viz_code
    
    html_output_folder = '/Users/aashmanrastogi/Desktop/quantumviz-backend/quantum_plots'
    try:
        exec(final_exec_code, globals())
        # List to store paths of generated HTML files
        generated_html_files = []
        
        # Search for HTML files in the specified folder
        for filename in os.listdir(html_output_folder):
            if filename.endswith('.html'):
                generated_html_files.append(os.path.join(html_output_folder, filename))
        
        return jsonify({
            "message": "code executed successfully",
            "html_files": generated_html_files
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Flask endpoint for processing the prompt
@app.route('/process-prompt', methods=['POST'])
def process_prompt():
    error_chain = []  # List to accumulate error messages
    try:
        # Get JSON data from the POST request
        data = request.json
        user_input = data.get('user_input')

        if not user_input:
            error_chain.append("user_prompt is required")
            return jsonify({"errors": error_chain}), 400

        # Process the user prompt
        try:
            response = process_user_prompt(user_input)
        except Exception as e:
            error_chain.append(f"Error in process_user_prompt: {str(e)}")

        # Return the JSON response
        return jsonify(response), 200
    except Exception as e:
        error_chain.append(f"Error in process_prompt: {str(e)}")

    # If there are any accumulated errors, return them
    return jsonify({"errors": error_chain}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6000', debug=True)