from flask import Flask, request, jsonify
import os
import json
import openai
from prompt import *
from viz_code import *
from circuit_gen import *
from dotenv import load_dotenv

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
import os

app = Flask(__name__)
global_assistant: Optional[Assistant] = None
global_run_id: Optional[str] = None
# Database configuration
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
Qiskit_links = ["https://docs.quantum.ibm.com/guides", "https://docs.quantum.ibm.com/guides/install-qiskit", "https://docs.quantum.ibm.com/guides/setup-channel", "https://docs.quantum.ibm.com/guides/online-lab-environments", "https://docs.quantum.ibm.com/guides/install-qiskit-source", "https://docs.quantum.ibm.com/guides/configure-qiskit-local", "https://docs.quantum.ibm.com/guides/hello-world", "https://docs.quantum.ibm.com/guides/intro-to-patterns", "https://docs.quantum.ibm.com/guides/map-problem-to-circuits", "https://docs.quantum.ibm.com/guides/optimize-for-hardware", "https://docs.quantum.ibm.com/guides/execute-on-hardware", "https://docs.quantum.ibm.com/guides/post-process-results", "https://docs.quantum.ibm.com/guides/latest-updates", "https://docs.quantum.ibm.com/guides/functions", "https://docs.quantum.ibm.com/guides/ibm-circuit-function", "https://docs.quantum.ibm.com/guides/algorithmiq-tem", "https://docs.quantum.ibm.com/guides/q-ctrl-performance-management", "https://docs.quantum.ibm.com/guides/qedma-qesem", "https://docs.quantum.ibm.com/guides/q-ctrl-optimization-solver", "https://docs.quantum.ibm.com/guides/circuit-library", "https://docs.quantum.ibm.com/guides/construct-circuits", "https://docs.quantum.ibm.com/guides/classical-feedforward-and-control-flow", "https://docs.quantum.ibm.com/guides/measure-qubits", "https://docs.quantum.ibm.com/guides/synthesize-unitary-operators", "https://docs.quantum.ibm.com/guides/bit-ordering", "https://docs.quantum.ibm.com/guides/save-circuits", "https://docs.quantum.ibm.com/guides/operators-overview", "https://docs.quantum.ibm.com/guides/specify-observables-pauli", "https://docs.quantum.ibm.com/guides/operator-class", "https://docs.quantum.ibm.com/guides/pulse", "https://docs.quantum.ibm.com/guides/introduction-to-qasm", "https://docs.quantum.ibm.com/guides/interoperate-qiskit-qasm2", "https://docs.quantum.ibm.com/guides/interoperate-qiskit-qasm3", "https://docs.quantum.ibm.com/guides/qasm-feature-table", "https://docs.quantum.ibm.com/guides/transpile", "https://docs.quantum.ibm.com/guides/transpiler-stages", "https://docs.quantum.ibm.com/guides/transpile-with-pass-managers", "https://docs.quantum.ibm.com/guides/dynamical-decoupling-pass-manager", "https://docs.quantum.ibm.com/guides/defaults-and-configuration-options", "https://docs.quantum.ibm.com/guides/set-optimization", "https://docs.quantum.ibm.com/guides/common-parameters", "https://docs.quantum.ibm.com/guides/represent-quantum-computers", "https://docs.quantum.ibm.com/guides/qiskit-transpiler-service", "https://docs.quantum.ibm.com/guides/ai-transpiler-passes", "https://docs.quantum.ibm.com/guides/transpile-rest-api", "https://docs.quantum.ibm.com/guides/custom-transpiler-pass", "https://docs.quantum.ibm.com/guides/custom-backend", "https://docs.quantum.ibm.com/guides/transpiler-plugins", "https://docs.quantum.ibm.com/guides/create-transpiler-plugin", "https://docs.quantum.ibm.com/guides/debugging-tools", "https://docs.quantum.ibm.com/guides/simulate-with-qiskit-sdk-primitives", "https://docs.quantum.ibm.com/guides/simulate-with-qiskit-aer", "https://docs.quantum.ibm.com/guides/local-testing-mode", "https://docs.quantum.ibm.com/guides/build-noise-models", "https://docs.quantum.ibm.com/guides/simulate-stabilizer-circuits", "https://docs.quantum.ibm.com/guides/primitives", "https://docs.quantum.ibm.com/guides/get-started-with-primitives", "https://docs.quantum.ibm.com/guides/primitive-input-output", "https://docs.quantum.ibm.com/guides/primitives-examples", "https://docs.quantum.ibm.com/guides/primitives-rest-api", "https://docs.quantum.ibm.com/guides/noise-learning", "https://docs.quantum.ibm.com/guides/runtime-options-overview", "https://docs.quantum.ibm.com/guides/specify-runtime-options", "https://docs.quantum.ibm.com/guides/error-mitigation-and-suppression-techniques", "https://docs.quantum.ibm.com/guides/configure-error-mitigation", "https://docs.quantum.ibm.com/guides/configure-error-suppression", "https://docs.quantum.ibm.com/guides/execution-modes", "https://docs.quantum.ibm.com/guides/sessions", "https://docs.quantum.ibm.com/guides/run-jobs-session", "https://docs.quantum.ibm.com/guides/run-jobs-batch", "https://docs.quantum.ibm.com/guides/repetition-rate-execution", "https://docs.quantum.ibm.com/guides/execution-modes-rest-api", "https://docs.quantum.ibm.com/guides/execution-modes-faq", "https://docs.quantum.ibm.com/guides/monitor-job", "https://docs.quantum.ibm.com/guides/estimate-job-run-time", "https://docs.quantum.ibm.com/guides/minimize-time", "https://docs.quantum.ibm.com/guides/minimize-time", "https://docs.quantum.ibm.com/guides/job-limits", "https://docs.quantum.ibm.com/guides/save-jobs", "https://docs.quantum.ibm.com/guides/processor-types", "https://docs.quantum.ibm.com/guides/qpu-information", "https://docs.quantum.ibm.com/guides/get-qpu-information", "https://docs.quantum.ibm.com/guides/native-gates", "https://docs.quantum.ibm.com/guides/retired-qpus", "https://docs.quantum.ibm.com/guides/dynamic-circuits-considerations", "https://docs.quantum.ibm.com/guides/instances", "https://docs.quantum.ibm.com/guides/fair-share-scheduler", "https://docs.quantum.ibm.com/guides/manage-cost", "https://docs.quantum.ibm.com/guides/visualize-circuits", "https://docs.quantum.ibm.com/guides/plot-quantum-states", "https://docs.quantum.ibm.com/guides/visualize-results", "https://docs.quantum.ibm.com/guides/serverless", "https://docs.quantum.ibm.com/guides/serverless-first-program", "https://docs.quantum.ibm.com/guides/serverless-run-first-workload", "https://docs.quantum.ibm.com/guides/serverless-manage-resources", "https://docs.quantum.ibm.com/guides/serverless-port-code", "https://docs.quantum.ibm.com/guides/addons", "https://docs.quantum.ibm.com/guides/qiskit-code-assistant", "https://docs.quantum.ibm.com/guides/qiskit-code-assistant-jupyterlab", "https://docs.quantum.ibm.com/guides/qiskit-code-assistant-vscode"]


# Initialize the assistant
def get_groq_assistant(collection_name: str) -> Assistant:
    embedder = OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536)
    return Assistant(
        name="groq_rag_assistant",
        llm=Groq(model="llama3-70b-8192"),
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
    global global_assistant, global_run_id
    
    if global_assistant is None:
        global_run_id = str(uuid.uuid4())
        global_assistant = get_groq_assistant(run_id=global_run_id)
        
        # Add Qiskit links to knowledge base
        scraper = WebsiteReader(max_links=1, max_depth=1)
        for url in Qiskit_links:
            web_documents: List[Document] = scraper.read(url)
            if web_documents:
                global_assistant.knowledge_base.load_documents(web_documents, upsert=True)
    
    return global_assistant, global_run_id

@app.route('/upload', methods=['POST'])
def upload():
    global global_assistant, global_run_id
    
    if global_assistant is None:
        global_assistant, global_run_id = initialize_knowledge_base()
    
    data = request.json
    urls = data.get('urls', [])
    pdf_urls = data.get('pdf_urls', [])
    
    scraper = WebsiteReader(max_links=2, max_depth=1)
    for url in urls:
        web_documents: List[Document] = scraper.read(url)
        if web_documents:
            global_assistant.knowledge_base.load_documents(web_documents, upsert=True)
    
    pdf_reader = PDFReader()
    for pdf_url in pdf_urls:
        pdf_documents: List[Document] = pdf_reader.read(pdf_url)
        if pdf_documents:
            global_assistant.knowledge_base.load_documents(pdf_documents, upsert=True)
    
    return jsonify({"message": "Knowledge base updated", "run_id": global_run_id}), 200

@app.route('/chat', methods=['POST'])
def chat():
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

@app.route('/get_qiskit_code', methods = ['POST'])
def get_code_utkarsh():
    data = request.get_json()
    user_input = data["user_input"]

    '''some code'''
    
    final_exec_code = qiskit_code + viz_code
    try:
        exec(final_exec_code)
        
        return jsonify("code executed successfully")
        
    except Exception as e:
        print(e)
        
    
# Flask endpoint for processing the prompt
@app.route('/process-prompt', methods=['POST'])
def process_prompt():
    error_chain = []  # List to accumulate error messages
    try:
        # Get JSON data from the POST request
        data = request.json
        user_prompt = data.get('user_prompt')
        image_urls = data.get('image_urls', None)

        if not user_prompt:
            error_chain.append("user_prompt is required")
            return jsonify({"errors": error_chain}), 400

        # Process the user prompt
        try:
            response = process_user_prompt(user_prompt, image_urls)
        except Exception as e:
            error_chain.append(f"Error in process_user_prompt: {str(e)}")

        # Return the JSON response
        return jsonify(response), 200
    except Exception as e:
        error_chain.append(f"Error in process_prompt: {str(e)}")

    # If there are any accumulated errors, return them
    return jsonify({"errors": error_chain}), 500

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = '6000', debug=True)