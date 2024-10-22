import re
import os
import json
import openai
import tempfile
import requests
from opeani_func import *
from flask_cors import CORS
from speech_to_text import *
from dotenv import load_dotenv
from viz_code import viz_code, qiskit_code
from flask import Flask, request, jsonify, send_from_directory
from prompt import chatbot_system_prompt, rag_system_prompt

import boto3  # Add this import at the top of your file

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

import base64
import wave
from pydub import AudioSegment
import io
import binascii

app = Flask(__name__, static_url_path='', static_folder='quantum_plots')
CORS(app, resources={r"/*": {"origins": "*"}})

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
    
    

@app.route('/chatbot-transcribe-audio', methods=['POST'])
def chatbot_transcribe_audio():
    try:
        data = request.get_json()
        if not data or 'audio_data' not in data:
            return jsonify({"error": "Audio data is required"}), 400

        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(data['audio_data'])
        except binascii.Error as e:
            return jsonify({"error": f"Invalid base64 encoding: {str(e)}"}), 400

        # Determine the audio format (you might need to send this info from the client)
        audio_format = data.get('audio_format', 'webm')  # Default to webm if not specified

        try:
            # Convert to WAV using pydub
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format)
        except Exception as e:
            return jsonify({"error": f"Error processing audio data: {str(e)}"}), 400
        
        try:
            # Save as temporary WAV file
            temp_audio_path = os.path.join(AUDIO_FILES_DIR, 'temp_audio.wav')
            audio.export(temp_audio_path, format="wav")
        except Exception as e:
            return jsonify({"error": f"Error saving temporary audio file: {str(e)}"}), 500

        try:
            # Check file size and duration
            file_size = os.path.getsize(temp_audio_path)
            duration = len(audio) / 1000.0  # pydub uses milliseconds

            print(f"Audio file size: {file_size} bytes")
            print(f"Audio duration: {duration} seconds")

            if duration < 0.1:
                os.remove(temp_audio_path)  # Clean up the file if it's too short
                return jsonify({"error": "Audio file is too short. Minimum audio length is 0.1 seconds."}), 400
        except Exception as e:
            return jsonify({"error": f"Error checking audio file: {str(e)}"}), 500

        try:
            transcription = transcribe_audio(temp_audio_path)
        except Exception as e:
            return jsonify({"error": f"Error in transcribe_audio: {str(e)}"}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        return jsonify({"error": f"Unexpected error in handle_transcribe_audio: {str(e)}"}), 500
 


##########################
#### RAG helper functions ####
##########################

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
@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory(app.static_folder, filename)


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
        # qiskit_code_v = qiskit_code
        
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
    s3_bucket_name = 'qubit-store-html'  # Define your S3 bucket name
    s3_client = boto3.client('s3',
                             aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),  
                             aws_secret_access_key=os.getenv('AWS_SECRET_KEY')  
    ) # Create an S3 client

    try:
        exec(final_exec_code, globals())
        generated_html_files = []
        
        for filename in os.listdir(html_output_folder):
            if filename.endswith('.html'):
                # Create a static URL for each HTML file
                generated_html_files.append(f"/static/{filename}")
                
                # Upload to S3
                s3_client.upload_file(os.path.join(html_output_folder, filename), s3_bucket_name, filename)

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

        try:
            response = process_user_prompt(user_input)
        except Exception as e:
            error_chain.append(f"Error in process_user_prompt: {str(e)}")

        return jsonify(response), 200
    except Exception as e:
        error_chain.append(f"Error in process_prompt: {str(e)}")

    return jsonify({"errors": error_chain}), 500


@app.route('/upload_images', methods=['POST'])
def upload_images():
    try:
        if 'image_files' not in request.files:
            return jsonify({"error": "No image files provided"}), 400

        image_files = request.files.getlist('image_files')  # Get the list of uploaded files
        temp_file_paths = []  # List to store paths of temporary files

        for image_file in image_files:
            # Create a temporary file to save the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(image_file.read())
                temp_file_path = temp_file.name  # Get the path of the temporary file
                temp_file_paths.append(temp_file_path)  # Add the path to the list

        response = openai_chat_image(System_image_prompt, temp_file_path)
        
        pattern = re.compile(r'-----FORMAT-----(.*?)-----FORMAT-----', re.DOTALL)
        matches = pattern.findall(response)
        
        if not matches:
            raise ValueError("No formatted text found in response")
        
        code = matches[0].strip()
        
        # Cleanup: Delete temporary files
        for temp_file_path in temp_file_paths:
            os.remove(temp_file_path)

        # Now you can pass temp_file_paths to the OpenAI chatbot or process them as needed
        # For example:
        # responses = [openai_chatbot.process_image(path) for path in temp_file_paths]

        return jsonify({"message": "Images uploaded successfully and code generated", "code": code}), 200
    except Exception as e:
        # Cleanup in case of an error
        for temp_file_path in temp_file_paths:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        return jsonify({"error": str(e)}), 500
    

# Define the audio files directory
AUDIO_FILES_DIR = os.path.join(os.path.dirname(__file__), 'audio_files')

# Ensure the audio files directory exists
os.makedirs(AUDIO_FILES_DIR, exist_ok=True)

@app.route('/transcribe-audio', methods=['POST'])
def handle_transcribe_audio():
    try:
        data = request.get_json()
        if not data or 'audio_data' not in data:
            return jsonify({"error": "Audio data is required"}), 400

        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(data['audio_data'])
        except binascii.Error as e:
            return jsonify({"error": f"Invalid base64 encoding: {str(e)}"}), 400

        # Determine the audio format (you might need to send this info from the client)
        audio_format = data.get('audio_format', 'webm')  # Default to webm if not specified

        try:
            # Convert to WAV using pydub
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=audio_format)
        except Exception as e:
            return jsonify({"error": f"Error processing audio data: {str(e)}"}), 400
        
        try:
            # Save as temporary WAV file
            temp_audio_path = os.path.join(AUDIO_FILES_DIR, 'temp_audio.wav')
            audio.export(temp_audio_path, format="wav")
        except Exception as e:
            return jsonify({"error": f"Error saving temporary audio file: {str(e)}"}), 500

        try:
            # Check file size and duration
            file_size = os.path.getsize(temp_audio_path)
            duration = len(audio) / 1000.0  # pydub uses milliseconds

            print(f"Audio file size: {file_size} bytes")
            print(f"Audio duration: {duration} seconds")

            if duration < 0.1:
                os.remove(temp_audio_path)  # Clean up the file if it's too short
                return jsonify({"error": "Audio file is too short. Minimum audio length is 0.1 seconds."}), 400
        except Exception as e:
            return jsonify({"error": f"Error checking audio file: {str(e)}"}), 500

        try:
            transcription = transcribe_audio(temp_audio_path)
        except Exception as e:
            return jsonify({"error": f"Error in transcribe_audio: {str(e)}"}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        return jsonify({"error": f"Unexpected error in handle_transcribe_audio: {str(e)}"}), 500

QUANTUM_PLOTS_DIR = os.path.join(os.getcwd(), "quantum_plots")

# Route to serve HTML files from the quantum_plots directory
@app.route("/plots/<path:filename>")
def serve_plot(filename):
    response = send_from_directory(QUANTUM_PLOTS_DIR, filename)
    response.headers['X-Frame-Options'] = 'ALLOW-FROM http://localhost:3000'
    response.headers['Content-Security-Policy'] = "frame-ancestors 'self' http://localhost:3000"
    return response


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = '8080', debug=True)