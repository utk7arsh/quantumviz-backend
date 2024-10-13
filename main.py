from flask import Flask, request, jsonify
import os
import json
import openai
from prompt import *
from circuit_gen import *
from dotenv import load_dotenv

app = Flask(__name__)


@app.route('/get_qiskit_code', methods = ['POST'])
def get_code_utkarsh():
    data = request.get_json()
    user_input = data["user_input"]
    
    
    '''some code'''
    
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