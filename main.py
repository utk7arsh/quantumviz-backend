from flask import Flask, request, jsonify
import os
import json
import openai
from dotenv import load_dotenv

app = Flask(__name__)


@app.route('/get_qiskit_code', methods = ['POST'])
def get_code_utkarsh():
    data = request.get_json()
    user_input = data["user_input"]
    
    
    '''some code'''