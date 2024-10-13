import os
import openai
from prompt import *
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# Check if the API key is provided
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

client = OpenAI(api_key = OPENAI_API_KEY)

def openai_chat(user_prompt, sys_prompt):
    try:
        # Call the OpenAI API for the chat completion
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=StructuredOutput
        )
        
        response_data = completion.choices[0].message.parsed
        
        return response_data
    except Exception as e:
        return f"An error occurred: {e}"
    
def openai_chat_image(user_prompt, sys_prompt, image_urls):
    try:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "text", "text": user_prompt if user_prompt is not None else sys_prompt}]}
        ]
        
        # Iterate over the list of image URLs and add them to the messages
        for image_url in image_urls:
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": image_url}})

        completion = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return completion.choices[0].message
    except Exception as e:
        return f"An error occurred: {e}"


def process_user_prompt(user_prompt = None):
    try:  
        print(system_prompt)
        
        response = openai_chat(user_prompt, system_prompt)

        print("Json Generated:")
        print(response)
        
        if hasattr(response, 'JSON_circuit'):
            json_circuit = response.JSON_circuit  # Parse the JSON circuit
        if hasattr(response, 'Quirk_Circuit_Link'):
            quirk_circuit_link = response.Quirk_Circuit_Link
  
        print("---------------------- ---------------------- ---------------------- ----------------------")
        print(json_circuit)
        print("---------------------- ---------------------- ---------------------- ----------------------")
        print(quirk_circuit_link)
        print("---------------------- ---------------------- ---------------------- ----------------------")
        
        return quirk_circuit_link
    except Exception as e:  # Added except block to handle errors
        print(f"An error occurred in process_user_prompt: {e}")
        return {"error": str(e)}  # Return error message as part of the response
