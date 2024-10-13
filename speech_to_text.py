import os
import pyaudio
import wave
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key= OPENAI_API_KEY)

# New code to record audio from the microphone
def record_audio(filename, duration=20):
    chunk = 1024  # Record in chunks of 1024 samples
    format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Single channel for microphone
    rate = 44100  # Sample rate
    p = pyaudio.PyAudio()

    stream = p.open(format=format, channels=channels,
                    rate=rate, input=True,
                    frames_per_buffer=chunk)

    print("Recording...")
    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


def transcribe_audio(audio_filename):
    # Open the recorded audio file for transcription
    audio_file = open(audio_filename, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"
    )
    
    return transcription



if __name__ == "__main__":
    audio_filename = "./audio_files/recorded_audio.wav"
    record_audio(audio_filename, duration=20)
    voice = transcribe_audio(audio_filename)
    print(voice)
    