from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import threading
import time
import base64
import numpy as np
import os
from dotenv import load_dotenv
import requests
import json
import pyaudio
import wave
import webrtcvad
import warnings
# Suppress audioop deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)
import audioop
import tempfile
from openai import OpenAI
from io import BytesIO
from PIL import Image
import queue
import socketio

# Load environment variables
load_dotenv()

# Flask ve SocketIO yapılandırması
flask_app = Flask(__name__)
sio = socketio.Server(async_mode='threading', cors_allowed_origins='*')
app = socketio.WSGIApp(sio, flask_app)

# Global variables
camera = None
lock = threading.Lock()
frame_buffer = None
last_processed_time = 0
processing_interval = 3  # Process frames every 3 seconds
is_listening = True      # Start with listening active
is_speaking = False      # Is the assistant currently speaking
audio_queue = queue.Queue()
RATE = 16000
CHUNK = 320  # 20ms at 16kHz (WebRTC VAD requires 10, 20, or 30 ms frames)
FORMAT = pyaudio.paInt16
CHANNELS = 1
VAD_MODE = 3  # VAD aggressiveness (0-3)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Voice Activity Detection
vad = webrtcvad.Vad(VAD_MODE)

# Audio processing thread variables
stop_listening_event = threading.Event()
pause_listening_event = threading.Event()
audio_processing_thread = None

# Keep track of the latest TTS audio file
latest_tts_file = None

def initialize_camera():
    global camera
    camera = cv2.VideoCapture(0)  # 0 is usually the default camera
    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return False
    return True

def generate_frames():
    global frame_buffer
    while True:
        with lock:
            if camera is None or not camera.isOpened():
                if not initialize_camera():
                    time.sleep(1)
                    continue

            success, frame = camera.read()
            if not success:
                print("Failed to read frame")
                time.sleep(0.1)
                continue

            # Store the current frame in buffer for processing
            frame_buffer = frame.copy()
            
            # Convert to jpg for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def analyze_frame_with_gpt(frame, prompt="What can you see in this image? If there's text, please read it."):
    """Send the frame to OpenAI's API for analysis"""
    try:
        # Convert the OpenCV frame to a format suitable for the API
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            return "Error encoding image"
        
        # Convert to base64 string
        image_binary = BytesIO(buffer).getvalue()
        
        # Send to OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using gpt-4o which supports vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(image_binary).decode('utf-8')}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return f"Error analyzing frame: {str(e)}"

def text_to_speech_openai(text, output_file=None):
    """Convert text to speech using OpenAI's TTS API"""
    global latest_tts_file
    
    try:
        # If no output file specified, create a temporary file
        if not output_file:
            temp_dir = "static/audio"
            # Create directory if it doesn't exist
            os.makedirs(temp_dir, exist_ok=True)
            output_file = f"{temp_dir}/tts_{int(time.time())}.mp3"
        
        # Call OpenAI TTS API
        response = client.audio.speech.create(
            model="tts-1", 
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )

        # Save the audio file
        response.stream_to_file(output_file)
        
        # Update the latest TTS file for frontend access
        latest_tts_file = output_file
        
        return output_file
    except Exception as e:
        print(f"Error in text to speech: {e}")
        return None

def audio_processing_worker():
    """Background thread to continuously process audio"""
    global is_listening, is_speaking
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    print("Starting continuous voice detection...")
    
    # Variables for voice detection
    local_is_speaking = False
    silent_chunks = 0
    voice_chunks = 0
    max_silent_chunks = 50  # ~1 second of silence (50 * 20ms)
    min_voice_chunks = 5    # At least ~100ms of voice to be considered speaking
    accumulated_audio = []
    
    try:
        while not stop_listening_event.is_set():
            # Skip processing if we're paused (assistant is speaking)
            if pause_listening_event.is_set():
                time.sleep(0.1)  # Short sleep to avoid CPU spike
                continue
                
            audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
            
            # Check if this chunk contains speech
            try:
                # Convert to proper format for WebRTC VAD
                is_speech = vad.is_speech(audio_chunk, RATE)
            except Exception as e:
                # If WebRTC VAD fails, fallback to energy-based detection
                rms = audioop.rms(audio_chunk, 2)  # Get RMS of the audio chunk
                is_speech = rms > 500  # Threshold for speech detection
            
            # State machine for speech detection
            if is_speech and not local_is_speaking:
                # Potential start of speech
                voice_chunks += 1
                accumulated_audio.append(audio_chunk)  # Start saving audio
                
                if voice_chunks >= min_voice_chunks:
                    print("Speech started")
                    local_is_speaking = True
            
            elif is_speech and local_is_speaking:
                # Continuing speech
                voice_chunks += 1
                silent_chunks = 0
                accumulated_audio.append(audio_chunk)
            
            elif not is_speech and local_is_speaking:
                # Potential end of speech
                silent_chunks += 1
                accumulated_audio.append(audio_chunk)
                
                if silent_chunks >= max_silent_chunks:
                    print(f"Speech ended, processing... (collected {len(accumulated_audio)} chunks)")
                    
                    # Process the accumulated audio if we have enough
                    if len(accumulated_audio) > 10:  # Minimum threshold
                        # Convert list of chunks to single bytes object
                        audio_data = b''.join(accumulated_audio)
                        
                        # Pause listening while processing - will be resumed after TTS finishes
                        pause_listening_event.set()
                        is_speaking = True
                        # Notify frontend that we're processing speech
                        sio.emit('listening_status', {'is_listening': False, 'reason': 'processing'})
                        
                        # Process in a separate thread
                        threading.Thread(target=process_speech, args=(audio_data,)).start()
                    
                    # Reset state
                    local_is_speaking = False
                    silent_chunks = 0
                    voice_chunks = 0
                    accumulated_audio = []
            
            elif not is_speech and not local_is_speaking:
                # No speech detected
                voice_chunks = 0
                # Keep a small buffer for better detection
                if len(accumulated_audio) > 10:
                    accumulated_audio = accumulated_audio[-10:]
                
    except Exception as e:
        print(f"Error in audio processing: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Stopped continuous voice detection")

def process_speech(audio_data):
    """Process detected speech and get response"""
    global is_speaking
    
    try:
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_audio_path = temp_file.name
            
        # Write WAV file with proper headers
        with wave.open(temp_audio_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit audio = 2 bytes
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        
        # Send to Whisper API
        print("Sending to Whisper API...")
        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=file,
                language="tr"  # Specify Turkish language
            )
        
        query_text = transcription.text
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        if not query_text or query_text.strip() == "":
            print("Empty transcription")
            # Resume listening
            resume_listening()
            return
            
        print(f"Transcribed: {query_text}")
        
        # Get current frame for analysis
        with lock:
            if frame_buffer is None:
                print("No frame available")
                # Resume listening
                resume_listening()
                return
            
            frame_to_analyze = frame_buffer.copy()
        
        # Analyze the frame with the query
        print("Sending to GPT-4o for analysis...")
        answer = analyze_frame_with_gpt(frame_to_analyze, query_text)
        print(f"GPT Response: {answer}")
        
        # Generate speech with OpenAI's TTS
        tts_file = text_to_speech_openai(answer)
        
        # Send result to frontend
        audio_queue.put({
            "type": "conversation",
            "query": query_text,
            "answer": answer,
            "audio_file": tts_file
        })
        
        # We'll keep listening paused until the frontend tells us audio playback is complete
        
    except Exception as e:
        print(f"Error in speech processing: {e}")
        audio_queue.put({
            "type": "error",
            "message": str(e)
        })
        # Resume listening in case of error
        resume_listening()

def resume_listening():
    """Resume listening after TTS is done playing"""
    global is_speaking
    is_speaking = False
    pause_listening_event.clear()
    # Notify frontend that we're listening again
    sio.emit('listening_status', {'is_listening': True, 'reason': 'tts_complete'})

def start_listening():
    """Start the continuous listening thread"""
    global audio_processing_thread, stop_listening_event, is_listening
    
    if audio_processing_thread is not None and audio_processing_thread.is_alive():
        print("Already listening")
        return False
    
    stop_listening_event.clear()
    pause_listening_event.clear()
    audio_processing_thread = threading.Thread(target=audio_processing_worker)
    audio_processing_thread.daemon = True
    audio_processing_thread.start()
    is_listening = True
    return True

def stop_listening():
    """Stop the continuous listening thread"""
    global audio_processing_thread, stop_listening_event, is_listening
    
    if audio_processing_thread is None or not audio_processing_thread.is_alive():
        print("Not currently listening")
        return False
    
    stop_listening_event.set()
    if audio_processing_thread.is_alive():
        audio_processing_thread.join(timeout=1.0)
    is_listening = False
    return True

def pause_listening():
    """Temporarily pause listening"""
    global is_listening
    pause_listening_event.set()
    is_listening = False

# Socket.IO event handlers
@sio.event
def connect(sid, environ):
    print(f"Client connected: {sid}")

@sio.event
def disconnect(sid):
    print(f"Client disconnected: {sid}")
    
@sio.event
def audio_completed(sid, data):
    """Client notifies server that TTS audio playback is complete"""
    print("Audio playback complete, resuming listening")
    resume_listening()

# Flask routes - now using flask_app instead of app
@flask_app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@flask_app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@flask_app.route('/analyze_current_frame', methods=['POST'])
def analyze_current_frame():
    """Analyze the current frame with GPT-4 Vision"""
    global frame_buffer, last_processed_time
    
    current_time = time.time()
    
    # Rate limiting to prevent too frequent API calls
    if current_time - last_processed_time < processing_interval:
        return jsonify({"error": "Please wait before sending another request"}), 429
    
    with lock:
        if frame_buffer is None:
            return jsonify({"error": "No frame available"}), 404
        
        # Make a copy to avoid thread issues
        frame_to_analyze = frame_buffer.copy()
    
    # Update the last processed time
    last_processed_time = current_time
    
    # Pause listening while processing
    pause_listening()
    is_speaking = True
    sio.emit('listening_status', {'is_listening': False, 'reason': 'processing'})
    
    # Analyze the frame
    analysis = analyze_frame_with_gpt(frame_to_analyze)
    
    # Generate speech with OpenAI's TTS
    tts_file = text_to_speech_openai(analysis)
    
    return jsonify({
        "analysis": analysis,
        "audio_file": tts_file
    })

@flask_app.route('/toggle_listening', methods=['POST'])
def toggle_listening_route():
    """Toggle continuous listening mode"""
    global is_listening
    
    if is_listening:
        success = stop_listening()
    else:
        success = start_listening()
    
    return jsonify({
        "success": success,
        "is_listening": is_listening
    })

@flask_app.route('/get_audio_queue', methods=['GET'])
def get_audio_queue():
    """Get audio processing results from queue (used for polling)"""
    try:
        # Non-blocking queue get with timeout
        data = audio_queue.get(block=False)
        return jsonify(data)
    except queue.Empty:
        return jsonify({"type": "empty"})
    except Exception as e:
        return jsonify({"type": "error", "message": str(e)})

@flask_app.route('/static/audio/<filename>')
def serve_audio(filename):
    """Serve the generated audio files"""
    return send_file(f"static/audio/{filename}")

@flask_app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up resources before closing"""
    global camera
    try:
        # Stop listening if active
        if is_listening:
            stop_listening()
            
        with lock:
            if camera is not None and camera.isOpened():
                camera.release()
                camera = None
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create static/audio directory if it doesn't exist
    os.makedirs("static/audio", exist_ok=True)
    
    initialize_camera()
    # Start listening automatically when the app starts
    start_listening()
    
    # Use a regular WSGI server instead of Flask's development server
    from waitress import serve
    print("Starting server on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)