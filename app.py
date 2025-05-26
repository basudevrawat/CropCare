import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
import tensorflow as tf

from model import PlantDiseaseModel
from utils import LanguageHelper

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'plant_disease_detection_key'

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/audio', exist_ok=True)

# Initialize model and language helper
model = PlantDiseaseModel()
language_helper = LanguageHelper()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Set default language if not set
    if 'language' not in session:
        session['language'] = 'English'
        session['language_code'] = 'en'
    
    # Get supported languages
    languages = language_helper.get_supported_languages()
    
    return render_template('index.html', 
                           languages=languages, 
                           current_language=session.get('language', 'English'))

@app.route('/set_language', methods=['POST'])
def set_language():
    language_name = request.form.get('language')
    languages = language_helper.get_supported_languages()
    
    if language_name in languages:
        session['language'] = language_name
        session['language_code'] = languages[language_name]
    
    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = model.predict(image_path=filepath)
        
        # Translate the results based on the selected language
        language_code = session.get('language_code', 'en')
        if language_code != 'en':
            result['name'] = language_helper.translate_text(result['name'], language_code)
            result['description'] = language_helper.translate_text(result['description'], language_code)
            result['treatment'] = language_helper.translate_text(result['treatment'], language_code)
        
        # Generate audio for the results
        audio_text = f"{result['name']}. {result['description']}. Treatment: {result['treatment']}"
        audio_file = language_helper.text_to_speech(audio_text, language_code)
        
        return render_template('result.html', 
                              result=result, 
                              image_file=filepath.replace('\\', '/'),
                              audio_file=audio_file.replace('\\', '/'),
                              languages=language_helper.get_supported_languages(),
                              current_language=session.get('language', 'English'))
    
    return redirect(url_for('index'))

@app.route('/voice_input')
def voice_input():
    return render_template('voice_input.html',
                          languages=language_helper.get_supported_languages(),
                          current_language=session.get('language', 'English'))

@app.route('/process_voice', methods=['POST'])
def process_voice():
    if 'audio_data' in request.files:
        audio_file = request.files['audio_data']
        temp_audio = os.path.join('static/audio', 'input.wav')
        audio_file.save(temp_audio)
        
        # Process the audio to text
        text = language_helper.speech_to_text(temp_audio)
        
        return jsonify({'status': 'success', 'text': text})
    
    return jsonify({'status': 'error', 'message': 'No audio data received'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 