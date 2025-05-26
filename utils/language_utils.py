import os
import speech_recognition as sr
from gtts import gTTS
from googletrans import Translator

class LanguageHelper:
    def __init__(self):
        self.translator = Translator()
        self.recognizer = sr.Recognizer()
        
        # Supported languages with their codes
        self.supported_languages = {
            'English': 'en',
            'Hindi': 'hi',
            'Bengali': 'bn',
            'Telugu': 'te',
            'Marathi': 'mr',
            'Tamil': 'ta',
            'Urdu': 'ur',
            'Gujarati': 'gu',
            'Kannada': 'kn',
            'Malayalam': 'ml',
            'Punjabi': 'pa'
        }
    
    def translate_text(self, text, target_language_code='en'):
        """
        Translate text to the target language.
        """
        try:
            translation = self.translator.translate(text, dest=target_language_code)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text if translation fails
    
    def text_to_speech(self, text, language_code='en', output_file='static/audio/output.mp3'):
        """
        Convert text to speech and save to an audio file.
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Create gTTS object
            tts = gTTS(text=text, lang=language_code, slow=False)
            
            # Save to file
            tts.save(output_file)
            return output_file
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return None
    
    def speech_to_text(self, audio_file=None):
        """
        Convert speech from microphone or audio file to text.
        """
        try:
            text = ""
            
            if audio_file:
                # Use audio file
                with sr.AudioFile(audio_file) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
            else:
                # Use microphone
                with sr.Microphone() as source:
                    print("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio_data = self.recognizer.listen(source)
                    text = self.recognizer.recognize_google(audio_data)
            
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition service error: {e}"
        except Exception as e:
            print(f"Speech-to-text error: {e}")
            return "An error occurred during speech recognition"
    
    def get_supported_languages(self):
        """
        Return a list of supported languages.
        """
        return self.supported_languages 