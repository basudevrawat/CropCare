import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

class PlantDiseaseModel:
    def __init__(self, model_path='trained_plant_disease_model.keras'):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        
        # Class names from the dataset
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
            'Apple___healthy', 'Blueberry___healthy', 
            'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
            'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]
        
        # Map disease classes to user-friendly descriptions and treatments
        self.disease_info = self._initialize_disease_info()
    
    def _initialize_disease_info(self):
        """Initialize disease information and treatment recommendations."""
        info = {}
        
        # Apple diseases
        info['Apple___Apple_scab'] = {
            'name': 'Apple Scab',
            'description': 'A fungal disease that causes dark, scabby lesions on leaves and fruit.',
            'treatment': 'Apply fungicides in early spring. Remove and destroy fallen leaves and fruit.'
        }
        
        info['Apple___Black_rot'] = {
            'name': 'Apple Black Rot',
            'description': 'A fungal disease causing leaf spots and fruit rot.',
            'treatment': 'Prune infected branches. Apply fungicides. Remove mummified fruits.'
        }
        
        info['Apple___Cedar_apple_rust'] = {
            'name': 'Cedar Apple Rust',
            'description': 'A fungal disease causing bright orange-yellow spots on leaves.',
            'treatment': 'Remove nearby cedar trees if possible. Apply fungicides in spring.'
        }
        
        info['Apple___healthy'] = {
            'name': 'Healthy Apple Plant',
            'description': 'This apple plant appears healthy.',
            'treatment': 'Continue regular care and monitoring.'
        }
        
        # Add more disease information for other plants...
        # For brevity, I'm only including some examples
        
        # Tomato diseases
        info['Tomato___Early_blight'] = {
            'name': 'Tomato Early Blight',
            'description': 'A fungal disease causing dark spots with concentric rings on lower leaves.',
            'treatment': 'Remove infected leaves. Apply fungicide. Mulch around plants.'
        }
        
        info['Tomato___Late_blight'] = {
            'name': 'Tomato Late Blight',
            'description': 'A serious fungal disease causing large, dark blotches on leaves and fruit rot.',
            'treatment': 'Apply copper-based fungicide. Remove infected plants. Avoid overhead watering.'
        }
        
        info['Tomato___healthy'] = {
            'name': 'Healthy Tomato Plant',
            'description': 'This tomato plant appears healthy.',
            'treatment': 'Continue regular care and monitoring.'
        }
        
        # Default for any disease not specifically defined
        for class_name in self.class_names:
            if class_name not in info:
                plant, condition = class_name.split('___')
                if condition == 'healthy':
                    info[class_name] = {
                        'name': f'Healthy {plant} Plant',
                        'description': f'This {plant.lower()} plant appears healthy.',
                        'treatment': 'Continue regular care and monitoring.'
                    }
                else:
                    info[class_name] = {
                        'name': f'{plant} {condition.replace("_", " ")}',
                        'description': f'A disease affecting {plant.lower()} plants.',
                        'treatment': 'Consult with a local agricultural expert for specific treatment recommendations.'
                    }
        
        return info
    
    def preprocess_image(self, image_path=None, image=None):
        """
        Preprocess the image for model prediction.
        Accept either a file path or an image object.
        """
        if image_path:
            # Load and preprocess image from path
            img = load_img(image_path, target_size=(128, 128))
            img_array = img_to_array(img)
        elif image is not None:
            # Preprocess the provided image
            img = cv2.resize(image, (128, 128))
            img_array = img_to_array(img)
        else:
            raise ValueError("Either image_path or image must be provided")
        
        # Convert to batch format
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path=None, image=None):
        """
        Predict the disease from an image.
        Return the disease class, name, description, and treatment.
        """
        # Preprocess the image
        img_array = self.preprocess_image(image_path, image)
        
        # Make prediction
        predictions = self.model.predict(img_array)
        
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions[0])
        
        # Get the class name
        predicted_class = self.class_names[predicted_class_index]
        
        # Get the confidence score
        confidence = float(predictions[0][predicted_class_index])
        
        # Get disease information
        disease_info = self.disease_info.get(predicted_class, {
            'name': predicted_class,
            'description': 'No detailed information available.',
            'treatment': 'Consult with a local agricultural expert.'
        })
        
        return {
            'class': predicted_class,
            'name': disease_info['name'],
            'description': disease_info['description'],
            'treatment': disease_info['treatment'],
            'confidence': confidence
        } 