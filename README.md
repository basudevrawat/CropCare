# CropCare - Plant Disease Detection System

CropCare is an AI-powered plant disease detection system developed for the Cyfuture AI Hackathon 1.0. The application uses deep learning to identify 38 different plant diseases from leaf images, providing farmers with instant diagnosis and treatment recommendations.

## Technology Stack

- **Backend**: Python, Flask, TensorFlow
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **AI/ML**: Convolutional Neural Network (CNN) with 7.8 million parameters
- **Accessibility**: Google Text-to-Speech (gTTS), Speech Recognition
- **Translation**: Google Translate API

## Model Performance

- **Training Accuracy**: 98.19%
- **Validation Accuracy**: 96.98%
- **Dataset**: 70,295 training images across 38 classes

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/basudevrawat/CropCare.git
   cd CropCare
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Create a Kaggle account if you don't have one
   - Download the [Plant Disease Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
   - Extract the dataset folders (train, valid, test) to the project root

4. Train the model or use pre-trained:
   ```bash
   python train_model.py
   ```
   Or download a pre-trained model and rename it to `trained_plant_disease_model.keras`

5. Run the application:
   ```bash
   python app.py
   ```

6. Access the application at `http://127.0.0.1:8080`

## Team Members

- [Kaberi Acharya](https://www.linkedin.com/in/contactkaberi/)
- [Namita Devi](https://www.linkedin.com/in/namita-devi-b104a828a/)
- [Irene Therese Joseph](https://www.linkedin.com/in/irenetjoseph/)
- [Niva Rani Deka](https://www.linkedin.com/in/niva-rani-deka-846605314/)
- [Basudev Rawat](https://www.linkedin.com/in/basudevrawat/)

## Project Structure

```
cropcare/
├── model/                  # Contains the disease detection model
├── static/                 # Static files (CSS, JS, images)
├── templates/              # HTML templates
├── utils/                  # Utility functions
├── app.py                  # Main Flask application
├── train_model.py          # Script to train the model
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Future Enhancements

- Mobile application development
- Offline functionality for areas with limited connectivity
- Integration with agricultural extension services
- Expansion to more crop varieties and diseases
- Real-time camera-based detection

## Acknowledgments

- [Plant Village Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) for providing the training data
- [Google Text-to-Speech (gTTS)](https://pypi.org/project/gTTS/) for voice assistance functionality
- [Google Translate API](https://cloud.google.com/translate) for multilingual support
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for voice input processing
- [Flask](https://flask.palletsprojects.com/) for web framework
- [TensorFlow](https://www.tensorflow.org/) for machine learning capabilities
- Cyfuture AI Hackathon for the opportunity to develop this solution
