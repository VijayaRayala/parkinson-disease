from flask import Flask, request, jsonify, render_template
import cv2
from skimage import feature
import numpy as np
import joblib
import tempfile  # Add this import for tempfile support

application = Flask(__name__)

# Load RandomForestClassifier models
loaded_rf_spiral_classifier = joblib.load('rf_spiral_model.pkl')
loaded_rf_wave_classifier = joblib.load('rf_wave_model.pkl')

# Load XGBClassifier models
loaded_xgb_spiral_classifier = joblib.load('xgb_spiral_model.pkl')
loaded_xgb_wave_classifier = joblib.load('xgb_wave_model.pkl')

# Load LabelEncoder
le = joblib.load('label_encoder.pkl')  # Load the label encoder

# Define a function to quantify the image
def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

# Define a function to process the uploaded image
def process_image(image_path):
    # Load and preprocess the input image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (200, 200))
    _, image = cv2.threshold(image, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    features = quantify_image(image)  # Use the quantify_image function you defined
    return features

# Define a function to predict the label and probability
def predict_label(image_path, classifier):
    # Process the input image
    features = process_image(image_path)

    # Make predictions
    prediction = classifier.predict([features])[0]
    probability = np.max(classifier.predict_proba([features]))

    # Inverse transform the label using the label encoder
    label = le.inverse_transform([prediction])[0]

    return label, probability

@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the selected option
        selected_option = request.form['prediction_option']

        # Get the uploaded image file from the request
        uploaded_file = request.files['image']

        if uploaded_file.filename != '':
            # Save the uploaded file to a temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            uploaded_file.save(temp_file.name)

            # Make predictions based on the selected option
            if selected_option == 'spiral':
                prediction, probability = predict_label(temp_file.name, loaded_rf_spiral_classifier)
            elif selected_option == 'wave':
                prediction, probability = predict_label(temp_file.name, loaded_rf_wave_classifier)
            
            else:
                return render_template('index.html', error='Invalid option selected')

            return render_template('index.html', prediction=prediction, probability=probability)

    return render_template('index.html')

@application.route('/first')
def home():
    return 'HI  '
    
@application.route('/second')
def second():
    return 'HI  hello'

if __name__ == '__main__':
    application.run(debug=True)
