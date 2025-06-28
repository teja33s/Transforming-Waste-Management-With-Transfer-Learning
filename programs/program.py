import numpy as np
import os

# Initialize Flask app
app = Flask(_name_)
UPLOAD_FOLDER = 'static/assets/forms/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model('model/waste_classifier.h5')


# Class labels (edit as per your model's output)
class_labels = ['Biodegradable', 'Non-Biodegradable']

# Image preprocessing and prediction
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    label_index = np.argmax(preds)
    confidence = round(preds[label_index] * 100, 2)

    return class_labels[label_index], confidence, preds.tolist()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result, confidence, probs = model_predict(filepath)

        return render_template(
            'result.html',
            result=result,
            confidence=confidence,
            image_file=filename,
            class_names=class_labels,
            class_probs=probs
        )

# Start app
if _name_ == '_main_':
    app.run(debug=True)
