#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# Load trained Random Forest model
model = joblib.load("rf_model.pkl")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({"message": "🎯 Random Forest API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        features = data['features']  # Expecting a list of numerical features
        input_array = np.array(features).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_array)[0]

        return jsonify({'prediction': str(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)


# In[2]:


get_ipython().system('jupyter nbconvert --to script app.ipynb')


# In[ ]:




