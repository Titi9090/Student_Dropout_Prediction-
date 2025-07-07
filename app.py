#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Secure CORS setup — only allow specific origins
CORS(app, resources={
    r"/predict": {"origins": ["http://localhost:3000", "https://your-app-frontend.com"]}
})

model = joblib.load("rf_model.pkl")

@app.route('/')
def home():
    return jsonify({"message": "Model API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(debug=False)  # Set debug=False in production

CORS(app, resources={
    r"/predict": {
        "origins": ["https://your-app.com"],
        "methods": ["POST"],
        "allow_headers": ["Content-Type"]
    }
})


# In[ ]:


get_ipython().system('jupyter nbconvert --to script app.ipynb')


# In[ ]:




