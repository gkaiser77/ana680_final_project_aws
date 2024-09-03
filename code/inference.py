import os
import joblib
import numpy as np
import json

def model_fn(model_dir):
    """
    Load the model from the model directory.
    """
    print("Loading model.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, "car_evaluation_model.joblib"))
    return loaded_model

def input_fn(request_body, content_type='application/json'):
    """
    Deserialize the input data.
    """
    if content_type == 'application/json':
        data = json.loads(request_body)
        # Convert to numpy array if needed
        return np.array(data)
    else:
        raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(input_object, model):
    """
    Make predictions using the loaded model.
    """
    print("Calling model to make predictions.")
    predictions = model.predict(input_object)
    return predictions

def output_fn(predictions, content_type='application/json'):
    """
    Serialize the predictions.
    """
    if content_type == 'application/json':
        # Convert predictions to list if needed
        return json.dumps(predictions.tolist())
    else:
        raise ValueError(f'Unsupported content type: {content_type}')
