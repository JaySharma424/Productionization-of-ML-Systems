from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model_filename = 'random_forest_model.pkl'
model = joblib.load(model_filename)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])

        # In a real-world scenario, you would perform the same preprocessing
        # on input_df as you did on your training data (X_train).
        # For this example, we assume the input JSON matches the expected column order/names.
        # If X_train is available, reindex to ensure column order and handle missing columns.

        # Placeholder for obtaining original column names if X_train isn't directly accessible here.
        # In a production setup, it's better to explicitly define the expected feature columns
        # or save X_train.columns alongside the model.
        # For now, if X_train was in the global scope of the Colab session where app.py was generated,
        # and this code is run as part of that session, it might be available.
        # However, for a standalone app.py, X_train won't be in scope.
        # A robust solution would involve saving the columns with the model or a separate file.

        # Let's assume the input_df from the API call will have the correct columns and order
        # based on the `X` dataframe used for training. If not, this is where feature engineering
        # and column reordering/filling would be done for the new input data.

        prediction = model.predict(input_df)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
