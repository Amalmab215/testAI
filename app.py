from flask import Flask, request, jsonify
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from scikeras.wrappers import KerasRegressor
import ast
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import subprocess
import os

#app = Flask(__name__)

# Charger les données d'entraînement
data = pd.read_csv('data.csv')
data['functions'] = data['functions'].fillna('aucune fonction n\'est modifiée')

# Sélection des features et de la cible
features = ['Year', 'Month', 'Day', 'Author', 'message', 'functions']
target_classification = 'Classification'
target_regression = 'is_bug'

# Séparation des données en ensembles d'entraînement et de test
X = data[features]
y_classification = data[target_classification]
y_regression = data[target_regression]

X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

# Charger les modèles
classification_pipeline = joblib.load('Regression_logistique.pkl')
preprocessor = joblib.load('preprocessor.pkl')

def create_model():
    # Charger le modèle préalablement sauvegardé
    model = load_model('NeuralNetwork_trained.h5')
    # Recompiler le modèle avec un nouvel optimiseur
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

regressor = KerasRegressor(build_fn=create_model, verbose=0)

regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])

regression_pipeline.fit(X_train, y_train_reg)

#@app.route('/predict', methods=['POST'])
def predict():
    #data = request.get_json()
    new_data = [
    {
    
        "commit": "abc123",
        "message": "Drop labels in Image and Audio folders",
        "functions": "[\"login\", \"authenticate\", \"session\"]",
        "User": "JohnDoe",
        "Author": "John Doe",
        "Created At": "2023-07-01 12:34:56",
        "Updated At": "2023-07-02 12:34:56",
        "Labels": "bug",
        "State": "closed",
        "Duration": "2 days",
        "Date": "2024-07-20"
    }
]

    df = pd.DataFrame(new_data)
    
    # Prétraiter les données si nécessaire
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop('Date', axis=1)
    df['functions'] = df['functions'].apply(lambda x: ' '.join(ast.literal_eval(x)))
    df['functions'] = df['functions'].fillna('aucune fonction n\'est modifiée')


    # Prédiction de classification
    y_pred_class = classification_pipeline.predict(df)

    # Prédiction de régression
    y_pred_reg = regression_pipeline.predict(df)

    response = jsonify({
        'classification': y_pred_class.tolist(),
        'regression': y_pred_reg.tolist(),
        'modified_functions': df['functions'].tolist()
    })
    
#     # Stop the server
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func:
#         func()

#     return response

# if __name__ == '__main__':
#     app.run(debug=True)
predict()
