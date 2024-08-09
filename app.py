import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from scikeras.wrappers import KerasRegressor
import ast
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import subprocess
import re
import json 
import Flask 

app = Flask(__name__)

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
regressor = KerasRegressor(model=create_model, verbose=0)
regression_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', regressor)
])
regression_pipeline.fit(X_train, y_train_reg)


def extract_functions_from_diff(diff):
    functions = []
    function_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(', re.MULTILINE)
    matches = function_pattern.findall(diff)
    for indent, func_name in matches:
        functions.append(func_name)
    return functions

def extract_function_calls(diff):
    calls = []
    call_pattern = re.compile(r'\b(\w+)\(')
    matches = call_pattern.findall(diff)
    for func_call in matches:
        calls.append(func_call)
    return calls


def identify_impacted_functions(modified_functions, commit_files):
    impacted_functions = set(modified_functions)

    for file in commit_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file, 'r', encoding='latin1') as f:
                content = f.read()

        for function in modified_functions:
            if function in content:
                calls = extract_function_calls(content)
                impacted_functions.update(calls)

    return list(impacted_functions)


# def identify_impacted_functions(modified_functions, commit_files):
#     impacted_functions = set(modified_functions)
#
#     for file in commit_files:
#         with open(file, 'r') as f:
#             content = f.read()
#             for function in modified_functions:
#                 if function in content:
#                     calls = extract_function_calls(content)
#                     impacted_functions.update(calls)
#
#     return list(impacted_functions)

def get_commit_data():
    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    commit_message = subprocess.check_output(['git', 'log', '-1', '--pretty=%B']).strip().decode('utf-8')
    commit_author = subprocess.check_output(['git', 'log', '-1', '--pretty=%an']).strip().decode('utf-8')
    commit_date = subprocess.check_output(['git', 'log', '-1', '--pretty=%ad', '--date=iso']).strip().decode('utf-8')

    commit_files = subprocess.check_output(['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash]).strip().decode('utf-8').split('\n')
    commit_files = [f.strip() for f in commit_files if f.strip()]
    modified_functions = []
    for file in commit_files:
        diff_bytes = subprocess.check_output(['git', 'diff', '--unified=0', commit_hash + '^', commit_hash, '--', file])
        diff = diff_bytes.decode('utf-8', errors='replace')
        functions = extract_functions_from_diff(diff)
        modified_functions.extend(functions)


    # for file in commit_files:
    #     diff = subprocess.check_output(['git', 'diff', '--unified=0', commit_hash + '^', commit_hash, '--', file]).decode('utf-8')
    #     functions = extract_functions_from_diff(diff)
    #     modified_functions.extend(functions)

    impacted_functions = identify_impacted_functions(modified_functions, commit_files)

    new_data = {
        "commit": commit_hash,
        "message": commit_message,
        "Author": commit_author,
        "Date": commit_date.split(' ')[0],
        "functions": modified_functions,
        "impacted_functions": impacted_functions
    }

    df = pd.DataFrame([new_data])
    
    return df
@app.route('/predict', methods=['POST'])
def predict():
    df = get_commit_data()

    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop('Date', axis=1)
    df['functions'] = df['functions'].apply(lambda x: ' '.join(x) if isinstance(x, list) else 'aucune fonction n\'est modifiée')
    df['impacted_functions'] = df['impacted_functions'].apply(lambda x: ' '.join(x) if isinstance(x, list) else 'aucune fonction n\'est modifiée')

    df['functions'] = df['functions'].fillna('aucune fonction n\'est modifiée')
    df['impacted_functions'] = df['impacted_functions'].fillna('aucune fonction n\'est modifiée')

    y_pred_class = classification_pipeline.predict(df)
    y_pred_reg = regression_pipeline.predict(df)

    response = {
        'Classification': y_pred_class.tolist(),
        'Regression Probability': y_pred_reg.tolist(),
        'Modified_functions': df['functions'].tolist(),
        'Impacted_functions': df['impacted_functions'].tolist()
    }
    
    print(json.dumps(response))

# if __name__ == '__main__':
#     predict()

if __name__ == '__main__':
    app.run(debug=True)
