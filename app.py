import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from jinja2 import Template

import webbrowser

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

X_train = pd.read_csv('./filename.csv')


@app.route('/')
def home():
    return "bonjour"

@app.route('/explanation_api',methods=['POST'])
def image():
    data = request.get_json()
    webbrowser.open('http://127.0.0.1:5500/' + data['name'])
    return 'good'


@app.route('/explanations_api',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    data = request.get_json(force=True)
    data = [[0.,  4.,  1.,  4.,  4.,  1.,  9.,  9.,  9.,  9.,  9.,  2.,  9.,
        3.,  4.,  4.,  9.,  9.,  9., 19.]]

    target_features = ['RIFAMPICIN', 'ISONIAZIDA', 'ETAMBUTOL',
        'ESTREPTOMI', 'PIRAZINAMI', 'ETIONAMIDA', 'OUTRAS']
    
    X_train_cols = ['CS_SEXO', 'CS_RACA', 'TRATAMENTO', 'RAIOX_TORA', 'TESTE_TUBE', 'FORMA',
        'AGRAVAIDS', 'AGRAVALCOO', 'AGRAVDIABE', 'AGRAVDOENC', 'AGRAVOUTRA',
        'BACILOSC_E', 'BACILOS_E2', 'BACILOSC_O', 'CULTURA_ES', 'HIV',
        'DOENCA_TRA', 'AGRAVDROGA', 'AGRAVTABAC', 'IDADE']

    # LIME
    explainers = []
    explanations = []
    list_names = []

    base_classifier = RandomForestClassifier(n_estimators=5, max_depth=5)

    print('start exe ======>')
    # Create a separate explainer for each label
    for label_idx, label in  enumerate(target_features):
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values.astype(int), mode='classification', feature_names = X_train_cols, discretize_continuous=True)
        # explain_fn = lambda x: base_classifier.predict_proba(x)[label_idx]
        explain_fn = lambda x: model.predict_proba(x)[label_idx]
        # explanations.append((label, explainer.explain_instance(data, explain_fn, num_features=len(X_train_cols))))
        # explanations.append((label, explainer.explain_instance(X_train.iloc[0], explain_fn, num_features=len(X_train_cols))))
        # explanations.append((label, explainer.explain_instance(X_train.iloc[0], explain_fn, num_features=len(X_train_cols))))
        exp = explainer.explain_instance(X_train.iloc[0], explain_fn, num_features=len(X_train_cols))
        explanations.append(exp)
        name = 'explanation-' + label + '.html'
        exp.save_to_file(name)
        list_names.append(name)

    print(list_names)
    return json.dumps(list(list_names))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json()
    print('data:', data)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    print(output)
    return json.dumps(list(output))
    # return data

if __name__ == "__main__":
    app.run(debug=True, port=8000)