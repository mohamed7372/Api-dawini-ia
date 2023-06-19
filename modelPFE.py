import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# data = {'experience':45, 'test_score':4, 'interview_score':1}

# prediction = model.predict([np.array(list(data.values()))])

data = {
        "CS_SEXO":        0.0,
        "CS_RACA":     4.0,
        "TRATAMENTO":  1.0,
        "RAIOX_TORA":  4.0,
        "TESTE_TUBE":  4.0,
        "FORMA":       1.0,
        "AGRAVAIDS":   9.0,
        "AGRAVALCOO":  9.0,
        "AGRAVDIABE":  9.0,
        "AGRAVDOENC":  9.0,
        "AGRAVOUTRA":  9.0,
        "BACILOSC_E":  2.0,
        "BACILOS_E2":  9.0,
        "BACILOSC_O":  3.0,
        "CULTURA_ES":  4.0,
        "HIV":         4.0,
        "DOENCA_TRA":  9.0,
        "AGRAVDROGA":  9.0,
        "AGRAVTABAC":  9.0,
        "IDADE":      19.0
    }
prediction = model.predict([np.array(list(data.values()))])


# data = [[ 0.,  4.,  1.,  4.,  4.,  1.,  9.,  9.,  9.,  9.,  9.,  2.,  9.,
#     3.,  4.,  4.,  9.,  9.,  9., 19.]]
# prediction = model.predict(data)

output = prediction[0]

print(output)