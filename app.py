from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('Daily_Average_Losis_Data.csv')
input_columns = ['NORMA AMPAS PRESS', 'LOSIS % AMPAS PRESS', 'NORMA BIJI', 'LOSIS % BIJI',
                 'NORMA TANDAN KOSONG', 'LOSIS % TANDAN KOSONG', 'NORMA DRAB AKHIR',
                 'LOSIS % DRAB AKHIR', 'NORMA SOLID DECANTER', 'LOSIS % SOLID DECANTER']
output_columns = ['NORMA PKS PER DATE', 'LOSIS PKS']

X = data[input_columns]
y = data[output_columns]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Neural Network Model
def create_nn_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(len(input_columns),)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

nn_model = create_nn_model()
nn_model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, verbose=0)

# Fuzzy Logic System
def create_fuzzy_system():
    fuzzy_params = {
        'Ampas Press': {'norm': 0.52, 'mf': {'Green': [0, 0, 0.52, 0.65], 'Orange': [0.52, 0.65, 0.78, 0.91], 'Red': [0.78, 0.91, 1.04, 1.04]}},
        'Tandan Kosong': {'norm': 0.44, 'mf': {'Green': [0, 0, 0.44, 0.55], 'Orange': [0.44, 0.55, 0.66, 0.77], 'Red': [0.66, 0.77, 0.88, 0.88]}},
        'Biji': {'norm': 0.10, 'mf': {'Green': [0, 0, 0.10, 0.125], 'Orange': [0.10, 0.125, 0.15, 0.175], 'Red': [0.15, 0.175, 0.20, 0.20]}},
        'Drab Akhir': {'norm': 0.26, 'mf': {'Green': [0, 0, 0.26, 0.325], 'Orange': [0.26, 0.325, 0.39, 0.455], 'Red': [0.39, 0.455, 0.52, 0.52]}},
        'Solid Decanter': {'norm': 0.07, 'mf': {'Green': [0, 0, 0.07, 0.0875], 'Orange': [0.07, 0.0875, 0.105, 0.1225], 'Red': [0.105, 0.1225, 0.14, 0.14]}}
    }

    input_vars = {}
    for param, details in fuzzy_params.items():
        universe = np.linspace(0, details['mf']['Red'][3], 100)
        input_vars[param] = ctrl.Antecedent(universe, param)
        input_vars[param]['Green'] = fuzz.trapmf(input_vars[param].universe, details['mf']['Green'])
        input_vars[param]['Orange'] = fuzz.trapmf(input_vars[param].universe, details['mf']['Orange'])
        input_vars[param]['Red'] = fuzz.trapmf(input_vars[param].universe, details['mf']['Red'])

    norma_pks = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'NORMA PKS PER DATE')
    losis_pks = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'LOSIS PKS')

    norma_pks['low'] = fuzz.trimf(norma_pks.universe, [0, 0, 0.5])
    norma_pks['medium'] = fuzz.trimf(norma_pks.universe, [0, 0.5, 1])
    norma_pks['high'] = fuzz.trimf(norma_pks.universe, [0.5, 1, 1])

    losis_pks['low'] = fuzz.trimf(losis_pks.universe, [0, 0, 0.5])
    losis_pks['medium'] = fuzz.trimf(losis_pks.universe, [0, 0.5, 1])
    losis_pks['high'] = fuzz.trimf(losis_pks.universe, [0.5, 1, 1])

    rules = [
        ctrl.Rule(input_vars['Ampas Press']['Green'], (norma_pks['high'], losis_pks['low'])),
        ctrl.Rule(input_vars['Ampas Press']['Red'], (norma_pks['low'], losis_pks['high'])),
        ctrl.Rule(input_vars['Tandan Kosong']['Green'], (norma_pks['high'], losis_pks['low'])),
        ctrl.Rule(input_vars['Tandan Kosong']['Red'], (norma_pks['low'], losis_pks['high'])),
        ctrl.Rule(input_vars['Biji']['Green'], (norma_pks['high'], losis_pks['low'])),
        ctrl.Rule(input_vars['Biji']['Red'], (norma_pks['low'], losis_pks['high'])),
        ctrl.Rule(input_vars['Drab Akhir']['Green'], (norma_pks['high'], losis_pks['low'])),
        ctrl.Rule(input_vars['Drab Akhir']['Red'], (norma_pks['low'], losis_pks['high'])),
        ctrl.Rule(input_vars['Solid Decanter']['Green'], (norma_pks['high'], losis_pks['low'])),
        ctrl.Rule(input_vars['Solid Decanter']['Red'], (norma_pks['low'], losis_pks['high']))
    ]

    fz_ctrl = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(fz_ctrl)

fuzzy_system = create_fuzzy_system()

def hybrid_predict_with_time(nn_model, fuzzy_system, input_data, hours_ahead):
    adjusted_input_data = input_data.copy()
    
    nn_input = scaler_X.transform(adjusted_input_data)
    nn_pred = nn_model.predict(nn_input)
    nn_pred = scaler_y.inverse_transform(nn_pred)

    fuzzy_input = {
        'Ampas Press': adjusted_input_data['LOSIS % AMPAS PRESS'].iloc[0],
        'Tandan Kosong': adjusted_input_data['LOSIS % TANDAN KOSONG'].iloc[0],
        'Biji': adjusted_input_data['LOSIS % BIJI'].iloc[0],
        'Drab Akhir': adjusted_input_data['LOSIS % DRAB AKHIR'].iloc[0],
        'Solid Decanter': adjusted_input_data['LOSIS % SOLID DECANTER'].iloc[0]
    }

    for param, value in fuzzy_input.items():
        fuzzy_system.input[param] = value

    try:
        fuzzy_system.compute()
        fz_pred = np.array([[fuzzy_system.output['NORMA PKS PER DATE'],
                             fuzzy_system.output['LOSIS PKS']]])
    except ValueError as e:
        print(f"Fuzzy system error: {e}")
        return nn_pred

    final_pred = (nn_pred + fz_pred) / 2
    return final_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    if request.method == 'POST':
        try:
            input_data = {
                'NORMA AMPAS PRESS': [float(request.form['norma_ampas_press'])],
                'LOSIS % AMPAS PRESS': [float(request.form['losis_ampas_press'])],
                'NORMA BIJI': [float(request.form['norma_biji'])],
                'LOSIS % BIJI': [float(request.form['losis_biji'])],
                'NORMA TANDAN KOSONG': [float(request.form['norma_tandan_kosong'])],
                'LOSIS % TANDAN KOSONG': [float(request.form['losis_tandan_kosong'])],
                'NORMA DRAB AKHIR': [float(request.form['norma_drab_akhir'])],
                'LOSIS % DRAB AKHIR': [float(request.form['losis_drab_akhir'])],
                'NORMA SOLID DECANTER': [float(request.form['norma_solid_decanter'])],
                'LOSIS % SOLID DECANTER': [float(request.form['losis_solid_decanter'])]
            }
            hours_ahead = int(request.form['hours_ahead'])
            
            new_data = pd.DataFrame(input_data)
            prediction = hybrid_predict_with_time(nn_model, fuzzy_system, new_data, hours_ahead)
            
            result = {
                'NORMA PKS PER DATE': prediction[0][0],
                'LOSIS PKS': prediction[0][1],
                'STATUS': 'MERAH' if prediction[0][1] > prediction[0][0] else 'HIJAU'
            }
            
            return render_template('evaluate.html', result=result, input_data=input_data, hours_ahead=hours_ahead)
        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template('evaluate.html', error=True)

if __name__ == '__main__':
    app.run(debug=True)