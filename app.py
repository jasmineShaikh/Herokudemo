
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from numpy import asarray

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    classifier = int(request.form.get('classifier'))

    int_features = [int(request.form.get('satisfaction_level')),
                    int(request.form.get('last_evaluation')),
                    int(request.form.get('number_project')),
                    int(request.form.get('average_montly_hours')),
                    int(request.form.get('time_spend_company')),
                    int(request.form.get('Work_accident')),
                    int(request.form.get('promotion_last_5years')),
                    int(request.form.get('sales')),
                    int(request.form.get('salary'))]

    print(classifier)
    final_features = [np.array(int_features)]
    print(final_features)
    if classifier == 0:
        prediction = model.predict(final_features)
    elif classifier == 1:
        prediction = model1.predict(final_features)
    else:
        prediction = model2.predict(final_features)

    if prediction == 0:
        return render_template('index.html', prediction_text='Employee Might not Leave The Job')

    else:
        return render_template('index.html', prediction_text='Employee Might Leave The Job')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
