
import pandas as pd
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
    df = pd.DataFrame([int_features])
    #final_features = [np.array(int_features)]
    print(df)
    if classifier == 0:
        prediction = model.predict(df)
    elif classifier == 1:
        prediction = model1.predict(df)
    else:
        prediction = model2.predict(df)

    if prediction == 0:
        return render_template('index.html', prediction_text='Employee Might not Leave The Job')

    else:
        return render_template('index.html', prediction_text='Employee Might Leave The Job')



if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
