# import numpy as np
# from flask import Flask, request, jsonify, render_template
# import pickle
# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# import statsmodels.api as sm
# from sklearn import linear_model

# app = Flask(__name__)
# model = pickle.load(open('trained_model.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict',methods=['POST'])
# def predict():

#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = model.predict(final_features)
#     output = round(prediction[0], 2)

#     return render_template("index.html", prediction_text='Your predicted annual Healthcare Expense is $ {}'.format(output))

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

# if __name__ == "__main__":
#     app.run(debug=True)



# ====================================================================


# percobaan 1
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model_file = open('trained_model.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', IQA_value=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    PM10,PM25,SO2,CO,O3,NO2 = [x for x in request.form.values()]

    data = []
    # untuk PM10
    data.append(float(PM10))
    # untuk PM25
    data.append(float(PM25))
    # untuk SO2
    data.append(float(SO2))
    # untuk CO
    data.append(float(CO))
    # untuk O3
    data.append(float(O3))
    # untuk NO2
    data.append(float(NO2)) 
    
    prediction = model.predict([data])
    output = float(prediction[0])

    return render_template('index.html', IQA_value=output)

if __name__ == '__main__':
    app.run(debug=True)

