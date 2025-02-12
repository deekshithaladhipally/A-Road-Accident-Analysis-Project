from flask import Flask, request, render_template, url_for
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load and prepare data
df = pd.read_csv('accidents_india.csv')
df.dropna(inplace=True)
df['Day'] = df['Day_of_Week'].astype('category').cat.codes
df['Light'] = df['Light_Conditions'].astype('category').cat.codes
df['Severity'] = df['Accident_Severity'].astype('category').cat.codes
df.drop(['Day_of_Week', 'Light_Conditions', 'Accident_Severity'], axis=1, inplace=True)

x = df.drop(['Pedestrian_Crossing', 'Special_Conditions_at_Site', 'Severity'], axis=1)
y = df['Severity']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.86)
model = DecisionTreeClassifier(criterion='gini')
model.fit(x_train, y_train)

@app.route('/')
def hello_world():
    return render_template("t.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        prediction = model.predict(final)

        if prediction == 0:
            return render_template('t.html', pred="Probability of accident severity is: Minor")
        else:
            return render_template('t.html', pred="Probability of accident severity is: Major")

    except Exception as e:
        return render_template('t.html', pred="Error: " + str(e))

@app.route('/Map')
def map1():
    return render_template("map.html")

@app.route('/Graphs')
def graph():
    return render_template("graph.html")

@app.route('/Pie')
def pie():
    return render_template("pie.html")

@app.route('/Map1')
def map2():
    return render_template("ur.html")

@app.route('/Map2')
def map3():
    return render_template("bs.html")

@app.route('/Map3')
def map4():
    return render_template("hm.html")

@app.route('/2020_analysis_report')
def powerbi_report():
    return render_template('2020_analysis_report.html')


if __name__ == "__main__":
    app.run(debug=True)
