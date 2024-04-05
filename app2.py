from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
Bootstrap(app)

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Pheneas7#'
app.config['MYSQL_DB'] = 'flask_app_db'
mysql = MySQL(app)

# Generate random data for demo
np.random.seed(0)
age = np.random.randint(6, 25, 1000)
weight = np.random.uniform(3, 25, 1000)
height = np.random.uniform(50, 120, 1000)

# Adjusted malnutrition status based on BMI calculation
bmi = weight / ((height / 100) ** 2)
malnutrition_status = np.where(bmi < 18.5, 'Underweight', np.where(bmi < 25, 'Normal', 'Overweight'))

malnutrition_data = pd.DataFrame({'age': age, 'weight': weight, 'height': height, 'malnutrition_status': malnutrition_status})
features = malnutrition_data[['age', 'weight', 'height']]
labels = malnutrition_data['malnutrition_status']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')  # Use class_weight='balanced' to handle class imbalance
model.fit(X_train, y_train)

# Dietary recommendations based on malnutrition status
recommendations = {
    'Underweight': {
        'recommendation': 'Increase calorie intake with nutrient-dense foods like nuts, seeds, and healthy oils. Focus on high-protein foods like lean meats, fish, eggs, and legumes.',
        'recovery_measure': 'Gradually increase calorie intake and engage in resistance training exercises to build muscle mass.'
    },
    'Normal': {
        'recommendation': 'Maintain a balanced diet with a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats.',
        'recovery_measure': None  # No specific recovery measure needed for normal status
    },
    'Overweight': {
        'recommendation': 'Reduce calorie intake by limiting sugary and fatty foods. Increase consumption of fruits, vegetables, lean proteins, and whole grains. Incorporate regular physical activity.',
        'recovery_measure': 'Follow a structured weight loss plan focusing on calorie deficit, portion control, and increased physical activity.'
    }
}

# Login route
@app.route('/')
def login():
    return render_template('login.html')

# Registration route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists in the database
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cur.fetchone()
        cur.close()

        # If the username already exists, return a message indicating that the user already exists
        if existing_user:
            return "Username already exists. Please choose a different username."
        

        # Hash the password before storing it in the database
        hashed_password = generate_password_hash(password)

        # Store user credentials in the database
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('login'))

    return render_template('register.html')

# Authentication route
@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']

    # Retrieve user credentials from the database
    cur = mysql.connection.cursor()
    cur.execute("SELECT password FROM users WHERE username = %s", (username,))
    user = cur.fetchone()
    cur.close()

    if user and check_password_hash(user[0], password):
        return redirect(url_for('dashboard'))
    else:
        return "Invalid credentials. Please try again."

# Dashboard route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Predict malnutrition route (retain your original code)
@app.route('/predict_malnutrition', methods=['POST'])
def predict_malnutrition():
    try:
        age_val = float(request.form['age'])
        weight_val = float(request.form['weight'])
        height_val = float(request.form['height'])

        # Make prediction using the trained model
        prediction = model.predict([[age_val, weight_val, height_val]])
        recommendation = recommendations[prediction[0]]

        return render_template('dashboard.html', prediction=prediction[0], recommendation=recommendation)
    except ValueError:
        return "Please enter valid numerical values for age, weight, and height."

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
