# Import necessary libraries
from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the machine learning model
model = load_model('learning_style_model_tf.h5')

# Load the LabelEncoder
le = LabelEncoder()
df = pd.read_csv('data.csv')  # Assuming you have the original dataset available
le.fit(df['LEARNINGSTYLE'])
# Load a dummy dataset for content recommendation
content_data = pd.DataFrame({
    'Content': ['Dummy Content 1', 'Dummy Content 2', 'Dummy Content 3', 'Dummy Content 4', 'Dummy Content 5'],
    'Visual': [10, 5, 15, 8, 12],
    'Auditorial': [15, 10, 5, 8, 12],
    'Kinesthetic': [5, 15, 10, 12, 8],
    'LearningStyle': ['V', 'A', 'K', 'V', 'A'],
    'YouTubeLink': [
        'https://www.youtube.com/watch?v=AjgD3CvWzS0',
        'https://www.youtube.com/watch?v=dqTTojTija8',
        'https://www.youtube.com/watch?v=HaEmIakO7f4',
        'https://www.youtube.com/watch?v=WtWxOhhZWX0',
        'https://www.youtube.com/watch?v=6_2hzRopPbQ'
    ]
})


# Placeholder for user scores
user_scores = {'Visual': 0, 'Auditorial': 0, 'Kinesthetic': 0}

# Standardize the input features for content recommendation
scaler = StandardScaler()
content_scaled = scaler.fit_transform(content_data[['Visual', 'Auditorial', 'Kinesthetic']])

# Function to get content recommendation based on learning style
def get_recommendation(learning_style):
    similar_content = content_data[content_data['LearningStyle'] == learning_style]
    return similar_content['Content'].tolist()


# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle user clicks and update scores
# ... (previous code)

@app.route('/update_scores', methods=['POST'])
def update_scores():
    global user_scores

    # Get scores from the clicked content
    clicked_content = request.json['content']
    user_scores['Visual'] += content_data.loc[content_data['Content'] == clicked_content, 'Visual'].values[0]
    user_scores['Auditorial'] += content_data.loc[content_data['Content'] == clicked_content, 'Auditorial'].values[0]
    user_scores['Kinesthetic'] += content_data.loc[content_data['Content'] == clicked_content, 'Kinesthetic'].values[0]

    # Convert int64 types to Python int
    user_scores = {key: int(value) for key, value in user_scores.items()}

    # Predict learning style based on updated scores
    input_data = np.array([[user_scores['Visual'], user_scores['Auditorial'], user_scores['Kinesthetic']]])
    input_data_scaled = scaler.transform(input_data)
    predicted_probabilities = model.predict(input_data_scaled)
    predicted_class = le.inverse_transform([predicted_probabilities.argmax()])[0]

    # Get content recommendation based on learning style
    recommendation = get_fresh_recommendation(predicted_class)

    # Send back the predicted learning style, recommendation, and current scores
    response_data = {
        'learning_style': predicted_class,
        'recommendation': recommendation,
        'current_scores': user_scores
    }

    return jsonify(response_data)
@app.route('/get_initial_recommendation', methods=['GET'])
def get_initial_recommendation():
    initial_recommendation = get_fresh_recommendation('V')  # Assuming 'V' as the default learning style
    response_data = {'initial_recommendation': initial_recommendation}
    return jsonify(response_data)

# Function to get fresh content recommendation based on learning style
def get_fresh_recommendation(learning_style):
    # You can modify this function to fetch recommendations from a database or external API.
    # For simplicity, we'll use a predefined list of fresh content.
    fresh_content = content_data.sample(n=3)  # Sample 3 random items for variety
    recommendation = fresh_content['Content'].tolist()
    return recommendation


# ... (remaining code)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
