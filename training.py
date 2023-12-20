import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import time

# Load the data
file_path = 'data.csv'
df = pd.read_csv(file_path)

# Split into features and target
X = df[['VISUAL', 'AUDITORIAL', 'KINESTHETIC']]
y = df['LEARNINGSTYLE']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
accuracy = model.evaluate(X_test_scaled, y_test)[1]
print(f"Accuracy: {accuracy:.2f}")

# Save the model
# saved_model_path = "D:\Coding Folder\Capstone".format(int(time.time()))
# tf.contrib.saved_model.saved_keras_model(model, saved_model_path)
model.save('learning_style_model_tf.h5')
print("Model saved as 'learning_style_model_tf.h5'")

# Example prediction
new_data = pd.DataFrame({'VISUAL': [15], 'AUDITORIAL': [15], 'KINESTHETIC': [15]})
new_data_scaled = scaler.transform(new_data)
predicted_probabilities = model.predict(new_data_scaled)
predicted_class = le.inverse_transform([predicted_probabilities.argmax()])[0]
print(f"Predicted Learning Style: {predicted_class}")
