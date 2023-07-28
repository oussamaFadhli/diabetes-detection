import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
# Assuming you have downloaded the Pima Indians Diabetes Database and it's saved as 'diabetes.csv'.
data = pd.read_csv('diabetes.csv')

# Step 2: Prepare the data
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']  # Target variable (0 - non-diabetic, 1 - diabetic)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Preprocess the data (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Create the Neural Network model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Step 6: Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Step 8: Make predictions
y_prob = model.predict(X_test_scaled)
y_pred = (y_prob > 0.5).astype(int).flatten()

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

# Calculate percentage risk for each patient in the test set
percentage_risk = y_prob * 100

# Display percentage risk for each patient in the test set
risk_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Probability of Diabetes (%)': percentage_risk.flatten()})
print("\nPatient Diabetes Probability:")
print(risk_df)

# Save risk_df to a CSV file
risk_df.to_csv('diabetes_risk_predictions.csv', index=False)
