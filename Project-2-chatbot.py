import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Hours_Studied']]
y = df['Pass']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Test Model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("AI DATA CLASSIFICATION PROJECT")
print("------------------------------")
print("Accuracy =", accuracy * 100, "%")

# User Input
hours = float(input("\nEnter Study Hours: "))

result = model.predict([[hours]])

if result[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
    
