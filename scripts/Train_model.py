import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pydantic import BaseModel

# 1. Define your schema (from your previous code)
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

def train():
    # 2. Load the dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 3. Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Initialize and train the model
    print("Training the Random Forest model...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # 5. Evaluate
    score = model.score(X_test, y_test)
    print(f"Model trained with accuracy: {score:.2%}")

    # 6. Save the model to a file
    joblib.dump(model, r"C:\Users\hp\Desktop\PY4E\ML-OPS\Test1\iris_model.pkl")
    print("Model saved as iris_model.pkl")

if __name__ == "__main__":
    train()