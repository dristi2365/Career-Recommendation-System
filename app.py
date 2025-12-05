from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# -----------------------------
# Load and train model
# -----------------------------
df = pd.read_csv("career_data.csv", sep='\t')

X = df.drop("career", axis=1)
y = df["career"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy =", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html", columns=X.columns, prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    input_data = {col: [int(request.form.get(col, 0))] for col in X.columns}
    new_input = pd.DataFrame(input_data)
    new_input = new_input[X.columns]

    career = model.predict(new_input)[0]

    return render_template("index.html", columns=X.columns, prediction=career)


if __name__ == "__main__":
    app.run(debug=True)
