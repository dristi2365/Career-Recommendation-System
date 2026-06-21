from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, '../data/career_data.csv'), sep="\t")
X = df.drop("career", axis=1)
careers = df["career"]

@app.route("/", methods=["GET", "POST"])
def index():
    top5 = None
    if request.method == "POST":

        # Get user inputs from form
        user_input = {}
        for col in X.columns:
            user_input[col] = [int(request.form.get(col, 1))]

        new_input = pd.DataFrame(user_input, columns=X.columns)

        # Compute cosine similarity
        similarity_scores = cosine_similarity(new_input, X)[0]

        # Use a copy to avoid modifying the global dataframe
        results = df.copy()
        results["similarity"] = similarity_scores

        # Get Top 5 careers
        top5 = results.sort_values(
            by="similarity", ascending=False
        ).head(5)[["career", "similarity"]].values.tolist()

    return render_template("index.html", top5=top5)

if __name__ == "__main__":
    app.run(debug=True)