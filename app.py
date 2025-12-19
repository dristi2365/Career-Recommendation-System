from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("career_data.csv", sep="\t")
X = df.drop("career", axis=1)
careers = df["career"]

@app.route("/", methods=["GET", "POST"])
def index():
    top5 = None
    if request.method == "POST":
        # Get user inputs from form
        user_input = {}
        for col in X.columns:
            user_input[col] = [int(request.form.get(col, 1))]  # default 1

        new_input = pd.DataFrame(user_input, columns=X.columns)

        # Compute cosine similarity
        similarity_scores = cosine_similarity(new_input, X)[0]
        df["similarity"] = similarity_scores

        # Get Top-5 careers
        top5 = df.sort_values(by="similarity", ascending=False).head(5)[["career", "similarity"]].values.tolist()

    return render_template("index.html", top5=top5)

if __name__ == "__main__":
    app.run(debug=True)
