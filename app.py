# importing required libraries
from flask import Flask, request, render_template
import numpy as np
import pickle
import warnings
from feature import FeatureExtraction

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model safely
with open("pickle/model.pkl", "rb") as f:
    gbc = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")

        try:
            # Extract features using your FeatureExtraction class
            obj = FeatureExtraction(url)
            features = np.array(obj.getFeaturesList()).reshape(1, -1)

            # Predict
            prediction = gbc.predict(features)[0]
            phishing_prob = gbc.predict_proba(features)[0, 0]   # class -1
            safe_prob = gbc.predict_proba(features)[0, 1]       # class 1

            # Format result
            result_percentage = round(safe_prob * 100, 2)

            return render_template(
                "index.html",
                url=url,
                xx=result_percentage,
            )

        except Exception as e:
            print("Error:", e)
            return render_template("index.html", xx=-1, url=url)

    # GET request
    return render_template("index.html", xx=-1)
    

if __name__ == "__main__":
    app.run()
