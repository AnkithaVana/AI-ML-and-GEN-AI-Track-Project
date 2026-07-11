from __future__ import annotations

import os

from flask import Flask, render_template, request

from src.config import DEFAULT_FORM_VALUES, FORM_OPTIONS
from src.data_utils import coerce_form_payload
from src.modeling import load_model, predict_approval


app = Flask(__name__)
model_bundle = load_model()


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        options=FORM_OPTIONS,
        defaults=DEFAULT_FORM_VALUES,
        best_model=model_bundle["best_model"],
        data_source=model_bundle["data_source"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    features = coerce_form_payload(request.form.to_dict())
    result = predict_approval(model_bundle, features)
    applicant = features.iloc[0].to_dict()
    return render_template("result.html", result=result, applicant=applicant)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=port)
