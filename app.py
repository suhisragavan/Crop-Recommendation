from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_PATH = "best_crop_model.pkl"

# Load model once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def to_float(value: str, field_name: str) -> float:
    """Convert input to float with friendly error message."""
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is required.")
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a number.")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    # Keep old values so user doesn’t lose inputs on submit
    values = {
        "N": "",
        "P": "",
        "K": "",
        "temperature": "",
        "humidity": "",
        "ph": "",
        "rainfall": ""
    }

    if request.method == "POST":
        try:
            # Read form values
            for k in values.keys():
                values[k] = request.form.get(k, "").strip()

            N = to_float(values["N"], "Nitrogen (N)")
            P = to_float(values["P"], "Phosphorus (P)")
            K = to_float(values["K"], "Potassium (K)")
            temperature = to_float(values["temperature"], "Temperature")
            humidity = to_float(values["humidity"], "Humidity")
            ph = to_float(values["ph"], "pH")
            rainfall = to_float(values["rainfall"], "Rainfall")

            # Optional: basic sanity checks (avoid crazy values)
            if not (0 <= ph <= 14):
                raise ValueError("pH must be between 0 and 14.")
            if humidity < 0:
                raise ValueError("Humidity cannot be negative.")
            if rainfall < 0:
                raise ValueError("Rainfall cannot be negative.")

            # Predict
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
            prediction = model.predict(features)[0]
            result = str(prediction)

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error, values=values)


if __name__ == "__main__":
    
    app.run(debug=True)
