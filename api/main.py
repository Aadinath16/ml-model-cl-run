from flask import Flask, request, jsonify, render_template
from app.model import predict

app = Flask(__name__, template_folder='templates')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        # Check if from form or API
        if request.content_type == "application/json":
            data = request.get_json(force=True)
            instances = data.get("instances", [])
            preds = predict(instances)
            return jsonify({"predictions": preds})
        else:
            # Form submission
            sl = float(request.form["sepal_length"])
            sw = float(request.form["sepal_width"])
            pl = float(request.form["petal_length"])
            pw = float(request.form["petal_width"])
            result = predict([[sl, sw, pl, pw]])[0]
            classes = ["Setosa", "Versicolor", "Virginica"]
            return render_template("index.html", prediction=classes[result])
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


# from flask import Flask, request, jsonify
# from app.model import predict

# app = Flask(__name__)

# @app.route("/")
# def index():
#     return "🎯 Iris model is running!"

# @app.route("/predict", methods=["POST"])
# def predict_endpoint():
#     try:
#         data = request.get_json(force=True)
#         instances = data.get("instances", [])
#         preds = predict(instances)
#         return jsonify({"predictions": preds})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)
