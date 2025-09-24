from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import uvicorn, os

# Train Model
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# FastAPI app
app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    x = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction = model.predict(x)[0]
    return {"prediction": iris.target_names[prediction]}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Iris Predictor</title></head>
    <body>
      <h2>Iris Flower Prediction</h2>
      <input id="sl" placeholder="Sepal length">
      <input id="sw" placeholder="Sepal width">
      <input id="pl" placeholder="Petal length">
      <input id="pw" placeholder="Petal width">
      <button onclick="predict()">Predict</button>
      <p id="result"></p>
      <script>
        async function predict() {
          let data = {
            sepal_length: parseFloat(document.getElementById("sl").value),
            sepal_width: parseFloat(document.getElementById("sw").value),
            petal_length: parseFloat(document.getElementById("pl").value),
            petal_width: parseFloat(document.getElementById("pw").value)
          };
          let res = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
          });
          let result = await res.json();
          document.getElementById("result").innerText = "Prediction: " + result.prediction;
        }
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="127.0.0.1", port=port)