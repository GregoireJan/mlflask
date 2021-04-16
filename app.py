from flask import Flask, jsonify, request
from modeler.modeler import Modeler
import pandas as pd
import os 

app = Flask(__name__)

modelpath = os.path.join(os.path.join(os.path.dirname(__file__), "/models/RDFModel.joblib"))

@app.route('/predict', methods=['POST'])
def post():
    data = request.get_json()
    df = pd.DataFrame.from_dict(data)
    #
    m = Modeler()
    if not os.path.isfile(modelpath):
        m.fit()
    prediction = m.predict(df)
    
    return jsonify({
        "Prediction": prediction})

if __name__ == "__main__":
    ## Uncomment for flask only (no docker container)
    # app.run(debug=True)
    ## Comment out for flask only (no docker container)
    app.run(host="0.0.0.0", port=8080)