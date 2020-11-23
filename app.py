from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from modeler.modeler import Modeler

app = Flask(__name__)
api = Api(app)

class Predict(Resource):
    @staticmethod
    def post():
        data = request.get_json()
        meanradius = data['mean radius']
        meantexture = data['mean texture']
        meanperimeter = data['mean perimeter']
        meanarea = data['mean area']
        meansmoothness = data['mean smoothness']
        meancompactness = data['mean compactness']
        meanconcavity = data['mean concavity']
        meanconcavepoints = data['mean concave points']
        meansymmetry = data['mean symmetry']
        radiuserror = data['radius error']
        perimetererror = data['perimeter error']
        areaerror = data['area error']
        compactnesserror = data['compactness error']
        concavityerror = data['concavity error']
        concavepointserror = data['concave points error']
        worstradius = data['worst radius']
        worsttexture = data['worst texture']
        worstperimeter = data['worst perimeter']
        worstarea = data['worst area']
        worstsmoothness = data['worst smoothness']
        worstcompactness = data['worst compactness']
        worstconcavity = data['worst concavity']
        worstconcavepoints = data['worst concave points']
        worstsymmetry = data['worst symmetry']
        worstfractaldimension = data['worst fractal dimension']
        #
        m = Modeler()
        m.prepro()
        m.fit()
        prediction = m.predict([meanradius, meantexture, meanperimeter, meanarea,
                                meansmoothness, meancompactness, meanconcavity,
                                meanconcavepoints, meansymmetry, radiuserror,
                                perimetererror, areaerror, compactnesserror, concavityerror,
                                concavepointserror, worstradius, worsttexture,
                                worstperimeter, worstarea, worstsmoothness,
                                worstcompactness, worstconcavity, worstconcavepoints,
                                worstsymmetry, worstfractaldimension])
        return jsonify({
            "Input": {
                'mean radius': meanradius,
                'mean texture':meantexture,
                'mean perimeter':meanperimeter,
                'mean area':meanarea,
                'mean smoothness':meansmoothness,
                'mean compactness':meancompactness,
                'mean concavity':meanconcavity,
                'mean concave points':meanconcavepoints,
                'mean symmetry':meansymmetry,
                'radius error':radiuserror,
                'perimeter error':perimetererror,
                'area error':areaerror,
                'compactness error':compactnesserror,
                'concavity error':concavityerror,
                'concave points error':concavepointserror,
                'worst radius':worstradius,
                'worst texture':worsttexture,
                'worst perimeter':worstperimeter,
                'worst area':worstarea,
                'worst smoothness':worstsmoothness,
                'worst compactness':worstcompactness,
                'worst concavity':worstconcavity,
                'worst concave points':worstconcavepoints,
                'worst symmetry':worstsymmetry,
                'worst fractal dimension':worstfractaldimension},
            "Prediction": prediction
        })

api.add_resource(Predict, "/predict")

if __name__ == "__main__":
    ## Uncomment for flask only (no docker container)
    # app.run(debug=True)
    ## Comment out for flask only (no docker container)
    app.run(host="0.0.0.0", port=8080)