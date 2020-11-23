# mlflask

* Machine learning model - no hyperparameter tuning 
* Served on flask and/or on Docker container  

Notebook exploration.ipynd contains the EDA.

Run app.py for launching flask rest api  
Run **docker build --tag name .** for building docker image then **docker run -p 8080:8080 name** to launch flask rest api  

Run testapp notebook to call the rest api (port 5000 for flask only/port 8080 for flask in docker container)
