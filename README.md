# BotOrNotDSCI470
Final project for Intro to Machine Learning

Clone the git repository and add the data folder containing the csv files from the bot repository. Add a twitter_credentials.py file to the folder containing:

```
# Variables that contain the user credentials to access Twitter API
ACCESS_TOKEN = "XXXXX"
ACCESS_TOKEN_SECRET = "XXXXX"
CONSUMER_KEY = "XXXXX"
CONSUMER_SECRET = "XXXXX"
```

Create a virtual environment with requirements.txt and activate it
```
python -m venv env
env\Scripts\activate.bat
```
Install all dependencies in requirements.txt
```
python -m pip install -r requirements.txt
```
Train the model using trainmodel.py
```
python trainmodel.py
```
Run the Flask/Dash based user interface and go to [http://127.0.0.1:8050/](http://127.0.0.1:8050/)
```
python app.py
```
