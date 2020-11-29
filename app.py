import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import pickle
import warnings
import tweepy
from twitter_credentials import *
import json
from pandas.io.json import json_normalize
import pandas as pd
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

model = pickle.load(open('model.pickle.dat', "rb"))


# assign the values accordingly
consumer_key = CONSUMER_KEY
consumer_secret = CONSUMER_SECRET
access_token = ACCESS_TOKEN
access_token_secret = ACCESS_TOKEN_SECRET

# authorization of consumer key and consumer secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# set access to user's access key and access secret
auth.set_access_token(access_token, access_token_secret)

# calling the api
api = tweepy.API(auth)

def predictUser(username):
    try:
        user = api.get_user(username)


        def jsonify_tweepy(tweepy_object):
            json_str = json.dumps(tweepy_object._json)
            return json.loads(json_str)

        ###
        apiKeep = ['favourites_count','url','followers_count','friends_count','lang','listed_count','protected','statuses_count','verified','created_at']
        features = ['friend_follower_ratio','favourites_count','followers_count','friends_count','listed_count','protected','statuses_count','verified', 'tweets_per_day', 'favourites_per_day']

        dfTest = json_normalize(jsonify_tweepy(user))
        dfTest = dfTest[apiKeep]
        dfTest['created_at'] = datetime.strftime(datetime.strptime(dfTest['created_at'][0],'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
        dfTest['created_at'] = pd.to_datetime(dfTest['created_at'])

        dfTest['friend_follower_ratio']=dfTest['friends_count']/(dfTest['followers_count']+1)
        dfTest['days_exist'] = (pd.to_datetime("today") - dfTest['created_at']).dt.days
        dfTest['tweets_per_day'] = dfTest['statuses_count']/dfTest['days_exist']
        dfTest['favourites_per_day'] = dfTest['favourites_count']/dfTest['days_exist']
        dfTest.head()

        Xnew = dfTest[features]

        prediction = model.predict_proba(Xnew)[0]
        # print()
        # print(username)
        # print("%.2f" % (prediction[0]*100),"% ",'bot')
        # print("%.2f" % (prediction[1]*100),"% ",'not')
        pred1 = prediction[0]*100
        pred1 = "{:.2f}".format(pred1)+"% "+'bot'
        pred2 = prediction[1]*100
        pred2 = "{:.2f}".format(pred2)+"% "+'not'
        retPred = pred1 + '\n' + pred2
        return retPred
    except tweepy.TweepError as e:
        # print(e.args[0][0]['message'])  # prints 34
        return(e.args[0][0]['message'])

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.I("Enter Username:"),
        html.Br(),
        dcc.Input(id="input1", type="text", placeholder="",debounce=True),
        html.Div(id="output",children = ""),
        # dash_table.DataTable(
        #     id='table',

        # )
        # dcc.Textarea(id='output'),
    ]
)



@app.callback(
    Output("output", "children"),

    Input("input1", "value"),

)
def update_output(input1):
    # print(input1)
    if input1 == None:
        return ''
    else:
        # print(input1)

        bot = predictUser(input1)
        # print(Xnew.head())
        return bot


if __name__ == "__main__":
    app.run_server(debug=True)