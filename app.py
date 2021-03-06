import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
from dash_table.Format import Format
import pickle
import warnings
import tweepy
from twitter_credentials import *
import json
from pandas.io.json import json_normalize
import pandas as pd
from datetime import datetime
from string import capwords
from sigfig import round
import plotly.express as px
import plotly.graph_objects as go

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')
model = pickle.load(open('model.pickle.dat', "rb"))
means = pickle.load(open('means.pickle.dat', "rb"))
stds = pickle.load(open('stds.pickle.dat', "rb"))
medians = pickle.load(open('medians.pickle.dat', "rb"))
mediansNotBot = pickle.load(open('mediansNotBot.pickle.dat', "rb"))
logBotData = pickle.load(open('logBotData.pickle.dat', "rb"))
logNotBotData = pickle.load(open('logNotBotData.pickle.dat', "rb"))
BotData = pickle.load(open('BotData.pickle.dat', "rb"))
NotBotData = pickle.load(open('NotBotData.pickle.dat', "rb"))

features = ['friend_follower_ratio', 'favourites_count', 'friends_count', 'statuses_count', 'tweets_per_day', 'favourites_per_day']
means = pd.DataFrame(means).T
sig_fig = 3
for col in means.columns:
    means[col] = round(float(means[col][0]),sigfigs= sig_fig)

medians = pd.DataFrame(medians).T
sig_fig = 3
for col in medians.columns:
    medians[col] = round(float(medians[col][0]),sigfigs= sig_fig)
medians['verified'] = medians['verified'].replace(0,False)
medians['protected'] = medians['protected'].replace(0,False)

mediansNotBot = pd.DataFrame(mediansNotBot).T
sig_fig = 3
for col in mediansNotBot.columns:
    mediansNotBot[col] = round(float(mediansNotBot[col][0]),sigfigs= sig_fig)
mediansNotBot['verified'] = mediansNotBot['verified'].replace(0,False)
mediansNotBot['protected'] = mediansNotBot['protected'].replace(0,False)
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
df = pd.DataFrame(columns = features)

def style_table_by_z_value(df,means,stds):
    if 'id' in df:
        numeric_columns = df.select_dtypes('number').drop(['id'], axis=1)
    else:
        numeric_columns = df.select_dtypes('number')
    max_across_numeric_columns = numeric_columns.max()
    max_across_table = max_across_numeric_columns.max()
    zs = df - means/stds
    zs = abs(zs.where((zs >3) | (zs <-3),0))
    # print(zs[''])
    styles = []
    for col in numeric_columns:
        # print(col)
        # print(zs[col][0])
        if zs[col][0] != 0:
            styles.append({
                'if': {
                    'filter_query': '{{{col}}} != {value}'.format(
                        col=0, value=zs[col][0]
                    ),
                    'column_id': col
                },
                'backgroundColor': '#F45D5D',
                'color': 'white'
            })
    return styles


def predictUser(username):
    # Allows user to use @ as part of the handle
    if(username != "" and username[0]=="@"):
        username = username[1:]
    try:
        user = api.get_user(username)


        def jsonify_tweepy(tweepy_object):
            json_str = json.dumps(tweepy_object._json)
            return json.loads(json_str)

        ###
        apiKeep = ['profile_image_url_https','favourites_count','url','followers_count','friends_count','lang','listed_count','protected','statuses_count','verified','created_at']
        features = ['friend_follower_ratio', 'favourites_count', 'friends_count', 'statuses_count', 'tweets_per_day', 'favourites_per_day']

        dfTest = json_normalize(jsonify_tweepy(user))
        dfTest = dfTest[apiKeep]
        dfTest['created_at'] = datetime.strftime(datetime.strptime(dfTest['created_at'][0],'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
        dfTest['created_at'] = pd.to_datetime(dfTest['created_at'])

        dfTest['friend_follower_ratio']=dfTest['friends_count']/(dfTest['followers_count']+1)
        dfTest['days_exist'] = (pd.to_datetime("today") - dfTest['created_at']).dt.days
        dfTest['tweets_per_day'] = dfTest['statuses_count']/dfTest['days_exist']
        dfTest['favourites_per_day'] = dfTest['favourites_count']/dfTest['days_exist']
        # print(dfTest['days_exist'][0])
        # dfTest.head()
        imurl = dfTest['profile_image_url_https']
        # print(imurl)

        Xnew = dfTest[features]

        prediction = model.predict_proba(Xnew)[0]
        # print()
        # print(username)
        # print("%.2f" % (prediction[0]*100),"% ",'bot')
        # print("%.2f" % (prediction[1]*100),"% ",'not')
        pred1 = prediction[0]*100
        pred1 = "Bot: {:.2f}%".format(pred1)
        pred2 = prediction[1]*100
        pred2 = "Not: {:.2f}%".format(pred2)
        retPred = pred1 + '\n' + pred2
        legit = True
        return retPred,Xnew,imurl,legit
    except tweepy.TweepError as e:
        # print(e.args[0][0]['message'])  # prints 34
        legit = False
        return(e.args[0][0]['message']),df,'',legit

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [html.H1("Bot or Not"),
    html.Div(

        [html.Div([
        html.I("Enter Username:"),
        html.Br(),
        dcc.Input(id="input1", type="text", placeholder="",debounce=True),
        html.Br(),
        html.Div(id="output", children = "")
        ],className = 'three columns'),
        html.Div([
            html.Img(id = 'img',height = 100),
            ],className = 'nine columns')

        ,],className = 'row',
        ),
    html.Div([
        html.H5("User's Features:"),
        html.I("Red indicates outlier compared to bot features"),
        dash_table.DataTable(
            id='table',
            style_cell={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            editable = False,
            columns = [{"name": capwords(i.replace('_', ' ')), "id": i,'type': 'numeric',"format":Format(group=',')} for i in df.columns],
            ),
        html.Hr(),
        html.H4("Data Set Information"),
        html.H5("Median Bot Features:"),
        dash_table.DataTable(
            id='medians',
            style_cell={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            data = medians.to_dict('records'),
            columns = [{"name": capwords(i.replace('_', ' ')), "id": i,'type': 'numeric',"format":Format(group=',')} for i in df.columns]
        ),
        html.H5("Median Not Bot Features:"),

        dash_table.DataTable(
            id='mediansNotBot',
            style_cell={
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            data = mediansNotBot.to_dict('records'),
            columns = [{"name": capwords(i.replace('_', ' ')), "id": i,'type': 'numeric',"format":Format(group=',')} for i in df.columns]
        ),
        html.Br(),
        html.Div([
            html.Div([
                html.Label(["Select Bots or Not Bots:",
                dcc.Dropdown(id='graph_dropdown_df',
                            value = 'Bots',
                            options=[{'label': i, 'value':i}for i in ['Bots','Not Bots']]
                        ),]),

                ],className = "six columns"),
            html.Div([
                html.Label(["Select Metric: ",
                dcc.Dropdown(id='graph_dropdown',
                            value = 'friend_follower_ratio',
                            options=[{'label': capwords(i.replace('_', ' ')), 'value':i}for i in df.columns]
                            ),]),
                ],className = "six columns"),

            ],className="row"),
            html.Div([
                dcc.Graph(id = 'hist'),
                ],className = "row")



            ],className = "row")

    ]
)



@app.callback(
    Output("output", "children"),
    Output("table", 'data'),
    Output("img",'src'),
    Output("table",'style_data_conditional'),
    Input("input1", "value"),

)
def update_output(input1):
    # print(input1)
    df = pd.DataFrame(columns = ['friend_follower_ratio','favourites_count','followers_count','friends_count','listed_count','protected','statuses_count','verified', 'tweets_per_day', 'favourites_per_day'])
    if input1 == None:
        return '',df.round(1).to_dict('records'),'',style_table_by_z_value(df,means,stds)
    else:
        # print(input1)

        bot,df,imurl,legit = predictUser(input1)

        if legit:

            # print('days_exist',df['days_exist'][0])
            sig_fig = 2
            # print(df.columns)
            df['friend_follower_ratio'] =round(float(df['friend_follower_ratio'][0]), sigfigs = sig_fig)#.round(6)
            df['tweets_per_day'] = round(float(df['tweets_per_day'][0]),sigfigs=sig_fig)
            df['favourites_per_day'] = round(float(df['favourites_per_day'][0]),sigfigs=sig_fig)

            zs = df- means/stds
            zs = abs(zs.where((zs >3) | (zs <-3),0))
            # print(zs)

            # layout =
        # print(Xnew.head())
        try:
            url = imurl[0][:-11]+imurl[0][-4:]
        except:
            url = ''
        return bot, df.to_dict('records'), url, style_table_by_z_value(df,means,stds)

@app.callback(
    Output("hist", "figure"),
    Input("graph_dropdown", "value"),
    Input("graph_dropdown_df","value")

)
def updateHistogram(column,data):
    # print(input2)
    if data == 'Bots':
        label = capwords(column.replace('_', ' '))
        # print(label)
        return px.histogram(BotData,x=column,labels={column:label})
    else:
        label = capwords(column.replace('_', ' '))
        return px.histogram(NotBotData,x=column,labels={column:label})

if __name__ == "__main__":
    app.run_server(debug=True)
