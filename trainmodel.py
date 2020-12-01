import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import pickle

def importData(genuine_folders, bot_folders):
    #Import genuine users and tweets
    list_genuine_users = []
    list_genuine_tweets = []

    for folder in genuine_folders:
        path = ''
        path = os.path.join(path, folder)
        # warnings.simplefilter(action='ignore', category=FutureWarning)
        df_users = pd.read_csv(path+'/users.csv',index_col='id',low_memory=False)
        df_users['source_f'] = folder
        df_users['botornot'] = 'not'
        list_genuine_users.append(df_users)
        df_tweets = pd.read_csv(path+'/tweets.csv',index_col='id',low_memory=False)
        df_tweets['source_f'] = folder
        df_tweets['botornot'] = 'not'
        list_genuine_tweets.append(df_tweets)

    genuine_tweets = pd.concat(list_genuine_tweets,sort = False)
    genuine_users = pd.concat(list_genuine_users,sort = False)

    #Import bot users and tweets
    list_bot_users = []
    list_bot_tweets = []
    for folder in bot_folders:
        path = ''
        path = os.path.join(path, folder)
        # print(path)
        # warnings.simplefilter(action='ignore', category=FutureWarning)
        df_users = pd.read_csv(path+'/users.csv',index_col='id',low_memory=False)
        df_users['source_f'] = folder
        df_users['botornot'] = 'bot'
        list_bot_users.append(df_users)
        df_tweets = pd.read_csv(path+'/tweets.csv',low_memory=False)
        df_tweets = df_tweets.set_index('id')
        df_tweets['source_f'] = folder
        df_tweets['botornot'] = 'bot'
        list_bot_tweets.append(df_tweets)

    bot_tweets = pd.concat(list_bot_tweets,sort = False)
    bot_users = pd.concat(list_bot_users,sort = False)

    #Combine genuine users, tweets and bot users, tweets into same dataframes
    users = pd.concat([bot_users,genuine_users])
    tweets = pd.concat([bot_tweets,genuine_tweets])

    #Extract labels from users dataframe
    labels = users['botornot']

    return (tweets, users, labels)

genuine_folders = ['data/genuine_accounts.csv']

bot_folders = [
        'data/social_spambots_1.csv',
        'data/social_spambots_2.csv',
        'data/social_spambots_3.csv',
        'data/traditional_spambots_1.csv',
        #'data/traditional_spambots_2.csv',
        #'data/traditional_spambots_3.csv',
        #'data/traditional_spambots_4.csv',
        'data/fake_followers.csv',
    ]
tweets, users, labels = importData(genuine_folders,bot_folders)

def convertTime(row):
  try:
    date = datetime.strftime(datetime.strptime(row,'%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d %H:%M:%S')
    return date
  except:
    return 0


usersThomas = users
tweetsThomas = tweets
labelsThomas = labels


warnings.simplefilter(action='ignore')
# cols = usersThomas.columns.sort_values()
# for i in cols:
#   print(i)
keep = ['botornot','favourites_count','url','followers_count','friends_count','lang','listed_count','protected','statuses_count','verified','updated','created_at']
features = ['friend_follower_ratio','favourites_count','friends_count','statuses_count','tweets_per_day','favourites_per_day']
usersThomasSel = usersThomas[keep]

usersThomasSel['friend_follower_ratio']=usersThomasSel['friends_count']/(usersThomasSel['followers_count']+1)
usersThomasSel['updated'] = pd.to_datetime(usersThomasSel['updated'])
usersThomasSel['created_at'] = usersThomasSel['created_at'].apply(lambda row: convertTime(row))
usersThomasSel = usersThomasSel[usersThomasSel.created_at != 0]
usersThomasSel['created_at'] = pd.to_datetime(usersThomasSel['created_at'])
# usersThomasSel.head()

usersThomasSel['protected'] = usersThomasSel['protected'].fillna(0)
usersThomasSel['verified'] = usersThomasSel['verified'].fillna(0)

usersThomasSel['days_exist'] = (usersThomasSel['updated']-usersThomasSel['created_at']).dt.days
usersThomasSel['tweets_per_day'] = usersThomasSel['statuses_count']/usersThomasSel['days_exist']
usersThomasSel['favourites_per_day'] = usersThomasSel['favourites_count']/usersThomasSel['days_exist']
X = usersThomasSel[features]
Y = usersThomasSel['botornot']


seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = y_pred
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# plot_importance(model, max_num_features = 10)
# pyplot.show()
# X.head()


# corrdf = X
# corrdf['botornot'] = Y

# def correlation_heatmap(train):
#     correlations = train.corr()

#     fig, ax = plt.subplots(figsize=(10,10))
#     sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
#                 square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70}
#                 )
#     plt.show();

# correlation_heatmap(corrdf)

# print(confusion_matrix(y_test, predictions))
# print(y_test)
pickle.dump(model, open("model.pickle.dat", "wb"))
