# Spotify Data Science Project

## Requirements

This project relies on Anaconda, which you can download:
	
- For mac: https://conda.io/docs/user-guide/install/macos.html
- For window: https://conda.io/docs/user-guide/install/windows.html
- For linux: https://conda.io/docs/user-guide/install/linux.html
    Be sure to downloard the “Anaconda” version

If you want to use your own data, make sure you have a login for a Spotify account. You will also need to use `pip`, a package manager of Python. Check if you have it installed by going into the Terminal/Command Line and type 
```
pip -V
```
 This will tell you what version of pip, if at all, is on your machine. 

Download `spotipy` by executing in Terminal/Command line:
```
pip3 install spotipy
```
## Overview

1. Get set up and collect data.
2. How do we operate on our data? What is `pandas`? `numpy`?
3. Simple visualization
4. Supervised learning: Classification and Regression
5. Unsupervised learning: Clustering and Dimensionality Reduction


## Getting the Data

Go to https://developer.spotify.com/dashboard and make a client.

For this project, we recommend working in a Jupyter notebook. In order to do this, navigate to a folder you want to do work in and type 
```
jupyter notebook
```

This should open up a browser window for you to work in. Hit New > Notebook > Python 3 to open a new file to work in.

You can type all of your code into the cells you see and run by hitting Shift+Enter or hitting the "Run" key. 

Let's start by importing what we need (and making sure all the imports work)

```
import sys
import spotipy
import spotipy.util as util
import pandas as pd
```

Now we need to get an access token to authenticate ourselves with Spotify (so they'll give us data about ourselves).  

Go to developer.spotify.com/dashboard and create a new client. You should see a client_id and client_secret section. Now we get our token into our program:

```
scope = 'user-library-read'
username = 'redacted'
cid = 'redacted'
csecret = 'redacted'
ruri = 'https://www.spotify.com'
token = util.prompt_for_user_token(username, scope, client_id=cid,client_secret=csecret,redirect_uri=ruri)
```

This will redirect you to Spotify's homepage. But if you look in the url, you'll see that your access token is written there! Copy the entire url back into jupyter, and you'll have your token! Let's check it with

```
token
```

Now we're ready to make requests against the API. Let's try

```
sp = spotipy.Spotify(auth=token)
results = sp.current_user_saved_tracks(limit=50)
```

`results` holds the data of the first 50 songs in your library. We need to parse it.

```
artists = []
songs = []
ids = []

for item in results['items']:
    track = item['track']
    ids.append(track['id'])
    songs.append(track['name'])
    artists.append(track['artists'][0]['name'])
```

To get audio features, we call the following function. We also put it into a pandas dataframe, a handy object that will help us play with our data later

```
df = pd.DataFrame(sp.audio_features(ids))
```

We can only request 50 items at a time. If we want to iterate through, we have to do something more complex

```
offset = 50
results = sp.current_user_saved_tracks(limit=50,offset=50)

while(len(results['items']) > 0):
    ids = []
    for item in results['items']:
        track = item['track']
        ids.append(track['id'])
        songs.append(track['name'])
        artists.append(track['artists'][0]['name'])
    df = df.append(pd.DataFrame(sp.audio_features(ids)), ignore_index=True)
    offset = offset + 50
    results = sp.current_user_saved_tracks(limit=50,offset=offset)
```
Finally, we select the columns we want, and add on the song names and artist names. 

```
library = df[['acousticness', 'danceability', 'energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','valence']].copy()
library['song_title'] = songs
library['artist'] = artists
```

To see what the dataframe looks like, try
```
library.head()
```

If you want to share your data, write it to a csv file with

```
library.to_csv("filename.csv", index=None)
```


## Operating on DataFrames

In this section, we will play with the package called `pandas`, which is the primary package for data clearning, wrangling and analysis in Python. We will introduce the concept of _DataFrame_ and _Series_ and cover basic commands, ways to do indexing, and operations on columns. Open `02-data-cleaning-and-wrangling` to get started!

## Visualization

In this section, we will talk about the packages `matplotlib` and `seaborn`, both used for data visualization in Python. We will cover a range of plots that display different aspects of the dataset. We will also learn how to adjust aesthetics, axis labels and legends, and how to output your plot. Open `03-simple-visualizations` to get started! 

## Supervised Learning

In this section, we will talk about supervised learning - classification and regression. We will use the package `sklearn` to train machine learning models and understand the basic machine learning pipeline. We will explore:
1. Classification - using Decision Trees to predict if a song comes from your library or Harry's
2. Regression - using linear regression to predict loudness of the songs
We will also cover ways to perform model assessment through techniques like train-test-split and confusion matrix. Open `04-regression-and-classification` to get started!


## Unsupervised Learning

## Conclusion

Now you're ready to play with your data and see what you can find. Best of all, whatever you find is specific to you! Here are some fun and/or useful links to better understand the data:

- [Spotify API documentation for what the audio features are](https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/)
- [Data Science article discussing why pop music all sounds the same](https://pudding.cool/2018/05/similarity/)

