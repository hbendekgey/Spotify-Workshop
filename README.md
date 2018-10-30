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

In this section, we will play with the package called `pandas`, which is the primary package for data clearning, wrangling and analysis in Python. We will introduce the concept of _DataFrame_ and _Series_ and cover the basic commands, ways to do indexing, some operations on different columns, and how to combine dataframes. Open `02-data-cleaning-and-wrangling` to get started!  

We will start by importing the necessary packages:
```
import pandas as pd
import numpy as np
%matplotlib inline
```

`pandas` is the package used for handling datasets and  `numpy` is used for numeric and matrix operations. `%matplotlib inline` is there so that the plots are rendered directly below each cell.

Then, we will start by loading in `harry.csv` by calling:

```
df = pd.read_csv('<file_name>.csv')
```

This should read in the `.csv` file into the object called `df`. We try print out the type of `df` by calling `type(df)`. You will see that it is of type `pandas.core.frame.DataFrame`. Now there are a lot of very basic, but useful commands you could call on this `df` object. Explore each of the following commands in __separate__ cells:

```
# Print all column names
df.columns
# Or
list(df)

# Print head/tail 5 rows of df
df.head() # Or df.tail()

# Print df info
df.info()

# Print shape (dimensions) of df 
df.shape

# Print data types of each column
df.dtypes

# Print summary stats of df's numeric columns
df.describe()

# Print correlations in between numeric columns in df
df.corr()
```

Explore the output of each command and see what they do! In the following, we move on to talk about indexing using `pandas`. Indexing basically means "selecting" :^) . Again, execute each of the commands below in __separate__ cells:

```
# Selecting one column: 'acousticness'
df['acousticness'] # This might be too long, to just print out the first 5 by 'df['acousticness'][:5]'

# Print out the type of this column:
type(df['acousticness'])

# Index both: 'acousticness', 'danceability'
df[['acousticness', 'danceability']]

# Indexing the first row using .loc
df.loc[0, :]

# Indexing using a logic: acousticness > 0.95
df[df['acousticness'] > 0.95]

# Indexing using combos of logic: acousticness > 0.95 && mode == 0
df[(df['acousticness'] > 0.95) & (df['mode']== 1)]
```

Now we have explored the commands for indexing, let's move on to talk about commands on the columns, such as how to extract information from existing columns and how to create new columns:

```
# Create a new column called "acousticness_2" that is 2 * values in acousticness
df['acousticness_2'] = df['acousticness'] * 2
df.head()

# Print the unique 'key', number of unique artists
df['key'].unique()

# The value counts for 'key'
df['key'].unique()

# Percentage of artists
df['key'].unique()/len(df)

# Quick histogram of 'acousticness'
df['acousticness'].hist()

# Groupby 'key', agg fuctions
df.groupby('key').agg(['mean', 'median'])

# Crosstabs on 'key' and 'mode'
pd.crosstab(df['key'], df['mode'])
```

These are all useful commands you may use in your Data Science journey onwards! Finally, we will look at how to concatenate and merge DataFrames:

```
# Read in YOUR data set
me = pd.read_csv('me.csv')

# Set a 'label' column for each dataframe
df['label'] = 0
me['label'] = 1

# Append/cancat two datasets!
append = df.append(me)
append.shape
``` 

This concludes section 2!

## Visualization

In this section, we will talk about the packages `matplotlib` and `seaborn`, both used for data visualization in Python. We will cover a range of plots that display different aspects of the dataset. We will also learn how to adjust aesthetics, axis labels and legends, and how to output your plot. Open `03-simple-visualizations` to get started! 

Again, we start by importing the necessary packages and loading in the data set:
```
# Import pandas, matplotlib.pyplot, seaborn, set matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

```
# Load in the data set from harry and you!
ha = pd.read_csv('harry.csv')
me = pd.read_csv('me.csv')
```

We will first look at the more basic (and flexible) package called `matplotlib`. Below, we will explore a number of useful graphs:
```
# Histogram of 'acousticness' in harry.csv
plt.hist(ha['acousticness'])

# Add x, y-axis, title
plt.hist(ha['acousticness'])
plt.xlabel('acousticness')
plt.ylabel('Frequency')
plt.title('Distribution of Acousticness')

# Scatter plot between 'energy', 'loudness'
plt.scatter(ha['energy'], ha['loudness'])

# Change color of points and the markers
plt.scatter(ha['energy'], ha['loudness'], c = 'r', marker = '+') # Explore c = 'b', 'g', 'k'..., and marker = '*', ',', 'v', '2'...

# Add labels and display labels using legend
plt.scatter(ha['energy'], ha['loudness'], c = 'r', marker = '+', label = 'harry')
plt.scatter(me['energy'], me['loudness'], c = 'g', marker = '2', label = 'me')
plt.legend()

# Save a plot to disk
plt.scatter(ha['energy'], ha['loudness'], c = 'r', marker = '+', label = 'harry')
plt.scatter(me['energy'], me['loudness'], c = 'g', marker = '2', label = 'me')
plt.legend()
plt.savefig('plots/cool_plot', fmt = 'png', dpi = 300)
```

This is already pretty cool, but we have something coolER - `seaborn`. `seaborn` is a package built on top of `matplotlib` and it incorporates amazing functionalities. Let's get started:

```
# Plot distribution of 'acousticness'
sns.distplot(ha['acousticness'])

# jointplot: fancy joint-distributions for 'energy', 'loudness', explore kind = 'kde'
sns.jointplot(data = ha, x = 'energy', y = 'loudness', kind = 'kde')

# countplot: count number of occurrences for each 'key', explore using x = 'key' and y = 'key'
sns.countplot(data = ha, x = 'key')

# barplot: plot averge 'energy' per 'key'
sns.barplot(data = ha, x = 'key', y = 'energy')

# swarmplot: for 'key' and 'energy', explore hue = 'mode'
sns.swarmplot(data = ha, x = 'key', y = 'energy')

# pairplot: scatter plot for multiple columns at the same time
# use: 'acousticness', 'loudness', 'energy', 'key', explore hue = 'key'
sns.pairplot(ha[['acousticness', 'loudness', 'energy', 'key']])

# regplot: 'energy' and 'loudness'
sns.regplot(data = ha, x = 'energy', y = 'loudness')
```

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

