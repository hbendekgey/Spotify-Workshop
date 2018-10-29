# Spotify Data Science Project

### Getting the Data

1. Go to https://developer.spotify.com/dashboard and make a client.

2. Go to https://accounts.spotify.com/authorize?client_id={yourclientid}&redirect_uri=https://www.spotify.com&scope=user-library-read&response_type=token

3. notice url is https://www.spotify.com/us/#access_token={token}&token_type=Bearer&expires_in=3600 
Save that token.

4. Make a curl request for songs by going to the command line (terminal) and writing:
```
curl -X GET "https://api.spotify.com/v1/me/tracks" -H "Authorization: Bearer {yourtoken}" > songs.json
```

Now you have a file named songs.json which contains all songs in your music library!

5. bla

```
curl -X GET "https://api.spotify.com/v1/audio-features/{songid}" -H "Authorization: Bearer {yourtoken}"
```

Now we have all of the song data!

6. Get that into a dataframe in Python and play around with it!


### Topics

Ahead of time: get Anaconda (necessary)

If you want your own data: get pip, use pip to install spotipy, and make sure you can log in to your spotify account


1. Get set up, get data (20 min) [Harry]
2. How do we operate on our data? (pandas, numpy, etc.) (20-30 min) [Zihao]

    a. How to combine data frames
3. Simple visualization of two coordinates (plot danceability vs tempo) (5-10 min) [Zihao]
4. Classification and Regression, and packages that will be helpful [Zihao]
5. Clustering and Dimensionality Reduction, and packages that will be helpful [Harry]
6. Have example usage up, and set people loose

### Need to be made:

Zihao: make notebook file for 2-4

Harry: figure out data collecting details, make notebook for 5/add it to end of Zihao's notebook file when sent

### Helpful Sites:

1. https://developer.spotify.com/documentation/web-api/reference/tracks/get-audio-features/

2. https://pudding.cool/2018/05/similarity/

