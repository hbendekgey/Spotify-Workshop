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