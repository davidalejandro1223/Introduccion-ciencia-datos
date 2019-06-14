import tweepy
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def autenticacion():
    consumer_key = 'stKuSL7WMShoaR0Se9fwHIkjW'
    consumer_secret = 'aIvT6hm6GqxaW0BfPe0VUZcwJJEHDebIawU9N9B8KGAckiog1Q'
    access_token = '293588703-08HTWRdhrTGkI4zjdlApfsy8tmPsqrGeigDubOqf'
    access_token_secret = 'Pvkfl4zA5RsGo8uBbX2GKHxEUnRKR01kYGh7IQVj805sv'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    return api

def home(request):
    return render(request, 'analiticas/index.html', context=None)

def conexion(request):
    api = autenticacion()

    public_tweets = api.home_timeline()
    for tweet in public_tweets:
        print(tweet.text)

    return HttpResponse(public_tweets)


def prueba(request):
    api = autenticacion()

