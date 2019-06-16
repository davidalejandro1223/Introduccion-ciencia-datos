import tweepy
from django.shortcuts import render
from django.http import HttpResponse
from .models import Hashtags, Usuarios
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

def analitica1(request):
    usuarios = Usuarios.objects.all()
    hashtags = Hashtags.objects.all()

    context = {
        'usuarios': usuarios,
        'hashtags': hashtags,
    }

    if request.method == 'POST':
        usuario = request.POST.get('usuario')
        hashtag = request.POST.get('hashtag')

        print (usuario, hashtag)
        
    return render(request, 'analiticas/analitica1.html', context=context)

def analitica2(request):
    usuarios = Usuarios.objects.all()

    context = {
        'usuarios': usuarios
    } 

    if request.method == 'POST':
        usuario = request.POST.get('usuario')
        palabra1 = request.POST.get('palabra1')
        palabra2 = request.POST.get('palabra2')
        palabra3 = request.POST.get('palabra3')
        palabra4 = request.POST.get('palabra4')

        print(usuario, palabra1, palabra2, palabra3, palabra4)
    
    return render(request, 'analiticas/analitica2.html', context=context)
    


def conexion(request):
    api = autenticacion()

    public_tweets = api.home_timeline()
    for tweet in public_tweets:
        print(tweet.text)

    return HttpResponse(public_tweets)


def prueba(request):
    api = autenticacion()

