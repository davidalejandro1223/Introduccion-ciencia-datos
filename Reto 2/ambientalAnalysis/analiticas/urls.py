from django.urls import include, path
from .views import conexion, prueba, home, analitica1

app_name='analiticas'

urlpatterns = [
    #path("pruebas/", .as_view(), name=""),
    path("", home, name="home"),
    path("prueba/", prueba, name='prueba'),
    path('analitica1/', analitica1, name='analitica1'),
]
