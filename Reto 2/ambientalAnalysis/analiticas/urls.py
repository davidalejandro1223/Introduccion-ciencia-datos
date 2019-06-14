from django.urls import include, path
from .views import conexion, prueba, home

urlpatterns = [
    #path("pruebas/", .as_view(), name=""),
    path("", home, name="home"),
    path("prueba", prueba, name='prueba'),
]
