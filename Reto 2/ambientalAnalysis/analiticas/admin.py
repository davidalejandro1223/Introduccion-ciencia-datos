from django.contrib import admin
from .models import Usuarios, Hashtags

# Register your models here.

@admin.register(Usuarios)
class AdminUsuarios(admin.ModelAdmin):
    list_display = ('id', 'arroba', 'nombre_cuenta')


@admin.register(Hashtags)
class AdminHashtags(admin.ModelAdmin):
    list_display = ('id', 'hashtag')