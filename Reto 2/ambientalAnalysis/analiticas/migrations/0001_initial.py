# Generated by Django 2.2.2 on 2019-06-14 17:41

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Hashtags',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('hashtag', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Usuarios',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('arroba', models.CharField(max_length=50)),
                ('nombre_cuenta', models.CharField(max_length=50)),
            ],
        ),
    ]