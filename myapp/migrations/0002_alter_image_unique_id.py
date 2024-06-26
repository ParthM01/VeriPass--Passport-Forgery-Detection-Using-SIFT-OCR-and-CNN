# Generated by Django 4.2.6 on 2024-01-20 13:23

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='unique_id',
            field=models.UUIDField(default=uuid.uuid4),
        ),
    ]