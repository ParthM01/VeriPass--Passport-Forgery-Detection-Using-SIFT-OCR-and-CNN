from django.db import models

# Create your models here.
import uuid
from django.db import models

class Image(models.Model):
    unique_id = models.CharField(primary_key=True, max_length=50, unique=True)
    name = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    photo = models.ImageField(upload_to='myimage')

    def __str__(self):
        return f"{self.name} {self.surname}"
