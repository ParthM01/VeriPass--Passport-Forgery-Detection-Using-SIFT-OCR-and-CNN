from django.contrib import admin
from .models import Image

class ImageAdmin(admin.ModelAdmin):
    list_display = ('unique_id', 'name', 'surname', 'photo')


admin.site.register(Image, ImageAdmin)
