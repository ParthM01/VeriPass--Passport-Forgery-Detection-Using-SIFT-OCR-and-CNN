from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from myapp import views
from myapp.views import your_success_view
from django.urls import reverse

urlpatterns = [
    path('admin/', admin.site.urls),
    path('success/', your_success_view, name='your_success_url'),
    path('', views.home)
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# print(static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT))
