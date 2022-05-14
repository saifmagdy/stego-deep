"""boardmaster357 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from webapp import views

urlpatterns = [


    path('image-encode/', views.ImageEncode, name="image-encode"),
    path('image-decodeTwoLeast/', views.ImagedecodeTwoLeast, name="image-decodeTwoLeast"),
    path('image-decodeLeast/', views.ImagedecodeLeast, name="image-decodeLeast"),
    path('Audio-Least/', views.AudioLeast, name="Audio-Least"),
    path('Audio-two-Least/', views.AudioTwoLeast, name="Audio-two-Least"),
    path('Audio-Encode/', views.AudioEncode, name="Audio-Encode"),
    path('Audio-two-Encode/', views.AudioTwoEncode, name="Audio-two-Encode"),
    
    
]