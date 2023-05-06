# api/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('video_feed/', views.VideoFeedView.as_view(), name='video_feed'),
    path('path/', views.PathView.as_view(), name='path'),

]
