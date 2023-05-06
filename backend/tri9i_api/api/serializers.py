# api/serializers.py

from rest_framework import serializers

class VideoFeedSerializer(serializers.Serializer):
    url = serializers.URLField()

class PathRequestSerializer(serializers.Serializer):
    point_a = serializers.CharField()
    point_b = serializers.CharField()
    desired_arrival_datetime = serializers.DateTimeField()
