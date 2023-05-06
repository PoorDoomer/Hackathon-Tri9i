# api/views.py

import cv2
import numpy as np
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import VideoFeedSerializer,PathRequestSerializer
import json
import networkx as nx
from geopy.distance import great_circle
from datetime import timedelta
import requests
# from retinaface import RetinaFace as RetinaFaceModel
from retinaface import RetinaFace
def process_video_feed(video_url):
    # Load the DNN Face Detection model
    # model_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    # config_file = "models/deploy.prototxt"
    # net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    # model = RetinaFaceModel(quality="normal")
    cap = cv2.VideoCapture(video_url, cv2.CAP_FFMPEG)


    # Initialize the number of faces
    num_faces = 0

    # Number of frames to capture for face detection
    num_frames_to_capture = 50

    
        # Read the video feed
    ret, frame = cap.read()
    if ret :

        # Detect faces using RetinaFace
        # faces = model.predict(frame)
        resp = RetinaFace.detect_faces(frame)
            # Count the number of faces
        num_faces += len(resp)

    cap.release()

    return num_faces, frame

class VideoFeedView(APIView):

    def post(self, request):
        serializer = VideoFeedSerializer(data=request.data)

        if serializer.is_valid():
            video_url = serializer.validated_data['url']
            num_faces, frame = process_video_feed(video_url)

            # Determine if the bus is empty or full
            bus_status = "empty" if num_faces <= 5 else "full"

            return Response({"num_faces": num_faces, "bus_status": bus_status}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



def build_graph(stations, routes):
    graph = nx.Graph()

    for station in stations:
        graph.add_node(station['id'], name=station['name'], location=(station['location']['latitude'], station['location']['longitude']), type=station['type'])

    for route in routes:
        for i in range(len(route['stations']) - 1):
            station_a = route['stations'][i]
            station_b = route['stations'][i + 1]

            location_a = graph.nodes[station_a]['location']
            location_b = graph.nodes[station_b]['location']
            distance = great_circle(location_a, location_b).meters

            arrival_time_a = route['arrival_times'][i]
            arrival_time_b = route['arrival_times'][i + 1]

            travel_time = (arrival_time_b - arrival_time_a).total_seconds()

            graph.add_edge(station_a, station_b, vehicle_id=route['vehicle_id'], travel_time=travel_time, distance=distance)

    return graph
def find_nearest_station(graph, point):
    nearest_station = None
    min_distance = float('inf')

    for node in graph.nodes:
        station_location = graph.nodes[node]['location']
        distance = great_circle(point, station_location).meters

        if distance < min_distance:
            min_distance = distance
            nearest_station = node

    return nearest_station


def find_best_route(graph, point_a, point_b, desired_arrival_datetime):
    start_station = find_nearest_station(graph, point_a)
    end_station = find_nearest_station(graph, point_b)

    try:
        shortest_path = nx.shortest_path(graph, start_station, end_station, weight='travel_time')
    except nx.NetworkXNoPath:
        return None

    travel_time = 0
    route = []

    for i in range(len(shortest_path) - 1):
        station_a = shortest_path[i]
        station_b = shortest_path[i + 1]
        edge_data = graph.edges[station_a, station_b]

        route.append({
            'from_station': station_a,
            'to_station': station_b,
            'vehicle_id': edge_data['vehicle_id'],
            'distance': edge_data['distance'],
            'travel_time': edge_data['travel_time']
        })

        travel_time += edge_data['travel_time']

    arrival_time = desired_arrival_datetime - timedelta(seconds=travel_time)

    return {
        'route': route,
        'arrival_time': arrival_time
    }



class PathView(APIView):

    def post(self, request):
        serializer = PathRequestSerializer(data=request.data)

        if serializer.is_valid():
            point_a = serializer.validated_data['point_a']
            point_b = serializer.validated_data['point_b']
            desired_arrival_datetime = serializer.validated_data['desired_arrival_datetime']

            # Request the relevant stations and routes data from the Java Spring application
            java_spring_url = 'http://java_spring_api_address/get_relevant_data/'
            java_spring_response = requests.post(java_spring_url, json={"point_a": point_a, "point_b": point_b})

            if java_spring_response.status_code == 200:
                java_spring_data = java_spring_response.json()
                stations = java_spring_data['stations']
                routes = java_spring_data['routes']

                graph = build_graph(stations, routes)
                best_route = find_best_route(graph, point_a, point_b, desired_arrival_datetime)

                if best_route:
                    # Get the vehicle URLs used in the path
                    relevant_vehicle_urls = {}
                    for step in best_route['route']:
                        vehicle_id = step['vehicle_id']
                        vehicle_url = step['url']
                        relevant_vehicle_urls[vehicle_id] = vehicle_url

                    # Get the vehicle statuses
                    vehicle_statuses = {}
                    for vehicle_id, video_url in relevant_vehicle_urls.items():
                        num_faces, _ = process_video_feed(video_url)
                        vehicle_statuses[vehicle_id] = "full" if num_faces >= 10 else "empty"

                    return Response({"path": best_route['route'], "arrival_time": best_route['arrival_time'], "vehicle_statuses": vehicle_statuses}, status=status.HTTP_200_OK)
                else:
                    return Response({"error": "No path found"}, status=status.HTTP_404_NOT_FOUND)
            else:
                return Response({"error": "Failed to fetch data from Java Spring API"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)