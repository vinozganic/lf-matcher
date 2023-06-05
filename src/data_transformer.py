import os
import json
import logging
import requests
from datetime import datetime

import numpy as np
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from gensim.models import KeyedVectors
from pyproj import CRS, Transformer
from shapely.geometry import Point, MultiLineString, LineString
from shapely.ops import unary_union, transform
from sklearn.metrics.pairwise import cosine_similarity


from constants import API_URL
from contracts import item_to_process
from exceptions import APIException, UnknownGeometryType

def patch_asscalar(a):
    return a.item()

setattr(np, 'asscalar', patch_asscalar)

class DataTransformer:
    def __init__(self):
        self.type_similarity_matrix = self._load_type_similarity_matrix()

    def prepare_data(self, lost_item : item_to_process, found_item: item_to_process):
        type_similarity = self._compute_type_similarity(lost_item.type, found_item.type)
        color_distance = self._compute_color_distance(lost_item.color, found_item.color)
        same_transport_line_usage, lost_path_presence, found_path_presence, lost_public_transport_lines_presence, found_public_transport_lines_presence, path_overlap_ratio, public_transport_lines_overlap_ratio, weighted_path_overlap_ratio, weighted_public_transport_lines_overlap_ratio, min_distance, centroid_distance, overlap_ratio, = self._compute_location_parameters(lost_item.location, found_item.location)
        date_distance = self._compute_date_distance(lost_item.date, found_item.date)

        return {
            "type_similarity": type_similarity,
            "color_distance": color_distance,
            "same_transport_line_usage": same_transport_line_usage,
            "lost_path_presence": lost_path_presence,
            "found_path_presence": found_path_presence,
            "lost_public_transport_lines_presence": lost_public_transport_lines_presence,
            "found_public_transport_lines_presence": found_public_transport_lines_presence,
            "path_overlap_ratio": path_overlap_ratio,
            "public_transport_lines_overlap_ratio": public_transport_lines_overlap_ratio,
            # "weighted_path_overlap_ratio": weighted_path_overlap_ratio,
            # "weighted_public_transport_lines_overlap_ratio": weighted_public_transport_lines_overlap_ratio,
            "location_min_distance": min_distance,
            "location_centroid_distance": centroid_distance,
            # "location_overlap_area": overlap_area,
            # "location_overlap_ratio": overlap_ratio,
            "date_distance": date_distance,
        }

    def _compute_type_similarity(self, lost_item_type, found_item_type):
        if self.type_similarity_matrix.get(lost_item_type) is None:
            return 0
        similarity = self.type_similarity_matrix[lost_item_type].get(found_item_type, 0)
        if not similarity:
            similarity = self.type_similarity_matrix[found_item_type].get(lost_item_type, 0)
        return similarity

    def _load_type_similarity_matrix(self):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        matrix = json.load(open(current_dir + "/model/type_similarity_matrix_gpt4.json"))
        return matrix

    def _compute_color_distance(self, lost_item_color, found_item_color):
        color1 = sRGBColor(lost_item_color[0], lost_item_color[1], lost_item_color[2])
        color2 = sRGBColor(found_item_color[0], found_item_color[1], found_item_color[2])

        color1_lab = convert_color(color1, LabColor)
        color2_lab = convert_color(color2, LabColor)

        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        return delta_e
    
    def _compute_date_distance(self, lost_item_date, found_item_date):
        return (found_item_date - lost_item_date).days

    def _compute_location_parameters(self, lost_item_location, found_item_location):
        same_transport_line_usage = self._get_same_transport_line_usage(lost_item_location, found_item_location)

        path_overlap_ratio = self._get_path_overlap_ratio(lost_item_location, found_item_location)
        public_transport_lines_overlap_ratio = self._get_public_transport_lines_overlap_ratio(lost_item_location, found_item_location)

        lost_path_presence = self._get_path_presence(lost_item_location)
        found_path_presence = self._get_path_presence(found_item_location)
        
        lost_public_transport_lines_presence = self._get_public_transport_lines_presence(lost_item_location)
        found_public_transport_lines_presence = self._get_public_transport_lines_presence(found_item_location)

        weighted_path_overlap_ratio = self._get_weighted_path_overlap_ratio(lost_path_presence, found_path_presence, lost_public_transport_lines_presence, found_public_transport_lines_presence, path_overlap_ratio)
        weighted_public_transport_lines_overlap_ratio = self._get_weighted_public_transport_lines_overlap_ratio(lost_path_presence, found_path_presence, lost_public_transport_lines_presence, found_public_transport_lines_presence, public_transport_lines_overlap_ratio)

        lost_geom = self._get_geometry(lost_item_location)
        lost_geom = self._transform_geometry_to_wgs84(lost_geom)
        found_geom = self._get_geometry(found_item_location)
        found_geom = self._transform_geometry_to_wgs84(found_geom)

        lost_geom_buffer = lost_geom.buffer(500)
        found_geom_buffer = found_geom.buffer(500)

        min_distance = self._get_min_distance(lost_geom, found_geom)
        centroid_distance = self._get_centroid_distance(lost_geom, found_geom)
        overlap_area = self._get_overlap_area(lost_geom_buffer, found_geom_buffer)
        overlap_ratio = self._get_overlap_ratio(lost_geom_buffer, found_geom_buffer, overlap_area)

        return same_transport_line_usage, lost_path_presence, found_path_presence, lost_public_transport_lines_presence, found_public_transport_lines_presence, path_overlap_ratio, public_transport_lines_overlap_ratio, weighted_path_overlap_ratio, weighted_public_transport_lines_overlap_ratio, min_distance, centroid_distance, overlap_ratio
    
    def _get_path_presence(self, item_location):
        if item_location.get("path"):
            return 1
        return 0
    
    def _get_public_transport_lines_presence(self, item_location):
        if item_location.get("publicTransportLines"):
            return 1
        return 0

    def _get_same_transport_line_usage(self, lost_item_location, found_item_location):
        lost_transport_lines = lost_item_location.get("publicTransportLines")
        found_transport_lines = found_item_location.get("publicTransportLines")
        if not lost_transport_lines or not found_transport_lines:
            return 0
        lost_transport_lines = [line["coordinates"] for line in lost_transport_lines]
        found_transport_lines = [line["coordinates"] for line in found_transport_lines]
        for lost_line in lost_transport_lines:
            for found_line in found_transport_lines:
                if lost_line == found_line:
                    return 1
        return 0

    def _get_geometry(self, item_location):
        path_geom = None
        if item_location.get("path") is not None:
            if item_location["path"]["type"] == "Point":
                path_geom = Point(item_location["path"]["coordinates"])
            elif item_location["path"]["type"] == "MultiLineString":
                path_geom = MultiLineString(item_location["path"]["coordinates"])

        public_transport_lines_geom = None
        if item_location.get("publicTransportLines") is not None:
            public_transport_lines = []
            for line in item_location["publicTransportLines"]:
                public_transport_lines.append(line["coordinates"])
            public_transport_lines_geom = MultiLineString(public_transport_lines)

        geom = None
        if path_geom is not None and public_transport_lines_geom is not None:
            geom = unary_union([path_geom, public_transport_lines_geom])
        elif path_geom is not None:
            geom = path_geom
        elif public_transport_lines_geom is not None:
            geom = public_transport_lines_geom

        if geom is None:
            raise UnknownGeometryType("Unknown geometry type")
        
        return geom
    
    def _transform_geometry_to_wgs84(self, geom):
        source_crs = CRS.from_epsg(4326)
        target_crs = CRS.from_epsg(32633)

        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        geom_wgs84 = transform(transformer.transform, geom)

        return geom_wgs84
        
    def _get_min_distance(self, lost_geom, found_geom):
        return lost_geom.distance(found_geom)
    
    def _get_centroid_distance(self, lost_geom, found_geom):
        return lost_geom.centroid.distance(found_geom.centroid)
    
    def _get_overlap_area(self, lost_geom_buffer, found_geom_buffer):
        return lost_geom_buffer.intersection(found_geom_buffer).area
    
    def _get_overlap_ratio(self, lost_geom_buffer, found_geom_buffer, overlap_area):
        return overlap_area / (lost_geom_buffer.area + found_geom_buffer.area - overlap_area)
    
    def _get_path_overlap_ratio(self, lost_item_location, found_item_location):
        # the possible geometry types are Point and MultiLineString
        # create buffer around the them and calculate the overlap ratio
        lost_path = lost_item_location.get("path")
        found_path = found_item_location.get("path")
        if lost_path is None or found_path is None:
            return 0
        lost_path_geom = None
        found_path_geom = None
        if lost_path["type"] == "Point":
            lost_path_geom = Point(lost_path["coordinates"])
        elif lost_path["type"] == "MultiLineString":
            lost_path_geom = MultiLineString(lost_path["coordinates"])
        if found_path["type"] == "Point":
            found_path_geom = Point(found_path["coordinates"])
        elif found_path["type"] == "MultiLineString":
            found_path_geom = MultiLineString(found_path["coordinates"])
        if lost_path_geom is None or found_path_geom is None:
            return 0
        
        lost_path_geom = self._transform_geometry_to_wgs84(lost_path_geom)
        found_path_geom = self._transform_geometry_to_wgs84(found_path_geom)

        lost_path_geom_buffer = lost_path_geom.buffer(500)
        found_path_geom_buffer = found_path_geom.buffer(500)
        overlap_area = lost_path_geom_buffer.intersection(found_path_geom_buffer).area
        return overlap_area / (lost_path_geom_buffer.area + found_path_geom_buffer.area - overlap_area)

    from shapely.ops import unary_union

    def _get_public_transport_lines_overlap_ratio(self, lost_item_location, found_item_location):
        """
        Calculate the overlap ratio between public transport lines of a lost and found item.
        
        Parameters:
        - lost_item_location: Location information of the lost item
        - found_item_location: Location information of the found item
        
        Returns:
        - The overlap ratio between the public transport lines of the lost and found items
        """
        lost_public_transport_lines = lost_item_location.get("publicTransportLines")
        found_public_transport_lines = found_item_location.get("publicTransportLines")
        if not lost_public_transport_lines or not found_public_transport_lines:
            return 0

        lost_public_transport_lines_geom = [
            self._transform_geometry_to_wgs84(LineString(line["coordinates"]))
            for line in lost_public_transport_lines
        ]
        found_public_transport_lines_geom = self._transform_geometry_to_wgs84(
            LineString(found_public_transport_lines[0]["coordinates"])
        )

        lost_public_transport_lines_geom_buffer = [
            line.buffer(500)
            for line in lost_public_transport_lines_geom
        ]
        found_public_transport_lines_geom_buffer = found_public_transport_lines_geom.buffer(500)

        lost_public_transport_lines_geom_buffer_union = unary_union(lost_public_transport_lines_geom_buffer)

        overlap_area = lost_public_transport_lines_geom_buffer_union.intersection(
            found_public_transport_lines_geom_buffer).area

        total_area = lost_public_transport_lines_geom_buffer_union.area + found_public_transport_lines_geom_buffer.area - overlap_area

        return overlap_area / total_area
    
    def _get_weighted_path_overlap_ratio(self, lost_path_presence, found_path_presence, lost_public_transport_lines_presence, found_public_transport_lines_presence, path_overlap_ratio):
        if lost_path_presence == 1 and found_path_presence == 1:
            return path_overlap_ratio
        return 0
    
    def _get_weighted_public_transport_lines_overlap_ratio(self, lost_path_presence, found_path_presence, lost_public_transport_lines_presence, found_public_transport_lines_presence, public_transport_lines_overlap_ratio):
        if lost_public_transport_lines_presence == 1 and found_public_transport_lines_presence == 1:
            return public_transport_lines_overlap_ratio
        return 0

    def get_types_from_db(self):
        response = requests.get(f"{API_URL}/config/types").json()
        if response["success"]:
            data = response["data"]
            return data
        else:
            raise APIException("Could not get items from database") 

    def calculate_similarity_matrixes(self):
        if self.type_similarity_matrix:
            return self.type_similarity_matrix

        all_types = self.get_types_from_db()
        print("All types: ", all_types)

        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model = KeyedVectors.load_word2vec_format(current_dir + "/model/glove.42B.300d.txt", binary=False, no_header=True)

        type_vectors = {}
        for type in all_types:
            if not type in model:
                logging.info(f"Type {type} not in model")
                continue
            type_vectors[type] = model[type]

        type_similarity_matrix = {}
        for type1 in all_types:
            type_similarity_matrix[type1] = {}
            type_similarity_matrix[type1][type1] = 1.0
            for type2 in all_types:
                if not type1 in model:
                    logging.info(f"Type {type1} not in model")
                    break
                if not type2 in model:
                    logging.info(f"Type {type2} not in model")
                    continue
                similarity = cosine_similarity([type_vectors[type1]], [type_vectors[type2]])[0][0]
                type_similarity_matrix[type1][type2] = float(similarity)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        with open(current_dir + "/model/type_similarity_matrix.json", "w") as fp:
            json.dump(type_similarity_matrix, fp)


        self.type_similarity_matrix = type_similarity_matrix
        return type_similarity_matrix