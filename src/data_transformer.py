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
from shapely.geometry import Point, MultiLineString
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
        min_distance, centroid_distance, overlap_area, overlap_ratio_lost, overlap_ratio_found = self._compute_location_parameters(lost_item.location, found_item.location)
        date_distance = self._compute_date_distance(lost_item.date, found_item.date)

        return {
            "type_similarity": type_similarity,
            "color_distance": color_distance,
            "location_min_distance": min_distance,
            "location_centroid_distance": centroid_distance,
            "location_overlap_area": overlap_area,
            "location_overlap_ratio_lost": overlap_ratio_lost,
            "location_overlap_ratio_found": overlap_ratio_found,
            "date_distance": date_distance,
        }

    def _compute_type_similarity(self, lost_item_type, found_item_type):
        if self.type_similarity_matrix.get(lost_item_type) is None:
            return 0
        return self.type_similarity_matrix[lost_item_type].get(found_item_type, 0)
    
    def _load_type_similarity_matrix(self):
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        matrix = json.load(open(current_dir + "/model/subtype_similarity_matrix.json"))
        return matrix

    def _compute_color_distance(self, lost_item_color, found_item_color):
        color1 = sRGBColor(lost_item_color[0], lost_item_color[1], lost_item_color[2])
        color2 = sRGBColor(found_item_color[0], found_item_color[1], found_item_color[2])

        color1_lab = convert_color(color1, LabColor)
        color2_lab = convert_color(color2, LabColor)

        delta_e = delta_e_cie2000(color1_lab, color2_lab)
        return delta_e
    
    def _compute_date_distance(self, lost_item_date, found_item_date):
        # lost_date = datetime.strptime(lost_item_date, "%Y-%m-%dT%H:%M:%S.%fZ")
        # found_date = datetime.strptime(found_item_date, "%Y-%m-%dT%H:%M:%S.%fZ")
        return abs((lost_item_date - found_item_date).days)

    def _compute_location_parameters(self, lost_item_location, found_item_location):
        lost_geom = self._get_geometry(lost_item_location)
        found_geom = self._get_geometry(found_item_location)
        
        lost_geom = self._transform_geometry_to_wgs84(lost_geom)
        found_geom = self._transform_geometry_to_wgs84(found_geom)

        min_distance = self._get_min_distance(lost_geom, found_geom)
        centroid_distance = self._get_centroid_distance(lost_geom, found_geom)
        overlap_area, overlap_ratio_lost, overlap_ratio_found = self._get_overlap_ratio(lost_geom, found_geom)

        return min_distance, centroid_distance, overlap_area, overlap_ratio_lost, overlap_ratio_found

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
    
    def _get_overlap_ratio(self, lost_geom, found_geom):
        lost_geom_buffer = lost_geom.buffer(800)
        found_geom_buffer = found_geom.buffer(800)
        
        overlap_area = self._get_overlap_area(lost_geom_buffer, found_geom_buffer)

        overlap_ratio_lost = overlap_area / lost_geom_buffer.area
        overlap_ratio_found = overlap_area / found_geom_buffer.area

        return overlap_area, overlap_ratio_lost, overlap_ratio_found
    
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
        with open(current_dir + "/model/subtype_similarity_matrix.json", "w") as fp:
            json.dump(type_similarity_matrix, fp)


        self.type_similarity_matrix = type_similarity_matrix
        return type_similarity_matrix