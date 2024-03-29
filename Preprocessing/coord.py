# ******************************************************************************************
# Author: Luis Ernesto Colchado Soncco
# Email: luis.colchado@ucsp.edu.pe / luisernesto.200892@gmail.coom
# Description: Class Coord (lat, long)
# ******************************************************************************************
from pyproj import Proj
import math

class Coord():
    def __init__(self,lat,long):
        self.lat = lat
        self.long = long
        

    def convertToPoint(self):
        point_proj = Proj('epsg:3857')
        x, y = point_proj(self.long, self.lat)
        return x, y
        

    def haversineDistance(self, other_coord):
        R = 6356.752
        lat1, lon1 = self.lat, self.long
        lat2, lon2 = other_coord.lat, other_coord.long

        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))