
  
from pyproj import Proj, transform
import math

class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def convertToCoord(self):
        in_proj = Proj('epsg:3857') # China data
        #in_proj = Proj(init='epsg:32723') # Brazil data
        out_proj = Proj('epsg:4326')
        long, lat = transform(in_proj, out_proj, self.x, self.y)
        return long, lat

    def isWithinBB(self, bb):
        isWithin = False
        if (self.x >= bb.left_top.x and self.y <= bb.left_top.y):
            if (self.x <= bb.right_bottom.x and self.y >= bb.right_bottom.y):
                isWithin = True
        return isWithin
    
    def euclidean_distance(self, other_point):
        distance = math.sqrt(math.pow(self.x - other_point.x,2) + math.pow(self.y - other_point.y,2))
        return distance
