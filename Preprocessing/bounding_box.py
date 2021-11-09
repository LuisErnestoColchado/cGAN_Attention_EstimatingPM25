class Bounding_box():
    def __init__(self, id, dim, left_top, right_bottom):
        self.id = id
        self.dim = dim
        self.left_top = left_top
        self.right_bottom = right_bottom

    def getCenter(self):
        x_center = (self.left_top.x + self.right_bottom.x) / 2
        y_center = (self.left_top.y + self.right_bottom.y) / 2
        return x_center, y_center