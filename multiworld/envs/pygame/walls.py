"""
Basic usage:
```

v_wall = VerticalWall(width, x_pos, bottom_y, top_y)
ball = Ball(radius=width)

init_xy = ...
new_xy = init_xy + velocity

if v_wall.collides_with(init_xy, new_xy):
    new_xy = v_wall.handle_collision(init_xy, new_xy)
```
"""
import abc


class Wall(object, metaclass=abc.ABCMeta):
    def __init__(self, min_x, max_x, min_y, max_y, min_dist, thickness,
                 epsilon_from_wall=0.01):
        self.top_segment = Segment(
            min_x,
            max_y,
            max_x,
            max_y,
        )
        self.bottom_segment = Segment(
            min_x,
            min_y,
            max_x,
            min_y,
        )
        self.left_segment = Segment(
            min_x,
            min_y,
            min_x,
            max_y,
        )
        self.right_segment = Segment(
            max_x,
            min_y,
            max_x,
            max_y,
        )
        self.segments = [
            self.top_segment,
            self.bottom_segment,
            self.right_segment,
            self.left_segment,
        ]
        self.min_dist = min_dist
        self.thickness = thickness
        self.epsilon_from_wall = epsilon_from_wall
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y

    def contains_point(self, point):
        return (self.min_x < point[0] < self.max_x) and (
                    self.min_y < point[1] < self.max_y)

    def handle_collision(self, start_point, end_point):
        trajectory_segment = (
            start_point[0],
            start_point[1],
            end_point[0],
            end_point[1],
        )
        if (self.top_segment.intersects_with(trajectory_segment) and
                end_point[1] <= start_point[1] >= self.max_y):
            end_point[1] = self.max_y + self.epsilon_from_wall
        if (self.bottom_segment.intersects_with(trajectory_segment) and
                end_point[1] >= start_point[1] <= self.min_y):
            end_point[1] = self.min_y - self.epsilon_from_wall
        if (self.right_segment.intersects_with(trajectory_segment) and
                end_point[0] <= start_point[0] >= self.max_x):
            end_point[0] = self.max_x + self.epsilon_from_wall
        if (self.left_segment.intersects_with(trajectory_segment) and
                end_point[0] >= start_point[0] <= self.min_x):
            end_point[0] = self.min_x - self.epsilon_from_wall
        return end_point


class Segment(object):
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def intersects_with(self, s2):
        # return do_intersect(
        #     (self.x0, self.y0),
        #     (self.x1, self.y1),
        #     (s2[0], s2[1]),
        #     (s2[2], s2[3]),
        # )
        left = max(min(self.x0, self.x1), min(s2[0], s2[2]))
        right = min(max(self.x0, self.x1), max(s2[0], s2[2]))
        bottom = max(min(self.y0, self.y1), min(s2[1], s2[3]))
        top = min(max(self.y0, self.y1), max(s2[1], s2[3]))

        if bottom > top or left > right:
            return False

        return True


class VerticalWall(Wall):
    def __init__(self, min_dist, x_pos, bottom_y, top_y, thickness=0.0):
        min_y = bottom_y - min_dist - thickness
        max_y = top_y + min_dist + thickness
        assert min_y < max_y
        min_x = x_pos - min_dist - thickness
        max_x = x_pos + min_dist + thickness
        super().__init__(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_dist=min_dist,
            thickness=thickness,
        )
        self.endpoint1 = (x_pos + thickness, top_y + thickness)
        self.endpoint2 = (x_pos + thickness, bottom_y - thickness)
        self.endpoint3 = (x_pos - thickness, bottom_y - thickness)
        self.endpoint4 = (x_pos - thickness, top_y + thickness)


class HorizontalWall(Wall):
    def __init__(self, min_dist, y_pos, left_x, right_x, thickness=0.0):
        min_y = y_pos - min_dist - thickness
        max_y = y_pos + min_dist + thickness
        min_x = left_x - min_dist - thickness
        max_x = right_x + min_dist + thickness
        assert min_x < max_x
        super().__init__(
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            min_dist=min_dist,
            thickness=thickness,
        )
        self.endpoint1 = (right_x + thickness, y_pos + thickness)
        self.endpoint2 = (right_x + thickness, y_pos - thickness)
        self.endpoint3 = (left_x - thickness, y_pos - thickness)
        self.endpoint4 = (left_x - thickness, y_pos + thickness)


# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def on_segment(p, q, r):
    """
    Given three colinear points p, q, r, the function checks if point q lies on
    line segment "pr"
    """
    if (max(p[0], r[0]) >= q[0] >= min(p[0], r[0]) and
            max(p[1], r[1]) >= q[1] >= min(p[1], r[1])):
        return True
    return False


def orientation(p, q, r):
    """
    Find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    """

    val = ((q[1] - p[1]) * (r[0] - q[0]) -
           (q[0] - p[0]) * (r[1] - q[1]))
    if val == 0:
        return 0  # colinear
    elif val > 0:
        return 1  # clockwise
    else:
        return 2  # counter-clockwise


def do_intersect(p1, q1, p2, q2):
    """
    Main function to check whether the closed line segments p1 - q1 and p2 - q2
    intersect
    """
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False  # Doesn't fall in any of the above cases
