import math

from shapely.geometry import LineString

from internnav.utils.common_log_util import common_logger as log


class AStarPlanner:
    def __init__(self, map_width=500, map_height=500, max_step=10000):
        """
        Initialize grid map for a star planning.
        Note that this class does not consider the robot's radius. So the given obs_map should be expanded
        """
        self.resolution = 1
        self.max_step = max_step
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = map_width, map_height
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = map_width, map_height
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + ',' + str(self.y) + ',' + str(self.cost) + ',' + str(self.parent_index)

    def get_angle_cost(self, gx, gy, current, x, y):
        import math

        # calculate the vector from current to goal
        vector_current_to_goal = (gx - current.x, gy - current.y)

        # calculate the vector from current to new point
        vector_current_to_new = (x - current.x, y - current.y)

        # calculate the dot product of the two vectors
        dot_product = (
            vector_current_to_goal[0] * vector_current_to_new[0] + vector_current_to_goal[1] * vector_current_to_new[1]
        )

        # calculate the magnitude of the two vectors
        magnitude_current_to_goal = math.sqrt(vector_current_to_goal[0] ** 2 + vector_current_to_goal[1] ** 2)
        magnitude_current_to_new = math.sqrt(vector_current_to_new[0] ** 2 + vector_current_to_new[1] ** 2)

        # calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_current_to_goal * magnitude_current_to_new)

        # ensure the cosine of the angle is between -1 and 1, prevent numerical errors
        cos_theta = max(-1.0, min(1.0, cos_theta))

        # calculate the angle
        theta = math.acos(cos_theta)

        # convert the angle to cost: the larger the angle, the higher the cost
        angle_cost = 100 * theta  # you can adjust this weight as needed

        return angle_cost

    def planning(
        self, sx, sy, gx, gy, obs_map, min_final_meter=1, use_new_cost=True
    ) -> tuple[list[tuple[float, float]], bool]:
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
            min_final_meter: number of pixels

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0,
            -1,
        )
        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0,
            -1,
        )
        motion = self.get_motion_model()
        reason = None
        if obs_map[goal_node.x, goal_node.y] == 255:
            reason = 'goal_in_obstacle'
            return [], False, reason

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        step = 0
        while step < self.max_step:
            step += 1
            if len(open_set) == 0:
                reason = 'open_set_empty'
                break
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]
            to_final_dis = self.calc_heuristic(current, goal_node)
            if to_final_dis <= min_final_meter:
                # print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(motion):
                x = current.x + motion[i][0]
                y = current.y + motion[i][1]
                if use_new_cost:
                    obs_cost = self.get_cost_new(x, y, obs_map)
                    obs_cost += self.get_angle_cost(gx, gy, current, x, y)
                else:
                    obs_cost = self.get_cost_old(x, y, obs_map)

                node = self.Node(x, y, current.cost + motion[i][2] + obs_cost, c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node, obs_map):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        find_flag = True
        if step == self.max_step:
            reason = 'plan_max_step'
            goal_node = current
            find_flag = False

        rx, ry = self.calc_final_path(goal_node, closed_set)
        points_list = list(zip(rx, ry))

        if len(points_list) > 1:
            points = self.simplify_path(points_list)
            points.append((gx, gy))
        else:
            log.warning(f'Path planning results only contain {len(points_list)} points.')
            points = []

        return points, find_flag, reason

    def get_cost_new(self, x, y, obs_map):
        # if the point is out of the map, return the maximum value
        if x < 0 or x >= self.max_x or y < 0 or y >= self.max_y:
            return 255

        # initialize the total cost and counter
        total_cost = 0
        count = 0

        # iterate the 5x5 region centered at (x, y)
        for dx in range(-2, 3):  # from -2 to 2 (inclusive)
            for dy in range(-5, 5):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.max_x and 0 <= ny < self.max_y:  # ensure the point is in the map
                    if obs_map[nx][ny] == 0:
                        cost = 240
                    elif obs_map[nx][ny] == 2:
                        cost = 0
                    else:
                        cost = obs_map[nx][ny]
                    total_cost += cost
                    count += 1

        # prevent count from being 0, theoretically it will not happen, defensive programming
        if count == 0:
            return 255

        # return the average value
        return total_cost // count

    def get_cost_old(self, x, y, obs_map):
        if x < self.max_x and y < self.max_y:
            if obs_map[x][y] == 0:
                cost = 240
            elif obs_map[x][y] == 2:
                cost = 0
            else:
                cost = obs_map[x][y]
            return cost
        else:
            return 255

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        rx.reverse()  # from begin to end
        ry.reverse()

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node, obs_map):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if obs_map[node.x][node.y] == 255:
            return False

        return True

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
        ]

        return motion

    def simplify_path(self, points, tolerance=0.01):
        """The tolerance sets sampling distance. The smaller the tolerance, the more points in the simplified line."""
        line = LineString(points)
        simplified_line = line.simplify(tolerance, preserve_topology=False)
        return list(simplified_line.coords)
