import math
from enum import Enum

import numpy as np

from internnav.utils.common_log_util import common_logger as log


class AStarDiscretePlanner:
    class Action(Enum):
        stop = 0
        forward = 1
        turn_left = 2
        turn_right = 3

    class Node:
        def __init__(self, x, y, cost, parent_index, angle):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index
            self.angle = angle

        def __str__(self):
            return (
                str(self.x)
                + ','
                + str(self.y)
                + ','
                + str(self.cost)
                + ','
                + str(self.parent_index)
                + ','
                + str(self.angle)
            )

    def __init__(
        self,
        map_width=500,
        map_height=500,
        aperture=200,
        step_unit_meter=0.25,
        angle_unit=15,
        max_step=10000,
    ):
        self.resolution = 1
        self.max_step = max_step
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = map_width, map_height
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        self.aperture = aperture
        self.step_unit_meter = step_unit_meter
        self.angle_unit = angle_unit
        if 360 % angle_unit != 0:
            raise ValueError('angle_unit needs to be divided by 360 degrees')
        self.x_step_pixels = step_unit_meter * 10 * map_width / aperture
        self.y_step_pixels = step_unit_meter * 10 * map_height / aperture

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def calc_grid_position(self, index, min_position):
        pos = index * self.resolution + min_position
        return pos

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

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

    def get_motions(self, yaw):
        motion = []
        base_angle = round(yaw * (180 / math.pi))
        for i in range(360 // self.angle_unit):
            angle = base_angle + i * self.angle_unit
            if angle > 360:
                angle = angle - 360
            dx = self.x_step_pixels * math.cos(math.radians(angle))
            dy = self.y_step_pixels * math.sin(math.radians(angle))
            dx = round(dx)
            dy = round(dy)
            cost = 1  # math.hypot(dx, dy)
            motion.append([dx, dy, cost, angle])
        return motion

    def get_cost(self, cost_map, start_node, end_node, goal_node, dilation=5):
        start_x = start_node.x
        start_y = start_node.y
        end_x = end_node.x
        end_y = end_node.y

        if end_x >= self.max_x or end_y >= self.max_y:
            return 255 + 12

        def min_with_dilation(xs, dilation, min_value):
            if min(xs) < min_value:
                return min_value
            if min(xs) - dilation < min_value:
                return min(xs)
            return min(xs) - dilation

        def max_with_dilation(xs, dilation, max_value):
            if max(xs) > max_value:
                return max_value
            if max(xs) + dilation > max_value:
                return max(xs)
            return max(xs) + dilation

        min_x = min_with_dilation((start_x, end_x), dilation, self.min_x)
        max_x = max_with_dilation((start_x, end_x), dilation, self.max_x)
        min_y = min_with_dilation((start_y, end_y), dilation, self.min_y)
        max_y = max_with_dilation((start_y, end_y), dilation, self.max_y)
        cost = np.mean(cost_map[min_x : max_x + 1, :][:, min_y : max_y + 1])
        if math.isnan(cost):
            cost = 500
            log.error(
                f'math.isnan(cost) min_x:{min_x},max_x:{max_x},min_y:{min_y},max_y:{max_y},cost_map:{cost_map[min_x:max_x + 1,:][:, min_y:max_y + 1]}'
            )
        cost = round(cost)

        angle_diff = max(start_node.angle, end_node.angle) - min(start_node.angle, end_node.angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        start_distance = pow((start_node.x - goal_node.x), 2) + pow((start_node.y - goal_node.y), 2)
        end_distance = pow((end_node.x - goal_node.x), 2) + pow((end_node.y - goal_node.y), 2)
        angle_cost = angle_diff // 15
        if start_distance <= 8:
            angle_cost = angle_cost // 12
        distance_cost = end_distance / start_distance
        cost = cost + angle_cost + distance_cost
        return cost

    def calc_final_path_and_actions(self, goal_node, closed_set):
        # generate final course
        x = self.calc_grid_position(goal_node.x, self.min_x)
        y = self.calc_grid_position(goal_node.y, self.min_y)
        points = [(x, y)]
        actions = []
        last_node = goal_node
        while last_node.parent_index != -1:
            current_node = closed_set[last_node.parent_index]
            x = self.calc_grid_position(current_node.x, self.min_x)
            y = self.calc_grid_position(current_node.y, self.min_y)
            points.append((x, y))
            actions.append(self.Action.forward.value)
            angle_diff = max(current_node.angle, last_node.angle) - min(current_node.angle, last_node.angle)
            if angle_diff == 0:
                pass
            elif angle_diff <= 180:
                if last_node.angle > current_node.angle:
                    actions.extend([self.Action.turn_left.value for _ in range(angle_diff // self.angle_unit)])
                else:
                    actions.extend([self.Action.turn_right.value for _ in range(angle_diff // self.angle_unit)])
            else:
                if last_node.angle > current_node.angle:
                    actions.extend([self.Action.turn_right.value for _ in range((360 - angle_diff) // self.angle_unit)])
                else:
                    actions.extend([self.Action.turn_left.value for _ in range((360 - angle_diff) // self.angle_unit)])
            last_node = current_node
        # from begin to end
        points.reverse()
        actions.reverse()

        return points, actions

    def planning(self, sx, sy, gx, gy, obs_map, yaw, min_final_meter=6) -> tuple[list[tuple[float, float]], bool]:
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]
            min_final_meter: number of pixels (0.25 meters is about 6 pixels)

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        # the angle in the image is opposite to the actual angle
        if yaw > math.pi:
            yaw = yaw - math.pi
        else:
            yaw = yaw + math.pi

        angle = round(yaw * (180 / math.pi))
        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0,
            -1,
            angle,
        )
        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0,
            -1,
            0,
        )
        reason = None
        if goal_node.x < self.min_x or goal_node.x > self.max_x or goal_node.y < self.min_y or goal_node.y > self.max_y:
            reason = 'goal_out_of_map'
            return [], [], False, reason

        if obs_map[goal_node.x, goal_node.y] == 255:
            reason = 'goal_in_obstacle'
            return [], [], False, reason

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        motions = self.get_motions(yaw)
        cost_map = np.where(obs_map == 0, 240, obs_map)
        cost_map = np.where(cost_map == 2, 0, cost_map)

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
                goal_node.angle = current.angle
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for _, motion in enumerate(motions):
                next_x = current.x + motion[0]
                next_y = current.y + motion[1]
                next_cost = motion[2]
                next_angle = motion[3]
                next_node = self.Node(next_x, next_y, -1, c_id, next_angle)
                obs_cost = self.get_cost(cost_map, current, next_node, goal_node)
                next_node.cost = current.cost + next_cost + obs_cost
                n_id = self.calc_grid_index(next_node)
                # If the node is not safe, do nothing
                if not self.verify_node(next_node, obs_map):
                    continue
                if n_id in closed_set:
                    continue
                if n_id not in open_set:
                    open_set[n_id] = next_node  # discovered a new node
                else:
                    if open_set[n_id].cost > next_node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = next_node
        find_flag = True
        if step == self.max_step:
            reason = 'plan_max_step'
            goal_node = current
            find_flag = False
        actions = []
        points, actions = self.calc_final_path_and_actions(goal_node, closed_set)

        return points, actions, find_flag, reason
