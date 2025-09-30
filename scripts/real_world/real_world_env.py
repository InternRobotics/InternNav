from cam import CameraSubscriber
from control import DiscreteRobotController


class RealWorldEnv:
    def __init__(self, camera_topic="/camera/image_raw"):
        self.node = DiscreteRobotController()
        self.cam = CameraSubscriber(camera_topic)

    def get_observation(self):
        frame = self.cam.get_latest()
        return frame

    def step(self, action):

        '''
        action (int): Discrete action to apply:
                    - 0: no movement (stand still)
                    - 1: move forward
                    - 2: rotate left
                    - 3: rotate right
        '''
        if action == 0:
            self.node.stand_still()
        elif action == 1:
            self.node.move_forward()
        elif action == 2:
            self.node.turn_left()
        elif action == 3:
            self.node.turn_right()
