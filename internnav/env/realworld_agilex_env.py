import threading
import time

from internnav.env import Env
from internnav.env.utils.agilex_extensions.cam import AlignedRealSense
from internnav.env.utils.agilex_extensions.control import DiscreteRobotController


@Env.register('realworld')
class RealWorldEnv(Env):
    def __init__(
        self,
        fps: int = 30,
        duration: float = 0.5,
        distance: float = 0.25,
        angle: int = 15,
        turn_speed: float = 0.5,
        move_speed: float = 0.3,
    ):

        self.node = DiscreteRobotController()
        self.cam = AlignedRealSense()
        self.latest_obs = None
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.fps = fps

        # 启动相机
        self.cam.start()
        # 启动采集线程
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        # control setting
        self.duration = duration
        self.distance = distance
        self.angle = angle
        self.turn_speed = turn_speed  # rad/s
        self.move_speed = move_speed  # m/s

    def reverse(self):
        self.distance = -self.distance

    def _capture_loop(self):
        """keep capturing frames"""
        interval = 1.0 / self.fps
        while not self.stop_flag.is_set():
            t0 = time.time()
            try:
                obs = self.cam.get_observation(timeout_ms=1000)
                with self.lock:
                    self.latest_obs = obs
            except Exception as e:
                print("Camera capture failed:", e)
                time.sleep(0.05)
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    def get_observation(self):
        """return most recent frame"""
        with self.lock:
            return self.latest_obs

    def step(self, action: int):
        """
        action:
            0: stand still
            1: move forward
            2: turn left
            3: turn right
        """
        if action == 0:
            self.node.stand_still(self.duration)
        elif action == 1:
            self.node.move_feedback(self.distance, self.move_speed)
        elif action == 2:
            self.node.turn(self.angle, self.turn_speed)
        elif action == 3:
            self.node.turn(self.angle, -self.turn_speed)

    def close(self):
        self.stop_flag.set()
        self.thread.join(timeout=1.0)
        self.cam.stop()
