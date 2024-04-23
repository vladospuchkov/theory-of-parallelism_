import cv2
import threading
import time
import argparse
from queue import LifoQueue, Queue
import logging
import os
import sys


class Sensor:
    def get(self):
        raise NotImplementedError("subclasses must implement method get()")


class SensorX(Sensor):
    """Sensor X"""

    def __init__(self, delay: float):
        self.delay = delay
        self.data = 0

    def get(self) -> int:
        time.sleep(self.delay)
        self.data += 1
        return self.data


class SensorCam(Sensor):
    def __init__(self, camera_index, width, height, glb_event):
        try:
            self._cap = cv2.VideoCapture(camera_index)
            if not self._cap.isOpened():
                logging.error(
                    "We don't have a camera with this camera index: %d", 
                    camera_index
                )
                glb_event.set()
                self.stop()
            else:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except Exception as e:
            logging.error(
                "We have an error: %s", str(e)
                )
            glb_event.set()
            self.stop()
            raise

    def get(self):
        ret, frame = self._cap.read()
        if not ret:
            logging.error(
                "we have no img from cam"
                )
            glb_event.set()
            self.stop()
            return None
        return frame

    def stop(self):
        self._cap.release()

    def __del__(self):
        self.stop()


def sensor_worker(sensor: SensorX, queue: LifoQueue, event_funk):
    while event_funk.is_set():
        a = sensor.get()
        if queue.full():
            queue.get()
        queue.put(a)
    print("!!! Sensor worker done !!!")


def camera_worker(queue: Queue, args, glb_event, event_funk):
    cam = SensorCam(args.camIndex, args.size[0], args.size[1], glb_event)
    while event_funk.is_set():
        a = cam.get()
        if a is None:
            break
        if queue.full():
            queue.get()
        queue.put(a)
    cam.stop()


class ImageWindow:
    def __init__(self, fps, height):
        self._sensor_data = [0, 0, 0]
        self.frame = None
        self.fps = fps
        self._height = height

    def show(self, cam_queue: Queue, queues: list):
        try:
            for i in range(3):
                if not queues[i].empty():
                    self._sensor_data[i] = queues[i].get()
                    print(self._sensor_data[i])
            if not cam_queue.empty():
                self.frame = cam_queue.get()
                cv2.putText(
                    self.frame,
                    f"Sensor1: {self._sensor_data[0]}  Sensor2: {self._sensor_data[1]}  Sensor3: {self._sensor_data[2]}",
                    (10, self._height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1,
                )
                cv2.imshow("camera and data", self.frame)
        except Exception as e:
            logging.error(
                "We have an error at show(): %s", str(e)
                )

    def stop(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="width, height, fps")
    
    parser.add_argument("--camIndex", 
                        type=int, 
                        default=0)
    
    parser.add_argument("--size", 
                        nargs=2, 
                        type=int, 
                        default=[720, 480], 
                        help="Size as 'width height'")
    
    parser.add_argument("--fps", 
                        type=int,
                        default=60)
    
    args = parser.parse_args()

    if not os.path.exists("log"):
        os.makedirs("log")
    log_file = os.path.join("log", "error.log")
    logging.basicConfig(filename=log_file, level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

    glb_event = threading.Event()
    event_funk = threading.Event()
    
    glb_event.clear()
    event_funk.set()

    sensors = [SensorX(i) for i in [0.01, 0.1, 1]]
    sensor_queues = [LifoQueue(maxsize=1) for _ in range(3)]
    cam_queue = Queue(maxsize=1)

    sensor_workers = [
        threading.Thread(target=sensor_worker, args=(sensors[i], sensor_queues[i], event_funk))
        for i in range(3)
    ]
    cam_worker = threading.Thread(target=camera_worker, args=(cam_queue, args, glb_event, event_funk))
    time.sleep(1)
    window_imager = ImageWindow(fps=args.fps, height=args.size[1])

    for worker in sensor_workers:
        worker.start()
    cam_worker.start()

    while True:
        window_imager.show(cam_queue, sensor_queues)

        if ((cv2.waitKey(1) & 0xFF == ord("q")) 
            or glb_event.is_set()):
            window_imager.stop()
            event_funk.clear()
            cam_worker.join()
            
            for worker in sensor_workers:
                worker.join()
            break
        
        time.sleep(1 / window_imager.fps)
