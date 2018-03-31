import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from io import BytesIO


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

#
# class SimplePIController:
#     def __init__(self, Kp, Ki):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.set_point = 0.
#         self.error = 0.
#         self.integral = 0.
#
#     def set_desired(self, desired):
#         self.set_point = desired
#
#     def update(self, measurement):
#         # proportional error
#         self.error = self.set_point - measurement
#
#         # integral error
#         self.integral += self.error
#
#         return self.Kp * self.error + self.Ki * self.integral
#
#
# controller = SimplePIController(0.1, 0.002)
# set_speed = 9
# controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        # imgString = data["image"]
        # image = Image.open(BytesIO(base64.b64decode(imgString)))
        # image_array = np.asarray(image)
        # steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        #
        # throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        # send_control(steering_angle, throttle)

        # save frame
    #     if args.image_folder != '':
    #         timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    #         image_filename = os.path.join(args.image_folder, timestamp)
    #         image.save('{}.jpg'.format(image_filename))
    # else:
    #     # NOTE: DON'T EDIT THIS.
    #     sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    # send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
