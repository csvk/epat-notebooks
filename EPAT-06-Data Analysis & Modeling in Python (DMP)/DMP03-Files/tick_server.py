#
# Simple Tick Data Server
#
import zmq
import time
import random

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('tcp://127.0.0.1:5555')

PRICE = 100.

while True:
    PRICE += random.gauss(0, 1) / 2
    msg = f'PRICE {PRICE:.3f}'
    socket.send_string(msg)
    print(msg)
    time.sleep(random.random() * 2)
