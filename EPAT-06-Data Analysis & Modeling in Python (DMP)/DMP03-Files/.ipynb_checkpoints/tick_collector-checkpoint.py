#
# Simple Tick Data Collector
#
import zmq
import datetime
import pandas as pd

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect('tcp://127.0.0.1:5555')
socket.setsockopt_string(zmq.SUBSCRIBE, '')

raw =pd.DataFrame()

while True:
    msg = socket.recv_string()
    t = datetime.datetime.now()
    print(str(t) + ' | ' + msg)
    symbol, price = msg.split()
    tick = pd.DataFrame({'symbol': symbol,
        'price': float(price)}, index=[t,])
    raw = raw.append(tick)
    data = raw.resample('5s', label='right').last().ffill()
