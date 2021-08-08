# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:52:32 2020

@author: Jay Parmar

@doc: http://interactivebrokers.github.io/tws-api/connection.html

@goal: Connect Python script to TWS
"""

# Import necessary libraries
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from threading import Thread
import time

# Define strategy class - inherits from EClient and EWrapper
class strategy(EClient, EWrapper):
    
    # Initialize the class - and inherited classes
    def __init__(self):
        EClient.__init__(self, self)
        
    # This Callback method is available from the EWrapper class
    def currentTime(self, time):
        print('Current time on server:', time)
        
# -------------------------x-----------------------x---------------------------

# Create object of the strategy class
app = strategy()

# Connect strategy to IB TWS
app.connect(host='127.0.0.1', port=7497, clientId=1)

# Wait for sometime to connect to the server
time.sleep(1)

# Start a separate thread that will receive all responses from the TWS
Thread(target=app.run, daemon=True).start()

print('Is application connected to IB TWS:', app.isConnected())

# This method comes from EClient class
# Example of sending a request to TWS
app.reqCurrentTime()

time.sleep(2)

# Disconnect the app
app.disconnect()


