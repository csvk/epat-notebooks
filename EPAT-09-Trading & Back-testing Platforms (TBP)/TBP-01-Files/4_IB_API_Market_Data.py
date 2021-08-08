# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:39:38 2020

@author: Jay Parmar

@doc: https://interactivebrokers.github.io/tws-api/md_request.html

@goal: Request live data from the TWS
"""

# Import necessary libraries
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from threading import Thread
import time

# Define strategy class - inherits from EClient and EWrapper
class Strategy(EClient, EWrapper):
    
    # Initialize the class - and inherited classes
    def __init__(self):
        EClient.__init__(self, self)

    def tickPrice(self, reqId, tickType, price, attrib):
        """
        Handle ticks from the TWS
        1 : Bid Price
        2 : Ask Price
        6 : Day's High Price
        7 : Day's Low Price
        9 : Yesterday's Close Price
        """
        cTime = time.strftime("%H:%M:%S", time.localtime())
        print(f'Id: {reqId}, Time: {cTime} TickType: {tickType}, Price: {price}')
        
# -------------------------x-----------------------x---------------------------

# Create object of the strategy class
app = Strategy()

# Connect strategy to IB TWS
app.connect(host='127.0.0.1', port=7497, clientId=12)

# Wait for sometime to connect to the server
time.sleep(1)

# Start a separate thread that will receive all responses from the TWS
Thread(target=app.run, daemon=True).start()

print('Is application connected to IB TWS:', app.isConnected())

# Create object for contract
contract = Contract()
contract.symbol = 'EUR'
contract.currency = 'USD'
contract.secType = 'CASH'
contract.exchange = 'IDEALPRO'

reqId = 73

# Request market data from the TWS
app.reqMktData(reqId=reqId, 
               contract=contract, 
               genericTickList='', 
               snapshot=False, 
               regulatorySnapshot=False, 
               mktDataOptions=[])

# Sleep for few seconds
time.sleep(30)

# Send request to cancel market data
app.cancelMktData(reqId)

# Disconnect the application
app.disconnect()