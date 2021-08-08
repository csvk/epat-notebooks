# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:36:13 2020

@author: Jay Parmar

@doc: http://interactivebrokers.github.io/tws-api/historical_bars.html

@goal: Fetch historical data of a financial instrument
"""

# Import necessary libraries
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from threading import Thread
import pandas as pd
import time

# Define strategy class - inherits from EClient and EWrapper
class Strategy(EClient, EWrapper):
    
    # Initialize the class - and inherited classes
    def __init__(self):
        EClient.__init__(self, self)
        self.df = pd.DataFrame(columns=['Time', 'Open', 'Close'])
           
    # Receive historical bars from TWS
    def historicalData(self, reqId, bar):
        print('Req Id:', reqId)
        dictionary = {'Time':bar.date,'Open': bar.open, 'Close': bar.close}
        self.df = self.df.append(dictionary, ignore_index=True)
        print(f'Time: {bar.date}, Open: {bar.open}, Close: {bar.close}')
        
    # Display a message once historical data is retreived
    def historicalDataEnd(self, reqId, start, end):
        print('\nHistorical Data Retrieved\n')
        print(self.df.head())
        # self.df.to_csv('Historical_data.csv')

# -------------------------x-----------------------x---------------------------

# Create object of the strategy class
app = Strategy()

# Connect strategy to IB TWS
app.connect(host='127.0.0.1', port=7497, clientId=2)

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

# Request for historical data
app.reqHistoricalData(reqId=33, 
                      contract=contract,
                      endDateTime='',
                      durationStr='300 S',
                      barSizeSetting='10 secs',
                      whatToShow='MIDPOINT',
                      useRTH=True,
                      formatDate=1,
                      keepUpToDate=True,
                      chartOptions=[])

# Wait for sometime to receive the response
time.sleep(5)

# Disconnect the app
app.disconnect()