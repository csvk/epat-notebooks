# -*- coding: utf-8 -*-
"""
Created on Mon May 25 22:01:06 2020

@author: Jay Parmar

@doc: https://interactivebrokers.github.io/tws-api/basic_orders.html

@goal: Order placement and cancellation
"""

# Import necessary libraries
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from threading import Timer, Thread
import time

# Define strategy class - inherits from EClient and EWrapper
class Strategy(EClient, EWrapper):
    
    # Initialize the class - and inherited classes
    def __init__(self):
        EClient.__init__(self, self)

    def openOrder(self, orderId, contract, order, orderState):
        print(f'\nOrder open for {contract} with orderid {orderId}.')
        
    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, \
                    permId, parentIt, lastFillPrice, clientId, whyHeld, \
                    mktCapPrice):
        print(f'\nOrder Status - OrderId: {orderId}, Status: {status}, Filled: {filled}, FilledPrice: {avgFillPrice}')

    # Receive details when orders are executed        
    def execDetails(self, reqId, contract, execution):
        print('\nOrder Executed:', execution.orderId)

# -------------------------x-----------------------x---------------------------
        
# Create object of the strategy class
app = Strategy()

# Connect strategy to IB TWS
app.connect(host='127.0.0.1', port=7497, clientId=123)

# Wait for sometime to connect to the server
time.sleep(1)

# Start a separate thread that will receive all responses from the TWS
Thread(target=app.run, daemon=True).start()

print('Is application connected to IB TWS:', app.isConnected())

# Define order id. This is not same as the request id
oid = 11111

# Create object for contract
contract = Contract()
contract.symbol = 'EUR'
contract.currency = 'USD'
contract.secType = 'CASH'
contract.exchange = 'IDEALPRO'

# Create object for order
order = Order()
order.action = 'BUY'
order.totalQuantity = 20000
order.orderType = 'MKT'
#order.lmtPrice = '1.1718'

# Request function
app.placeOrder(orderId=oid, 
               contract=contract, 
               order=order)

# 
Timer(10, app.cancelOrder, [oid]).start()

# Sleep for few seconds
time.sleep(5)

# Disconnect the application
app.disconnect()
