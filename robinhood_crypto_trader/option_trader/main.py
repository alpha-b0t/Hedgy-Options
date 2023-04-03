#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import robin_stocks.robinhood as rh

from option import Option

def logout():
    try:
        rh.authentication.logout()
        print("successfully logged out")
    except:
        print("already logged out")

rh.authentication.login(by_sms=False, store_session=False)

def see_top_calls(option_data, symbol, number=10):
    for i in range(number):
        print(i+1, option_data[symbol]["calls"][i])

def see_top_puts(option_data, symbol, number=10):
    for i in range(number):
        print(i+1, option_data[symbol]["puts"][i])

try:
    options = dict()
    ranked_options = dict()
    symbol = input("Enter stock ticker ('exit' to exit): ")
    
    while symbol.lower() != "exit":
        
        tradable_options = rh.options.find_tradable_options(symbol)
        
        calls, puts = [], []
        
        for i in range(len(tradable_options)):
            print(symbol + ": " + str(i+1) + "/" + str(len(tradable_options)))
            
            if tradable_options[i]["type"] == "call":
                calls.append(Option(tradable_options[i]))
            elif tradable_options[i]["type"] == "put":
                puts.append(Option(tradable_options[i]))
            else:
                raise Exception("Option is neither a call nor a put")
        
        options[symbol.upper()] = {"calls": calls, "puts": puts}
        
        # Best first, worst last
        calls.sort(reverse=True)
        puts.sort(reverse=True)
        
        ranked_options[symbol.upper()] = {"calls": calls, "puts": puts}
        
        symbol = input("Enter stock ticker ('exit' to exit): ")
    
    logout()
except:
    logout()
