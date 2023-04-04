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

def see_top_calls(option_data, symbol, number=10):
    for i in range(number):
        print(i+1, option_data[symbol]["calls"][i])

def see_top_puts(option_data, symbol, number=10):
    for i in range(number):
        print(i+1, option_data[symbol]["puts"][i])

try:
    rh.authentication.login(by_sms=False, store_session=False)
    
    options, ranked_options = dict(), dict()
    
    symbol = input("Enter stock ticker(s) (if multiple, separate with a space) ('exit' to exit): ")
    tickers = symbol.split()
    
    stop = False
    
    while True:
        for n in range(len(tickers)):
            
            if tickers[n].lower() == 'exit':
                stop = True
                break
            
            tradable_options = rh.options.find_tradable_options(tickers[n])
            
            calls, puts = [], []
            
            for i in range(len(tradable_options)):
                print(tickers[n] + ": " + str(i+1) + "/" + str(len(tradable_options)))
                
                if tradable_options[i]["type"] == "call":
                    calls.append(Option(tradable_options[i]))
                elif tradable_options[i]["type"] == "put":
                    puts.append(Option(tradable_options[i]))
                else:
                    raise Exception("Option is neither a call nor a put")
            
            options[tickers[n].upper()] = {"calls": calls, "puts": puts}
            
            # Best first, worst last            
            ranked_options[tickers[n].upper()] = {"calls": sorted(calls, reverse=True), "puts": sorted(puts, reverse=True)}
        
        if stop:
            break
        
        symbol = input("Enter stock ticker(s) (if multiple, separate with a space) ('exit' to exit): ")
        tickers = symbol.split()
    
    logout()
except Exception as ex:
    logout()
    print(type(ex))
    print(ex.args)
    print(ex)
