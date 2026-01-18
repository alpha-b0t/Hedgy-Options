#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import robin_stocks.robinhood as rh

from option import Option

def login():
    try:
        rh.authentication.login(by_sms=False, store_session=False)
        print("successfully logged in")
    except:
        print("login failed")

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
    login()
    
    options, ranked_options, ask_sorted_ranked_options = dict(), dict(), dict()
    
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
                
                option = Option(tradable_options[i])
                
                if option.retrieved_market_data:
                    if tradable_options[i]["type"] == "call":
                        calls.append(option)
                    elif tradable_options[i]["type"] == "put":
                        puts.append(option)
                    else:
                        raise Exception("Option is neither a call nor a put")
                else:
                    continue
            
            options[tickers[n].upper()] = {"calls": calls, "puts": puts}
            
            # Best first, worst last            
            ranked_options[tickers[n].upper()] = {"calls": sorted(calls, reverse=True), "puts": sorted(puts, reverse=True)}
    
            # Best first, worst last
            ask_sorted_ranked_options[tickers[n].upper()] = {"calls": sorted(sorted(calls, reverse=True), key=lambda obj: obj.long_ask_breakeven_price, reverse=False), "puts": sorted(sorted(puts, reverse=True), key=lambda obj: obj.long_ask_breakeven_price, reverse=True)}
        
        if stop:
            break
        
        symbol = input("Enter stock ticker(s) (if multiple, separate with a space) ('exit' to exit): ")
        tickers = symbol.split()
    
    logout()
except Exception as ex:
    logout()
    raise ex
