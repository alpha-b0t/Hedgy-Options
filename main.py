#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import robin_stocks.robinhood as rh
from dotenv import load_dotenv
import os
from option import Option

def login(**kwargs):
    try:
        if 'username' in kwargs and 'password' in kwargs:
            rh.authentication.login(username=kwargs['username'], password=kwargs['password'], store_session=False)
        else:
            rh.authentication.login(store_session=False)
        print("successfully logged in")
    except Exception as e:
        print("login failed")
        raise e

def logout():
    try:
        rh.authentication.logout()
        print("successfully logged out")
    except Exception as e:
        print("already logged out")
        raise e

if __name__ == '__main__':
    try:
        # Load variables from .env file
        load_dotenv()

        # Access variables
        rh_username = os.getenv('RH_USERNAME')
        rh_password = os.getenv('RH_PASSWORD')
        login(username=rh_username, password=rh_password)
        del rh_username, rh_password
        
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

                    print(f"{option}\n")
                    
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
            
            if stop:
                break
            
            symbol = input("Enter stock ticker(s) (if multiple, separate with a space) ('exit' to exit): ")
            tickers = symbol.split()
        
        logout()
    except Exception as ex:
        logout()
        raise ex
    except:
        logout()

