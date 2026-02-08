#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import robin_stocks.robinhood as rh

class Option():
    def __init__(self, config):
        """
        config = {'chain_id': '71b2a769-4781-4df1-a776-222206504c43',
         'chain_symbol': 'XPEV',
         'created_at': '2023-03-02T02:05:57.413583Z',
         'expiration_date': '2023-04-14',
         'id': 'fff615b6-9334-4da9-85ba-337acfaa8ca7',
         'issue_date': '2023-03-02',
         'min_ticks': {'above_tick': '0.05',
          'below_tick': '0.01',
          'cutoff_price': '3.00'},
         'rhs_tradability': 'tradable',
         'state': 'active',
         'strike_price': '9.0000',
         'tradability': 'tradable',
         'type': 'put',
         'updated_at': '2023-03-02T02:05:57.413588Z',
         'url': 'https://api.robinhood.com/options/instruments/fff615b6-9334-4da9-85ba-337acfaa8ca7/',
         'sellout_datetime': '2023-04-14T19:30:00+00:00',
         'long_strategy_code': 'fff615b6-9334-4da9-85ba-337acfaa8ca7_L1',
         'short_strategy_code': 'fff615b6-9334-4da9-85ba-337acfaa8ca7_S1'}
        """
        self.chain_id = config['chain_id']
        self.chain_symbol = config['chain_symbol']
        self.created_at = config['created_at']
        self.expiration_date = config['expiration_date']
        self.id = config['id']
        self.issue_date = config['issue_date']
        self.min_ticks = config['min_ticks']
        self.rhs_tradability = config['rhs_tradability']
        self.state = config['state']
        self.strike_price = float(config['strike_price'])
        self.tradability = config['tradability']
        self.type = config['type']
        self.updated_at = config['updated_at']
        self.url = config['url']
        self.sellout_datetime = config['sellout_datetime']
        self.long_strategy_code = config['long_strategy_code']
        self.short_strategy_code = config['short_strategy_code']
        
        """
        market_data = {'adjusted_mark_price': '1.610000',
         'adjusted_mark_price_round_down': '1.610000',
         'ask_price': '1.640000',
         'ask_size': 3,
         'bid_price': '1.580000',
         'bid_size': 234,
         'break_even_price': '9.390000',
         'high_price': '1.570000',
         'instrument': 'https://api.robinhood.com/options/instruments/e6324dbb-a2d4-43d8-b4b4-821881b640ea/',
         'instrument_id': 'e6324dbb-a2d4-43d8-b4b4-821881b640ea',
         'last_trade_price': '1.550000',
         'last_trade_size': 3,
         'low_price': '1.550000',
         'mark_price': '1.610000',
         'open_interest': 2033,
         'previous_close_date': '2023-03-30',
         'previous_close_price': '1.550000',
         'updated_at': '2023-03-31T19:59:59.9403136Z',
         'volume': 19,
         'symbol': 'XPEV',
         'occ_symbol': 'XPEV  230616P00011000',
         'state': 'active',
         'chance_of_profit_long': '0.399222',
         'chance_of_profit_short': '0.600778',
         'delta': '-0.402638',
         'gamma': '0.088542',
         'implied_volatility': '0.862730',
         'rho': '-0.012650',
         'theta': '-0.010370',
         'vega': '0.019607',
         'high_fill_rate_buy_price': '1.628000',
         'high_fill_rate_sell_price': '1.591000',
         'low_fill_rate_buy_price': '1.599000',
         'low_fill_rate_sell_price': '1.620000'}
        """
        
        market_data_response = rh.options.get_option_market_data(self.chain_symbol, self.expiration_date, config['strike_price'], self.type)
        
        self.retrieved_market_data = True

        try:
            market_data = market_data_response[0][0]
        except IndexError:
            try:
                attempt_count = 1
                while market_data_response == [[]] and attempt_count < 10:
                    market_data_response = rh.options.get_option_market_data(self.chain_symbol, self.expiration_date, self.strike_price, self.type)
                    attempt_count += 1
            
                market_data = market_data_response[0][0]
            except:
                self.retrieved_market_data = False
                return
        
        try:
            self.adjusted_mark_price = float(market_data['adjusted_mark_price'])
        except:
            self.adjusted_mark_price = market_data['adjusted_mark_price']
        
        try:
            self.adjusted_mark_price_round_down = float(market_data['adjusted_mark_price_round_down'])
        except:
            self.adjusted_mark_price_round_down = market_data['adjusted_mark_price_round_down']
        
        try:
            self.ask_price = float(market_data['ask_price'])
        except:
            self.ask_price = market_data['ask_price']
        
        try:
            self.ask_size = int(market_data['ask_size'])
        except:
            self.ask_size = market_data['ask_size']
        
        try:
            self.bid_price = float(market_data['bid_price'])
        except:
            self.bid_price = market_data['bid_price']
        
        try:
            self.bid_size = int(market_data['bid_size'])
        except:
            self.bid_size = market_data['bid_size']
        
        self.break_even_price = float(market_data['break_even_price'])
        
        try:
            self.high_price = float(market_data['high_price'])
        except:
            self.high_price = market_data['high_price']
        
        self.instrument = market_data['instrument']
        self.instrument_id = market_data['instrument_id']
        
        try:
            self.last_trade_price = float(market_data['last_trade_price'])
        except:
            self.last_trade_price = market_data['last_trade_price']
        
        try:
            self.last_trade_size = int(market_data['last_trade_size'])
        except:
            self.last_trade_size = market_data['last_trade_size']
        
        try:
            self.low_price = float(market_data['low_price'])
        except:
            self.low_price = market_data['low_price']
        
        try:
            self.mark_price = float(market_data['mark_price'])
        except:
            self.mark_price = market_data['mark_price']
        
        try:
            self.open_interest = int(market_data['open_interest'])
        except:
            self.open_interest = market_data['open_interest']
        
        self.previous_close_date = market_data['previous_close_date']
        
        try:
            self.previous_close_price = float(market_data['previous_close_price'])
        except:
            self.previous_close_price = market_data['previous_close_price']
        
        self.updated_at = market_data['updated_at']
        
        try:
            self.volume = int(market_data['volume'])
        except:
            self.volume = market_data['volume']
        
        self.symbol = market_data['symbol']
        self.occ_symbol = market_data['occ_symbol']
        self.state = market_data['state']
        
        try:
            self.chance_of_profit_long = float(market_data['chance_of_profit_long'])
        except:
            self.chance_of_profit_long = market_data['chance_of_profit_long']
        
        try:
            self.chance_of_profit_short = float(market_data['chance_of_profit_short'])
        except:
            self.chance_of_profit_short = market_data['chance_of_profit_short']
        
        try:
            self.delta = float(market_data['delta'])
        except:
            self.delta = market_data['delta']
        
        try:
            self.gamma = float(market_data['gamma'])
        except:
            self.gamma = market_data['gamma']
        
        try:
            self.implied_volatility = float(market_data['implied_volatility'])
        except:
            self.implied_volatility = market_data['implied_volatility']
        
        try:
            self.rho = float(market_data['rho'])
        except:
            self.rho = market_data['rho']
        
        try:
            self.theta = float(market_data['theta'])
        except:
            self.theta = market_data['theta']
        
        try:
            self.vega = float(market_data['vega'])
        except:
            self.vega = market_data['vega']
        
        try:
            self.high_fill_rate_buy_price = float(market_data['high_fill_rate_buy_price'])
        except:
            self.high_fill_rate_buy_price = market_data['high_fill_rate_buy_price']
        
        try:
            self.high_fill_rate_sell_price = float(market_data['high_fill_rate_sell_price'])
        except:
            self.high_fill_rate_sell_price = market_data['high_fill_rate_sell_price']
        
        try:
            self.low_fill_rate_buy_price = float(market_data['low_fill_rate_buy_price'])
        except:
            self.low_fill_rate_buy_price = market_data['low_fill_rate_buy_price']
        
        try:
            self.low_fill_rate_sell_price = float(market_data['low_fill_rate_sell_price'])
        except:
            self.low_fill_rate_sell_price = market_data['low_fill_rate_sell_price']
        
        try:
            self.stock_price = float(rh.stocks.get_latest_price(self.chain_symbol)[0])
        except:
            self.stock_price = 0.00
        
        try:
            # TO-DO: Implement
            if self.type == 'call':
                self.long_ask_breakeven_price = self.strike_price + self.ask_price
                # self.short_breakeven_price = None
            else:
                self.long_ask_breakeven_price = self.strike_price - self.ask_price
                # self.short_breakeven_price = None
        except:
            self.long_ask_breakeven_price = None
            # self.short_breakeven_price = None
    
    def __repr__(self):
        return "{" + self.chain_symbol + " " + self.type + " " + self.expiration_date + " $" + str(self.strike_price) + " (ask_price=$" + str(self.ask_price) + ") (adj_mark_price=$" + str(self.adjusted_mark_price) + ") (long_ask_breakeven_price=$" + str(self.long_ask_breakeven_price) + ") (breakeven_price=$" + str(self.break_even_price) + ") (stock_price=$" + str(self.stock_price) + ")}"
    
    def is_later_date(self, str1, str2):
        """
        Returns True if str1 is later than str2, and False otherwise.
        Assumes that str1 and str2 are strings representing dates in the format of "YYYY-MM-DD".
        """
        date1 = [int(x) for x in str1.split("-")]
        date2 = [int(x) for x in str2.split("-")]
        
        if date1[0] > date2[0]:
            return True
        elif date1[0] == date2[0] and date1[1] > date2[1]:
            return True
        elif date1[0] == date2[0] and date1[1] == date2[1] and date1[2] > date2[2]:
            return True
        else:
            return False
    
    def get_latest_stock_price(self):
        try:
            price = rh.stocks.get_latest_price(self.chain_symbol)[0]
            
            if price == None:
                return self.stock_price
            else:
                self.stock_price = float(price)
                
                return self.stock_price
        except:
            return self.stock_price
