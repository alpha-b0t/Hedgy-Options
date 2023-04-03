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
        self.strike_price = config['strike_price']
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
        
        market_data = rh.options.get_option_market_data(self.chain_symbol, self.expiration_date, self.strike_price, self.type)[0][0]
        
        self.adjusted_mark_price = market_data['adjusted_mark_price']
        self.adjusted_mark_price_round_down = market_data['adjusted_mark_price_round_down']
        self.ask_price = market_data['ask_price']
        self.ask_size = market_data['ask_size']
        self.bid_price = market_data['bid_price']
        self.bid_size = market_data['bid_size']
        self.break_even_price = market_data['break_even_price']
        self.high_price = market_data['high_price']
        self.instrument = market_data['instrument']
        self.instrument_id = market_data['instrument_id']
        self.last_trade_price = market_data['last_trade_price']
        self.last_trade_size = market_data['last_trade_size']
        self.low_price = market_data['low_price']
        self.mark_price = market_data['mark_price']
        self.open_interest = market_data['open_interest']
        self.previous_close_date = market_data['previous_close_date']
        self.previous_close_price = market_data['previous_close_price']
        self.updated_at = market_data['updated_at']
        self.volume = market_data['volume']
        self.symbol = market_data['symbol']
        self.occ_symbol = market_data['occ_symbol']
        self.state = market_data['state']
        self.chance_of_profit_long = market_data['chance_of_profit_long']
        self.chance_of_profit_short = market_data['chance_of_profit_short']
        self.delta = market_data['delta']
        self.gamma = market_data['gamma']
        self.implied_volatility = market_data['implied_volatility']
        self.rho = market_data['rho']
        self.theta = market_data['theta']
        self.vega = market_data['vega']
        self.high_fill_rate_buy_price = market_data['high_fill_rate_buy_price']
        self.high_fill_rate_sell_price = market_data['high_fill_rate_sell_price']
        self.low_fill_rate_buy_price = market_data['low_fill_rate_buy_price']
        self.low_fill_rate_sell_price = market_data['low_fill_rate_sell_price']
        
        try:
            self.stock_price = rh.stocks.get_latest_price(self.chain_symbol)[0]
        except:
            self.stock_price = "0.00"
    
    def __repr__(self):
        return "{" + self.chain_symbol + " " + self.type + " " + self.expiration_date + " $" + self.strike_price + " (adj_mark_price=$" + self.adjusted_mark_price + ") (breakeven_price=$" + self.break_even_price + ") (stock_price=$" + self.stock_price + ")}"
    
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

    
    def __eq__(self, other):
        # ==
        if self.break_even_price == other.break_even_price and self.adjusted_mark_price == other.adjusted_mark_price and self.expiration_date == other.expiration_date:
            return True
        else:
            return False
    
    def __lt__(self, other):
        # <
        if self.type == 'call':
            # Want lower breakeven price for calls
            if self.break_even_price < other.break_even_price:
                return False
            elif self.break_even_price > other.break_even_price:
                return True
            else:
                # self and other have the same breakeven price
                # rank by risk (premium)
                if self.adjusted_mark_price < other.adjusted_mark_price:
                    return False
                elif self.adjusted_mark_price > other.adjusted_mark_price:
                    return True
                else:
                    # self and other have the same breakeven price and risk
                    # rank by expiration date
                    return not self.is_later_date(self.expiration_date, other.expiration_date)
        else:
            # Want high breakeven price for puts
            if self.break_even_price < other.break_even_price:
                return True
            elif self.break_even_price > other.break_even_price:
                return False
            else:
                # self and other have the same breakeven price
                # rank by risk (premium)
                if self.adjusted_mark_price < other.adjusted_mark_price:
                    return False
                elif self.adjusted_mark_price > other.adjusted_mark_price:
                    return True
                else:
                    # self and other have the same breakeven price and risk
                    # rank by expiration date
                    return not self.is_later_date(self.expiration_date, other.expiration_date)
    
    def __le__(self, other):
        # <=
        if self < other or self == other:
            return True
        else:
            return False
    
    def __gt__(self, other):
        # >
        if self.type == 'call':
            # Want lower breakeven price for calls
            if self.break_even_price < other.break_even_price:
                return True
            elif self.break_even_price > other.break_even_price:
                return False
            else:
                # self and other have the same breakeven price
                # rank by risk (premium)
                if self.adjusted_mark_price < other.adjusted_mark_price:
                    return True
                elif self.adjusted_mark_price > other.adjusted_mark_price:
                    return False
                else:
                    # self and other have the same breakeven price and risk
                    # rank by expiration date
                    return self.is_later_date(self.expiration_date, other.expiration_date)
        else:
            # Want high breakeven price for puts
            if self.break_even_price < other.break_even_price:
                return False
            elif self.break_even_price > other.break_even_price:
                return True
            else:
                # self and other have the same breakeven price
                # rank by risk (premium)
                if self.adjusted_mark_price < other.adjusted_mark_price:
                    return True
                elif self.adjusted_mark_price > other.adjusted_mark_price:
                    return False
                else:
                    # self and other have the same breakeven price and risk
                    # rank by expiration date
                    return self.is_later_date(self.expiration_date, other.expiration_date)
    
    def __ge__(self, other):
        # >=
        if self > other or self == other:
            return True
        else:
            return False
    
    def get_latest_stock_price(self):
        try:
            price = rh.stocks.get_latest_price(self.chain_symbol)[0]
            
            if price == None:
                return self.stock_price
            else:
                self.stock_price = price
                
                return self.stock_price
        except:
            return self.stock_price
