import gymnasium as gym
import yfinance as yf
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
import pandas as pd
import numpy as np
from typing import List

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

class Portfolio(object):
    def __init__(self, init_cash, ticker_keys):
        self.portfolio = dict() # ticker, number of shares
        self.cash = init_cash
        self.ticker_keys = ticker_keys
        for ticker in ticker_keys:
            self.portfolio[ticker] = 0

    def sell(self, ticker, num_shares, all_tickers, time):
        closes = all_tickers['Close']

        if ticker not in self.portfolio.keys():
            return
        
        if closes[ticker].iloc[time] < 0:
            return

        if num_shares > self.portfolio[ticker]:
            num_shares = self.portfolio[ticker]

        self.portfolio[ticker] -= num_shares

        self.cash += num_shares * closes[ticker].iloc[time]

    def buy(self, ticker, num_shares, all_tickers, time):
        closes = all_tickers['Close']
        current_price = closes[ticker].iloc[time]

        if current_price < 0:
            return

        cost = num_shares * current_price

        if cost > self.cash:
            num_shares = int(self.cash / current_price)
            cost = num_shares * current_price

        if ticker in self.portfolio.keys():
            self.portfolio[ticker] += num_shares
        else:
            self.portfolio[ticker] = num_shares

        self.cash -= cost

    def get_obs(self):
        obs = []
        for ticker in self.ticker_keys:
            obs.append(self.portfolio[ticker])
        obs.append(self.cash)
        return np.array(obs)
    
    

    def get_change_in_value(self, all_tickers, time):
        value = 0
        closes = all_tickers['Close']
        for ticker, num_shares in self.portfolio.items():
            current_price = closes[ticker].iloc[time]
            next_price = closes[ticker].iloc[time + 1]

            if next_price < 0 or current_price < 0:
                continue

            value += (next_price - current_price) * num_shares

        return value


class Tickers(object):
    def __init__(self, ticker_keys: List[str], use_cache, test):
        self.ticker_keys = ticker_keys
        if use_cache:
            print("Using cache")
            if test:
                self.df = pd.read_pickle("/Users/jh/dev/tickers/data/tickers-test.pkl")
            else:
                self.df = pd.read_pickle("/Users/jh/dev/tickers/data/tickers-train.pkl")
        else:
            session = CachedLimiterSession(
            limiter=Limiter(RequestRate(2, Duration.SECOND*5)),  # max 2 requests per 5 seconds
                bucket_class=MemoryQueueBucket,
                backend=SQLiteCache("yfinance.cache"),
            )
            if test:
                self.df = yf.download(" ".join(ticker_keys), start="2023-01-01", session=session)
                self.df.fillna(value=-1, inplace=True)
                self.df.to_pickle("/Users/jh/dev/tickers/data/tickers-test.pkl")
            else:
                self.df = yf.download(" ".join(ticker_keys), start="2000-01-01", end="2022-12-31", session=session)
                self.df.fillna(value=-1, inplace=True)
                self.df.to_pickle("/Users/jh/dev/tickers/data/tickers-train.pkl")
            

    def get_obs(self, time):
        obs = []
        closes = self.df['Close']
        opens = self.df['Open']
        highs = self.df['High']
        lows = self.df['Low']

        for ticker in self.ticker_keys:
            ticker_obs = []
            for t in range(time - 2, time + 1):
                if t < 0 or t >= len(closes):
                    ticker_obs.extend([-1, -1, -1, -1])
                else:
                    ticker_obs.extend([
                        closes[ticker].iloc[t] / 1000,
                        opens[ticker].iloc[t] / 1000,
                        highs[ticker].iloc[t] / 1000,
                        lows[ticker].iloc[t] / 1000,
                    ])
            obs.extend(ticker_obs)

        obs = np.array(obs)

        return obs


class TickersEnv(gym.Env):
    def __init__(self, init_cash=10000, render_mode=None, ticker_keys=["AAPL", "MSFT", "GOOG"], use_cache=True):
        self.init_cash = init_cash
        self.current_time = 0
        self.ticker_keys = ticker_keys
        self.use_cache = use_cache
        self.tickers = Tickers(ticker_keys, use_cache, False)
        self.portfolio = Portfolio(init_cash, ticker_keys)


        num_actions = 11 ** len(ticker_keys)
        self.action_space = gym.spaces.Discrete(num_actions)

        market_data_dim = len(ticker_keys) * 4 * 3 
        portfolio_dim = len(ticker_keys) + 1  # Number of shares for each ticker + cash
        total_dim = market_data_dim + portfolio_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_dim,), 
            dtype=np.float32
        )

        self.set_timeframe(0, False)

    def set_timeframe(self, seed, test):
        if test:
            self.current_time = 0
            self.end_time = len(self.tickers.df) - 1
        else:
            np.random.seed(seed)
            self.current_time = np.random.randint(0, len(self.tickers.df) - 30)
            self.end_time = self.current_time + 30

    def reset(self, seed, options):

        self.portfolio = Portfolio(self.init_cash, self.ticker_keys)

        if options is not None and "test" in options.keys() and options["test"]:
            self.tickers = Tickers(self.ticker_keys, self.use_cache, True)
            self.set_timeframe(seed, True)
        else:
            self.tickers = Tickers(self.ticker_keys, self.use_cache, False)
            self.set_timeframe(seed, False)

        return self.get_obs(self.current_time), {}
    
    def get_obs(self, time):
        market_data = self.tickers.get_obs(time)
        portfolio_data = self.portfolio.get_obs() / 10000

        return np.concatenate([market_data, portfolio_data])
    

    def _decode_action(self, action):
        actions = []
        for _ in range(len(self.ticker_keys)):
            actions.append(action % 11)
            action //= 11
        return actions
    
    def step(self, action):
        cash_before = self.portfolio.cash

        actions = self._decode_action(action)

        for i, amount in enumerate(actions):
            ticker = self.ticker_keys[i]
            if amount > 5:  # Sell action
                self.portfolio.sell(ticker, amount - 5, self.tickers.df, self.current_time)
            else:  # Buy action
                self.portfolio.buy(ticker, amount, self.tickers.df, self.current_time)

        cash_after = self.portfolio.cash

        portfolio_change = self.portfolio.get_change_in_value(self.tickers.df, self.current_time)

        reward = portfolio_change + (cash_after - cash_before)

        obs = self.get_obs(self.current_time)

        self.current_time += 1

        done = self.current_time >= self.end_time

        return obs, reward, done, False, {}

