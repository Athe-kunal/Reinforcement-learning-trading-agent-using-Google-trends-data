from finrl.config_tickers import DOW_30_TICKER
from config import INDICATORS
from finrl import config, config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv as StockTradingEnv_numpy 
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.meta.data_processor import DataProcessor
# from stable_baselines3.common.vec_env import DummyVecEnv
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import ray
from pprint import pprint
import time
from typing import Dict, Optional, Any
import psutil

technical_indicator_list = INDICATORS

model_name = 'ppo'
env = StockTradingEnv_numpy
# ticker_list = ['SPY','TSLA','AAPL','GOOGL']
ticker_list = DOW_30_TICKER
data_source = 'yahoofinance'
time_interval = '1D'
TRAIN_START_DATE = '2014-01-01'
TRAIN_END_DATE = '2019-07-30'

VAL_START_DATE = '2019-08-01'
VAL_END_DATE = '2020-07-30'

TEST_START_DATE = '2020-08-01'
TEST_END_DATE = '2021-10-01'

def get_train_env(start_date, end_date, ticker_list, data_source, time_interval, 
          technical_indicator_list, env, model_name, if_vix = True,
          **kwargs):
    
    #fetch data
    DP = DataProcessor(data_source, **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = DP.add_vix(data)
    price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)
    train_env_config = {'price_array':price_array,
              'tech_array':tech_array,
              'turbulence_array':turbulence_array,
              'if_train':True}
    
    return train_env_config

def calculate_sharpe(episode_reward:list):
  perf_data = pd.DataFrame(data=episode_reward,columns=['reward'])
  perf_data['daily_return'] = perf_data['reward'].pct_change(1)
  if perf_data['daily_return'].std() !=0:
    sharpe = (252**0.5)*perf_data['daily_return'].mean()/ \
          perf_data['daily_return'].std()
    return sharpe
  else:
    return 0

def get_test_config(start_date, end_date, ticker_list, data_source, time_interval, 
         technical_indicator_list, env, model_name, if_vix = True,
         **kwargs):
  
  DP = DataProcessor(data_source, **kwargs)
  data = DP.download_data(ticker_list, start_date, end_date, time_interval)
  data = DP.clean_data(data)
  data = DP.add_technical_indicator(data, technical_indicator_list)
  
  if if_vix:
      data = DP.add_vix(data)
  
  price_array, tech_array, turbulence_array = DP.df_to_array(data, if_vix)
  test_env_config = {'price_array':price_array,
            'tech_array':tech_array,
            'turbulence_array':turbulence_array,'if_train':False}
  return test_env_config
