# Importing the libraries
import pandas as pd
import numpy as np

from finrl import config, config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading_np import (
    StockTradingEnv as StockTradingEnv_numpy,
)
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.data_processor import DataProcessor
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
import ray
from pprint import pprint
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.algorithms.ddpg import ddpg
from ray.rllib.algorithms.a2c import a2c
from ray.rllib.algorithms.td3 import td3
from ray.rllib.algorithms.ppo import ppo
from ray.rllib.algorithms.sac import sac
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
from ray.tune import run, sample_from
from ray.tune.registry import register_env

import os
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.registry import register_env

from typing import Dict, Optional, Any
import psutil
import numpy as np

psutil_memory_in_bytes = psutil.virtual_memory().total
ray._private.utils.get_system_memory = lambda: psutil_memory_in_bytes


class hyperparams_opt:
    """
    Returns the hyperparameter search space, mutation range and hyperparameter bounds

        Parameter:
          model_name (str): Name of the RL algorithm
        Returns:
          (sample_hyperparameters, mutate_hyperparameters, params_bounds) (tuple): Search space ranges
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def loguniform(self, low=0, high=1):
        return np.exp(np.random.uniform(low, high))

    def sample_ddpg_params(self):

        return {
            "buffer_size": tune.choice([int(1e4), int(1e5), int(1e6)]),
            "lr": tune.loguniform(1e-5, 1),
            "train_batch_size": tune.choice([32, 64, 128, 256, 512]),
        }

    def sample_a2c_params(self):

        return {
            "lambda": tune.choice([0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
            "entropy_coeff": tune.loguniform(0.00000001, 0.1),
            "lr": tune.loguniform(1e-5, 1),
        }

    def sample_ppo_params(self):
        return {
            "lr": tune.loguniform(5e-5, 0.0001),
            "sgd_minibatch_size": tune.choice([64, 128, 256]),
            "entropy_coeff": tune.loguniform(0.00000001, 0.1),
            "clip_param": tune.choice([0.1, 0.2, 0.3, 0.4]),
            "vf_loss_coeff": tune.uniform(0, 1),
            "lambda": tune.choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999]),
            "kl_target": tune.choice([0.001, 0.01, 0.1]),
        }

    def mutate_ddpg_hyperparams(self):

        return {
            "buffer_size": [int(1e4), int(1e5), int(1e6)],
            "lr": lambda: self.loguniform(1e-5, 1),
            "train_batch_size": [32, 64, 128, 256, 512],
        }

    def mutate_ppo_hyperparams(self):

        return {
            "entropy_coeff": lambda: self.loguniform(0.00000001, 0.1),
            "lr": lambda: self.loguniform(5e-5, 0.0001),
            "sgd_minibatch_size": [32, 64, 128, 256, 512],
            "lambda": [0.9, 0.95, 0.98, 0.99, 0.995, 0.999],
            "clip_param": [0.1, 0.2, 0.3, 0.4],
            "vf_loss_coeff": lambda: np.random.uniform(0, 1),
            "kl_target": [0.001, 0.01, 0.1],
        }

    def mutate_a2c_hyperparams(self):
        return {
            "lambda": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
            "entropy_coeff": lambda: self.loguniform(0.00000001, 0.1),
            "lr": lambda: self.loguniform(1e-5, 1),
        }

    def a2c_params_bounds(self):
        return {
            "buffer_size": [int(1e4), int(1e6)],
            "lr": [1e-5, 1],
            "train_batch_size": [32, 512],
        }

    def ppo_param_bounds(self):
        return {
            "lr": [5e-5, 0.0001],
            "sgd_minibatch_size": [64, 256],
            "entropy_coeff": [0.00000001, 0.1],
            "clip_param": [0.1, 0.4],
            "vf_loss_coeff": [0, 1],
            "lambda": [0.9, 0.999],
            "kl_target": [0.001, 0.1],
        }

    def ddpg_param_bounds(self):
        return {
            "buffer_size": [int(1e4), int(1e6)],
            "lr": [1e-5, 1],
            "train_batch_size": [32, 512],
        }

    def choose_space(self):
        if self.model_name == "ddpg":
            sample_hyperparameters = self.sample_ddpg_params()
            mutate_hyperparameters = self.mutate_ddpg_hyperparams()
            params_bounds = self.ddpg_param_bounds()
        elif self.model_name == "ppo":
            sample_hyperparameters = self.sample_ppo_params()
            mutate_hyperparameters = self.mutate_ppo_hyperparams()
            params_bounds = self.ppo_param_bounds()
        elif self.model_name == "a2c":
            sample_hyperparameters = self.sample_a2c_params()
            mutate_hyperparameters = self.mutate_a2c_hyperparams()
            params_bounds = self.a2c_params_bounds()

        return (sample_hyperparameters, mutate_hyperparameters, params_bounds)


class PopulationBasedRayTune:
    """
    Returns the analysis of the Hyperparameter optimization

      Parameters:
        env_config (dict): Environment arguments
        env_class: FinRL environment class
        env_name (str) : Name of the environment to register it
        model_name (str) : Name of the RL algorithm
        num_samples (int) : Population size
        training_iterations (int): Number of time ray.tune is reported
        log_dir (str): Directory to save the tensorboard logs and model weights

      Returns:
        analysis: Ray tune generated analysis
        You can do analysis.df to get all the results in a dataframe
    """

    def __init__(
        self,
        env_config: dict,
        env_class,
        env_name: str,
        model_name: str,
        num_samples: int,
        training_iterations: int,
        log_dir: str,
    ):
        self.env_config = env_config
        self.env_class = env_class
        self.env_name = env_name
        self.register_env()
        self.model_name = model_name
        (
            self.sample_hyperparameters,
            self.mutate_hyperparameters,
            self.params_bounds,
        ) = hyperparams_opt(self.model_name).choose_space()
        self.num_samples = num_samples
        self.training_iterations = training_iterations
        self.log_dir = log_dir
        self.MODEL_TRAINER = {"a2c": A2CTrainer, "ppo": PPOTrainer, "ddpg": DDPGTrainer}
        self.MODELS = {"a2c": a2c, "ddpg": ddpg, "td3": td3, "sac": sac, "ppo": ppo}

    def register_env(self):
        register_env(self.env_name, lambda config: self.env_class(self.env_config))

    def run_PBT(self):
        pbt_scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=self.training_iterations / 10,
            burn_in_period=0.0,
            quantile_fraction=0.25,
            hyperparam_mutations=self.mutate_hyperparameters,
        )
        analysis = tune.run(
            self.MODEL_TRAINER[self.model_name],
            scheduler=pbt_scheduler,  # To prune bad trials
            metric="episode_reward_mean",
            mode="max",
            config={
                **self.sample_hyperparameters,
                "env": self.env_name,
                "num_workers": 1,
                "num_gpus": 1,
                "framework": "torch",
                "log_level": "DEBUG",
            },
            num_samples=self.num_samples,  # Number of hyperparameters to test out
            stop={
                "training_iteration": self.training_iterations
            },  # Time attribute to validate the results
            verbose=1,
            local_dir="./" + self.log_dir,  # Saving tensorboard plots
            # resources_per_trial={'gpu':1,'cpu':1},
            max_failures=1,  # Extra Trying for the failed trials
            raise_on_failed_trial=False,  # Don't return error even if you have errored trials
            # keep_checkpoints_num = 2,
            checkpoint_score_attr="episode_reward_mean",  # Only store keep_checkpoints_num trials based on this score
            checkpoint_freq=self.training_iterations,  # Checpointing all the trials,
        )
        print("Best hyperparameter: ", analysis.best_config)
        return analysis

    def run_PB2(self):

        pb2_scheduler = PB2(
            time_attr="training_iteration",
            perturbation_interval=self.training_iterations / 10,
            quantile_fraction=0.25,
            hyperparam_bounds={**self.params_bounds},
        )

        analysis = tune.run(
            self.MODEL_TRAINER[self.model_name],
            scheduler=pb2_scheduler,  # To prune bad trials
            metric="episode_reward_mean",
            mode="max",
            config={
                **self.sample_hyperparameters,
                "env": self.env_name,
                "num_workers": 1,
                "num_gpus": 1,
                "framework": "torch",
                "log_level": "DEBUG",
            },
            num_samples=self.num_samples,  # Number of hyperparameters to test out
            stop={
                "training_iteration": self.training_iterations
            },  # Time attribute to validate the results
            verbose=1,
            local_dir="./" + self.log_dir,  # Saving tensorboard plots
            # resources_per_trial={'gpu':1,'cpu':1},
            max_failures=1,  # Extra Trying for the failed trials
            raise_on_failed_trial=False,  # Don't return error even if you have errored trials
            # keep_checkpoints_num = 2,
            checkpoint_score_attr="episode_reward_mean",  # Only store keep_checkpoints_num trials based on this score
            checkpoint_freq=self.training_iterations,  # Checpointing all the trials
        )
        print("Best hyperparameter: ", analysis.best_config)
        return analysis

 if __name__ == '__main__':
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
    
    train_env_config = get_train_env(TRAIN_START_DATE, VAL_END_DATE, 
                     ticker_list, data_source, time_interval, 
                        technical_indicator_list, env, model_name)
    test_config = get_test_config(TEST_START_DATE, TEST_END_DATE, ticker_list, data_source, time_interval, 
         technical_indicator_list, env, model_name, if_vix = True)
    
    pbt = PopulationBasedRayTune(
        env_config=train_env_config,
        env_class=StockTradingEnv_numpy,
        env_name="StockTrainingEnv",
        model_name="ppo",
        num_samples=100,
        training_iterations=100,
        log_dir="PBT Dir",
        )
    pb_analysis = pb.run_PBT()
    pb_analysis.to_csv('PBT.csv')
