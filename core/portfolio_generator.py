import pandas as pd
import numpy as np
import scipy.optimize as sco
import logging
import math
from core import logging_config
from misc.utils import *
from core.finance_adapter import FinanceAdapter

class PortfolioGenerator:
    def __init__(self, returns):
        self.rets = returns
        try:
            self.ticks = returns.columns
        except:
            pass
        self.params = read_json("parameter.json")

    def _init_params(self):
        """Function inits constant parameters
        for optimization problem
        """
        noa = len(self.rets.columns)
        bounds = tuple((0,1) for x in range(noa))
        eweights = np.array(noa * [1. / noa])
        constraints = (
            {'type': 'eq', 
             'fun': lambda x: np.sum(x) - 1}
        )
        self.noa = noa
        self.bounds = bounds
        self.eweights = eweights
        self.constraints = constraints

    def get_returns(self, weights):
        """Function calculates the daily portfolio 
        return by inherited returns and given weights

        :param weights: Weights for return calculation
        :type weights: Dictionary
        :return: Daily portfolio returns
        :rtype: Series
        """

        port_rets = pd.DataFrame(index=self.rets.index)
        for tick, weight in weights.items():
            try:
                weighted_rets = self.rets[tick] * weight
                port_rets[tick] = weighted_rets
            except:
                logging.warning(f"Ticker {tick} not in return portfolio constituents")
        port_rets = port_rets.sum(axis=1)
        return port_rets
    
    def get_max_sharpe_weights(self):
        """Function calculates optimized weights
        for maximum sharpe ratio portfolio

        :return: Portfolio weights
        :rtype: Dictionary
        """
        self._init_params()
        opt = sco.minimize(self._neg_sharpe_ratio,
                           self.eweights,
                           bounds=self.bounds,
                           constraints=self.constraints,
                           method='SLSQP')
        opt_weights = opt['x'].round(3)
        weight_dict = self._create_weight_dict(
            ticks=self.ticks,
            weights=opt_weights
        )
        return weight_dict
    
    def get_min_var_weights(self):
        """Function calculates optimized weights
        for minimum variance portfolio

        :return: Portfolio weights
        :rtype: Dictionary
        """
        self._init_params()
        opt = sco.minimize(self._annualized_volatility,
                           self.eweights,
                           bounds=self.bounds,
                           constraints=self.constraints,
                           method='SLSQP')
        opt_weights = opt['x'].round(3)
        weight_dict = self._create_weight_dict(
            ticks=self.ticks,
            weights=opt_weights
        )
        return weight_dict
    
    def get_custom_weights(self):
        custom = self.params["custom_weights"]
        custom_weights = custom.copy()
        custom_weight_sum = 0
        for tick, custom_weight in custom.items():
            if tick not in self.ticks:
                logging.warning(f"Ticker {tick} not in ticker list for historical returns")
                custom_weights.pop(tick)
                logging.info(f"Ticker {tick} was removed from custom weights")
            else:
                custom_weight_sum += custom_weight
        custom_weighs_ticks = custom_weights.keys()
        if custom_weight_sum != 1 or len(self.ticks) != len(custom_weighs_ticks):
            logging.warning("Sum of custom weights are not equal to 1 or length of custom weights not match with portfolio constituents") 
        return custom_weights
    
    def get_equal_weights(self):
        equal_weights = {}
        equal_weight = 1 / len(self.ticks)
        for tick in self.ticks:
            equal_weights[tick] = equal_weight
        return equal_weights
    
    def get_actual_invest(self, weights, actual_quotes):
        weight_dict = {}
        long_pos_dict = {}
        invest = self.params["investment"]
        for tick, weight in weights.items():
            long_pos_list = []
            actual_quote = actual_quotes[tick]
            raw_amount = invest * weight
            n_shares = raw_amount / actual_quote
            n_shares_adj = int(math.floor(n_shares))
            weight_adj = actual_quote * n_shares_adj
            weight_adj = weight_adj / invest
            tick_invest = n_shares_adj * actual_quote
            long_pos_list.append(n_shares_adj)
            long_pos_list.append(tick_invest)
            weight_dict[tick] = weight_adj
            long_pos_dict[tick] = long_pos_list
        return weight_dict, long_pos_dict
    
    def get_portfolio_performance(self, bench_rets):
        ann_mean_ret = self.rets.mean() * 252
        ann_mean_vol = np.sqrt(self.rets.std() * 252)
        sharpe_ratio = ann_mean_ret / ann_mean_vol
        bench_corr = self.rets.corr(bench_rets)
        return ann_mean_ret, ann_mean_vol, sharpe_ratio, bench_corr

    def _annualized_volatility(self, weights):
        """Function calculates the annualized
        portfolio volatility by inherited returns
        and given weights. Function is only used
        for maximization problem

        :param weights: Weights for volatility calculation
        :type weights: Array
        :return: Annualized portfolio volatility
        :rtype: Float
        """
        ann_std = np.dot(
            weights.T,
            np.dot(
                self.rets.cov() * 252,
                weights
            )
        )
        ann_vola = np.sqrt(ann_std)
        return ann_vola

    def _annualized_return(self, weights):
        """Function calculates the annualized mean 
        portfolio return by inherited returns and 
        given weights. Function is only used
        for maximization problem

        :param weights: Weights for mean return calculation
        :type weights: Array
        :return: Annualized mean portfolio return
        :rtype: Float
        """
        ann_rets = (self.rets * weights).sum()
        ann_ret = ann_rets.mean() * 252
        return ann_ret

    def _neg_sharpe_ratio(self, weights):
        """Function calculates the negative
        portfolio sharpe ratio. Function is only
        used for maximization problem

        :param weights: Weights for sharpe ratio calculation
        :type weights: Array
        :return: Negative sharpe ratio
        :rtype: Float
        """
        ann_ret = self._annualized_return(weights=weights)
        ann_vola = self._annualized_volatility(weights=weights)
        sharpe_ratio = ann_ret / ann_vola
        return -sharpe_ratio
    
    def _create_weight_dict(self, ticks, weights):
        """Function generates dictionary for weight
         to ticker mapping for given ticker symbols

        :param ticks: Ticker symbols
        :type ticks: Index
        :param weights: Optimized weights
        :type weights: Array
        :return: Optimized weight for respective ticker
        :rtype: Dictionary
        """
        weight_dict = {}
        keys = list(ticks)
        values = list(weights)
        for key, value in zip(keys, values):
            weight_dict[key] = value
        return weight_dict