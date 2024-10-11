import pandas as pd
import numpy as np
import scipy.optimize as sco

class PortfolioGenerator:
    def __init__(self, returns):
        self.rets = returns
        self.ticks = returns.columns

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
        port_rets = (self.rets * weights).sum()
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


class PortfolioGenerator_OLD:
    def __init__(self, rets):
        self.rets = rets
        self.noa = len(self.rets.columns)
        self.bounds = tuple((0,1) for x in range(self.noa))
        self.eweights = np.array(self.noa * [1. / self.noa])
        self.constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        self.ticks = list(self.rets.columns)

    def _port_ret(self, weights):
        """
        :param weights: List of weights
        :return: Float with yearly scaled average return for given weights
        """
        y_port_ret = (self.rets.mean() * weights).sum(skipna=False) * 252
        return y_port_ret

    def _annualized_mean(self, rets):
        """
        :param rets: Series of returns to calculate the annualized mean return
        :return: FLoat with annualized mean return
        """

        mean_ret = rets.mean()
        annualized_mean_ret = mean_ret * 252
        return annualized_mean_ret

    def _port_vol(self, weights):
        """
        :param weights: List of weights
        :return: Float with yearly scaled volatility for given weights
        """
        y_port_std_dev = np.dot(weights.T, np.dot(self.rets.cov() * 252, weights))
        y_port_vol = np.sqrt(y_port_std_dev)
        return y_port_vol

    def _annulized_vola(self, rets):
        """
        :param rets: Series of returns to calculate the annualized volatility
        :return: Float with annualized volatility
        """
        volatility = rets.std()
        annualized_volatility = volatility * np.sqrt(252)
        return annualized_volatility

    def _min_func_sharpe(self, weights):
        """
        :param weights: List of weights
        :return: Float with the negative sharpe ratio for given weights
        """
        neg_sharpe_ratio = -self._port_ret(weights) / self._port_vol(weights)
        return neg_sharpe_ratio

    def get_max_sharpe_weights(self):
        """
        :return: Array of optimized weights for maximum sharpe ratio
        """
        opt = sco.minimize(self._min_func_sharpe,
                           self.eweights,
                           bounds=self.bounds,
                           constraints=self.constraints,
                           method='SLSQP')
        opt_weights = opt['x'].round(3)
        return opt_weights

    def get_min_var_weights(self):
        """
        :return: Array of optimized weights for minimum volatility
        """
        opt = sco.minimize(self._port_vol,
                           self.eweights,
                           bounds=self.bounds,
                           constraints=self.constraints,
                           method='SLSQP')
        opt_weights = opt['x'].round(3)
        return opt_weights

    def get_port_returns(self, weights):
        """
        :param weights: Array of weights for given tickers to calculate portfolio returns
        :return: Series of portfolio returns calcualated by the sum of the stock returns multiplied with the given
        weights
        """
        weighted_rets = self.rets * weights
        port_rets = weighted_rets.sum(axis=1, skipna=False)
        port_rets.name = 'Portfolio Returns'
        return port_rets

    def get_weights_analysis(self, benchmark_rets, custom_weights):
        """
        :param benchmark_rets: Optional dataframe of benchmark returns
        :param custom_weights: Optional list of custom weights used for calculation of custom portfolio returns
        :return: Dataframe containing annualy mean return, volatility and sharpe ration for different portfolio types
        """

        total_rets = {}
        max_sharpe_rets = self.get_port_returns(self.get_max_sharpe_weights())
        min_var_rets = self.get_port_returns(self.get_min_var_weights())
        equal_weight_rets = self.get_port_returns(self.eweights)
        total_rets['Maximum Sharpe Ratio'] = max_sharpe_rets
        total_rets['Minimum Variance'] = min_var_rets
        total_rets['Equal Weights'] = equal_weight_rets
        if not benchmark_rets is None:
            total_rets['Benchmark'] = benchmark_rets.squeeze()
        if not custom_weights is None:
            try:
                custom_rets = self.get_port_returns(custom_weights)
                sum_weights = np.array(custom_weights).sum()
                if sum_weights != 1:
                    print('Warning: Sum of custom weights unequal to 1')
                total_rets['Custom Weights'] = custom_rets
            except:
                print('Warning: Length of custom weights does not match to the number of portfolio constituents')

        rets_analysis = pd.DataFrame(index=total_rets.keys())
        for key, value in total_rets.items():
            mean = self._annualized_mean(value)
            vola = self._annulized_vola(value)
            sharpe_ratio = mean / vola
            rets_analysis.loc[key, 'Anually Mean Return'] = mean
            rets_analysis.loc[key, 'Anually Volatility'] = vola
            rets_analysis.loc[key, 'Annualy Sharpe Ratio'] = sharpe_ratio

        rets_analysis.sort_values('Annualy Sharpe Ratio', inplace=True, ascending=False)
        return rets_analysis

    def get_weights_distribution(self):
        """
        :return: Dataframe with portfolio constituents and the distributed weights from minimum variance and maximum
        sharpe ration optimization
        """

        max_sharpe_weights = self.get_max_sharpe_weights()
        min_var_weights = self.get_min_var_weights()
        weights_dist = pd.DataFrame(index=self.ticks)
        weights_dist['Minimum Variance'] = min_var_weights
        weights_dist['Maximum Sharpe Ratio'] = max_sharpe_weights
        return weights_dist

    def get_portfolio_investment(self, investment, ticks, weights, quotes):
        """
        :param investment: Float for portfolio investment
        :param ticks: List of ticker symbols
        :param weights: Numpy array containing portfolio weights
        :param quotes: Numpy array containing the last valid quotes for stock ticker in list ticks
        :return: Dataframe with list ticks as index for portfolio information
        """
        investment_results = pd.DataFrame(index=ticks)
        stock_invest = investment * weights
        stock_count = stock_invest / quotes
        stock_count = stock_count.round()
        stock_count = stock_count.astype(int)
        effective_weights = quotes * stock_count
        effective_weights = effective_weights / investment
        investment_results.loc[:, 'Investment'] = investment * weights
        investment_results.loc[:, 'Number Shares'] = stock_count
        investment_results.loc[:, 'Effective Weights'] = effective_weights
        investment_results.loc[:, 'Optimal Weights'] = weights
        return investment_results
