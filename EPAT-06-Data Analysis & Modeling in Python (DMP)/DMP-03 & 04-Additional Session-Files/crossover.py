import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from itertools import product
import random
from typing import Dict, List, Tuple, Union, Optional, Any, Iterable


class CROSSOVER:
    """
    class attribute: We can know the possible values for moving_average and metrics
                     before creating an object of the class
    """

    moving_average = ("Simple","exponential") 
    MAX_ATTEMPTS = 100
    metrics = {'Sharpe': 'Sharpe Ratio',
               'CAGR': 'Compound Average Growth Rate',
               'MDD': 'Maximum drawdown',
               'NHR': 'Normalized Hit Ratio',
               'OTS': 'Optimal trade size'}
    
    def __init__(self,
                 dataframe: pd.DataFrame,
                 window_sizes: Dict[int, Tuple[int, int, int]],
                 moving_average: str = "Simple",
                 regime: bool = True,
                 allocation: int = 100,
                 data_name: Optional[str] = None,
                 start: Optional[str] = None,
                 end: Optional[str] = None
                 ) -> None:

        """
         Doc string of the class. We define what class is about here
        """
        data = dataframe.copy()
        # Sanity check
        if regime:
            assert data.regime.nunique() == len(window_sizes)
            assert set(data.regime.unique()) == set(window_sizes.keys())

        assert set(data.columns) == {'Date', 'Close', 'regime', 'data_category'}

        for k,v in window_sizes.items(): # v:short, long
            if v[0] == v[1]:
                AssertionError(f"(Short window,Long window) as ({v}) is not allowed")
            elif v[0] > v[1]:
                window_sizes[k] = v[1],v[0],v[2]

        # Initializations
        self.data = data  # the dataframe
        self.data['yr'] = self.data['Date'].dt.year
        
        self.moving_average = moving_average
        if data_name is None:  # the name that will appear on plots
            self.data_name = moving_average
        else:
            self.data_name = data_name
            
        self.n_regimes = len(window_sizes)
        self.window_sizes = window_sizes
        self.n_days = (self.data.Date.iloc[-1] - self.data.Date.iloc[0])
        self.allocation = allocation
        self.start = start  # the beginning date of the sample period
        self.end = end  # the ending date of the sample period

    def generate_signals(self, charts: bool= True) -> None:
        """
        This is an instance method. Only works on an object.
        """
        self.data['signal'] = np.nan
        for k,v in self.window_sizes.items():
            _sw, _lw, behaviour = v
            if self.moving_average == "Simple":
                    self.data[f"SMA_{k}"] = self.data.Close.rolling(_sw).mean()
                    self.data[f"LMA_{k}"] = self.data.Close.rolling(_lw).mean()
            elif self.moving_average == "exponential":
                self.data[f"SMA_{k}"] = self.data.Close.ewm(span=_sw, adjust=True).mean()
                self.data[f"LMA_{k}"] = self.data.Close.ewm(span=_lw, adjust=True).mean()
            else:
                raise Exception("Not a valid type")

            self.data[f"_lag1_SMA_{k}"] = self.data[f"SMA_{k}"].shift(1)
            self.data[f"_lag1_LMA_{k}"] = self.data[f"LMA_{k}"].shift(1)

            # creating signal
            buy_mask = (self.data[f"SMA_{k}"] > self.data[f"LMA_{k}"]) & (self.data[f"_lag1_SMA_{k}"] < self.data[f"_lag1_LMA_{k}"])
            sell_mask = (self.data[f"SMA_{k}"] < self.data[f"LMA_{k}"]) & (self.data[f"_lag1_SMA_{k}"] > self.data[f"_lag1_LMA_{k}"])

            self.data[f'signal_{k}'] = np.nan
            self.data.loc[buy_mask,f'signal_{k}'] = behaviour
            self.data.loc[sell_mask,f'signal_{k}'] = -1*behaviour
            self.data[f'signal_{k}'] = self.data[f'signal_{k}'].fillna(method="ffill")
            if self.n_regimes == 1:
                regime_mask = self.data.regime.notnull()
            else:
                regime_mask = (self.data.regime == k)
            self.data['signal'] = np.where(regime_mask, self.data[f'signal_{k}'], self.data['signal'])
#         if burn:
#             burn_period = 2*self.long_window
#             self.data.loc[:burn_period,"signal"] = np.nan

        mask = (self.data.signal != self.data.signal.shift(1)) & (self.data.signal.notnull())
        self.data['trade_num'] = np.where(mask,1,0).cumsum()

        # display chart
        if charts:
            fig = plt.figure()
            ax = {}
            for k, v in self.window_sizes.items():
                _sw, _lw, behaviour = v
                behaviour = "Regular" if behaviour==1 else "Inverse"
                ax[k] = plt.subplot(self.n_regimes*100 + 11+k)
                ax[k].plot(self.data['Date'],self.data[f"SMA_{k}"],color='black', label=f'SMA ={_sw}')
                ax[k].plot(self.data['Date'],self.data[f"LMA_{k}"],color='blue', label=f'LMA ={_lw}')
                ax[k].set_title(f"Regime {k} ({behaviour})")
                ax[k].set_xlim(self.data['Date'].values[0],self.data['Date'].values[-1])
                ax[k].legend(loc=0)
                d_color = {}
                d_color[1] = '#90ee90'  # light green
                d_color[-1] = "#ffcccb" # light red

                # create long and short vertical spans
                anchor = 0
                for i in range(1,self.data.shape[0]):
                    if np.isnan(self.data.signal[i-1]):
                        anchor=i
                    elif (self.data.signal[i-1] == self.data.signal[i]) and (i< (self.data.shape[0]-1)):
                        continue
                    else:
                        ax[k].axvspan(self.data['Date'][anchor], self.data['Date'][i], 
                                   alpha=0.5, color=d_color[self.data.signal[i-1]], label="interval")
                        anchor = i
                        
                # create white space
                if self.n_regimes>1:
                    anchor = 0
                    anchor_set =0
                    for i in range(1,self.data.shape[0]):
                        if (self.data.regime[i]!=k) and (anchor_set==0):
                            anchor=i
                            anchor_set = 1
                        elif (self.data.regime[i-1] == self.data.regime[i]) and (i< (self.data.shape[0]-1)):
                            continue
                        else:
                            ax[k].axvspan(self.data['Date'][anchor], self.data['Date'][i],color='white', label="interval")
                            anchor_set = 0
            if self.n_regimes>1: 
                ax[0].get_shared_x_axes().join(ax[0], ax[1])
                ax[0].set_xticklabels([])      
                ax[1].tick_params(which='major', length=17)
            plt.show()
        
    def signal_performance(self):
        """
        Another instance method
        """

        # creating returns and portfolio value series
        self.data['Return'] = self.data['Close']/self.data['Close'].shift(1) - 1
        self.data['S_Return'] = self.data['signal'].shift(1)*self.data['Return']
        self.data['Market_Return'] = (1+self.data['Return']).cumprod() - 1
        self.data['Strategy_Return'] = (1+self.data['S_Return']).cumprod() - 1
        self.data['Portfolio Value'] = ((self.data['Strategy_Return']+1)*self.allocation)
        self.data['Market_Portfolio'] = ((self.data['Market_Return']+1)*self.allocation)

        self.data['Wins'] = np.where(self.data['S_Return'] > 0,1,0)
        self.data['Losses'] = np.where(self.data['S_Return']<0,1,0)
        train_mask = self.data.data_category == "train"
        val_mask = self.data.data_category == "val"
        self.n_days = {'train': (self.data[train_mask].Date.iloc[-1] - self.data[train_mask & self.data.signal.notnull()].Date.iloc[0]),
                       'val': (self.data[val_mask].Date.iloc[-1] - self.data[val_mask].Date.iloc[0])}

        # self.n_train_days = (self.data[train_mask].Date.iloc[-1] - self.data[train_mask & self.data.signal.notnull()].Date.iloc[0])
        # self.n_valid_days = (self.data[val_mask].Date.iloc[-1] - self.data[val_mask].Date.iloc[0])

        ## Daywise train Performance
        # d_train_perform = {}
        # d_train_perform['TotalWins']=self.data[train_mask]['Wins'].sum()
        # d_train_perform['TotalLosses']=self.data[train_mask]['Losses'].sum()
        # d_train_perform['TotalTrades']=d_train_perform['TotalWins']+d_train_perform['TotalLosses']
        # d_train_perform['HitRatio']=round(d_train_perform['TotalWins']/d_train_perform['TotalTrades'],2)
        # d_train_perform['CAGR'] = ((1+self.data[train_mask]['S_Return']).cumprod()).iloc[-1]**(365.25/self.n_train_days.days) -1
        # d_train_perform['SharpeRatio'] = d_train_perform['CAGR']/ self.data[train_mask]["S_Return"].std() / (252**.5)        
        # d_train_perform['MaxDrawdown']=(1.0-self.data[train_mask]['Portfolio Value']/self.data[train_mask]['Portfolio Value'].cummax()).max()
        # self.market_CAGR_train = ((1+self.data[train_mask]['Return']).cumprod()).iloc[-1]**(365.25/self.n_train_days.days) -1
        # self.market_Sharpe_train = d_train_perform['Market_CAGR']/ self.data[train_mask]["Return"].std() / (252**.5)
        # self.market_drawdown_train=(1.0-self.data[train_mask]['Market_Portfolio']/self.data[train_mask]['Market_Portfolio'].cummax()).max()
        # d_train_perform['Market_Excess_CAGR'] = d_train_perform['CAGR'] - self.market_CAGR_train
        # d_train_perform['Market_Excess_Sharpe'] = d_train_perform['SharpeRatio'] - self.market_Sharpe_train
        # d_train_perform['Market_Excess_Drawdown'] = d_train_perform['MaxDrawdown'] - self.market_drawdown_train
        # self.daywise_train_performance = pd.Series(d_train_perform)

        def get_day_wise_performance(mask, category):
            d_perform = {}
            d_perform['TotalWins']=self.data[mask]['Wins'].sum()
            d_perform['TotalLosses']=self.data[mask]['Losses'].sum()
            d_perform['TotalTrades']=d_perform['TotalWins']+d_perform['TotalLosses']
            d_perform['HitRatio']=round(d_perform['TotalWins']/d_perform['TotalTrades'],2)
            d_perform['CAGR'] = ((1+self.data[mask]['S_Return']).cumprod()).iloc[-1]**(365.25/self.n_days[category].days) -1
            d_perform['SharpeRatio'] = d_perform['CAGR']/ self.data[mask]["S_Return"].std() / (252**.5)        
            d_perform['MaxDrawdown']=(1.0-self.data[mask]['Portfolio Value']/self.data[mask]['Portfolio Value'].cummax()).max()

            # market stats
            self.market['CAGR'][category] = ((1+self.data[mask]['Return']).cumprod()).iloc[-1]**(365.25/self.n_days[category].days) -1
            self.market['Sharpe'][category] = self.market['CAGR'][category]/ self.data[mask]["Return"].std() / (252**.5)
            self.market['DrawDown'][category] =(1.0-self.data[mask]['Market_Portfolio']/self.data[mask]['Market_Portfolio'].cummax()).max()
            d_perform['Market_Excess_CAGR'] = d_perform['CAGR'] - self.market['CAGR'][category]
            d_perform['Market_Excess_Sharpe'] = d_perform['SharpeRatio'] - self.market['Sharpe'][category]
            d_perform['Market_Excess_Drawdown'] = d_perform['MaxDrawdown'] - self.market['DrawDown'][category]
            return pd.Series(d_perform)

        # Trade wise train performance
        def get_trade_wise_performance(mask, category):
            _df = self.data[mask].groupby(["signal","trade_num"]).S_Return.sum().reset_index()
            _df['Wins']=np.where(_df['S_Return'] > 0,1,0)
            _df['Losses']=np.where(_df['S_Return']<0,1,0)
            d_tp = {}
            d_tp.update(_df[["Wins","Losses"]].sum().rename({'Wins': 'TotalWins','Losses': 'TotalLosses'}).to_dict())
            d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
            d_tp['HitRatio'] = np.round(d_tp["TotalWins"] / d_tp['TotalTrades'],4)
            d_tp['AvgWinRet'] = np.round(_df[_df.Wins == 1].S_Return.mean(),4)
            d_tp['AvgLossRet'] = np.round(_df[_df.Losses == 1].S_Return.mean(),4)
            d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet']/d_tp['AvgLossRet']),2)
            d_tp['RetVar'] = np.round(_df.S_Return.std(),4)
            try:  # breaks when just Wins are present
                _sum = _df.groupby("Wins").S_Return.sum()
                d_tp['NormHitRatio'] = np.round(_sum[1]/_sum.abs().sum(),4)
            except:
                d_tp['NormHitRatio'] = np.nan
            d_tp['OptimalTradeSize'] = self.kelly(p = d_tp['HitRatio'], b = d_tp['WinByLossRet'])
            return pd.Series(d_tp)

        self.market = {'CAGR': {}, 'Sharpe': {}, 'DrawDown': {}}
        self.daywise_performance = {}
        self.tradewise_performance = {}
        self.daywise_performance['train'] = get_day_wise_performance(train_mask, category='train')
        self.tradewise_performance['train'] = get_trade_wise_performance(train_mask, category='train')
        self.daywise_performance['val'] = get_day_wise_performance(val_mask, category='val')
        self.tradewise_performance['val'] = get_trade_wise_performance(val_mask, category='val')

        ## Daywise val Performance
        # d_val_perform = {}
        # d_val_perform['TotalWins']=self.data[val_mask]['Wins'].sum()
        # d_val_perform['TotalLosses']=self.data[val_mask]['Losses'].sum()
        # d_val_perform['TotalTrades']=d_val_perform['TotalWins']+d_val_perform['TotalLosses']
        # d_val_perform['HitRatio']=round(d_val_perform['TotalWins']/d_val_perform['TotalTrades'],2)
        # d_val_perform['CAGR'] = ((1+self.data[val_mask]['S_Return']).cumprod()).iloc[-1]**(365.25/self.n_train_days.days) -1
        # d_val_perform['SharpeRatio'] = d_val_perform['CAGR']/ self.data[val_mask]["S_Return"].std() / (252**.5)        
        # d_val_perform['MaxDrawdown']=(1.0-self.data[val_mask]['Portfolio Value']/self.data[val_mask]['Portfolio Value'].cummax()).max()
        # d_val_perform['Market_CAGR'] = ((1+self.data[val_mask]['Return']).cumprod()).iloc[-1]**(365.25/self.n_train_days.days) -1
        # d_val_perform['Market_SharpeRatio'] = d_val_perform['Market_CAGR']/ self.data[val_mask]["Return"].std() / (252**.5)        
        # d_val_perform['Market_MaxDrawdown']=(1.0-self.data[val_mask]['Market_Portfolio']/self.data[val_mask]['Market_Portfolio'].cummax()).max()
        # self.daywise_val_performance = pd.Series(d_val_perform)



        ## Tradewise val performance
        # _df = self.data[val_mask].groupby(["signal","trade_num"]).S_Return.sum().reset_index()
        # _df['Wins']=np.where(_df['S_Return'] > 0,1,0)
        # _df['Losses']=np.where(_df['S_Return']<0,1,0)
        # d_tp = {}
        # d_tp.update(_df[["Wins","Losses"]].sum().rename({'Wins':'TotalWins','Losses':'TotalLosses'}).to_dict())
        # d_tp['TotalTrades'] = d_tp["TotalWins"] + d_tp["TotalLosses"]
        # d_tp['HitRatio'] =  np.round(d_tp["TotalWins"] / d_tp['TotalTrades'],4)
        # d_tp['AvgWinRet'] = np.round(_df[_df.Wins==1].S_Return.mean(),4)
        # d_tp['AvgLossRet'] = np.round(_df[_df.Losses==1].S_Return.mean(),4)
        # d_tp['WinByLossRet'] = np.round(abs(d_tp['AvgWinRet']/d_tp['AvgLossRet']),2)
        # d_tp['RetVar'] = np.round(_df.S_Return.std(),4)
        # _sum = _df.groupby("Wins").S_Return.sum()
        # d_tp['NormHitRatio'] = np.round(_sum[1]/_sum.abs().sum(),4)
        # d_tp['OptimalTradeSize'] = self.kelly(p = d_tp['HitRatio'], b = d_tp['WinByLossRet'])
        # self.tradewise_val_performance = pd.Series(d_tp)

    @staticmethod
    def kelly(p: float,b: float) -> float:
        """
        Static method: No object or class related arguments
        p: win prob, b: net odds received on wager, output(f*) = p - (1-p)/b

        Spreadsheet example
            from sympy import symbols, solve, diff
            x = symbols('x')
            y = (1+3.3*x)**37 *(1-x)**63
            solve(diff(y, x), x)[1]
        Shortcut
            .37 - 0.63/3.3
        """
        return np.round(p - (1-p)/b,4)

    def plot_performance(self):

        self.signal_performance()
        
        # yearly performance
        self.yearly_performance()
        
        # Plotting the Performance of the strategy
        train_mask = self.data.data_category == "train"
        plt.plot(self.data[train_mask]['Date'],(1+self.data[train_mask]['Return']).cumprod(),color='black', label='Market Returns')
        plt.plot(self.data[train_mask]['Date'],(1+self.data[train_mask]['S_Return']).cumprod(),color='blue', label= 'Strategy Returns')
        plt.title('%s Strategy Backtest: Training dataset'%(self.data_name))
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

        val_mask = self.data.data_category == "val"
        plt.plot(self.data[val_mask]['Date'],(1+self.data[val_mask]['Return']).cumprod(),color='black', label='Market Returns')
        plt.plot(self.data[val_mask]['Date'],(1+self.data[val_mask]['S_Return']).cumprod(),color='blue', label= 'Strategy Returns')
        plt.title('%s Strategy Backtest: Validation dataset'%(self.data_name))
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()


    def yearly_performance(self):
        """
        Instance method
        Adds an instance attribute: yearly_df
        """
        _yearly_df = self.data.groupby(['yr','signal']).S_Return.sum().unstack()
        _yearly_df.rename(columns={-1.0:'Sell',1.0:'Buy'}, inplace=True)
        _yearly_df['Return'] = _yearly_df.sum(1)

        # yearly_df
        self.yearly_df = _yearly_df.style.bar(color=["#ffcccb",'#90ee90'], align = 'zero').format({
            'Sell': '{:,.2%}'.format,'Buy': '{:,.2%}'.format,'Return': '{:,.2%}'.format})
        
    @staticmethod
    def optimal(_df):
        """ 
        This is a function inside a function.
        We could have created it outside the class if needed, but that would made it harder to relate. 
        Lower rank is better
        """
        return np.abs(_df.val - _df.train).rank() + _df.train.rank(ascending=False)

    @property
    def update_metrics(self) -> pd.DataFrame:
        """
        Called from the SMA_LMA_matrix class method
        """
        d_field = {'train': {}, 'val': {}}

        for category in ['train','val']:
            d_field[category]['Sharpe'] = self.daywise_performance[category].SharpeRatio
            d_field[category]['CAGR'] = self.daywise_performance[category].CAGR
            d_field[category]['MDD'] = self.daywise_performance[category].MaxDrawdown
            d_field[category]['NHR'] = self.tradewise_performance[category].NormHitRatio
            d_field[category]['xs_Sharpe'] = self.daywise_performance[category].Market_Excess_Sharpe
            d_field[category]['xs_CAGR'] = self.daywise_performance[category].Market_Excess_CAGR
            d_field[category]['xs_MDD'] = self.daywise_performance[category].Market_Excess_Drawdown
            # d_field[category]['xs_NHR'] = self.tradewise_performance[category].NormHitRatio
            # d_field[category]['OTS'] = self.tradewise_performance[category].OptimalTradeSize

        return pd.DataFrame(d_field).stack()
    
    @classmethod
    def quick_compare_random(cls,
                             data: pd.DataFrame,
                             metrics: List[str],
                             within: Iterable = range(5,100),
                             moving_average: str = "Simple",
                             top_k: int =3
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # cols = ["TimeTaken","BestPair",'CAGR','SharpeRatio', 'MaxDrawdown',
        #         'Market_CAGR', 'Market_SharpeRatio', 'Market_MaxDrawdown']
        
        start_time = time.time()
        d_optimal, no_reg_seen_df = cls.random_no_regime_pairs(data, within=within, metrics=metrics,
                                                               moving_average=moving_average)
        elapsed_time = time.time() - start_time

        print(f"No regime: Found best pair as {d_optimal} from {no_reg_seen_df.shape[0]} "
              f"combinations in {int(elapsed_time)} seconds")

        start_time = time.time()
        d_optimal, dual_reg_seen_df = cls.random_2regime_pairs(data, within=within, metrics=metrics,
                                                               moving_average=moving_average)
        elapsed_time = time.time() - start_time

        print(f"2 regimes: Found best pair as {d_optimal} from {dual_reg_seen_df.shape[0]} "
              f"combinations in {int(elapsed_time)} seconds")

        # TODO: Add daywise val performance
        return no_reg_seen_df.head(top_k),dual_reg_seen_df.head(top_k)

    @classmethod
    def random_2regime_pairs(cls,
                             data: pd.DataFrame,
                             metrics: List[str],
                             within: Iterable = range(1,100),
                             moving_average: str = "Simple",
                             optimal_sol: bool = True
                             ) -> Tuple[Dict, pd.DataFrame]:
        """
        This is a class method. First argument is a class.
        """
        prod = [pair for pair in product(within, within) if abs(pair[0] - pair[1])>4 ]
        if data.regime.nunique() == 2:
            dbl_prod = product(prod, prod)
        else:
            AssertionError("Max 2 regimes supported")
        
        if len(prod)>cls.MAX_ATTEMPTS**0.5:
            dbl_prod = random.sample(list(dbl_prod),cls.MAX_ATTEMPTS)
        
        l_profiles = []
        l_pairs = []
        for pair1, pair2 in dbl_prod:
            pair1 = (*pair1,random.choice([-1,+1]))
            pair2 = (*pair2,random.choice([-1,+1]))
            window_sizes = {0:pair1, 1:pair2}
            obj = cls(data, window_sizes , moving_average) ## object being created from the class
            obj.generate_signals(charts=False)
            obj.signal_performance()
            l_pairs.append(str((pair1, pair2)))
            l_profiles.append(obj.update_metrics)

        d_df = pd.concat(l_profiles, axis=1)
        d_df.columns = l_pairs
        d_df = d_df.T

        if optimal_sol:
            d_df['Signal'] = 0
            if 'Sharpe' in metrics: d_df['Signal'] += 2*cls.optimal(d_df['Sharpe'])
            if 'NHR' in metrics: d_df['Signal'] += cls.optimal(d_df['NHR'])
            if 'CAGR' in metrics: d_df['Signal'] += cls.optimal(d_df['CAGR'])
            # if 'MDD' in metrics: d_df['Signal'] -= cls.optimal(d_df['xs_MDD'])
            
            d_df.sort_values("Signal", inplace=True)
            p0, p1 = eval(d_df.head(1).index.values[0])
            d_optimal = {0: p0, 1: p1}
        return d_optimal, d_df

    @classmethod
    def random_no_regime_pairs(cls,
                               data: pd.DataFrame,
                               metrics: List[str],
                               within: Iterable = range(1,100),
                               moving_average: str = "Simple",
                               optimal_sol:bool = True
                               ) -> Tuple[Dict, pd.DataFrame]:
        """
        This is a class method. First argument is a class.
        """
        prod = [pair for pair in product(within, within) if abs(pair[0] - pair[1])>4 ]
        if len(prod)>cls.MAX_ATTEMPTS//2:
            prod = random.sample(list(prod),cls.MAX_ATTEMPTS//2)
        
        l_profiles = []
        l_pairs = []
        for pair in prod:
            pair = (*pair,random.choice([-1,+1]))
            window_sizes = {0:pair}
            obj = cls(data, window_sizes, moving_average, regime=False)
            obj.generate_signals(charts=False)
            obj.signal_performance()
            l_pairs.append(str(pair))
            l_profiles.append(obj.update_metrics)

        d_df = pd.concat(l_profiles, axis=1)
        d_df.columns = l_pairs
        d_df = d_df.T
        # print(d_df)
        if optimal_sol:
            d_df['Signal'] = 0
            if 'Sharpe' in metrics: d_df['Signal'] += 2*cls.optimal(d_df['Sharpe'])
            if 'NHR' in metrics: d_df['Signal'] += cls.optimal(d_df['NHR'])
            if 'CAGR' in metrics: d_df['Signal'] += cls.optimal(d_df['CAGR'])
            # if 'MDD' in metrics: d_df['Signal'] -= 2*cls.optimal(d_df['xs_MDD'])
            
            d_df.sort_values("Signal", inplace=True)
            p0 = eval(d_df.head(1).index.values[0])
            d_optimal = {0: p0}
        return d_optimal, d_df
