import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from hierarchy import *


def topN_tw(n, start, end):
    cap = pd.read_excel('mktCap.xlsx')
    new_columns = {col: col.split(' ')[0] + '.TW' for col in cap.columns[1:]}
    cap = cap.rename(columns=new_columns)
    Top = pd.to_numeric(cap.iloc[0], errors='coerce').nlargest(n).index.tolist()
    close = yf.download(Top, start=start, end=end)
    close = close.loc[:, ('Adj Close', slice(None))]
    close.columns = close.columns.get_level_values(1)

    return close

def topN_us(n, start, end):
    us_stocks = pd.read_csv('usStockCAP_2023_12.csv')
    Top = us_stocks['Symbol'][0:n].tolist()
    Top.sort()
    close = yf.download(Top, start=start, end=end)
    close = close.loc[:, ('Adj Close', slice(None))]
    close.columns = close.columns.get_level_values(1)

    return close

def calculate_portfolio_value(prices_df, weights, initial_value):
    daily_returns = prices_df.pct_change().fillna(0)
    portfolio_daily_returns = daily_returns.dot(weights)
    portfolio_value_series = (1 + portfolio_daily_returns).cumprod() * initial_value

    return portfolio_value_series

def backtest(df, optimize, rolling_window, hold_period, share_df='none'):
    wt_all = pd.DataFrame()
    values = pd.Series(1)
    rolling_window = relativedelta(months = rolling_window)
    roll = relativedelta(months = hold_period)
    end = df.index[0] + rolling_window
    mons = df.index.strftime('%Y-%m').nunique()

    for i in range(mons-2):
        start = end - rolling_window
        start_p, end_p = start.strftime('%Y-%m'), end.strftime('%Y-%m')
        print(f'Backtest training data period : {start_p} - {end_p}')
        train_set = df[
            (df.index.strftime('%Y-%m') >= start.strftime('%Y-%m')) &
            (df.index.strftime('%Y-%m') < end.strftime('%Y-%m'))
            ]
        #train_set = train.dropna(axis=1)
    
        if optimize == 'EQW' :
            wt = {item: 1/len(train_set.columns) for item in train_set.columns} 
            wt = pd.DataFrame([wt],columns=wt.keys()).T

        elif optimize == 'MKT':
            if type(share_df)==str:
                print('Input share outstanding data')
            else:
                share_month = share_df.copy()
                latest_p = train_set.iloc[-1]
                share_month = share_month[train_set.columns]
                target_mon = (train_set.index[-1]+relativedelta(month=1)).strftime('%Y-%m')
                share_list = share_month.loc[target_mon]
                mcaps = pd.DataFrame(latest_p*share_list)
                wt = {item: (mcaps.loc[item]/(mcaps.values.sum())).values[0] for item in mcaps.index} #eqaul weight
                wt = pd.DataFrame([wt],columns=wt.keys()).T 
        
        elif optimize == 'HRP':
            ret = train_set.pct_change().dropna()
            hc = HierarchicalClutering(ret.T, method='average')
            hc.fit()
            link = hc.linkage_matrix
            wt = hierarchical_risk_parity(ret.cov(), link)
            wt = pd.DataFrame([wt],columns=wt.keys()).T

        elif optimize == 'W_HRP':
            ret = train_set.pct_change().dropna()
            whc = wasserstein_HC(ret.T, method='average')
            whc.fit()
            link = whc.linkage_matrix
            cov = whc.distances
            wt = hierarchical_risk_parity(ret.cov(), link)
            wt = pd.DataFrame([wt],columns=wt.keys()).T

        wt_all[end.strftime('%Y-%m')] = wt
        test_set = df[
            (df.index.strftime('%Y-%m') >= end.strftime('%Y-%m')) &
            (df.index.strftime('%Y-%m') < (end+roll).strftime('%Y-%m'))
            ]

        ini_val = values.iloc[-1]
        val = calculate_portfolio_value(test_set, wt, ini_val)
        print(f'Portfolio Value : {val}')
        values = pd.concat([values, val])
        end += roll

        if (end.strftime('%Y-%m') > df.index[-1].strftime('%Y-%m')) == True:
            break
        else:
            continue

    values = values[1:]
    values.index = pd.to_datetime(values.index)

    return values, wt_all

def performance_metric(portfolio_df, rf_df):
    trading_days_per_year = 252 
    sharpe_ratios, sortino_ratios, annual_returns = {}, {}, {}
    annual_volatilities, max_drawdowns, daily_vars = {}, {}, {}
    risk_free_rate_df = rf_df#.to_frame('Risk_Free_Rate')

    risk_free_rate_df.index = pd.to_datetime(risk_free_rate_df.index)
    daily_risk_free_rate_df = risk_free_rate_df.resample('D').ffill()

    portfolio_df.index = pd.to_datetime(portfolio_df.index)
    aligned_risk_free_rate = daily_risk_free_rate_df.loc[portfolio_df.index]

    for portfolio_name, portfolio_values in portfolio_df.items():
        portfolio_series = pd.Series(portfolio_values)
        
        # Daily returns
        daily_returns = portfolio_series.pct_change().dropna()
        risk_free_rate = aligned_risk_free_rate.loc[daily_returns.index].values.flatten()/100
        rf = (1 + risk_free_rate)**(1/252) - 1
        
        # Annual Return
        total_return = (portfolio_series.iloc[-1] - portfolio_series.iloc[0]) / portfolio_series.iloc[0]
        annual_return = (1 + total_return) ** (trading_days_per_year / len(portfolio_series)) - 1
        annual_returns[portfolio_name] = annual_return
        
        # Annual Volatility
        annual_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
        annual_volatilities[portfolio_name] = annual_volatility
        
        # Sharpe Ratio
        sharpe_ratio = (daily_returns - rf).mean() / daily_returns.std() * np.sqrt(trading_days_per_year)
        sharpe_ratios[portfolio_name] = sharpe_ratio
        
        # Sortino Ratio
        downside_deviation = daily_returns[daily_returns < 0].std()
        sortino_ratio = (daily_returns - rf).mean() / downside_deviation * np.sqrt(trading_days_per_year)
        sortino_ratios[portfolio_name] = sortino_ratio
        
        # Maximum Drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        max_drawdowns[portfolio_name] = max_drawdown
        
        # Daily VaR 
        confidence_level = 0.95
        daily_var = -np.percentile(daily_returns, 100 * (1 - confidence_level))
        daily_vars[portfolio_name] = daily_var

    results_df = pd.DataFrame({
        'Annual Return': annual_returns,
        'Annual Volatility': annual_volatilities,
        'Sharpe Ratio': sharpe_ratios,
        'Sortino Ratio': sortino_ratios,
        'Max Drawdown': max_drawdowns,
        'Daily VaR (95%)': daily_vars
    })

    return(results_df)

def plot_series(df, save_path=None):
    plt.figure(figsize=(10, 6))
    line = ['--', '-', '-.', ':']
    for i, column in enumerate(df.columns):
        plt.plot(df.index, df[column], label=column, linestyle=line[i])

    plt.xlabel('Time'), plt.ylabel('Value')
    quarters = pd.date_range(start=df.index.min(), end=df.index.max(), freq='6M')
    quarter_labels = [date.strftime('%Y-%m') for date in quarters]
    plt.xticks(quarters, quarter_labels)
    plt.xticks(rotation=45)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()



