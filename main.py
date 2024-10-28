from hierarchy import *
from finance_data_utils import *
import warnings
warnings.filterwarnings("ignore")

start = '2014-01-01'
end = '2024-05-01'
us_price = topN_us(20, start, end)
tw_price = topN_tw(20, start, end)

# Plot hierarchy clustering example
us_ret = us_price.pct_change().dropna()#['2022-01-01':'2024-01-01']

# with wasserstein distance
whc = wasserstein_HC(us_ret.T, method='average')
whc.fit()
w_link = whc.linkage_matrix
whc.plt_dendrogram('results/wass_hierarchy_plot')

# with euclidean distance
hc = HierarchicalClutering(us_ret.T, method='average')
hc.fit()
hc_link = hc.linkage_matrix
hc.plt_dendrogram('results/hierarchy_plot')


# Backtest
rolling_window = [24, 60]
hold_period = 12

for roll in rolling_window:
    # US
    val_list = []
    methods = ['EQW', 'HRP', 'W_HRP']
    for method in methods:
        val, wt = backtest(us_price, method, rolling_window=roll, hold_period=hold_period)
        val_list.append(val[0])

    val_df = pd.DataFrame(data=dict(zip(methods, val_list)))
    plot_series(val_df, save_path = f'results/us performance_plot_{roll}M')

    rf = pd.read_csv('data/US_risk_free_rate.csv', index_col='Date')
    us_perf = performance_metric(val_df, rf)
    us_perf.to_csv(f'results/us_performance_{roll}M.csv')

    # TW
    val_list = []
    methods = ['EQW', 'HRP', 'W_HRP']
    for method in methods:
        val, wt = backtest(tw_price, method, rolling_window=roll, hold_period=hold_period)
        val_list.append(val[0])

    val_df = pd.DataFrame(data=dict(zip(methods, val_list)))
    plot_series(val_df, save_path=f'results/tw performace plot_{roll}M')

    rf = pd.read_csv('data/TW_risk_free_rate.csv', index_col='Date')
    tw_perf = performance_metric(val_df, rf)
    tw_perf.to_csv(f'results/tw_performance_{roll}M.csv')
