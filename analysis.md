
<div style="text-align: center;">
  <h1 style="font-size: 30px;">Does Ethereum copy Bitcoin ?</h1>
</div>


## Intro + Short talk
You have definitely heard of Bitcoin, digital cryptocurrency that is supposed to make things easier, but somehow doesnt. Many novices that touched it, managed to lost quite bit of money and in the end it actually worsened their financial situation.
Memories remain and regrets arises every now and then for not investing and hauling crypto soon enough, so in remorse people join the so called "pump n dump" scheme or also reffered as "bubble", meaning they buy at high price and not knowing 2 bits about the infrastructure, panic sell due to high volatility and get sidetracked to more safe investment opportunities (usually despite taking loss) such as stocks fiasco, instead of do a little bit of research.

Maybe its just my point of view, but closing market for a weekend and enforce personal information in order to invest or to pay tax because you are in unrealised profit is just widely accepted scam.

Well either way, ethereum is here and with its zero knowledge proves, L2s, privacy protection and (recent ETH 2.0) proof of stake mechanism things are ideologically promosing. Or are they?
With these great things built, ETH should take its lead and diverge, stand out as a cryptocurrency and rise, but nothing really happened, in fact it seems as if it has no impact whatsoever, as if people didnt care, and only buy all the cryptocurrencies to DCA. This leads us to state our hypotheses:

## Zero hypothesis $H_0$ and altenative hypothesis $H_1$

**Does Ethereum only copy bitcoin?**

Or on the other hand: **Does ETH has its own drive and people distinguish these two cryptocurrencies and buy each independently based on some criteria?**

## Preparation
The fisrt thing i did was to gather data, there is a [blockchair](https://blockchair.com) which provided me with historical prices of both coins in ".tsv" format.\
Using python I parsed the data and thanks to wide range libraries, analysis was made easy - I used Engle-Granger 2step cointegration* test which consists of regressing ETH to BTC and testing residuals by ADF, of course the daily average trading prices (data) have been normalized beforehand. \
Bitcoin is older than Ethereum so I decided to just cut its leading trail and measure from the same point in history which is 30.7 in 2015 - the launch of Ethereum.

There are many models i could have used instead, each and every one of them is bad, however some actually tells us more relevant information than others. Engle-Granger was (in my opinion and knowledge) the most suitable one.

*[Cointegration](https://www.mathworks.com/discovery/cointegration.html) is an analytic technique for testing for common trends in multivariate time series and modeling long-run and short-run dynamics. Two or more predictive variables in a time-series model are cointegrated when they share a common stochastic drift. Variables are considered cointegrated if a linear combination of them produces a stationary time series.

**[stationary distribution / stationary time series](https://www.mathworks.com/help/symbolic/markov-chain-analysis-and-stationary-distribution.html?s_tid=srchtitle_site_search_1_stationary%20distribution)
Mathematically, if P is the transition matrix of the Markov chain, then the stationary distribution π satisfies the equation πP=π, where π is a row vector and P is the transition matrix.

## Procedure
The **Engle-Granger test** is generally used to determine whether two (or more) time series are cointegrated - they share a long-term equilibrium relationship despite being non-stationary on their own. \
Steps we take are simple enough, firstly we regress all time series onto one we choose as pivot (Ethereum is regressed onto Bitcoin prices in this case). We have obtained a residual on which we can do the second step, which is apply Augmented Dickey-Fuller (ADF) test to check for statioinary.

The **ADF test** is a statistical test that determines whether a given time series are stationary. We either deny or confirm the H0 we stated at the beggining depending on the result of the test, in this particular example, showing that the residuals of the regression between two cryptocurrencies are stationary means cointegration, indicating a longterm equilibrium.
1. Firstly the **ADF statistics** explained: If we dispone a negative value that more significantly diverges from zero suggests stronger evidence against the null hypothesis of a unit root.
2. Secondly **p-value**: assuming H0 holds true, it is an indication of how well is the sationary supported (typically and so in this project ) p < 0.05 leads to rejecting the null hypothesis.
3. Lastly **critical values** are thresholds at which ADF would lead to rejecting the H0 at different severity levels (in this case 1%, 3%, and 10%).


## Assumptions
1. **linear relationship inbetween both coins** - appears to hold true due to the Engle-Granger cointegration test results and linear regression is possible, ADF confirms it.
2. existence of *only* **single equation** (involving a constant term and a stochastic term) is sufficient to model out the relationship accurately - This assumption has been reasonably validated. The residuals from the regression analysis exhibit stationarity, as indicated by the ADF test results. This suggests that the single-equation model used here captures the essential relationship between Bitcoin and Ethereum without requiring more complex modeling approaches.
3. **variable order** - The choice to regress ETH prices on BTC prices without a predefined direction of dependency seems justified by the outcome. ADF statistic and low p-value imply that the direction of regression chosen does not adversely impact the validity of the statistic model. While this approach worked in this analysis, it's generally doesnt have to as varialbles are not interchangable in different contexts.

## Results & Hypothesis claim
I think the result ins't that shocking, based on this data we can confidently say that ethereum isn't as self-sustainable despite the technology it possesses in comparison to Bitcoin. Therefore we can claim our $H_0$ hypothesis, that is ETH and BTC are not so different despite the various differences both offer and confidently state it holds true.

Down below there is a graph of normalized prices of both coins and a result from the python code.

![image](graph.png)

```sh
ADF Statistic: -3.6455113831610215
p-value: 0.004948241017297639
Critical Values: {'1%': -3.4323793062321144, '5%': -2.8624366253389106, '10%': -2.567247293335737}
```

## Code
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def input_parsing(file):
    data = pd.read_csv(file, delimiter='\t', names=['Time', 'Price'], skiprows=1, parse_dates=['Time'], dayfirst=True)
    data.set_index('Time', inplace=True)
    return data

def normalize(data):
    return (data['Price'] - data['Price'].min()) / (data['Price'].max() - data['Price'].min())


def formatting(btc_data, eth_data):
    start_date = max(btc_data.index.min(), eth_data.index.min())
    end_date = min(btc_data.index.max(), eth_data.index.max())
    btc_aligned = btc_data.loc[start_date:end_date].copy()
    eth_aligned = eth_data.loc[start_date:end_date].copy()
    btc_aligned['Normalized Price'] = normalize(btc_aligned)
    eth_aligned['Normalized Price'] = normalize(eth_aligned)
    return btc_aligned, eth_aligned


'''Engle-Granger 2step cointegration test
1. regress projection ETH to BTC
2. test residuals by ADF test '''
def cointegration_test(btc_prices, eth_prices):
    y = eth_prices.values
    x = sm.add_constant(btc_prices.values)
    model = sm.OLS(y, x).fit()
    residuals = model.resid

    adf_result = adfuller(residuals)
    return adf_result


def graphing(btc_data, eth_data):
    plt.figure(figsize=(12, 6))
    plt.plot(btc_data.index, btc_data['Normalized Price'], label='BTC Normalized', color='orange')
    plt.plot(eth_data.index, eth_data['Normalized Price'], label='ETH Normalized', color='darkblue')
    plt.title('Normalized Price of BTC and ETH')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.grid(True)
    plt.legend()
    plt.show()


btc, eth = formatting(input_parsing('btc_prices.tsv'), input_parsing('eth_prices.tsv'))
res = cointegration_test(btc['Normalized Price'], eth['Normalized Price'])

print('ADF:', res[0])
print('p-value:', res[1])
print('critical values:', res[4])
graphing(btc, eth)

```