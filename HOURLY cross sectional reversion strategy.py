#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[1]:


def retrieve_data(stocks, start, end):
    data = get_pricing(stocks, start_date = start, end_date = end, fields = 'price', frequency = 'minute').pct_change()[1:].dropna()
    data.columns = stocks
    data = data.resample('60T').mean().dropna()#resample to hourly
    mavg = data.rolling(window = 1).mean()
    return data, mavg


# In[2]:


def calculate_weights(data):
    market_ret = data.mean(axis = 1)
    weights_temp = pd.DataFrame(columns = data.columns, index = data.index)
    for j in stocks:
        #weights_temp[j] = data[j]
        weights_temp[j] = market_ret - data[j]
    return weights_temp


# In[3]:


def digitalise(weights_temp):
    weights = weights_temp*1
    for j in stocks:
        for i in range(len(weights)):
            if weights[j][i] > 0:
                weights[j][i] = 1 
            else:
                weights[j][i] = -1 #-1 for long-short o for long only
    return weights

#print(digitalise(weights_temp))


# In[4]:


def calculate_market_return(data):
    market_ret = data.mean(axis = 1)
    return market_ret

#print(market_ret.head())


# In[5]:


def calculate_strategy_returns(data, weights):
    returns = pd.DataFrame(index = data.index, columns = data.columns)
    for j in stocks:
        for i in range(1,len(data)):
            returns[j][i] = np.multiply(weights[j][i-1], data[j][i])            
    return returns


# In[6]:


def calculate_portf_returns(returns):
    pmean_returns = returns.mean(axis = 1)
    #print(pmean_returns)
    pcum_returns = np.cumprod(pmean_returns+ 1) -1
    return pcum_returns


# In[7]:


stocks = ['mosy', 'sgyp', 'opk', 'tops', 'args'
        ,'jagx', 'plug', 'cccl', 'drys', 'mark'
        ,'oncs', 'gern', 'win', 'gluu', 'nvax'
        , 'fcel', 'veon', 'odp', 'itus']#small cap? or same indutry? cant rememebr basis of selection
        

#print(len(stocks))
start = '2013-06-01'
end = '2018-06-01'

data, mavg = retrieve_data(stocks, start, end)
weights_temp = calculate_weights(mavg)
weights = digitalise(weights_temp)
market_ret = calculate_market_return(data)
returns = calculate_strategy_returns(data, weights)
pcum_returns = calculate_portf_returns(returns)


# In[ ]:


returns = returns - (returns.mean()*0.3)#transactions cost? cant remember...doesnt work-- wrong cost model
pcum_returns = calculate_portf_returns(returns)


# In[ ]:


plt.plot(np.cumprod(market_ret+1)-1,'r')
plt.plot(pcum_returns,'b')
plt.legend(['market returns', 'strategy returns'])


# In[ ]:


def calculate_sharpe_ratio(returns):
    #pmean_returns = returns
    pmean_returns = returns.mean(axis = 1)
    sr = (pmean_returns.mean() / pmean_returns.std()) * np.sqrt(2016)# change with resampling period
    return sr

print(calculate_sharpe_ratio(returns))
    


# In[ ]:


returns.mean(axis = 1).mean()*252*8


# In[ ]:


returns.mean(axis = 1).std()*16*2.8


# In[ ]:


len(returns[returns>1])/ len(returns)


# In[ ]:


len(returns[returns>1])


# In[ ]:





# In[ ]:





# In[ ]:


returns


# In[ ]:





# In[ ]:


print(weights)


# In[62]:


print(weights_temp)


# In[ ]:




