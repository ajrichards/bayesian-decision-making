#!/usr/bin/python
"""
helper functions for supply chain case study
"""


import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import arviz as az
import matplotlib.pyplot as plt

SMALL_SIZE = 15
MEDIUM_SIZE = 16
LARGE_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title


font_name = 'sans'


def fix_layout(ax,buff=0.01):
    """
    use x and y to add well spaced margins
    """
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    xbuff = buff * (xmax - xmin)
    ybuff = buff * (ymax - ymin)
    ax.set_xlim(xmin-xbuff,xmax+xbuff)
    ax.set_ylim(ymin-ybuff,ymax+ybuff)


def plot_yield_data(data_df):
    """
    simple plot to visualize yield data
    """
    
    data_tidy = data_df.unstack().to_frame('yield')
    data_tidy.index = data_tidy.index.set_names(['supplier', 'obs'])
    g = sns.FacetGrid(data=data_tidy.reset_index().dropna(), col='supplier', height=6.0)
    g.map(sns.distplot, 'yield', kde=False, color="cyan");


def run_binomial_model(num_trials,num_successes,alpha=2,beta=2):
    """
    run the binomial model
    alpha and betap: beta prior takes two positive shape parameters 
    """
    
    # context management
    with pm.Model() as model: 
        p = pm.Beta('p', alpha=alpha, beta=beta)
        y = pm.Binomial('y', n=num_trials, p=p, observed=num_successes)

        start = pm.find_MAP()
        step = pm.Metropolis()
        trace = pm.sample(1000, step, start)

    return(trace)

    
def plot_binomial_model(trace, alpha, beta, ax=None):
    """
    plot the posterior and prior distributions
    """

    if not ax:
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111)
    ax.hist(trace['p'], 20, histtype='step', lw=2.0, density = True, label='post'); 
    x = np.linspace(0, 1, 100)
    ax.plot(x, stats.beta.pdf(x, alpha, beta), label='prior');
    ax.legend(loc='best');
    return(ax)


def calc_yield_and_price(orders, prices, supplier_yield):
    """
    function to convert the orders we place to each supplier, the yield we assume for each one, and what their prices are.    
    """
    orders = np.asarray(orders)
    
    full_yield = np.sum(supplier_yield * orders)
    price_per_item = np.sum(orders * prices) / np.sum(orders)
    
    return full_yield, price_per_item

    
def objective_fn(orders, supplier_yield, demand_samples, max_order_size):
    """
    Used to compute the total yield and effective price given a posterior predictive sample
    supplier_yield - posterior predictive samples


    """
    orders = np.asarray(orders)
    losses = []
    
    # Negative orders are impossible, indicated by np.inf
    if np.any(orders < 0):
        return np.inf
    # Ordering more than the supplier can ship is also impossible
    if np.any(orders > MAX_ORDER_SIZE):
        return np.inf
    
    # Iterate over post pred samples provided in supplier_yield
    for i, supplier_yield_sample in supplier_yield.iterrows():
        full_yield, price_per_item = calc_yield_and_price(
            orders,
            supplier_yield=supplier_yield_sample
        )
        
        # evaluate loss over each sample with one sample from the demand distribution
        loss_i = loss(full_yield, demand_samples[i], price_per_item)
        
        losses.append(loss_i)
        
    return np.asarray(losses)    


def plot_example_binomial(params,fig=None):
    """
    plot_example_binomial([(5,0.25),(5,0.5),(5,0.75)])
    """


    if not fig:
        fig = plt.figure(figsize=(13,8))
        
    subplots = len(params)
    
    # loop through parameterizations of the beta
    axes = []
    for k,_params in enumerate(params): #

        n,p = _params
        ax = fig.add_subplot(1,subplots,k+1)
    
        x = np.arange(stats.binom.ppf(0.01,n,p),stats.binom.ppf(0.99,n,p))
        ax.plot(x, stats.binom.pmf(x,n,p), 'bo', ms=8, label='pmf')
        ax.vlines(x, 0, stats.binom.pmf(x,n,p), colors='b', lw=5, alpha=0.5)
        rv = stats.binom(n,p)
    
        ax.set_title("n=%s,p=%s"%(n,p))
        ax.set_aspect(1./ax.get_data_ratio())

        ax.set_xlabel("Number of Deals Closed")
        ax.set_ylabel("Probability")
        
        fix_layout(ax)
        axes.append(ax)
        
        
if __name__ == "__main__":

    plot_example_binomial([(100,0.25)])
