# - Define the objects of interest

set S;                       # the set of stocks

param C := 1;                # capital; set 1 to find weights
param r_risk_free >= 0;       # return rate risk free assed
param r {i in S};            # return rate risky assets (stochastic)
param lower_divers;
param upper_divers;

param Sigma {i in S, j in S};# covariance matrix r = E[(r - mu)(r - mu)^T] 
param risk_aversion;         # risk aversion 

var x {i in S} >= 0;           # Portfolio weights
var x_risk_free >= 0;           # Number of bonds

# - The optimization problem

maximize return_on_investment: r_risk_free * x_risk_free + sum {i in S} x[i] * r[i];

s.t. total_capital: x_risk_free + sum {i in S} x[i] <= C;

s.t. risk_bound: sum {i in S, j in S} x[i] * x[j] * Sigma[i, j] <= risk_aversion;

s.t. diversity_lower_bound:  sum {i in S} (if x[i] > 0 then 1 else 0) >= lower_divers;
s.t. diversity_upper_bound:  sum {i in S} (if x[i] > 0 then 1 else 0) <= upper_divers;