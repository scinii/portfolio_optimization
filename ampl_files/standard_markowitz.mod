set S;                       # the set of stocks

param C := 1;                # capital; set 1 to find weights
param r {i in S};            # return rate risky assets

param Sigma {i in S, j in S}; # covariance matrix r = E[(r - mu)(r - mu)^T]
param risk_aversion;          # risk aversion

var x {i in S} >= 0;          # Portfolio weights



maximize return_on_investment: sum {i in S} x[i] * r[i];


subject to total_capital: sum {i in S} x[i] <= C;

subject to risk_bound: sum {i in S, j in S} x[i] * x[j] * Sigma[i, j] <= risk_aversion;
