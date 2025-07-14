set S;                       # the set of stocks

param C := 1;                # capital; set 1 to find weights
param r_risk_free >= 0;      # return rate risk-free asset
param r {i in S};            # return rate risky assets
param lower_divers;
param upper_divers;

param Sigma {i in S, j in S}; # covariance matrix r = E[(r - mu)(r - mu)^T]
param risk_tolerance;

var x {i in S} >= 0;   # Portfolio weights
var x_risk_free >= 0;         # Number of bonds

var y {i in S} binary;

maximize return_on_investment: r_risk_free * x_risk_free + (sum {i in S} x[i] * r[i]) - risk_tolerance * (sum {i in S, j in S} x[i] * x[j] * Sigma[i, j]);

subject to total_capital: x_risk_free + sum {i in S} x[i] <= C;

subject to diversity_count1{i in S}: x[i] <= y[i];

subject to diversity_count2{i in S}: x[i] >=  1e-6 * y[i];

subject to diversity_lower_bound:  sum {i in S} y[i] >= lower_divers;

subject to diversity_upper_bound:  sum {i in S} y[i] <= upper_divers;