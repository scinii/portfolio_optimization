set S;

param C := 1;

param r {i in S};
param Sigma {i in S, j in S};
param phi;
param alpha;                  # what is a "bad" return

var x {i in S} >= 0;

# - The optimization problem

maximize return_on_investment: sum {i in S} x[i] * r[i];

s.t. total_capital: sum {i in S} x[i] = C;
s.t. risk_bound: phi * (sum{i in S, j in S} x[i] * Sigma[i, j] * x[j]) <= sum{i in S} r[i] * x[i] - alpha;