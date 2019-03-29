X = [1 2 3 4];
y = 5;
theta = [0.1 0.2 0.3 0.4]';
[J g] = linearRegCostFunction(X, y, theta, 7);

fprintf(['Cost at theta: %f '...
         '\n(this value should be about 3.0150)\n'], J);