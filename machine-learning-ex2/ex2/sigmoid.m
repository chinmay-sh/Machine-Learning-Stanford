function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% for every element of z,raise e to the power of -(element) and add 1 to it
% now inverse all elements of temporary g ,i.e, (1 + e .^ -(b))
g = (1 + e .^ -(z)) .^ -1;



% =============================================================

end
