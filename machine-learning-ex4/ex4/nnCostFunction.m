function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add coloumn of 1 to X matrix

X = [ones(m,1) X];

% calculating a2 (activation of layer 2)
z1 = Theta1 * X';
a2 = sigmoid(z1);

%adding a row of 1s to a2
a2 = [ones(1,m);a2];
z2 = Theta2*a2;
h = sigmoid(z2);

% each row of new h contains classifications for each training example
h = h';

% converting the y vector to a m x 10 matrix of only 0 and 1
% very inefficient
%mod_y = zeros(m,10);
%for i=1:m,
%  for j=1:10,
%    if j==y(i),
%      mod_y(i,j) = 1;
%    end
%  end
%end

% creating a matrix of m rows and 1 to 10 columns
% correspondiong to each class
mod_y = ones(m,num_labels) .* [1:num_labels];
% using logical array to compare each element of matrix with y
% a new matrix is returned with 1 where elements were same and 0 where different
mod_y = mod_y == y;

% calculating cost without regularization
% currentElm_h will hold h vector for one training example
currentElm_h = 0;
% tempJ will act as an accumulator for each value of J of each training example
tempJ = 0;

for i=1:m,
  % assigning current example h vector (which is a row of h matrix
  currentElm_h = h(i,:);
  % assigning current example y vector (which is a row of mod_y matrix
  current_y = mod_y(i,:);
  % calculate cost for individual output class
  for j=1:num_labels,
    tempJ += (current_y(j) * log(currentElm_h(j)) + ((1-current_y(j)) * (log(1-currentElm_h(j)))));
  endfor
  
endfor

J = (-1 / m) * tempJ; 

% calculating regulatization term
tempJ = (lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2 (:))) + sum(sum(Theta2(:,2:end).^2 ))));
J += tempJ;

% backpropagation for one term in each iteration
cdelta_2 = 0;
cdelta_1 = 0;
for t=1:m,
    % step 1: calculating h for every term
    a_1 = X(t,:);
    a_1 = a_1(:); % vector with 1 input
    
    z_1 = z1'(t,:);
    z_1 = z_1(:);
    
    a_2 = a2'(t,:);
    a_2 = a_2(:);
    
    z_2 = z2'(t,:);
    z_2 = z_2(:);
   
    a_3 = h(t,:);
    
    a_3 = a_3(:);
    
    % step 2: backpropagate layer 2(output layer)
    % delta_3 vector
    delta_3 = zeros(num_labels,1);
    %for k=1:num_labels,
    
    delta_3 = a_3 - mod_y(t,:)(:);
    %endfor
    % step 3: delta_2 for hidden layer 2
    delta_2 = (Theta2' * delta_3)(2:end) .* sigmoidGradient(z_1);
    
    cdelta_2 = cdelta_2 + delta_3 * a_2';
    cdelta_1 = cdelta_1 + delta_2 * a_1';
    
    
    Theta2_grad = (1/m) * cdelta_2;
    Theta1_grad = (1/m) * cdelta_1;
    
    Theta2_grad(:,2:end) += (lambda/m) * Theta2(:,2:end);
    Theta1_grad(:,2:end) += (lambda/m) * Theta1(:,2:end);
endfor

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
