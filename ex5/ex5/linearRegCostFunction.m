function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


y_imitator = X * theta;
error = (y_imitator - y) .^ 2;
theta_reg = theta(2:end, :);
cost_reg = lambda/(2 * m) * sum((theta_reg) .^2);
J = (1/(2 * m) * sum(error)) + cost_reg;

grad = X' * (y_imitator -y)/m;
theta_grad_reg = theta;
theta_grad_reg(1,1) = 0;
grad_reg = lambda/m * theta_grad_reg;
grad += grad_reg;


% =========================================================================

grad = grad(:);

end
