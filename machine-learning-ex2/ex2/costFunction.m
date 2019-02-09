function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% avg2 = sum(X(:,2))/m;
% avg3 = sum(X(:,3))/m;
% X(:,2) = (X(:,2)-avg2)/(max(X(:,2))- min(X(:,2)));
% X(:,3) = (X(:,3)-avg3)/(max(X(:,3))- min(X(:,3)));

z = X*theta;          % X*theta implies (m x 3)*(3 x 1) = (m x 1)
h = sigmoid(z);
J = (1/m)*sum(-y.*log(h) - (1-y).*log(1-h));

% grad(1) = (1/m)*sum(h'-y);
% grad(2) = (1/m)*sum((h'-y).*X(:,2));
% grad(3) = (1/m)*sum((h'-y).*X(:,3));

grad = (X'*(h-y))/m;


% =============================================================

end
