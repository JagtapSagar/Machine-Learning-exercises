function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    h = X*theta;         % (m x 1) = (m x 2) * (2 x 1)
%     delta0 = sum(h-y)/m;
%     x1 = X(:,2);
%     delta1 = sum((h-y).*x1)/m;
%     temp0 = theta(1) - alpha*delta0;
%     temp1 = theta(2) - alpha*delta1;
%     theta = [temp0 ; temp1];

%     temp0 = theta(1) - (alpha/m)*sum((h-y));
%     temp1 = theta(2) - (alpha/m)*sum((h-y).*x1);
%     theta = [temp0 ; temp1];    

    theta = theta - alpha * (1 / m) * (X' * (h - y)); % (2 x 1) = (2 x 1) - (2 x 97)*(97 x 1)

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
