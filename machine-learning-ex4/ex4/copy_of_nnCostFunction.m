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
Theta1_grad = zeros(size(Theta1));        % 25 x 401
Theta2_grad = zeros(size(Theta2));        % 10 x 26
Delta1 = 0;
Delta2 = 0;

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



h = zeros(m,1);
X = [ones(m, 1) X];     % (5000 x 401)
X = X';                 % (5000 x 401) to (401 x 5000)
for i = 1:m
    %%    Part 1: Feedforward
    
    X_wb = X(:,i);        % (401 x 1)    wb - with bias
    z2 = Theta1*X_wb;     % (25 x (n+1))*((n+1) x 1) = (25 x 1)
    a2 = sigmoid(z2);       % (25 x 1)
    a2_wb = [1; a2];           % to (26 x 1)    wb - with bias
    z3 = Theta2*a2_wb;         % (10 x 26)*(26 x 1) = (10 x 1)
    a3 = sigmoid(z3);       % (10 x 1)
%     a3_wb = [1; a3];           % (11 x 1)
%     [~, ih] = max(a3, [], 1);
    yj = zeros(num_labels,1);
    for d = 1:num_labels
%         switch ih
%             case 1
%                 h(i) = 1;
%             case 2
%                 h(i) = 2;
%             case 3
%                 h(i) = 3;
%             case 4
%                 h(i) = 4;
%             case 5
%                 h(i) = 5;
%             case 6
%                 h(i) = 6;
%             case 7
%                 h(i) = 7;
%             case 8
%                 h(i) = 8;
%             case 9
%                 h(i) = 9;
%             case 10
%                 h(i) = 0;
%         end
        if d==y(i)
            yj(d) = 1;
        end
    end
    
    
    
    yj = zeros(10,1);
    for j = 1:num_labels
        if j==y(i)
            yj(j) = 1;
        end
    end
    
% Cost operation
    J = J - sum( yj.*log(a3) + (1-yj).*log(1-a3));
   
    %%         Part 2: Implement the backpropagation algorithm
    
%     delta3 = 0;
%     y_max_id = y(i);
    delta3 = a3 - yj;
    delta2 = Theta2(:,2:end)'*delta3.*sigmoidGradient(z2); % Formula confusion
    % Step 4 - Accumulate
% 	Delta2 += (d3 * a2');
% 	Delta1 += (d2 * a1');
    Delta2 = Delta2 + delta3*a2';
    Delta1 = Delta1 + delta2*X_wb';

    
end


    Theta2_grad = Delta2/m;
    Theta1_grad = Delta1/m;

% Cost w/o regularization
J = J/m;

Theta1_square_Vect = zeros(hidden_layer_size,1);
for j = 1:hidden_layer_size
        Theta1_square_Vect(j) = sum(Theta1(j,2:end).^2);   % Theta1(j,2:(input_layer_size+1)
end
Theta1_square = sum(Theta1_square_Vect);

Theta2_square_Vect = zeros(num_labels,1);
for j = 1:num_labels
        Theta2_square_Vect(j) = sum(Theta2(j,2:end).^2);   %Theta2(j,2:(hidden_layer_size+1))
end
Theta2_square = sum(Theta2_square_Vect);

% Cost with regularization
J = J + (Theta1_square + Theta2_square)*lambda/(2*m);





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
