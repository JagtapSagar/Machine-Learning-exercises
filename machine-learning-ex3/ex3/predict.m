function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X];     % (5000 x 401)
% X = X';                 % (5000 x 401) to (401 x 5000)

z1 = X*Theta1';         % (5000 x 401)*(401 x 25) = (5000 x 25)
a2 = sigmoid(z1);       % (5000 x 25)
a2 = [ones(m,1), a2];   % to (5000 x 26)
z2 = a2*Theta2';        % (5000 x 26)*(26 x 10) = (5000 x 10)
a3 = sigmoid(z2);       % (5000 x 10)
[~, ih] = max(a3, [], 2);

for k = 1:m
%     X_temp = X(:,k);
%     z1 = Theta1*X_temp;     % (25 x (n+1))*((n+1) x 1) = (25 x 1)
%     a2 = sigmoid(z1);       % (25 x 1)
%     a2 = [1; a2];           % to (26 x 1)
%     z2 = Theta2*a2;         % (10 x 26)*(26 x 1) = (10 x 1)
%     a3 = sigmoid(z2);       % (10 x 1)
%     [~, ih] = max(a3, [], 1);
    for i = 1:num_labels
        switch ih(k)
%         switch ih
            case 1
                p(k) = 1;
            case 2
                p(k) = 2;
            case 3
                p(k) = 3;
            case 4
                p(k) = 4;
            case 5
                p(k) = 5;
            case 6
                p(k) = 6;
            case 7
                p(k) = 7;
            case 8
                p(k) = 8;
            case 9
                p(k) = 9;
            case 10
                p(k) = 0;
        end
    end
end
    






% =========================================================================


end
