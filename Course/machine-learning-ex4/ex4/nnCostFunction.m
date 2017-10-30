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

hidden_layers = 1;
L = hidden_layers + 2;

Theta{1} = Theta1;
Theta{2} = Theta2;

delta{1} = zeros(hidden_layer_size,input_layer_size+1);
delta{2} = zeros(num_labels,hidden_layer_size+1);

J = 0;
for i = 1:m
  % Forward propagation
  a{1} = X(i,:)';
  for n = 2:L
    a{n-1} = [1;a{n-1}];
    z{n} = Theta{n-1}*a{n-1};
    a{n} = sigmoid(z{n});
  end
  % end of forward propagation
  
  yVec = [1:num_labels]' == y(i);
  
  % Back propagation
  sigma{L} = (a{L} - yVec);
  for n = L-1:2
    sigma{n} = Theta{n}'*sigma{n+1} .* [1;sigmoidGradient(z{n})];
  end
  
  for n = 1:L-1
    if n == L-1
      delta{n} = delta{n} + sigma{n+1}*a{n}';
    else
      delta{n} = delta{n} + sigma{n+1}(2:end)*a{n}';
    end    
  end
  
  costVec = yVec.*log(a{L}) + (1 - yVec).*log(1-a{L});
  J = J + sum(costVec);
end
J = -1/m * J;

% Gradient Regularization and Unrolling
grad = [];
for n = 1:L-1
  delta{n} = 1/m * delta{n} + lambda/m * [zeros(size(Theta{n},1),1) Theta{n}(:,2:end)];
  grad = [grad ; delta{n}(:)];
end


% Cost Function Regularization
for i = 1:L-1
  J = J + lambda/(2*m) * sum(sum(Theta{i}(:,2:end).^2));  
end



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
%grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
