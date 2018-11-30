function error = testForC_Sigma(C, sigma, X, y, Xval, yval)
%   error = testForC_Sigma(C, sigma, X, y, Xval, yval) returns an
%   error matrix with the error Er(i,j) equals to the error on 
%   validation set (Xval, yval) when the SVM model is trained with
%   C(i) and sigma(j) on the training set (X, y).


% Ensure that C and sigma are column vectors
C = C(:); sigma = sigma(:);

error = zeros(length(C),length(sigma));

% ====================== YOUR CODE HERE ======================
%

for i = 1:length(C)
  for j = 1:length(sigma)
    [model] = svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
    pred = svmPredict(model,Xval);
    error(i,j) = mean(pred ~= yval);
    printf("For C = %f, sigma = %f, error = %f\n",C(i),sigma(j),error(i,j));
  end
end

% =============================================================
    
end
