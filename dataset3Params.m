function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% Dummy variables for Gaussian Kernel 
x1 = [1 2 1]; x2 = [0 4 -1]; % TODO: Figure out what these are
number = 1;
m = length(Xval);
pred = 0;

% All possible values of C and sigma
C_grid = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_grid = [0.01 0.03 0.1 0.3 1 3 10 30];

total_models = size(C_grid,2) * size(sigma_grid,2);

model_predict = zeros(total_models, 3);

% You need to return the following variables correctly.
C = 0;
sigma = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


for i = 1:size(C_grid,2)
    for j = 1:size(sigma_grid,2)
        fprintf("Number of the model being trained %d/ %d \n",number, total_models);
        
        model = svmTrain(X, y, C_grid(i),...
            @(x1, x2) gaussianKernel(x1, x2, sigma_grid(j)));
        pred = svmPredict(model,Xval);
        model_predict(number, :) = [mean(double(pred ~= yval)) i j];
        number = number + 1;
    end
end

fprintf("===Validation Done===\n");
[m,i] = min(model_predict(:,1));
C = C_grid(model_predict(i, 2));
sigma = sigma_grid(model_predict(i, 3));

% =========================================================================

end
