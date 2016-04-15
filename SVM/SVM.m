function [accuracy, MSE] = SVM(train,test)

%TRAINING
model = libsvmtrain(train.y,train.x,'-c 100');
%PREDICTION
[y_hat] = libsvmpredict(test.y, test.x, model);

%ACCURACY
accuracy = sum(y_hat == test.y)/numel(test.y);

%MEAN SQUARED ERROR
MSE = mean((y_hat == test.y).^2);
end