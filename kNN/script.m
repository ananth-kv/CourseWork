clear
load MNIST_digit_data

% kNN classifier with 1000 Test Data and 10,000 training data
images_train = images_train(1:10000, :);
labels_train = labels_train(1:10000, :);
images_test = images_test(1:1000, :);
labels_test = labels_test(1:1000, :);

% Setting initial value of k as 3
k=3;

% function kNN() returns accV vector and acc scalar
[accV, acc] = kNN(images_test,images_train,labels_test,labels_train,k);