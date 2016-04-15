clear
load MNIST_digit_data
whos

%%% randomly permute data points
rand('seed', 1); %%just to make all random sequences on all computers the same.
inds = randperm(size(images_train, 1));
images_train = images_train(inds, :);
labels_train = labels_train(inds, :);

inds = randperm(size(images_test, 1));
images_test = images_test(inds, :);
labels_test = labels_test(inds, :);


% %%% if you want to use only the first 1000 data points.
images_train = images_train(1:5000, :);
labels_train = labels_train(1:5000, :);

images_test = images_test(1:2000, :);
labels_test = labels_test(1:2000, :);

% %%% show the 10'th train image
% i = 10;
% close all
% im = reshape(images_train(1, :), [28 28]);
% imshow(im)
% title(num2str(labels_train(i)));

train.x = images_train;
train.y = labels_train;
test.x = images_test;
test.y = labels_test;

%SVM without PCA
[accuracy, MSE] = SVM(train,test);

%Question 4 Visualizing Eigen Vectors
[~, V] = PCA(train.x,49);
for i=1:10
im = reshape(V(:,i),[28 28]);
subplot(5,5,i);
imagesc(im);
hold on;
end

%Question 4: Reconstruction Error of PCA
dimen = [1 5 50 500 784];
List_MSE = zeros(size(dimen,2),1);
for i=1:size(dimen,2)
    d = dimen(i);
    [Embed, V] = PCA(train.x,d);
    Embed = Embed*transpose(V);
    List_MSE(i) = mean(sum((train.x-Embed).^2));    
end
hold on;
plot(dimen, List_MSE,'g--');
hold off;

%SVM with PCA
[accuracy_PCA] = SVM_PCA(train,test);
plot(dimensions,accuracy_PCA,'r--');
