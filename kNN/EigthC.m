%8(c)
clear
load MNIST_digit_data

imagestest = images_test(1:1000, :);
labelstest = labels_test(1:1000, :);

Xaxis=[];
Yaxis=[];
k=1;

for index = 30:500:10000
    Xaxis(end+1) = index;
    
    imagestrain = images_train(1:index, :);
    labelstrain = labels_train(1:index, :);
    
    [avgAcc, acc] = kNN(imagestest,imagestrain,labelstest,labelstrain,k);
    Yaxis(end+1) = acc;
    index
end

plot(transpose(Xaxis),transpose(Yaxis))
hold on
xlabel('# of Training Data')
ylabel('Average Accuracy')