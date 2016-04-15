%clear
load MNIST_digit_data

imagestest = images_test(1:100, :);
labelstest = labels_test(1:100, :);

Xaxis=[];
Yaxis=[];

k=10;

for index = 50:500:3000
    Xaxis(end+1) = index;
    
    imagestrain = images_train(1:index, :);
    labelstrain = labels_train(1:index, :);
    
    [avgAcc, acc] = kNN(imagestest,imagestrain,labelstest,labelstrain,k);
    Yaxis(end+1) = acc;
    index
end

plot(transpose(Xaxis),transpose(Yaxis));
hold on

xlabel('# of training data');
ylabel('accuracy');
legend('k=1','k=3','k=5','k=10');
