clear
load MNIST_digit_data

imagestrain = images_train(1:8000, :);
labelstrain = labels_train(1:8000, :);

imagesValidate = images_train(8001:10000, :);
labelsValidate = labels_train(8001:10000, :);


Xaxis=[1,3,5,10];
Yaxis=[];

k=1;    
[avgAcc1, acc1] = kNN(imagesValidate,imagestrain,labelsValidate,labelstrain,k);
Yaxis(end+1) = acc1;
k=3;    
[avgAcc3, acc3] = kNN(imagesValidate,imagestrain,labelsValidate,labelstrain,k);
Yaxis(end+1) = acc3;
k=5;    
[avgAcc5, acc5] = kNN(imagesValidate,imagestrain,labelsValidate,labelstrain,k);
Yaxis(end+1) = acc5;
k=10;    
[avgAcc10, acc10] = kNN(imagesValidate,imagestrain,labelsValidate,labelstrain,k);

Yaxis(end+1) = acc10;
    
plot(transpose(Xaxis),transpose(Yaxis));
hold on

xlabel('k (# of nearest neighbors)');
ylabel('accuracy');
%legend('k=1','k=3','k=5','k=10');
