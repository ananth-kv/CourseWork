function [List_acc_PCA] = SVM_PCA(train,test)

dimensions = [2 5 10 20 30 50 70 100 150 200 250 300 400 500 748];
List_acc_PCA = zeros(1,size(dimensions,2));

for i=1:size(dimensions,2)
    dimension = dimensions(i);
    [pca_train.x, V] = PCA(train.x,dimension);
    
    %[pca_test.x, ~] = PCA(test.x,dimension);
    mu = mean(test.x);
    N = size(test.x,1);
    pca_test.x = test.x - ones(N,1)*mu;
    pca_test.x = pca_test.x*V;
    
    pca_train.y = train.y;
    pca_test.y = test.y;
    
    [accuracy] = SVM(pca_train,pca_test);
    List_acc_PCA(i) = accuracy;
    
end
plot(dimensions,List_acc_PCA,'r--');
end