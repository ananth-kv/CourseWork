function [accV, acc] = kNN(images_test,images_train,labels_test,labels_train,k)

acc = 0;
accV = zeros(10,1);
NoOfClass = zeros(10,1);

for i=1:size(images_test,1)
testData = repelem(images_test(i,:),size(images_train,1),1);
testData = images_train - testData;
testData = testData.^2;
oneMatrix = ones(size(testData,2),1);
testData = testData*oneMatrix;
testData = sqrt(testData);

[minTestData, minTestDataIndex] = sort(testData);

minTestDataIndex = minTestDataIndex(1:k);

classV = zeros(10,1);
for j=1:k
    classV(labels_train(minTestDataIndex(j))+1)=classV(labels_train(minTestDataIndex(j))+1)+1;    
end

[count, label] = max(classV());

NoOfClass(labels_test(i)+1) = NoOfClass(labels_test(i)+1)+1;

    if  (label-1) == labels_test(i)
    acc = acc+1;
accV(labels_test(i)+1) = accV(labels_test(i)+1)+1;    
    end

end

%output
for m=1:10
    accV(m) = (accV(m)*100)/NoOfClass(m);
end

acc = sum(accV)/10;

end
