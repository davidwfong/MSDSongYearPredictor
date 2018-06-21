function [YearPredTest, YearPredTrain, ETestpred6, ETrainpred6] = regressionkNN(nTrain, YearTrain, fTrain, nTest, YearTest, fTest, k, p)

%Find k Nearest Neighbours for test set
[IndicesNNTest,DistancesNNTest] = knnsearch(fTrain, fTest,'K',k,'Distance','minkowski','P',p);
YearNNTest = zeros(nTest,k);
YearPredTest = zeros(nTest,1);

for n = 1:nTest
    for i = 1:k
        YearNNTest(n,i) = YearTrain(IndicesNNTest(n,i));
    end
    %Take average
    YearPredTest(n) = (1./DistancesNNTest(n,:))*(YearNNTest(n,:)')/(sum(1./DistancesNNTest(n,:)));
end

%Find test error 
errorTest = zeros(nTest,1);
for n = 1:nTest
    errorTest(n) = (YearPredTest(n)-YearTest(n))^2;
end
ETestpred6 = mean(errorTest);

%Find k Nearest Neighbours for training set
[IndicesNNTrain,DistancesNNTrain] = knnsearch(fTrain, fTrain,'K',k,'Distance','minkowski','P',p);
YearNNTrain = zeros(nTrain,k);
YearPredTrain = zeros(nTrain,1);

for n = 1:nTrain
    for i = 1:k
        YearNNTrain(n,i) = YearTrain(IndicesNNTrain(n,i));
    end
    %Take average
    YearPredTrain(n) = (1./DistancesNNTrain(n,:))*(YearNNTrain(n,:)')/(sum(1./DistancesNNTrain(n,:)));
end

%Find training error 
errorTrain = zeros(nTrain,1);
for n = 1:nTrain
    errorTrain(n) = (YearPredTrain(n)-YearTrain(n))^2;
end
ETrainpred6 = mean(errorTrain);

end