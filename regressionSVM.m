function [yearmdlTrainSVMlow, yearmdlTestSVMlow, ETestpred5, ETrainpred5] = regressionSVM(nTrain, YearTrain, fTrain, nTest, YearTest, fTest)

%Create single input variable x for train and test sets
denom = 0;
for i = 1:size(fTrain,2)
    denom = denom + i;
end
c = zeros(size(fTrain,2),1);
for i = 1:size(fTrain,2)
    c(i)=(size(fTrain,2)-(i-1))/denom;
end
xTrain = zeros(nTrain,1);
for n = 1:nTrain
    xTrain(n) =  (c')*(fTrain(n,:)');
end
xTest = zeros(nTest,1);
for n = 1:nTest
    xTest(n) =  (c')*(fTest(n,:)');
end

%Create Regression model and use to predict response for DTest
baselinemodel = fitrsvm(xTrain,YearTrain,'KernelFunction','gaussian');
yearmdlTrainSVMlow = predict(baselinemodel,xTrain);
yearmdlTestSVMlow = predict(baselinemodel,xTest);

%Find Train and Test errors
errorTrain = zeros(nTrain,1);
errorTest = zeros(nTest,1);
for n = 1:nTrain
    errorTrain(n) = (yearmdlTrainSVMlow(n)-YearTrain(n))^2;
end
ETrainpred5 = mean(errorTrain);
for n = 1:nTest
    errorTest(n) = (yearmdlTestSVMlow(n)-YearTest(n))^2;
end
ETestpred5 = mean(errorTest);

end