function [yearmdlTrainLR, yearmdlTestLR, ETestpred2, ETrainpred2] = regressionBaseline(nTrain, YearTrain, fTrain, nTest, YearTest, fTest)

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
baselinemodel = fitlm(xTrain,YearTrain);
yearmdlTrainLR = predict(baselinemodel,xTrain);
yearmdlTestLR = predict(baselinemodel,xTest);

%Plot the Response
figure
plot(xTrain,YearTrain,'o',xTrain,yearmdlTrainLR,'x')
ylim([1920 2018])
legend('Original Data','Predicted values')
xlabel('timbre feature vector weighted average (x)')
ylabel('Song Release Year (y)')
title('Linear Regression Baseline Predictor of Song Release Year from Audio Features')

%Find Train and Test errors
errorTrain = zeros(nTrain,1);
errorTest = zeros(nTest,1);
for n = 1:nTrain
    errorTrain(n) = (yearmdlTrainLR(n)-YearTrain(n))^2;
end
ETrainpred2 = mean(errorTrain);
for n = 1:nTest
    errorTest(n) = (yearmdlTestLR(n)-YearTest(n))^2;
end
ETestpred2 = mean(errorTest);

end