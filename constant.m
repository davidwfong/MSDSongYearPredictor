function [yhatpred1, ETestpred1, ETrainpred1] = constant(nTrain, YearTrain, nTest, YearTest) %10000 rows, 13 cols)

%Find Average Year from DTrain and Training Error
yhatpred1 = mean(YearTrain); 

MSEtrainpred1 = zeros(nTrain,1);
for i = 1:nTrain
    MSEtrainpred1(i) = (yhatpred1 - YearTrain(i))^2;
end
ETrainpred1 = mean(MSEtrainpred1);

% Find Test Error from DTest
MSEtestpred1 = zeros(nTest,1);
for i = 1:nTest
    MSEtestpred1(i) = (yhatpred1 - YearTest(i))^2;
end
ETestpred1 = mean(MSEtestpred1);

end