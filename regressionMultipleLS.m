function [yearmdlTestLS, yearmdlTrainLS, yearmdlTestLSL1, yearmdlTrainLSL1, yearmdlTestLSL2, yearmdlTrainLSL2, ETestpred3, ETrainpred3, ETestpred3L1, ETrainpred3L1, ETestpred3L2, ETrainpred3L2] = regressionMultipleLS(YearTrain, fTrain, YearTest, fTest)
%Find test error for Multiple Linear Regression without penalty
multiplemodelLS = fitrlinear(fTrain, YearTrain,'Learner','leastsquares','Solver','sgd');

yearmdlTestLS = predict(multiplemodelLS, fTest); 
yearmdlTrainLS = predict(multiplemodelLS, fTrain);

ETestpred3 = loss(multiplemodelLS, fTest, YearTest); 
ETrainpred3 = loss(multiplemodelLS, fTrain, YearTrain);

%Find test error for Multiple Linear Regression LS predictor with Lasso penalty
multiplemodelLSL1 = fitrlinear(fTrain, YearTrain,'Learner','leastsquares','Lambda','auto','Regularization','lasso','Solver','sparsa');

yearmdlTestLSL1 = predict(multiplemodelLSL1, fTest); 
yearmdlTrainLSL1 = predict(multiplemodelLSL1, fTrain);

ETestpred3L1 = loss(multiplemodelLSL1, fTest, YearTest); 
ETrainpred3L1 = loss(multiplemodelLSL1, fTrain, YearTrain);

%Find test error for Multiple Linear Regression LS predictor with Ridge penalty
multiplemodelLSL2 = fitrlinear(fTrain, YearTrain,'Learner','leastsquares','Lambda','auto', 'Regularization','ridge','Solver','sgd');

yearmdlTestLSL2 = predict(multiplemodelLSL2, fTest); 
yearmdlTrainLSL2 = predict(multiplemodelLSL2, fTrain);

ETestpred3L2 = loss(multiplemodelLSL2, fTest, YearTest); 
ETrainpred3L2 = loss(multiplemodelLSL2, fTrain, YearTrain);

end