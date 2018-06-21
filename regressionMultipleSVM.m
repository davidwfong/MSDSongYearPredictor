function [yearmdlTestSVM, yearmdlTrainSVM, yearmdlTestSVML1, yearmdlTrainSVML1, ETestpred4, ETrainpred4, ETestpred4L1, ETrainpred4L1, yearmdlTestSVML2, yearmdlTrainSVML2, ETestpred4L2, ETrainpred4L2] = regressionMultipleSVM(YearTrain, fTrain, YearTest, fTest)
%Find test error for SVM predictor without penalty
multiplemodelSVM = fitrlinear(fTrain, YearTrain,'Learner','svm','Solver','sgd');

yearmdlTestSVM = predict(multiplemodelSVM, fTest); 
yearmdlTrainSVM = predict(multiplemodelSVM, fTrain);

ETestpred4 = loss(multiplemodelSVM, fTest, YearTest); 
ETrainpred4 = loss(multiplemodelSVM, fTrain, YearTrain);

%Find test error for SVM predictor with Lasso penalty
multiplemodelSVML1 = fitrlinear(fTrain, YearTrain,'Learner','svm','Lambda','auto','Regularization','lasso','Solver','sgd');

yearmdlTestSVML1 = predict(multiplemodelSVML1, fTest); 
yearmdlTrainSVML1 = predict(multiplemodelSVML1, fTrain);

ETestpred4L1 = loss(multiplemodelSVML1, fTest, YearTest);
ETrainpred4L1 = loss(multiplemodelSVML1, fTrain, YearTrain);

%Find test error for SVM predictor with Ridge penalty
multiplemodelSVML2 = fitrlinear(fTrain, YearTrain,'Learner','svm','Lambda','auto','Regularization','ridge','Solver','sgd');

yearmdlTestSVML2 = predict(multiplemodelSVML2, fTest); 
yearmdlTrainSVML2 = predict(multiplemodelSVML2, fTrain);

ETestpred4L2 = loss(multiplemodelSVML2, fTest, YearTest); 
ETrainpred4L2 = loss(multiplemodelSVML2, fTrain, YearTrain);

end