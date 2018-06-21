%Initialise script
close all
clear
clc
rng(9999999);
%Import the data
fileTrain = 'YearPredictionMSDTrain.xlsx';
sheetTrain = 1;
cellrangeTrain = 'A1:M8000';
fileTest = 'YearPredictionMSDTest.xlsx';
sheetTest = 1;
cellrangeTest = 'A1:M2000';
[DTrain, nTrain, YearTrain, fTrain, DTest, nTest, YearTest, fTest] = dataimport(fileTrain, sheetTrain, cellrangeTrain, fileTest, sheetTest, cellrangeTest);

% Cross validation of Data
KCross = 5;
[partition, DVal, nVal, YearVal, fVal, DTrainNew, nTrainNew, YearTrainNew, fTrainNew] = crossvalidate(DTrain, KCross);

% Constant base predictor
[yhatpred1 ,ETestpred1, ETrainpred1] = constant(nTrain, YearTrain, nTest, YearTest);

% Linear Regression Baseline
[yearmdlTestLR, yearmdlTrainLR, ETestpred2, ETrainpred2] = regressionBaseline(nTrain, YearTrain, fTrain, nTest, YearTest, fTest);

% Multiple Linear Regression Baseline with Least Squares (no penalty, L1 & L2 penalty)
[yearmdlTestLS, yearmdlTrainLS, yearmdlTestLSL1, yearmdlTrainLSL1, yearmdlTestLSL2, yearmdlTrainLSL2, ETestpred3, ETrainpred3, ETestpred3L1, ETrainpred3L1, ETestpred3L2, ETrainpred3L2] = regressionMultipleLS(YearTrain, fTrain, YearTest, fTest);

% SVM Regression (no penalty, L1 & L2 penalty)
[yearmdlTestSVM, yearmdlTrainSVM, yearmdlTestSVML1, yearmdlTrainSVML1, ETestpred4, ETrainpred4, ETestpred4L1, ETrainpred4L1, yearmdlTestSVML2, yearmdlTrainSVML2, ETestpred4L2, ETrainpred4L2] = regressionMultipleSVM(YearTrain, fTrain, YearTest, fTest);

% SVM Regression, low-dimensional data
[yearmdlTestSVMlow, yearmdlTrainSVMlow, ETestpred5, ETrainpred5] = regressionSVM(nTrain, YearTrain, fTrain, nTest, YearTest, fTest);

% k-Nearest Neighbours Regression
% Choose k using Cross-Validation
%[k, Errors, meanErrors] = choosek(KCross, DVal, nVal, YearVal, fVal, DTrainNew, nTrainNew, YearTrainNew, fTrainNew);
k = 54;
% Choose distance function norm p
%ETestpred6trial = zeros(12,1);
%for p = 1:12
%    [yearmdlTestkNN, yearmdlTrainkNN, ETestpred6trial(p), ETrainpred6] = regressionkNN(nTrain, YearTrain, fTrain, nTest, YearTest, fTest, k, p);
%end
p = 5;
[yearmdlTestkNN, yearmdlTrainkNN, ETestpred6, ETrainpred6] = regressionkNN(nTrain, YearTrain, fTrain, nTest, YearTest, fTest, k, p);

% Cross-validation Error Calculations
[Ecv1, Ecv2, Ecv3, Ecv3L1, Ecv3L2, Ecv4, Ecv4L1, Ecv4L2, Ecv5, Ecv6] = cverror(KCross, nVal, YearVal, fVal, nTrainNew, YearTrainNew, fTrainNew);

% Plot result of chosen predictor (k-NN)
figure
n = 1:1:nTest;
plot(n, yearmdlTestkNN, 'r')
hold on
plot(n, YearTest, 'b')
hold off
legend('Predicted Song Release Year','Actual Song Release Year')
title('Predicted vs Actual song release year for k-NN predictor')
xlabel('Song number')
ylabel('Year')
