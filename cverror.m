function [Ecv1, Ecv2, Ecv3, Ecv3L1, Ecv3L2, Ecv4, Ecv4L1, Ecv4L2, Ecv5, Ecv6] = cverror(KCross, nVal, YearVal, fVal, nTrainNew, YearTrainNew, fTrainNew)

EValpred1 = zeros(KCross,1);
EValpred2 = zeros(KCross,1);
EValpred3 = zeros(KCross,1);
EValpred3L1 = zeros(KCross,1);
EValpred3L2 = zeros(KCross,1);
EValpred4 = zeros(KCross,1);
EValpred4L1 = zeros(KCross,1);
EValpred4L2 = zeros(KCross,1);
EValpred5 = zeros(KCross,1);
EValpred6 = zeros(KCross,1);

for fold = 1:KCross 
    YearValFold = YearVal{1,fold};
    YearTrainNewFold = YearTrainNew{1,fold};
    fValFold = fVal{1,fold};
    fTrainNewFold = fTrainNew{1,fold};
 
    % Constant base predictor
    [~ ,EValpred1(fold), ~] = constant(nTrainNew, YearTrainNewFold, nVal, YearValFold);
    % Linear Regression Baseline
    [~, ~, EValpred2(fold), ~] = regressionBaseline(nTrainNew, YearTrainNewFold, fTrainNewFold, nVal, YearValFold, fValFold);
    % Multiple Linear Regression Baseline with Least Squares (no penalty, L1, & L2 penalty)
    [~, ~, ~, ~, ~, ~, EValpred3(fold), ~, EValpred3L1(fold), ~, EValpred3L2(fold), ~] = regressionMultipleLS(YearTrainNewFold, fTrainNewFold, YearValFold, fValFold);
    % Multiple Linear Regression with SVM (no penalty, L1 & L2 penalty)
    [~, ~, ~, ~, EValpred4(fold), ~, EValpred4L1(fold), ~, ~, ~, EValpred4L2(fold), ~] = regressionMultipleSVM(YearTrainNewFold, fTrainNewFold, YearValFold, fValFold);
    % Linear Regression with SVM, low-dimensional data
    [~, ~, EValpred5(fold), ~] = regressionSVM(nTrainNew, YearTrainNewFold, fTrainNewFold, nVal, YearValFold, fValFold);
    % k-Nearest Neighbours Regression
    [~, ~, EValpred6(fold), ~] = regressionkNN(nTrainNew, YearTrainNewFold, fTrainNewFold, nVal, YearValFold, fValFold, 54, 5);   
end

Ecv1 = mean(EValpred1);
Ecv2 = mean(EValpred2);
Ecv3 = mean(EValpred3);
Ecv3L1 = mean(EValpred3L1);
Ecv3L2 = mean(EValpred3L2);
Ecv4 = mean(EValpred4);
Ecv4L1 = mean(EValpred4L1);
Ecv4L2 = mean(EValpred4L2);
Ecv5 = mean(EValpred5);
Ecv6 = mean(EValpred6);

end