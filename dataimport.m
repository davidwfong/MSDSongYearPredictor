function [DTrain, nTrain, YearTrain, fTrain, DTest, nTest, YearTest, fTest] = dataimport(fileTrain, sheetTrain, cellrangeTrain, fileTest, sheetTest, cellrangeTest)

DTrain = xlsread(fileTrain,sheetTrain,cellrangeTrain);
YearTrain = DTrain(:,1);
fTrain = DTrain(:,2:size(DTrain,2));

DTest = xlsread(fileTest,sheetTest,cellrangeTest);
YearTest = DTest(:,1);
fTest = DTest(:,2:size(DTest,2));

nTrain = size(DTrain,1);                %size of training set
nTest = size(DTest,1);                  %size of test set

end



