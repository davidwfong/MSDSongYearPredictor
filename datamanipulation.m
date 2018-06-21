filename = 'YearPredictionMSD.xlsx';
sheet = 'YearPredictionMSD';
xlRange = 'A1:M515345';

MSDData = xlsread(filename,sheet,xlRange);  %Load data into matrix
[oldDataSize,attributes] = size(MSDData);   %Find size of matrix
trainSetSize = 463715;                      %TrainSetSize for full dataset
testSetSize = 51630;                        %TestSetSize for full dataset
trainSetData = MSDData(1:trainSetSize,:);   %Matrix with train data 
testSetData = MSDData(trainSetSize+1:end,:);%Matrix with test data  
newDataSize = 10000;                        %Subset
newTrainSetSize = 0.8*newDataSize;          %Chose 80:20 Train/Test split
newTestSetSize = 0.2*newDataSize;

pTrain = randperm(trainSetSize,newTrainSetSize);    %TrainSet matrix permutation
MSDTrainSet = zeros(newTrainSetSize,attributes);    %initialise 8000x13 matx
for i = 1:newTrainSetSize
    MSDTrainSet(i,:) = trainSetData(pTrain(i),:);   %fill in TrainSet matx  
end

pTest = randperm(testSetSize,newTestSetSize);       %TestSet matrix permutation
MSDTestSet = zeros(newTestSetSize,attributes);      %Initialise 2000x13 matx
for j = 1:newTestSetSize
    MSDTestSet(j,:) = testSetData(pTest(j),:);      %fill in TestSet matx
end

MSDFullSet = [MSDTrainSet; MSDTestSet];             %Concatenate Train and Test Set to form Full Data Subset

filenameTrain = 'YearPredictionMSDTrain.xlsx';      %save Train set to new file
xlswrite(filenameTrain,MSDTrainSet)
filenameTest = 'YearPredictionMSDTest.xlsx';        %save Test set to new file
xlswrite(filenameTest,MSDTestSet)
filenameFull = 'YearPredictionMSDFullSubset.xlsx';  %save Full set to new file
xlswrite(filenameFull,MSDFullSet)
