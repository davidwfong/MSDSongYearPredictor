function [partition, DVal, nVal, YearVal, fVal, DTrainNew, nTrainNew, YearTrainNew, fTrainNew] = crossvalidate(DTrain, KCross)

nTrain = length(DTrain);

partition = cvpartition(nTrain, 'kfold', KCross); %generated indices for K-fold CV
disp(partition)

for i = 1:KCross         
    indicesVal = test(partition, i);
    DVal{i} = DTrain([indicesVal],:);
    DVal_i = DVal{1,i};
    YearVal{i} = DVal_i(:,1);
    fVal{i} = DVal_i(:,2:end);
    
    indicesTrain = training(partition, i);
    DTrainNew{i} = DTrain([indicesTrain],:);
    DTrainNew_i = DTrainNew{1,i};
    YearTrainNew{i} = DTrainNew_i(:,1);
    fTrainNew{i} = DTrainNew_i(:,2:end);
end

nVal = length(DVal{1,1});
nTrainNew = length(DTrainNew{1,1});

end