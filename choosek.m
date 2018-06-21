function [k, Errors, meanErrors] = choosek(KCross, DVal, nVal, YearVal, fVal, DTrainNew, nTrainNew, YearTrainNew, fTrainNew)
kMax = 100;
Errors = zeros(KCross,kMax);
for fold = 1:KCross 
    YearValFold = YearVal{1,fold};
    YearTrainFold = YearTrainNew{1,fold};
    for i = 1:kMax
        [IndicesNN,DistancesNN] = knnsearch(fTrainNew{1,fold}, fVal{1,fold},'K',i,'Distance','minkowski','P',5);
        YearNN = zeros(nVal, i);
        YearPred = zeros(nVal,1);
        for n = 1:nVal
            for j = 1:i                
                YearNN(n,j) = YearTrainFold(IndicesNN(n,j));
            end
            %Take average
            YearPred(n) = (1./DistancesNN(n,:))*(YearNN(n,:)')/(sum(1./DistancesNN(n,:)));
        end
        
        %Find test error 
        error = zeros(nVal,1);
        for n = 1:nVal
            error(n) = (YearPred(n)-YearValFold(n))^2;
        end
        Errors(fold,i) = mean(error);
    end
end

meanErrors = mean(Errors,1);
[minimum,k] = min(meanErrors);

end