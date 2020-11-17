function [NN] = eigenappraiser(test, train)
    % Mean ctr data
    testMC = test - mean(test);
    trainMC = train - mean(train);
    
    % covariance matrix
    covar = trainMC' * trainMC;
    
    % EVD
    [V, D] = eig(covar);   % V = eigenvectors, D = eigenvalues
    
    % creating appraisalspace
    appraisalSpaceTrain = V' * trainMC';
    appraisalSpaceTest = V' * testMC';
    
    % project on to appraisalspace
    projAppraisalTrain = V * appraisalSpaceTrain;
    projAppraisalTest = V * appraisalSpaceTest;
    
    % NN finding
    nearestNeighbor = knnsearch(projAppraisalTrain', projAppraisalTest');
    NN = nearestNeighbor;
end