function [weights, biasesVis, biasesHid, errsum] = DBN_RBM(pathBatch1, pathBatch2, numNodes1, numNodes2, restart, PARAMS)
% DBN_RBM ...
%   DBN_RBM
%
%   Example
%   DBN_RBM

%   See also
%

%% AUTHOR    : Tushar Tank
%% $DATE     : 02-May-2013 13:30:36 $
%% $Revision : 1.00 $
%% DEVELOPED : 7.13.0.564 (R2011b)
%% FILENAME  : DBN_RBM.m
%% COPYRIGHT 2011 3 Phonenix Inc.

%% constants
maxEpoch                = PARAMS.maxEpoch;
learningRateW           = PARAMS.learningRateW;
learningRateBiasVis     = PARAMS.learningRateBiasVis;
learningRateBiasHid     = PARAMS.learningRateBiasHid;
weightCost              = PARAMS.weightCost;
initialMomentum         = PARAMS.initialMomentum;
finalMomentum           = PARAMS.finalMomentum;
numBatches              = PARAMS.numBatches;
batchSize               = PARAMS.batchSize;
epochToChangeMomentum   = PARAMS.epochToChangeMomentum;

%% update variables 
if restart == 1
    epoch=1;

    % Initializing symmetric weights and biases.
    weights     = 0.1*randn(numNodes1, numNodes2);
    biasesHid  = zeros(1,numNodes2);
    biasesVis  = zeros(1,numNodes1);
    deltaWeights  = zeros(numNodes1,numNodes2);
    deltaBiasesHid = zeros(1,numNodes2);
    deltaBiasesVis = zeros(1,numNodes1);
end

%% train RBM weights
for epoch = epoch:maxEpoch
    %fprintf(1,'epoch %d\r',epoch);
    errsum=0;
    
    for batch = 1:numBatches
        %fprintf(1,'epoch %d batch %d\r',epoch,batch);
        
        %% START POSITIVE PHASE
        S = load([pathBatch1 'batch' num2str(batch)]);
        data = S.batchData;
        posHidProbs = 1./(1 + exp(-data*weights - repmat(biasesHid,batchSize,1)));
        batchData = posHidProbs; % for next leve input
        posProds    = data' * posHidProbs;
        poshidact   = sum(posHidProbs);
        posvisact = sum(data);
        
        poshidstates = posHidProbs > rand(batchSize,numNodes2);
        
        %% START NEGATIVE PHASE
        negdata = 1./(1 + exp(-poshidstates*weights' - repmat(biasesVis,batchSize,1)));
        negHidProbs = 1./(1 + exp(-negdata*weights - repmat(biasesHid,batchSize,1)));
        negProds  = negdata'*negHidProbs;
        neghidact = sum(negHidProbs);
        negvisact = sum(negdata);
        
        %% Udate running error and momentum
        err = sum(sum( (data-negdata).^2 ));
        errsum = err + errsum;
        
        if epoch > epochToChangeMomentum
            momentum=finalMomentum;
        else
            momentum=initialMomentum;
        end;
        
        %%  UPDATE WEIGHTS AND BIASES 
        deltaWeights = momentum*deltaWeights + ...
            learningRateW*( (posProds-negProds)/batchSize - weightCost*weights);
        deltaBiasesVis = momentum*deltaBiasesVis + (learningRateBiasVis/batchSize)*(posvisact-negvisact);
        deltaBiasesHid = momentum*deltaBiasesHid + (learningRateBiasHid/batchSize)*(poshidact-neghidact);
        
        weights = weights + deltaWeights;
        biasesVis = biasesVis + deltaBiasesVis;
        biasesHid = biasesHid + deltaBiasesHid;
        
        % save layer 2 batch data as input to next layer
        if epoch == maxEpoch
            save([pathBatch2 'batch' num2str(batch)], 'batchData');
        end
    end
    fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
end

