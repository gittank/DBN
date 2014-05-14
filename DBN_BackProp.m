function DBN_BackProp(pathDir, pathBatch, pathValidate, PARAMS)
% DBN_BACKPROP ...
%   DBN_BACKPROP
%
%   Example
%   DBN_BackProp

%   See also
%

%% AUTHOR    : Tushar Tank
%% $DATE     : 02-May-2013 17:18:13 $
%% $Revision : 1.00 $
%% DEVELOPED : 7.13.0.564 (R2011b)
%% FILENAME  : DBN_BackProp.m
%% COPYRIGHT 2011 3 Phonenix Inc.

%% Constants
maxEpoch                = PARAMS.maxBackPropEpoch;
numNodes                = numel(PARAMS.nodes);
numTargetClass          = PARAMS.numTargets;
numBatches              = PARAMS.numBatches;
batchSize               = PARAMS.batchSize;
dataDimension           = PARAMS.dataLength;
numCombinedBatches      = PARAMS.numCombinedBatches;
maxIterations           = PARAMS.numberOfLineSearches;
numValidate             = PARAMS.numValidate;
combo                   = PARAMS.combo;

%% Load and initilize states
% initialize weights and biases to dummy val
[w{1:numNodes}] = deal(eye(2));
[v{1:numNodes}] = deal(eye(2));
idx = zeros(numNodes,1);

for ii = 1:numNodes
    S = load([pathDir 'state' num2str(ii)]);
    w{ii} = [S.weights; S.biasesHid];
    v{ii} = S.biasesVis;
    idx(ii) = size(w{ii},1)-1;
end
w{numNodes+1} = 0.1*randn(size(w{end},2)+1,numTargetClass);
idx(numNodes+1) = size(w{numNodes+1},1)-1;
idx(numNodes+2) = numTargetClass;

testError              = zeros(1,maxEpoch);
testErrorNormalized    = zeros(1,maxEpoch);
trainError             = zeros(1,maxEpoch);
trainErrorNormalized   = zeros(1,maxEpoch);


fprintf(1,'%d training batches of %d samples\n', numBatches, batchSize);
fprintf(1,'%d validation batches of %d samples\n', numValidate, batchSize);

for epoch = 1:maxEpoch
    
    %% training misclassification rate
    error = 0;
    counter = 0;
    
    for batch = 1:numBatches
        fprintf(1,'Epoch %d\tBatch %d\n', epoch, batch);

        S = load([pathBatch '\batch' num2str(batch)]);
        data = [S.batchData ones(batchSize,1)];
        label = S.batchLabel;
        clear S;
        
        for level = 1:numNodes
            temp = 1./(1 + exp(-data*w{level}));
            data = [temp ones(batchSize, 1)];
        end
        
        labelEst = exp(data*w{level+1});
        labelEst = labelEst./repmat(sum(labelEst,2), 1, numTargetClass);
        [~, idxEst]= max(labelEst,[],2);
        [~, idxTrue]= max(label,[],2);
        counter = counter + length(find(idxEst==idxTrue));
        error = error - sum(sum(label(:,1:end).*log(labelEst)));
    end
    trainError(epoch) = (batchSize*numBatches-counter);
    trainErrorNormalized(epoch)= error/numBatches;
    
    %% test misclassification rate
    error = 0;
    counter = 0;
    
    for batch = 1:numValidate
        S = load([pathValidate 'batch' num2str(batch)]);
        data = [S.batchData ones(batchSize,1)];
        label = S.batchLabel;
        clear S;
        
        for level = 1:numNodes
            temp = 1./(1 + exp(-data*w{level}));
            data = [temp ones(batchSize, 1)];
        end
        
        labelEst = exp(data*w{level+1});
        labelEst = labelEst./repmat(sum(labelEst,2), 1, numTargetClass);
        [~, idxEst]= max(labelEst,[],2);
        [~, idxTrue]= max(label,[],2);
        counter = counter + length(find(idxEst==idxTrue));
        error = error- sum(sum(label(:,1:end).*log(labelEst)));
    end
    testError(epoch) = (batchSize*numValidate-counter);
    testErrorNormalized(epoch)= error/numValidate;
    
    fprintf(1,'Before epoch %d\nTrain # misclassified: %d (from %d)\nTest # misclassified: %d (from %d) \t \t \n',...
            epoch,trainError(epoch),batchSize*numBatches,testError(epoch),batchSize*numValidate);

    %% gradient descent with three line searches
    
    for batch = 1:numCombinedBatches
        % make a bigger minibatch
        data = [];
        label = [];
        for count = 0:combo-1
            S = load([pathBatch '\batch' num2str(batch+count)]);
            temp = [S.batchData ones(batchSize,1)];
            data = [data; temp;];
            label = [label; S.batchLabel;];
            %label(batch:batch+comb-1) = S.label;
            clear S;
        end
        comboData = data(:,1:end-1);
        %% conjgate gradient descent with linesearches
        
        
        if epoch<6  % First update top-level weights holding other weights fixed.
            for level = 1:numNodes
                temp = 1./(1 + exp(-data*w{level}));
                data = [temp ones(batchSize*combo, 1)];
            end
            
            % remove bias
            data = data(:, 1:end-1);
            
            % vectorize final layer wieghts 
            temp = w{level+1};
            wFinal = temp(:);
            
            % CG and update weights
            dim = idx(numNodes+1:end);
            [X, ~] = minimize(wFinal,'DBN_ConjugateGradientInit',maxIterations,dim,data,label,PARAMS);
            w{end} = reshape(X,idx(end-1)+1,idx(end));
        
        else
            
            wFinal = [];
            for level = 1:numNodes+1
                temp = w{level};
                wFinal = [wFinal temp(:)'];
            end
            
            dim = idx;
            [X, ~] = minimize(wFinal,'DBN_ConjugateGradient',maxIterations,dim,comboData,label,PARAMS);
            
            delta = 0;
            for ii = 1:numNodes+1
                row = idx(ii) + 1;
                col = idx(ii+1);
                w{ii} = reshape(X(delta+1:delta+(row*col)),row,col);
                delta = row*col;
            end
        end
    end
end

