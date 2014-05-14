function [batchData, batchLabel] = DBN_MakeBatches(fileName, totalNum, offset, pathBatch, pathData, PARAMS)
% DBN_MAKEBATCHES ... 
%   DBN_MAKEBATCHES 
%  
%   Example 
%   DBN_MakeBatches 

%   See also 
% 

%% AUTHOR    : Tushar Tank 
%% $DATE     : 29-Apr-2013 12:05:03 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 7.13.0.564 (R2011b) 
%% FILENAME  : DBN_MakeBatches.m 
%% COPYRIGHT 2011 3 Phonenix Inc. 
%% constants
batchSize = PARAMS.batchSize;
numDimension  =  PARAMS.dataLength;
numBatches = PARAMS.numBatches;
batchData = zeros(batchSize, numDimension);
batchLabel = zeros(batchSize,PARAMS.numTargets);

%% random order
randomOrder = reshape(randperm(totalNum)+offset, [numBatches batchSize]);
S = load([pathData 'label']);
label = S.label;

%% create each batch and save it off
for ii = 1:numBatches
    for jj = 1:batchSize
        S = load([pathData fileName num2str(randomOrder(ii,jj))]);
        batchData(jj, :) = S.data;
        batchLabel(jj, :) = label(randomOrder(ii,jj), :);
    end
    save([pathBatch '\batch' num2str(ii)], 'batchData', 'batchLabel');
end
