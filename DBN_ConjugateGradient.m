function [f, df] = DBN_ConjugateGradient(VV,idx,XX,target,PARAMS)
% DBN_CONJUGATEGRADIENT ...
%   DBN_CONJUGATEGRADIENT
%
%   Example
%   DBN_ConjugateGradient

%   See also
%

%% AUTHOR    : Tushar Tank
%% $DATE     : 17-May-2013 13:47:57 $
%% $Revision : 1.00 $
%% DEVELOPED : 7.13.0.564 (R2011b)
%% FILENAME  : DBN_ConjugateGradient.m
%% COPYRIGHT 2011 3 Phonenix Inc.

numNodes = numel(PARAMS.nodes);
delta = 0;
for ii = 1:numNodes+1
    row = idx(ii)+1;
    col = idx(ii+1);
    w{ii} = reshape(VV(delta+1:delta+row*col), row, col);
    delta = row * col;
end

N = size(XX,1);
wProb{1} = [XX ones(N,1)];
for ii = 1:numNodes
    temp = 1./(1 + exp(-wProb{ii}*w{ii}));
    wProb{ii+1} = [temp ones(N,1)];
end

targetout = exp(wProb{end}*w{end});
targetout = targetout./repmat(sum(targetout,2),1,PARAMS.numTargets);
f = -sum(sum( target(:,1:end).*log(targetout))) ;

IO = (targetout-target(:,1:end));
Ix{numNodes+1} = IO; 
dw{numNodes+1} =  wProb{end}'*IO; 

% remove bias from XX (wProb{1})
temp = wProb{1};
wProb{1} = temp(:,end-1);

for ii = numNodes:-1:1
    temp = (Ix{ii+1} * w{ii+1}') .* wProb{ii+1} .* (1-wProb{ii+1});
    Ix{ii} = temp(:,1:end-1);
    dw{ii} = wProb{ii}' * Ix{ii};
end

df = [];
for ii = 1:numNodes+1
    temp = dw{ii};
    df = [df temp(:)'];
end
df = df';

% Ix3 = (Ix_class*w{end}').*wProb{end}.*(1-wProb{end});
% Ix3 = Ix3(:,1:end-1);
% dw3 =  wProb{end-1}'*Ix3;
% 
% Ix2 = (Ix3*w{end-1}').*wProb{end-1}.*(1-wProb{end-1}); 
% Ix2 = Ix2(:,1:end-1);
% dw2 =  wProb{end-2}'*Ix2;
% 
% Ix1 = (Ix2*w{end-2}').*wProb{end-2}.*(1-wProb{end-2}); 
% Ix1 = Ix1(:,1:end-1);
% dw1 =  XX'*Ix1;
% 
% df = [dw1(:)' dw2(:)' dw3(:)' dw_class(:)']'; 


