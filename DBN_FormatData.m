function DBN_FormatData(dirpath, outpathTrain, outpathTest, PARAMS)
% DBN_FORMATDATA ... 
%   DBN_FORMATDATA 
%  
%   Example 
%   DBN_FormatData 

%   See also 
% 

%% AUTHOR    : Tushar Tank 
%% $DATE     : 30-Apr-2013 13:54:04 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 7.13.0.564 (R2011b) 
%% FILENAME  : DBN_FormatData.m 


%% read in data from file and save off each processed file

% We are assuming a directory strucutre such that under the dirpath
% directory we have two subdirectories: test and train. Both these
% directories will have a set of files that are the input data. The
% filename for these files will be prepended with either train or test and
% concatenated with a number. The number will be sequential from 1 to the
% number of files in that directory. For example the subdirectory train
% with have files: train1, train2, ... trainN. The subdirectory test will
% have test1, test2, ... testN. In this example (kaggle whale detection
% example) the files are aiff files. We also expect a comma seperated value
% file (csv) where one the first column is the file name and the second
% colunm is the class label. The code below will need to be
% modified for your paticular file type. The variable dirpath will have to
% be modified to where you have downloaded your data. 

% constants
% dirpath = 'C:\Users\tushar.tank\Downloads\whale_data\data\';
% outpathTrain = 'C:\Users\tushar.tank\Downloads\whale_data\data\processed\train\';
% outpathTest = 'C:\Users\tushar.tank\Downloads\whale_data\data\processed\test\';

% read labels
[labelSingle, filename] = xlsread([dirpath 'train.csv']); % read in as xls file
filename = filename(2:end,1);                       % remove header
% format labels in matrix of indicators
labelSingle = labelSingle(1:PARAMS.trainSamples);
label = zeros(numel(labelSingle), PARAMS.numTargets);
for ii = 1:PARAMS.numTargets
    label(:,ii) = labelSingle == ii-1;
end

save([outpathTrain 'label'], 'label');

% train data
for ii = 1:PARAMS.trainSamples
     [~,name,~] = fileparts(filename{ii});
     data = double(aiffread([dirpath 'train\' filename{ii}]));
     [data, ~, ~] = DBN_Preprocess(data);
     save([outpathTrain name], 'data');
end

% test data
% filename = getAllFiles([dirpath 'test']);
% 
% for ii = 1:numel(filename)
%      [~,name,~] = fileparts(filename{ii});
%      data = double(aiffread([dirpath 'test\' filename{ii}]));
%      data = DBN_Preprocess(data);
%      save([outpathTest name], 'data');
% end
% 
end

function fileList = getAllFiles(dirName)
  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; getAllFiles(nextDir)];  %# Recursively call getAllFiles
  end

end
