function [out, n, m] = DBN_Preprocess(in)
% DBN_PREPROCESS ... 
%   DBN_PREPROCESS 
%  
%   Example 
%   DBN_Preprocess 

%   See also 
% 

%% AUTHOR    : Tushar Tank 
%% $DATE     : 30-Apr-2013 14:57:13 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 7.13.0.564 (R2011b) 
%% FILENAME  : DBN_Preprocess.m 
%% COPYRIGHT 2011 3 Phonenix Inc. 


%% this function will need to be modified for each application 

%% constants
DWN_SAMPLE = 2;
FS = 2000;
NFFT = 256;
MAX_FREQ = 300;
FFT_BIN = (FS/DWN_SAMPLE/2) / (NFFT/2+1);
LAST_BIN = ceil(MAX_FREQ/FFT_BIN);
DISP = 0;

%% spectrogram (STFT) and trunacte freq
y = decimate(in, DWN_SAMPLE);
z = spectrogram(y, 100, 50, NFFT, FS/DWN_SAMPLE);
z = z(1:LAST_BIN,:);
[n, m] = size(z);
out = z(:);

%% normalize (EACH DATA SEPERATELY!)
% this is important, depending on the cost function for each layer (ie
% sigmoid) your may not be able to recreate all inputs. The sigmoid
% requires all our values to be between [0,1] - thus we do things this way.
% Change the sigmoid, you can (and will need to) change the normalization.
out = abs(out);
out = out ./ max(out);



%% visualize
if DISP
    time = (1:size(z,2)) ./ FS/DWN_SAMPLE/2;
    freq = (1:LAST_BIN) .* FFT_BIN;
    imagesc(time, freq, abs(z));
    axis('xy')
    colorbar;
    breakpt = 1;
end