function outputCell = roi2mask(inputCell)
%ROI2MASK Convert region labels to a mask of labels of size equal to the
%size of the input ECG signal.
%
%   inputCell is a two-element cell array containing an ECG signal vector
%   and a table of region labels. 
%
%   outputCell is a two-element cell array containing the ECG signal vector
%   and a categorical label vector mask of the same length as the signal. 

% Copyright 2019 The MathWorks, Inc.

sig = inputCell{1};
labelsTable = inputCell{2};
L = length(sig);
mask = categorical(repmat("n/a",L,1),["T","P","QRS","n/a"]);

for idx = 1:height(labelsTable)
    limits = labelsTable.ROILimits(idx,:);
    mask(limits(1):limits(2)) = labelsTable.Value(idx);    
end
outputCell = {sig,mask};
end
