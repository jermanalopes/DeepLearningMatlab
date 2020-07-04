function displayWaveLabels(inputCell,convertToMaskFlag,numPoints)
%DISPLAYWAVEFORMLABELS displays the ECG signal data color coded based on
%its region label values.
%
%   inputCell is a two-element cell array containing an ECG signal vector
%   and a table of region labels if convertToMaskFlag is true, or a mask
%   vector of labels if convertToMaskFlag is false.
%
%   numPoints specifies the number of signal points to be plotted.

% Copyright 2019 The MathWorks, Inc.

% Convert the label regions to a mask of categorical labels with equal
% length as the ecgSignal
if (convertToMaskFlag)
    outCell = roi2mask(inputCell);
else
    outCell = inputCell;
end

ecgSignal = outCell{1};
ecgSignal = ecgSignal(:);
labelMask = outCell{2};

if nargin == 2
    numPoints = length(ecgSignal);
end
ecgSignal = ecgSignal(1:numPoints);
labelMask = labelMask(1:numPoints);

tECG = 1:length(ecgSignal);

% Find P regions
pRegionIndices = findRegions(labelMask,"P");

% Find QRS Regions
qrsRegionIndices = findRegions(labelMask,"QRS");

% Find T Regions
tRegionIndices = findRegions(labelMask,"T");

% Find N/A regions
naRegionIndices = findRegions(labelMask,"n/a");


% Plot labeled waveform
hAx = gca;
hold on

%Plot P wave regions
for ii = 1:length(pRegionIndices)
    iidx = pRegionIndices{ii};
    plot(tECG(iidx),ecgSignal(iidx),'Color',[0 0.447 0.741],'LineWidth',2)
end

%Plot QRS regions
for ii = 1:length(qrsRegionIndices)
    iidx = qrsRegionIndices{ii};
    plot(tECG(iidx),ecgSignal(iidx),'Color',[0.85 0.325 0.098],'LineWidth',2)
end

%Plot T wave regions
for ii = 1:length(tRegionIndices)
    iidx = tRegionIndices{ii};
    plot(tECG(iidx),ecgSignal(iidx),'Color',[0.929 0.694 0.125],'LineWidth',2)
end

%Plot N/A regions
for ii = 1:length(naRegionIndices)
    iidx = naRegionIndices{ii};
    plot(tECG(iidx),ecgSignal(iidx),'k','LineWidth',2)
end

% Setup plot options
grid on
box on
hold off
axis tight

cmap = [0 0 0;
    0.929 0.694 0.125;
    0.85 0.325 0.098;
    0 0.447 0.741];
classNames = {'N/A','T Wave','QRS','P Wave'};
colormap(hAx,cmap)

% Add colorbar to current figure.
c = colorbar(hAx);

% Use class names for tick marks.
c.TickLabels = classNames;
numClasses = size(cmap,1);

% Center tick labels.
c.Ticks = 1/(numClasses*2):1/numClasses:1;

% Remove tick mark.
c.TickLength = 0;
end

%--------------------------------------------------------------------------
function idxCell = findRegions(labelMask,labelName)
idx =  find(labelMask == labelName);
idxSep = find(diff(idx)~=1);

if ~isempty(idxSep)
    idxCell = cell(1,length(idxSep)+1);
    for ii = 1:length(idxSep)+1
        if ii == 1
            %Catch first T region
            idxCell{ii} = idx(1:idxSep(ii));
        elseif ii <= length(idxSep)
            idxCell{ii} = idx(idxSep(ii-1)+1:idxSep(ii));
        else
            %Catch last T region
            idxCell{ii} = idx(idxSep(ii-1)+1:end);
        end
    end
else
    if ~isempty(idx)
        %Special case of only a single region
        idxCell{1} = idx;
    else
        %No QRS Present
        idxCell = {};
    end
end
end