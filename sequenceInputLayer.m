function layer = sequenceInputLayer(varargin)
% sequenceInputLayer   Sequence input layer
%
%   layer = sequenceInputLayer(inputSize) defines a sequence input layer.
%   inputSize is the size of the input sequence at each time step,
%   specified as a positive integer or vector of positive integers.
%       - For vector sequence input, inputSize is a scalar corresponding to
%       the number of features.
%       - For 2-D image sequence input, inputSize is a vector of three
%       elements [H W C], where H is the image height, W is the image
%       width, and C is the number of channels of the image.
%       - For 3-D image sequence input, inputSize is a vector of four
%       elements [H W D C], where H is the image height, W is the image
%       width, D is the image depth, and C is the number of channels of the
%       image.
%
%   layer = sequenceInputLayer(inputSize, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%   specifies optional parameter name/value pairs for creating the layer:
%
%    'Normalization'     Data normalization applied when data is forward
%                        propagated through the input layer. Valid values
%                        are:
%                           'zerocenter'        - zero-center normalization
%                           'zscore'            - z-score normalization
%                           'rescale-symmetric' - rescale to [-1 1]
%                           'rescale-zero-one'  - rescale to [0 1]
%                           'none'              - do not normalize
%                           function handle     - custom normalization
%
%                        Default: 'none'
%
%    'NormalizationDimension'
%                        Dimension over which the same normalization is
%                        applied, specified as one of the following:
%                          - 'auto'    - If the ResetInputNormalization
%                                        training option is false and you
%                                        specify normalization statistics,
%                                        then normalize over dimensions
%                                        matching the statistics.
%                                        Otherwise, recompute the
%                                        statistics and apply channel-wise
%                                        normalization.
%                          - 'channel' - Channel-wise normalization
%                          - 'element' - Element-wise normalization
%                          - 'all'     - Normalize all values using
%                                        scalar statistics
%
%                        Default: 'auto'
%
%    'Mean'              The mean used for zero centering and z-score
%                        normalization. This can be [], a scalar, or an
%                        array of size inputSize. If inputSize is a [H W C]
%                        row vector or a [H W D C] row vector, then the
%                        mean can also be a 1-by-1-by-C or a
%                        1-by-1-by-1-by-C array, respectively.
%
%                        Default: []
%
%    'StandardDeviation' The standard deviation used for z-score
%                        normalization. This can be [], a scalar, or an
%                        array of size inputSize. If inputSize is a [H W C]
%                        row vector or a [H W D C] row vector, then the
%                        standard deviation can also be a 1-by-1-by-C or a
%                        1-by-1-by-1-by-C array, respectively.
%
%                        Default: []
%
%    'Min'               The minimum used for rescaling.
%                        This can be [], a scalar, or an array of size
%                        inputSize. If inputSize is a [H W C] row vector or
%                        a [H W D C] row vector, then the minimum can also
%                        be a 1-by-1-by-C or a 1-by-1-by-1-by-C array,
%                        espectively.
%
%                        Default: []
%
%    'Max'               The maxmimum used for rescaling.
%                        This can be [], a scalar, or an array of size
%                        inputSize. If inputSize is a [H W C] row vector or
%                        a [H W D C] row vector, then the maximum can also
%                        be a 1-by-1-by-C or a 1-by-1-by-1-by-C array,
%                        espectively.
%
%                        Default: []
%
%    'Name'              A name for the layer.
%
%                        Default: ''
%
%   Example:
%       % Create a sequence input layer for multi-dimensional time series
%       % with 5 dimensions per time step.
%
%       layer = sequenceInputLayer(5);
%
%   See also nnet.cnn.layer.SequenceInputLayer
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2017-2019 The MathWorks, Inc.

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

normalization = iCreateTransforms(...
    inputArguments.Normalization, inputArguments.InputSize);

% Create an internal representation of a sequence input layer.
internalLayer = nnet.internal.cnn.layer.SequenceInput( ...
    inputArguments.Name, ...
    inputArguments.InputSize, ...
    normalization );

% Assign statistics
internalLayer.Mean = inputArguments.Mean;
internalLayer.Std = inputArguments.StandardDeviation;
internalLayer.Min = inputArguments.Min;
internalLayer.Max = inputArguments.Max;
internalLayer.NormalizationDimension = inputArguments.NormalizationDimension;

% Pass the internal layer to a function to construct a user visible
% sequence input layer.
layer = nnet.cnn.layer.SequenceInputLayer(internalLayer);
end

function inputArguments = iParseInputArguments(varargin)
varargin = nnet.internal.cnn.layer.util.gatherParametersToCPU(varargin);
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser.Results);
end

function p = iCreateParser(varargin)
p = inputParser;

defaultName = '';
defaultTransform = 'none';
defaultDimension = 'auto';

addRequired(p, 'InputSize', @iAssertValidInputSize);
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
addParameter(p, 'Normalization', defaultTransform, @(x)~isempty(iCheckAndReturnValidNormalization(x)));
addParameter(p, 'NormalizationDimension', defaultDimension);
addParameter(p, 'Mean', []);
addParameter(p, 'StandardDeviation', []);
addParameter(p, 'Min', []);
addParameter(p, 'Max', []);
end

function iAssertValidInputSize(sz)
validateattributes(sz,{'numeric'},{'nonempty','real','finite','integer','positive','row'})
isValidSize = (isscalar(sz) || iIsRowVectorOfThreeOrFour(sz));
if ~isValidSize
    error(message('nnet_cnn:layer:SequenceInputLayer:InvalidInputSize'));
end
end

function x = iCheckAndReturnValidNormalization(x)
validateattributes(x,{'string','char','function_handle'},{})
if isa(x,'function_handle')
    % Checks are performed at train/inference time when normalization is
    % applied
else
    validTransforms = {'zerocenter', 'zscore', 'rescale-symmetric', ...
        'rescale-zero-one', 'none'};
    x = validatestring(x, validTransforms);
end
end

function tf = iIsRowVectorOfThreeOrFour(x)
tf = isrow(x) && (numel(x) == 3 || numel(x) == 4);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function args = iConvertToCanonicalForm(params)
try    
    args = struct;
    % Make sure integral values are converted to double
    args.InputSize = double(params.InputSize);
    args.Normalization = iCheckAndReturnValidNormalization(params.Normalization);
    args.NormalizationDimension = iCheckAndReturnValidDimension( ...
        params.NormalizationDimension, args.Normalization );
    args.Mean = iCheckAndReturnSingleStatistics( params.Mean, 'Mean', ...
        args.Normalization, args.NormalizationDimension, args.InputSize);
    args.StandardDeviation = iCheckAndReturnSingleStatistics( params.StandardDeviation, 'StandardDeviation', ...
        args.Normalization, args.NormalizationDimension, args.InputSize);
    args.Min = iCheckAndReturnSingleStatistics( params.Min, 'Min', ...
        args.Normalization, args.NormalizationDimension, args.InputSize);
    args.Max = iCheckAndReturnSingleStatistics( params.Max, 'Max', ...
        args.Normalization, args.NormalizationDimension, args.InputSize);
    % Make sure strings get converted to char vectors
    args.Name = convertStringsToChars(params.Name); 
catch e
    % Reduce the stack trace of the error message by throwing as caller
    throwAsCaller(e)
end
end

function statsValue = iCheckAndReturnSingleStatistics( userValue, name, ...
    normalization, normalizationDimension, inputSize)
% Validate user-provided statistics with respect to normalization method 
% and input size. Return the statistic in single precision.
if ~isempty(userValue)
    nnet.internal.cnn.layer.paramvalidation.validateNormalizationStatistics(...
        userValue, name, normalization, normalizationDimension, inputSize);
end
statsValue = single(userValue);
end

function transform = iCreateTransforms(type, dataSize)
transform = nnet.internal.cnn.layer.InputTransformFactory.create(type, dataSize);
end

function dimValue = iCheckAndReturnValidDimension(dimValue,normalization)
dimValue = nnet.internal.cnn.layer.paramvalidation.validateNormalizationDimension(dimValue,normalization);
end