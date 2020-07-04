function [trainedNet, info] = trainDeep(varargin)
% trainNetwork   Train a neural network
%
%   trainedNet = trainNetwork(imds, layers, options) trains and returns a
%   network trainedNet for a classification problem. imds is an
%   ImageDatastore with categorical labels, layers is an array of network
%   layers or a LayerGraph, and options is a set of training options.
%
%   trainedNet = trainNetwork(ds, layers, options) trains and returns a
%   network trainedNet using the datastore ds. For single-input networks,
%   the datastore read function must return a two-column table or
%   two-column cell array, where the first column specifies the inputs to
%   the network and the second column specifies the expected responses. For
%   networks with multiple inputs, the datastore read function must return
%   a cell array with N+1 columns, where N is the number of inputs. The
%   first N columns correspond to the N inputs and the final column
%   corresponds to the responses.
%
%   trainedNet = trainNetwork(X, Y, layers, options) trains and returns a
%   network, trainedNet. The format for X depends on the input layer. For
%   an image input layer, X is a numeric array of images arranged so that
%   the first three dimensions are the width, height and channels, and the
%   last dimension indexes the individual images. For a 3-D image input
%   layer, X is a numeric array of 3-D images with the dimensions width,
%   height, depth, channels, and the last dimension indexes the individual
%   observations. In a classification problem, Y specifies the labels for
%   the images as a categorical vector. In a regression problem, Y contains
%   the responses arranged as a matrix of size number of observations by
%   number of responses, or a four dimensional numeric array, where the
%   last dimension corresponds to the number of observations.
%
%   trainedNet = trainNetwork(sequences, Y, layers, options) trains an LSTM
%   network for classification and regression problems for sequence or
%   time-series data. layers must define a network with a sequence input
%   layer. sequences must be one of the following:
%      - A cell array of C-by-S matrices, where C is the number of features
%        and S is the number of time steps.
%      - A cell array of H-by-W-by-C-by-S arrays, where H-by-W-by-C is the
%        2-D image size and S is the number of time steps.
%      - A cell array of H-by-W-by-D-by-C-by-S arrays, where
%        H-by-W-by-D-by-C is the 3-D image size and S is the number of time
%        steps.
%   Y must be one of the following:
%      - For sequence-to-label classification, a categorical vector.
%      - For sequence-to-sequence classification, a cell array of
%        categorical sequences.
%      - For sequence-to-one regression, a matrix of targets.
%      - For sequence-to-sequence regression, a cell array of C-by-S
%        matrices.
%   For sequence-to-sequence problems, the number of time steps of the
%   sequences in Y must be identical to the corresponding predictor
%   sequences. For sequence-to-sequence problems with one observation, the
%   input sequence can be a numeric array, and Y must be a categorical
%   sequence of labels or a numeric array of responses.
%
%   trainedNet = trainNetwork(tbl, layers, options) trains and returns a
%   network, trainedNet. For networks with an image input layer, tbl is a
%   table containing predictors in the first column as a cell array of
%   image paths or images. Responses must be in the second column as
%   categorical labels for the images. In a regression problem, responses
%   must be in the second column as either vectors or cell arrays
%   containing 3-D arrays or in multiple columns as scalars. For networks
%   with a sequence input layer, tbl is a table containing a cell array of
%   MAT file paths of predictors in the first column. For a
%   sequence-to-label classification problem, the second column must be a
%   categorical vector of labels. For a sequence-to-one regression problem,
%   the second column must be a numeric array of responses or in multiple
%   columns as scalars. For a sequence-to-sequence classification problem,
%   the second column must be a cell array of MAT file paths with a
%   categorical response sequence. For a sequence-to-sequence regression
%   problem, the second column must be a cell array of MAT file paths with
%   a numeric response sequence. Support for tables and networks with a
%   sequence input layer will be removed in a future release. For
%   out-of-memory data, use a datastore instead.
%
%   trainedNet = trainNetwork(tbl, responseNames, ...) trains and returns a
%   network, trainedNet. responseNames is a character vector, a string
%   array, or a cell array of character vectors specifying the names of the
%   variables in tbl that contain the responses.
%
%   [trainedNet, info] = trainNetwork(...) trains and returns a network,
%   trainedNet. info contains information on training progress.
%
%   Example 1:
%       % Train a convolutional neural network on some synthetic images
%       % of handwritten digits. Then run the trained network on a test
%       % set, and calculate the accuracy.
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [ ...
%           imageInputLayer([28 28 1])
%           convolution2dLayer(5,20)
%           reluLayer
%           maxPooling2dLayer(2,'Stride',2)
%           fullyConnectedLayer(10)
%           softmaxLayer
%           classificationLayer];
%       options = trainingOptions('sgdm', 'Plots', 'training-progress');
%       net = trainNetwork(XTrain, YTrain, layers, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   Example 2:
%       % Train a long short-term memory network to classify speakers of a
%       % spoken vowel sounds on preprocessed speech data. Then make
%       % predictions using a test set, and calculate the accuracy.
%
%       [XTrain, YTrain] = japaneseVowelsTrainData;
%
%       layers = [ ...
%           sequenceInputLayer(12)
%           lstmLayer(100, 'OutputMode', 'last')
%           fullyConnectedLayer(9)
%           softmaxLayer
%           classificationLayer];
%       options = trainingOptions('adam', 'Plots', 'training-progress');
%       net = trainNetwork(XTrain, YTrain, layers, options);
%
%       [XTest, YTest] = japaneseVowelsTestData;
%
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   Example 3:
%       % Train a network on synthetic digit data, and measure its
%       % accuracy:
%
%       [XTrain, YTrain] = digitTrain4DArrayData;
%
%       layers = [
%           imageInputLayer([28 28 1], 'Name', 'input')
%           convolution2dLayer(5, 20, 'Name', 'conv_1')
%           reluLayer('Name', 'relu_1')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_2')
%           reluLayer('Name', 'relu_2')
%           convolution2dLayer(3, 20, 'Padding', 1, 'Name', 'conv_3')
%           reluLayer('Name', 'relu_3')
%           additionLayer(2,'Name', 'add')
%           fullyConnectedLayer(10, 'Name', 'fc')
%           softmaxLayer('Name', 'softmax')
%           classificationLayer('Name', 'classoutput')];
%
%       lgraph = layerGraph(layers);
%
%       lgraph = connectLayers(lgraph, 'relu_1', 'add/in2');
%
%       plot(lgraph);
%
%       options = trainingOptions('sgdm', 'Plots', 'training-progress');
%       [net,info] = trainNetwork(XTrain, YTrain, lgraph, options);
%
%       [XTest, YTest] = digitTest4DArrayData;
%       YPred = classify(net, XTest);
%       accuracy = sum(YTest == YPred)/numel(YTest)
%
%   See also nnet.cnn.layer, trainingOptions, SeriesNetwork, DAGNetwork, LayerGraph.

%   Copyright 2015-2019 The MathWorks, Inc.

narginchk(3,4);

try
    [layersOrGraph, opts, X, Y] = iParseInputArguments(varargin{:});
    [trainedNet, info] = doTrainNetwork(layersOrGraph, opts, X, Y);
catch e
    iThrowCNNException( e );
end

end

function [trainedNet, info] = doTrainNetwork(layersOrGraph, opts, X, Y)

% Infer parameters of layers
isaDAG = iHaveDAGNetwork(layersOrGraph);
analyzedLayers = iInferParameters(layersOrGraph);
internalLayers = analyzedLayers.InternalLayers;
networkInfo = nnet.internal.cnn.util.ComputeNetworkInfo(isaDAG,internalLayers);

% Set desired precision
precision = nnet.internal.cnn.util.Precision('single');

% Set up and validate parallel training
executionSettings = nnet.internal.cnn.assembler.setupExecutionEnvironment(...
    opts, networkInfo.IsRNN, X, precision );

% Assemble internal network
strategy = nnet.internal.cnn.assembler.NetworkAssemblerStrategyFactory...
    .createStrategy(~networkInfo.IsDAG);
assembler = nnet.internal.cnn.assembler.TrainingNetworkAssembler(strategy);
trainedNet = assembler.assemble(analyzedLayers, executionSettings);

% Optimize the network according to the input options (currently not
% supported, use default optimizer)
trainedNet = trainedNet.optimizeNetworkForTraining( ...
    nnet.internal.cnn.optimizer.DefaultNetworkOptimizer() );

% Retrieve the network input and output size for validation
networkInfo = setNetworkSize(networkInfo,trainedNet);

iValidateNoMISOAndParallel(networkInfo, opts.ExecutionEnvironment);
iValidateNoMISOAndValidationData(networkInfo, opts);

% Validate training data
if numel(networkInfo.InputSizes) == 1
    iValidateTrainingDataForProblem( X, Y, networkInfo );
else
    iValidateTrainingDataForMISOProblem( X, Y, networkInfo );
end

% Create a training dispatcher
if numel(networkInfo.InputSizes) == 1
    trainingDispatcher = iCreateTrainingDataDispatcher( X, Y, ...
        opts, executionSettings, networkInfo);
else
   trainingDispatcher = iCreateMISOTrainingDataDispatcher( ...
       X, opts, executionSettings, networkInfo, precision );
end

% Create a validation dispatcher if validation data was passed in
if numel(networkInfo.InputSizes) == 1
    validationDispatcher = iCreateValidationDispatcher( opts, executionSettings, networkInfo );
else
    validationDispatcher = iCreateMISOValidationDispatcher( ...
        X, opts, executionSettings, networkInfo, precision );
end

% Assert that training and validation data are consistent
if numel(networkInfo.InputSizes) == 1
    iValidateDataSizeForSISONetwork( trainingDispatcher, validationDispatcher, networkInfo, trainedNet );
else
    iValidateDataSizeForMISONetwork( trainingDispatcher, networkInfo, trainedNet );
end
    
% Instantiate reporters as needed
[reporters, trainingPlotReporter] = iOptionalReporters( opts, ...
    analyzedLayers, executionSettings, networkInfo, ...
    trainingDispatcher, validationDispatcher, assembler );
errorState = nnet.internal.cnn.util.ErrorState();
cleanup = onCleanup(@()iFinalizePlot(trainingPlotReporter, errorState));

% Always create the info recorder (because we will reference it later) but
% only add it to the list of reporters if actually needed.
infoRecorder = iInfoRecorder( opts, networkInfo );
if nargout >= 2
    reporters.add( infoRecorder );
end

% Create a trainer to train the network with dispatcher and options
trainer = iCreateTrainer( opts, executionSettings.precision, reporters,...
    executionSettings );

if isa(trainer, 'nnet.internal.mss.StatelessTrainer')
    % This call is needed to handle Ctrl+C events cleanly when a
    % StatelessTrainer instance is used. StatelessTrainer stops training on
    % workers via delete, which isn't called until a training window is
    % closed (when the training-plots option is enabled). Garbage
    % collection on Trainer instances is deferred because Trainer instances
    % listen for events from training plot windows, and don't get 
    % collected until training windows are closed.
    statelessCleanup = onCleanup(@()delete(trainer));
end

% Do pre-processing work required for input and output layers
trainedNet = trainer.initializeNetwork(trainedNet, trainingDispatcher);

% Do the training
trainedNet = trainer.train(trainedNet, trainingDispatcher);

% Do post-processing work (if any)
trainedNet = trainer.finalizeNetwork(trainedNet, trainingDispatcher);

trainedNet = iPrepareNetworkForOutput(trainedNet, analyzedLayers, assembler);
info = infoRecorder.Info;

% Update error state ready for the cleanup.
errorState.ErrorOccurred = false;
end

function [layers, opts, X, Y] = iParseInputArguments(varargin)
% iParseInputArguments   Parse input arguments of trainNetwork
%
% Output arguments:
%   layers  - An array of layers or a layer graph
%   opts    - An object containing training options
%   X       - Input data, this can be a data dispatcher, an image
%             datastore, a table, a numeric array or a cell array
%   Y       - Response data, this can be a numeric array or empty in case X
%             is a dispatcher, a table, an image datastore or a cell array

X = varargin{1};
if iIsADataDispatcher( X )
    % X is a custom dispatcher. The custom dispatcher api is for internal
    % use only.
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsAnImageDatastore( X )
    iAssertOnlyThreeArgumentsForIMDS( nargin );
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsADatastore(X)
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif iIsPixelLabelDatastore( X )
    Y = [];
    layers = varargin{2};
    opts = varargin{3};
elseif istable( X )
    secondArgument = varargin{2};
    if ischar(secondArgument) || iscellstr(secondArgument) || isstring(secondArgument)
        % ResponseName syntax
        narginchk(4,4);
        responseNames = convertStringsToChars(secondArgument);
        iAssertValidResponseNames(responseNames, X);
        X = iSelectResponsesFromTable( X, responseNames );
        Y = [];
        layers = varargin{3};
        opts = varargin{4};
    else
        narginchk(3,3);
        Y = [];
        layers = varargin{2};
        opts = varargin{3};
    end
elseif isnumeric( X ) || islogical( X )
    narginchk(4,4);
    Y = varargin{2};
    layers = varargin{3};
    opts = varargin{4};
elseif iscell( X )
    narginchk(4,4);
    Y = varargin{2};
    layers = varargin{3};
    opts = varargin{4};
else
    error(message('nnet_cnn:internal:cnn:util:TrainingDataErrorThrower:XIsNotValidType'));
end

% Bypass network analyzer check of layers since we do not accept networks.
if ~isa(layers, 'nnet.cnn.LayerGraph') && ~isa(layers, 'nnet.cnn.layer.Layer')
    error(message('nnet_cnn:internal:cnn:analyzer:NetworkAnalyzer:InvalidInput'))
end

% Validate options
iValidateOptions( opts );
end

function [X, Y] = iGetValidationDataFromOptions( opts )
X = opts.ValidationData;
if iIsADataDispatcher( X )
    % X is a custom dispatcher. The custom dispatcher api is for internal
    % use only.
    Y = [];
elseif iIsAnImageDatastore( X )
    Y = [];
elseif iIsADatastore( X )
    Y = [];
elseif istable( X )
    Y = [];
elseif iscell( X )
    Y = X{2};
    X = X{1};
else
    % Do nothing. Invalid type is already checked when creating
    % trainingOptions
end
end

function iValidateOptions( opts )
% iValidateOptions   Assert that opts is a valid training option object
if ~isa(opts, 'nnet.cnn.TrainingOptions')
    error(message('nnet_cnn:trainNetwork:InvalidTrainingOptions'))
end
end

function iValidateTrainingDataForProblem( X, Y, networkInfo )
% iValidateTrainingDataForProblem   Assert that the input training data X
% and response Y are valid for the class of problem considered
trainingDataValidator = iTrainingDataValidator;
trainingDataValidator.validateDataForProblem( X, Y, networkInfo );
end

function iValidateTrainingDataForMISOProblem( X, Y, networkInfo )
% iValidateTrainingDataForMISOProblem   Assert that input training data is
% valid for a multiple input network
trainingDataValidator = iTrainingDataValidator;
trainingDataValidator.validateDataForMISOProblem( X, Y, networkInfo );
end

function iValidateValidationDataForProblem( X, Y, networkInfo )
% iValidateValidationDataForProblem   Assert that the input validation data
% X and response Y are valid for the class of problem considered
validationDataValidator = iValidationDataValidator;
validationDataValidator.validateDataForProblem( X, Y, networkInfo );
end

function trainingDataValidator = iTrainingDataValidator()
trainingDataValidator = nnet.internal.cnn.util.NetworkDataValidator( ...
    nnet.internal.cnn.util.TrainingDataErrorThrower );
end

function validationDataValidator = iValidationDataValidator()
validationDataValidator = nnet.internal.cnn.util.NetworkDataValidator( ...
    nnet.internal.cnn.util.ValidationDataErrorThrower );
end

function iValidateDataSizeForSISONetwork(trainingDispatcher, validationDispatcher, networkInfo, internalNet)
% Assert that the training data is valid for the network architecture
trainingDataValidator = iTrainingDataValidator;
trainingDataValidator.validateDataSizeForSISONetwork( ...
    trainingDispatcher, networkInfo, internalNet);

% Assert that the validation data is valid for the network architecture
validationDataValidator = iValidationDataValidator;
validationDataValidator.validateDataSizeForSISONetwork( ...
    validationDispatcher, networkInfo, internalNet );

% For a classification problem, assert that responses in the training and
% validation data have the same labels
iAssertTrainingAndValidationDispatcherHaveSameClasses( ...
    trainingDispatcher, validationDispatcher);
end

function iValidateDataSizeForMISONetwork( trainingDispatcher, networkInfo, internalNet )
% Assert the training data is valid for the network architecture
trainingDataValidator = iTrainingDataValidator;
trainingDataValidator.validateDataSizeForMISONetwork( ...
    trainingDispatcher, networkInfo, internalNet);
end

function iAssertTrainingAndValidationDispatcherHaveSameClasses(trainingDispatcher, validationDispatcher)
if ~isempty(validationDispatcher)
    numResponses = numel(trainingDispatcher.ResponseMetaData);
    for i = 1:numResponses
        if iIsClassificationMetaData(trainingDispatcher.ResponseMetaData(i))
            iAssertClassNamesAreTheSame( ...
                trainingDispatcher.ResponseMetaData(i).Categories, ...
                validationDispatcher.ResponseMetaData(i).Categories);
            iAssertClassesHaveSameOrdinality( ...
                trainingDispatcher.ResponseMetaData(i).Categories, ...
                validationDispatcher.ResponseMetaData(i).Categories);
        end
    end
end
end

function tf = iIsClassificationMetaData(responseMetaData)
tf = isa(responseMetaData, 'nnet.internal.cnn.response.ClassificationMetaData');
end

function iAssertClassNamesAreTheSame(trainingCategories, validationCategories)
% iHaveSameClassNames   Assert that the class names for the training and
% validation responses are the same.
trainingClassNames = categories(trainingCategories);
validationClassNames = categories(validationCategories);
if ~isequal(trainingClassNames, validationClassNames)
    error(message('nnet_cnn:trainNetwork:TrainingAndValidationDifferentClasses'));
end
end

function iAssertClassesHaveSameOrdinality(trainingCategories, validationCategories)
if ~iHaveSameOrdinality(trainingCategories, validationCategories)
    error(message('nnet_cnn:trainNetwork:TrainingAndValidationDifferentOrdinality'));
end
end

function tf = iHaveSameOrdinality(trainingCategories, validationCategories)
% iHaveSameOrdinality   Return true if the categories from the training
% dispatcher have the same ordinality as those for the validation
% dispatcher
tf = isequal(isordinal(trainingCategories), isordinal(validationCategories));
end

function iThrowCNNException( exception )
% Wrap exception in a CNNException, which reports the error in a custom way
err = nnet.internal.cnn.util.CNNException.hBuildCustomError( exception );
throwAsCaller(err);
end

function externalNet = iPrepareNetworkForOutput(internalNet, ...
    analyzedLayers, assembler)
% If output network is on pool, retrieve it
if isa(internalNet, 'Composite')
    spmd
        [internalNet, labWithOutput] = iPrepareNetworkForOutputOnPool(internalNet);
    end
    internalNet = internalNet{labWithOutput.Value};
else
    internalNet = iPrepareNetworkForHostPrediction(internalNet);
end

% Convert to external network for user
externalNet = assembler.createExternalNetwork(internalNet, analyzedLayers);
end

function [internalNet, labWithResult] = iPrepareNetworkForOutputOnPool(internalNet)
if isempty(internalNet)
    labWithResult = gop(@min, inf);
else
    labWithResult = gop(@min, labindex);
end
if labindex == labWithResult
    % Convert to host network on pool, in case client has no GPU
    internalNet = iPrepareNetworkForHostPrediction(internalNet);
end
% Only labWithResult can be returned using AutoTransfer - network is too
% big
labWithResult = distributedutil.AutoTransfer( labWithResult, labWithResult );
end

function internalNet = iPrepareNetworkForHostPrediction(internalNet)
% Make sure any Acceleration settings do not apply to subsequent prediction
% by setting the optimization scheme back to noop
internalNet = internalNet.prepareNetworkForPrediction();
internalNet = internalNet.optimizeNetworkForPrediction( ...
    nnet.internal.cnn.optimizer.NoOpNetworkOptimizer() );
internalNet = internalNet.setupNetworkForHostPrediction();
end

function externalNetwork = iPrepareAndCreateExternalNetwork(...
    internalNetwork, analyzedLayers, assembler)
% Prepare an internal network for prediction, then create an external
% network
internalNetwork = internalNetwork.prepareNetworkForPrediction();
internalNetwork = internalNetwork.setupNetworkForHostPrediction();
externalNetwork = assembler.createExternalNetwork(internalNetwork, ...
    analyzedLayers);
end

function infoRecorder = iInfoRecorder( opts, networkInfo )
trainingInfoContent = iTrainingInfoContent( opts, networkInfo );
infoRecorder = nnet.internal.cnn.util.traininginfo.Recorder(trainingInfoContent);
end

function aContent = iTrainingInfoContent( opts, networkInfo )
isValidationSpecified = iIsValidationSpecified(opts);

if networkInfo.DoesClassification
    if isValidationSpecified
        aContent = nnet.internal.cnn.util.traininginfo.ClassificationWithValidationContent;
    else
        aContent = nnet.internal.cnn.util.traininginfo.ClassificationContent;
    end
else
    if isValidationSpecified
        aContent = nnet.internal.cnn.util.traininginfo.RegressionWithValidationContent;
    else
        aContent = nnet.internal.cnn.util.traininginfo.RegressionContent;
    end
end
end

function iAssertOnlyThreeArgumentsForIMDS( nArgIn )
if nArgIn~=3
    error(message('nnet_cnn:trainNetwork:InvalidNarginWithImageDatastore'));
end
end

function tf = iIsADataDispatcher(X)
tf = isa(X, 'nnet.internal.cnn.DataDispatcher');
end

function tf = iIsADatastore(X)
tf = isa(X,'matlab.io.Datastore') || isa(X,'matlab.io.datastore.Datastore');
end

function iValidateNoMISOAndParallel(networkInfo, executionEnvironment)
if  iIsMultiInput(networkInfo) && iIsParallel(executionEnvironment)
    error(message('nnet_cnn:trainNetwork:InvalidMultiInputExecutionEnvironment'));
end
end

function iValidateNoMISOAndValidationData(networkInfo, opts)
if  iIsMultiInput(networkInfo) && iHasValidationData(opts)
    error(message('nnet_cnn:trainNetwork:InvalidMultiInputValidationData'));
end
end

function tf = iIsMultiInput(networkInfo)
tf = numel(networkInfo.InputSizes) > 1;
end

function tf = iIsParallel(executionEnvironment)
tf = ismember( executionEnvironment, {'multi-gpu', 'parallel'} );
end

function tf = iHasValidationData(opts)
tf = ~isempty(opts.ValidationData);
end

function dispatcher = iCreateTrainingDataDispatcher(X, Y, opts, executionSettings, networkInfo)
% Create a dispatcher.
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts, 'discardLast', networkInfo, executionSettings );
end

function dispatcher = iCreateValidationDataDispatcher(X, Y, opts, trainingExecutionSettings, networkInfo)
% iCreateValidationDataDispatcher   Create a dispatcher for validation data

% Validation execution settings
executionSettings = iSetValidationExecutionSettings(trainingExecutionSettings);
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcher( ...
    X, Y, opts, 'truncateLast', networkInfo, executionSettings );
end

function executionSettings = iSetValidationExecutionSettings(trainingExecutionSettings)
% Copy training settings for use with validation
executionSettings = trainingExecutionSettings;
% If the training execution environment is parallel, prefetching cannot be
% used by the validation dispatcher
if trainingExecutionSettings.useParallel
    executionSettings.backgroundPrefetch = false;
end
% Validation dispatcher cannot be parallel
executionSettings.useParallel = false;
end

function [reporter, trainingPlotReporter] = iOptionalReporters( opts, ...
    analyzedLayers, executionSettings, networkInfo, ...
    trainingDispatcher, validationDispatcher, assembler )
% iOptionalReporters   Create a vector of Reporters based on the given
% training options and the network type
%
% See also: nnet.internal.cnn.util.VectorReporter
reporter = nnet.internal.cnn.util.VectorReporter();

isValidationSpecified = iIsValidationSpecified(opts);

if opts.Verbose
    % If verbose is true, add a progress displayer
    if networkInfo.DoesClassification
        if isValidationSpecified
            columnStrategy = nnet.internal.cnn.util.ClassificationValidationColumns;
        else
            columnStrategy = nnet.internal.cnn.util.ClassificationColumns;
        end
    else
        if isValidationSpecified
            columnStrategy = nnet.internal.cnn.util.RegressionValidationColumns;
        else
            columnStrategy = nnet.internal.cnn.util.RegressionColumns;
        end
    end
    progressDisplayerFrequency = opts.VerboseFrequency;
    if isValidationSpecified
        progressDisplayerFrequency = [progressDisplayerFrequency opts.ValidationFrequency];
    end
    progressDisplayer = nnet.internal.cnn.util.ProgressDisplayer(columnStrategy);
    progressDisplayer.Frequency = progressDisplayerFrequency;
    reporter.add( progressDisplayer );
end

if isValidationSpecified
    % Create a validation reporter
    validationPredictStrategy = iValidationPredictStrategy( validationDispatcher, ...
        networkInfo, executionSettings.precision, executionSettings, opts.Shuffle );
    validationReporter = iValidationReporter( validationDispatcher, executionSettings, opts.ValidationFrequency, opts.ValidationPatience, opts.Shuffle, validationPredictStrategy );
    reporter.add( validationReporter );
end

if ~isempty( opts.CheckpointPath )
    checkpointSaver = nnet.internal.cnn.util.CheckpointSaver( opts.CheckpointPath );
    checkpointSaver.ConvertorFcn = @(net)iPrepareAndCreateExternalNetwork(net,...
        analyzedLayers, assembler);
    reporter.add( checkpointSaver );
end

if ~isempty( opts.OutputFcn )
    userCallbackReporter = nnet.internal.cnn.util.UserCallbackReporter( opts.OutputFcn );
    reporter.add( userCallbackReporter );
end

% Training plot
config = nnet.internal.cnn.ui.TrainingPlotConfig(opts, networkInfo, trainingDispatcher, executionSettings);
if isa(opts.Plots, 'nnet.internal.cnn.ui.TrainingPlotter')
    iThrowTrainingPlotErrorInDeployedApplication();
    
    trainingPlotter = opts.Plots;
    trainingPlotter.configure(config);
    
    trainingPlotReporter = nnet.internal.cnn.util.TrainingPlotReporter(trainingPlotter);
    reporter.add( trainingPlotReporter );
    
elseif strcmp(opts.Plots, 'training-progress')
    iThrowTrainingPlotErrorInDeployedApplication();
    
    plotFactory = nnet.internal.cnn.ui.HGTrainingPlotFactory();
    trainingPlotter = nnet.internal.cnn.ui.CLITrainingPlotter(plotFactory);
    trainingPlotter.configure(config);
    
    trainingPlotReporter = nnet.internal.cnn.util.TrainingPlotReporter(trainingPlotter);
    reporter.add( trainingPlotReporter );
    
else
    trainingPlotReporter = nnet.internal.cnn.util.EmptyPlotReporter();
end

end

function iThrowTrainingPlotErrorInDeployedApplication()
if isdeployed
    error(message('nnet_cnn:internal:cnn:ui:trainingplot:TrainingPlotNotDeployable'))
end
end

function iFinalizePlot(trainingPlotReporter, errorState)
trainingPlotReporter.finalizePlot(errorState.ErrorOccurred);
end

function validationDispatcher = iCreateValidationDispatcher(opts, ...
    executionSettings, layers)
% iValidationDispatcher   Get validation data and create a dispatcher for it. Validate the
% data for the current problem and w.r.t. the current architecture.

% Return empty if no validation data was specified
if ~iIsValidationSpecified(opts)
    validationDispatcher = [];
else
    % There is no need to convert datastore into table, since validation
    % will be computed only on one worker
    [XVal, YVal] = iGetValidationDataFromOptions( opts );
    iValidateValidationDataForProblem( XVal, YVal, layers );
    % Create a validation dispatcher
    validationDispatcher = iCreateValidationDataDispatcher(XVal, YVal, ...
        opts, executionSettings, layers);
end
end

function tf = iIsValidationSpecified(opts)
tf = ~isempty(opts.ValidationData);
end

function strategy = iValidationPredictStrategy( validationDispatcher, networkInfo, precision, executionSettings, shuffle )
strategy = nnet.internal.cnn.util.ValidationPredictStrategyFactory.createStrategy(validationDispatcher, networkInfo, precision, executionSettings, shuffle);
end

function validator = iValidationReporter(validationDispatcher, executionEnvironment, frequency, patience, shuffle, validationPredictStrategy)
validator = nnet.internal.cnn.util.ValidationReporter(validationDispatcher, executionEnvironment, frequency, patience, shuffle, validationPredictStrategy);
end

function trainer = iCreateTrainer( ...
    opts, precision, reporters, executionSettings )
if executionSettings.useParallel
    summaryFcn = @(dispatcher,maxEpochs)nnet.internal.cnn.util. ...
        ParallelMiniBatchSummary(dispatcher,maxEpochs);
    trainer = nnet.internal.cnn.ParallelTrainer( ...
        opts, precision, reporters, executionSettings, summaryFcn);
elseif executionSettings.useStateless
    summaryFcn = @(dispatcher,maxEpochs)nnet.internal.cnn.util. ...
        MiniBatchSummary(dispatcher,maxEpochs);
    trainer = nnet.internal.mss.StatelessTrainer( ...
        opts, precision, reporters, executionSettings, summaryFcn);
else
    summaryFcn = @(dispatcher,maxEpochs)nnet.internal.cnn.util. ...
        MiniBatchSummary(dispatcher,maxEpochs);
    trainer = nnet.internal.cnn.Trainer( ...
        opts, precision, reporters, executionSettings, summaryFcn);
end
end

function tf = iIsAnImageDatastore(x)
tf = isa(x, 'matlab.io.datastore.ImageDatastore');
end

function iAssertValidResponseNames(responseNames, tbl)
% iAssertValidResponseNames   Assert that the response names are variables
% of the table and they do not refer to the first column.
variableNames = tbl.Properties.VariableNames;
refersToFirstColumn = ismember( variableNames(1), responseNames );
responseNamesAreAllVariables = all( ismember(responseNames,variableNames) );
if refersToFirstColumn || ~responseNamesAreAllVariables
    error(message('nnet_cnn:trainNetwork:InvalidResponseNames'))
end
end

function resTbl = iSelectResponsesFromTable(tbl, responseNames)
% iSelectResponsesFromTable   Return a new table with only the first column
% (predictors) and the variables specified in responseNames.
variableNames = tbl.Properties.VariableNames;
varTF = ismember(variableNames, responseNames);
% Make sure to select predictors (first column) as well
varTF(1) = 1;
resTbl = tbl(:,varTF);
end

function tf = iIsPixelLabelDatastore(x)
tf = isa(x, 'matlab.io.datastore.PixelLabelDatastore');
end

function haveDAGNetwork = iHaveDAGNetwork(lgraph)
haveDAGNetwork = isa(lgraph,'nnet.cnn.LayerGraph');
end

function analysis = iInferParameters(layersOrGraph)
[~, analysis] = nnet.internal.cnn.layer.util.inferParameters(layersOrGraph);
end

function dispatcher = iCreateMISOTrainingDataDispatcher( ...
    X, options, executionSettings, netInfo, precision)
mapping = iConstructMapping(netInfo);
collateFcns = nnet.internal.cnn.util.multipleInputSingleOutputCollateFcns( ...
    numel(netInfo.InputSizes) );
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcherMIMO( ...
    X, mapping, [], options.MiniBatchSize, 'discardLast', ...
    precision, executionSettings, options.Shuffle, collateFcns);
end

function dispatcher = iCreateMISOValidationDispatcher( ...
    X, options, trainingExecutionSettings, netInfo, precision)
mapping = iConstructMapping(netInfo);
executionSettings = iSetValidationExecutionSettings(trainingExecutionSettings);
dispatcher = nnet.internal.cnn.DataDispatcherFactory.createDataDispatcherMIMO( ...
    X, mapping, [], options.MiniBatchSize, 'truncateLast', ...
    precision, executionSettings, options.Shuffle, []);
end

function mapping = iConstructMapping(netInfo)
numInputLayers = numel(netInfo.InputSizes);
mapping = cell(1,5);
mapping{1} = num2cell(1:numInputLayers);
mapping{2} = {numInputLayers+1};
mapping{3} = netInfo.DoesClassification;
mapping{4} = netInfo.InputSizes;
mapping{5} = netInfo.OutputSizes;
end