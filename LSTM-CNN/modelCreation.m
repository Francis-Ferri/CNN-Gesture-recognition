%{
    LSTM
%}

%% SET DATASTORES PATHS
dataDirTraining = fullfile('DatastoresLSTM', 'training');
dataDirValidation = fullfile('DatastoresLSTM', 'validation');
dataDirTesting = fullfile('DatastoresLSTM', 'testing');

%% THE DATASTORES RE CREATED
% The classes are defined
withNoGesture = true;
classes = Shared.setNoGestureUse(withNoGesture);

% The datastores are created
trainingDatastore = SpectrogramDatastoreLSTM(dataDirTraining);
validationDatastore = SpectrogramDatastoreLSTM(dataDirValidation);
testingDatastore = SpectrogramDatastoreLSTM(dataDirTesting);
%dataSample = preview(trainingDatastore);
% Clean up variables
clear dataDirTraining dataDirValidation withNoGesture

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.FrameDimensions;

%% DEFINE THE AMOUNT OF DATA
% The amount of data to be used in the creation is specified ]0:1]
trainingDatastore = setDataAmount(trainingDatastore, 1);
validationDatastore = setDataAmount(validationDatastore, 1);
testingDatastore = setDataAmount(testingDatastore, 1);

%% THE DATA IS DIVIDED IN TRAINING-VALIDATION-TESTING
% The total data for training-validation-tests is obtained
numTrainingSamples = ['Training samples: ', num2str(trainingDatastore.NumObservations)];
numValidationSamples = ['Validation samples: ', num2str(validationDatastore.NumObservations)];
numTestingSamples = ['Testing samples: ', num2str(testingDatastore.NumObservations)];
% The amount of training-validation-tests data is printed
fprintf('\n%s\n%s\n%s\n', numTrainingSamples, numValidationSamples, numTestingSamples);
% Clean up variables
clear numTrainingSamples numValidationSamples numTestingSamples

%% THE NEURAL NETWORK ARCHITECTURE IS DEFINED
numClasses = trainingDatastore.NumClasses;
lgraph = setNeuralNetworkArchitecture(inputSize, numClasses);
analyzeNetwork(lgraph);
% Clean up variables
clear numClasses

%% THE OPTIONS ARE DIFINED
%gpuDevice(1);
maxEpochs = 1;%10
miniBatchSize = 32;%1024
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',3, ... %8
    'ExecutionEnvironment','cpu', ... %gpu
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'ValidationData', validationDatastore, ...
    'ValidationFrequency',floor(trainingDatastore.NumObservations/ miniBatchSize), ...
    'ValidationPatience',5, ...
    'Plots','training-progress');
% Clean up variables
clear maxEpochs miniBatchSize

%% NETWORK TRAINING
net = trainNetwork(trainingDatastore, lgraph, options);
% Clean up variables
clear options lgraph

%% ACCURACY FOR EACH DATASET
% The accuracy for training-validation-tests is obtained
[accTraining, flattenLabelsTraining ] = calculateAccuracy(net, trainingDatastore);
[accValidation, flattenLabelsValidation ] = calculateAccuracy(net, validationDatastore);
[accTesting, flattenLabelsTesting ] = calculateAccuracy(net, testingDatastore);

% The amount of training-validation-tests data is printed
strAccTraining = ['Training accuracy: ', num2str(accTraining)];
strAccValidation = ['Validation accuracy: ', num2str(accValidation)];
strAccTesting = ['Testing accuracy: ', num2str(accTesting)];
fprintf('\n%s\n%s\n%s\n', strAccTraining, strAccValidation, strAccTesting);

% Clean up variables
clear accTraining accValidation accTesting strAccTraining strAccValidation strAccTesting

%% CONFUSION MATRIX FOR EACH DATASET
calculateConfusionMatrix(flattenLabelsTraining, 'training');
calculateConfusionMatrix(flattenLabelsValidation, 'validation');
calculateConfusionMatrix(flattenLabelsTesting, 'testing');

%% SAVE MODEL
if ~exist("ModelsLSTM", 'dir')
   mkdir("ModelsLSTM");
end
save(['ModelsLSTM/model_', datestr(now,'dd-mm-yyyy_HH-MM-ss')], 'net');

%% FUNCTION TO STABLISH THE NEURAL NETWORK ARCHITECTURE
function lgraph = setNeuralNetworkArchitecture(inputSize, numClasses)
    % Create layer graph
    lgraph = layerGraph();
    
    % Add layer branches
    tempLayers = [
    sequenceInputLayer(inputSize,"Name","sequence")
    sequenceFoldingLayer("Name","seqfold")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1a-3x3_reduce")
        reluLayer("Name","Inception_1a-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1a-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1a-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1a-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1a-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1a-5x5_reduce")
        reluLayer("Name","Inception_1a-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1a-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1a")
        reluLayer("Name","Inception_1a_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1b-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1b-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1b-3x3_reduce")
        reluLayer("Name","Inception_1b-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1b-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1b-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1b-5x5_reduce")
        reluLayer("Name","Inception_1b-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1b-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1b")
        reluLayer("Name","Inception_1b")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1c-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1c-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1c-3x3_reduce")
        reluLayer("Name","Inception_1c-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1c-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1c-5x5_reduce")
        reluLayer("Name","Inception_1c-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1c-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1c-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = depthConcatenationLayer(4,"Name","depthcat_1c");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_1ac")
        reluLayer("Name","Inception_1c")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1d-5x5_reduce")
        reluLayer("Name","Inception_1d-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1d-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1d-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1d-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1d-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1d-3x3_reduce")
        reluLayer("Name","Inception_1d-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1d-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1d")
        reluLayer("Name","Inception_1d")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1e-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1e-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],18,"Name","Inception_1e-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1e-5x5_reduce")
        reluLayer("Name","Inception_1e-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1e-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1e-3x3_reduce")
        reluLayer("Name","Inception_1e-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1e-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = depthConcatenationLayer(4,"Name","depthcat_1e");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_1ce")
        reluLayer("Name","Inception_1e")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1f-3x3_reduce")
        reluLayer("Name","Inception_1f-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1f-3x3","Padding",[1 1 1 1])
        reluLayer("Name","Inception_1f-3x3_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1f-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1f-pool_proj")
        reluLayer("Name","Inception_1f-relu-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],18,"Name","Inception_1f-1x1")
        reluLayer("Name","Inception_1f-1x1_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1f-5x5_reduce")
        reluLayer("Name","Inception_1f-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1f-5x5","Padding",[2 2 2 2])
        reluLayer("Name","Inception_1f-5x5_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1f")
        crossChannelNormalizationLayer(5,"Name","crossnorm_1")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        sequenceUnfoldingLayer("Name","sequnfold")
        flattenLayer("Name","flatten")
        lstmLayer(128,'OutputMode','sequence',"Name","lstm")
        fullyConnectedLayer(numClasses,"Name","fc_1")
        softmaxLayer("Name","softmax")
        classificationLayer("Name","classoutput")];
    lgraph = addLayers(lgraph,tempLayers);

    % Connect Layer Branches
    lgraph = connectLayers(lgraph,"seqfold/out","Inception_1a-3x3_reduce");
    lgraph = connectLayers(lgraph,"seqfold/out","Inception_1a-1x1");
    lgraph = connectLayers(lgraph,"seqfold/out","Inception_1a-pool");
    lgraph = connectLayers(lgraph,"seqfold/out","Inception_1a-5x5_reduce");
    lgraph = connectLayers(lgraph,"seqfold/miniBatchSize","sequnfold/miniBatchSize");
    lgraph = connectLayers(lgraph,"Inception_1a-1x1","depthcat_1a/in4");
    lgraph = connectLayers(lgraph,"Inception_1a-5x5","depthcat_1a/in2");
    lgraph = connectLayers(lgraph,"Inception_1a-pool_proj","depthcat_1a/in3");
    lgraph = connectLayers(lgraph,"Inception_1a-3x3","depthcat_1a/in1");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-pool");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-1x1");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","Inception_1b-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1a_relu","addition_1ac/in1");
    lgraph = connectLayers(lgraph,"Inception_1b-pool_proj","depthcat_1b/in4");
    lgraph = connectLayers(lgraph,"Inception_1b-3x3","depthcat_1b/in3");
    lgraph = connectLayers(lgraph,"Inception_1b-1x1","depthcat_1b/in2");
    lgraph = connectLayers(lgraph,"Inception_1b-5x5","depthcat_1b/in1");
    lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-pool");
    lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1b","Inception_1c-1x1");
    lgraph = connectLayers(lgraph,"Inception_1c-5x5","depthcat_1c/in3");
    lgraph = connectLayers(lgraph,"Inception_1c-3x3","depthcat_1c/in2");
    lgraph = connectLayers(lgraph,"Inception_1c-pool_proj","depthcat_1c/in4");
    lgraph = connectLayers(lgraph,"Inception_1c-1x1","depthcat_1c/in1");
    lgraph = connectLayers(lgraph,"depthcat_1c","addition_1ac/in2");
    lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-pool");
    lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-1x1");
    lgraph = connectLayers(lgraph,"Inception_1c","Inception_1d-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1c","addition_1ce/in2");
    lgraph = connectLayers(lgraph,"Inception_1d-pool_proj","depthcat_1d/in4");
    lgraph = connectLayers(lgraph,"Inception_1d-1x1","depthcat_1d/in1");
    lgraph = connectLayers(lgraph,"Inception_1d-5x5","depthcat_1d/in3");
    lgraph = connectLayers(lgraph,"Inception_1d-3x3","depthcat_1d/in2");
    lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-pool");
    lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-1x1");
    lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1d","Inception_1e-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1e-1x1","depthcat_1e/in1");
    lgraph = connectLayers(lgraph,"Inception_1e-pool_proj","depthcat_1e/in4");
    lgraph = connectLayers(lgraph,"Inception_1e-5x5","depthcat_1e/in3");
    lgraph = connectLayers(lgraph,"Inception_1e-3x3","depthcat_1e/in2");
    lgraph = connectLayers(lgraph,"depthcat_1e","addition_1ce/in1");
    lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-pool");
    lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-1x1");
    lgraph = connectLayers(lgraph,"Inception_1e","Inception_1f-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1f-1x1_relu","depthcat_1f/in1");
    lgraph = connectLayers(lgraph,"Inception_1f-relu-pool_proj","depthcat_1f/in4");
    lgraph = connectLayers(lgraph,"Inception_1f-3x3_relu","depthcat_1f/in2");
    lgraph = connectLayers(lgraph,"Inception_1f-5x5_relu","depthcat_1f/in3");
    lgraph = connectLayers(lgraph,"crossnorm_1","sequnfold/in");
end

%% FUNCTION TO CALCULATE ACCURACY OF A DATASTORE
function [accuracy, flattenLabels] = calculateAccuracy(net, datastore)
    % Configure options
    realVsPredData = cell(datastore.NumObservations, 2);
    datastore.MiniBatchSize = 1; % No padding

    % Read while the datastore has data
    totalLabels = 0;
    while hasdata(datastore)
        % Get sample
        position = datastore.CurrentFileIndex;
        data = read(datastore);
        labels = data.labelsSequences;
        sequence = data.sequences;
        % Cassify sample
        labelsPred = classify(net,sequence);
        % Save labels to flatten later and calculate accuracy
        realVsPredData(position, :) = [labels, labelsPred];
        totalLabels = totalLabels + length(labels{1,1});
    end
    
    % Allocate space for flatten labels
    flattenLabels = cell(totalLabels,2);
    idx = 0;
    % Flat labels
    for i = 1:length(realVsPredData)
        labels = realVsPredData{i, 1};
        labelsPred = realVsPredData{i, 2};
        for j = 1:length(labels)
            flattenLabels{idx+j, 1} = char(labels(1,j));
            flattenLabels{idx+j, 2} = char(labelsPred(1, j));
        end
        idx = idx + length(labels);
    end
    
    % Count labels that match with its prediction
    matches = 0;
    for i = 1:length(flattenLabels)
        if isequal(flattenLabels{i, 1}, flattenLabels{i, 2})
            matches = matches + 1;
        end
    end

    % Calculate accuracy
    accuracy = matches / length(flattenLabels);
    reset(datastore);
end

%% FUNCTION TO CALCULATE AD PLOT A CONFUSION MATRIX
function calculateConfusionMatrix(flattenLabels, datasetName)
    % Stablish clases
    classes = categorical(Shared.setNoGestureUse(true));

    % Transform labels into categorical
    realLabels = categorical(flattenLabels(:,1), Shared.setNoGestureUse(true));
    predLabels = categorical(flattenLabels(:,2), Shared.setNoGestureUse(true));
    
    % Create the confusion matrix
    confusionMatrix = confusionmat(realLabels, predLabels, 'Order', classes);
    figure('Name', ['Confusion Matrix - ' datasetName])
        matrixChart = confusionchart(confusionMatrix, classes);
        % Chart options
        matrixChart.ColumnSummary = 'column-normalized';
        matrixChart.RowSummary = 'row-normalized';
        matrixChart.Title = ['Hand gestures - ' datasetName];
        sortClasses(matrixChart,classes);
end
