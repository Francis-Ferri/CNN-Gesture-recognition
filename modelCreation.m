%{

%}

%% SET DATASTORES PATHS
dataDirTraining = fullfile('Datastores', 'training');
dataDirValidation = fullfile('Datastores', 'validation');
dataDirTesting = fullfile('Datastores', 'testing');

%% THE DATASTORES RE CREATED
% The classes are defined
withNoGesture = true;
classes = Shared.setNoGestureUse(withNoGesture);
trainingDatastore = SpectrogramDatastore(dataDirTraining, withNoGesture);
validationDatastore = SpectrogramDatastore(dataDirValidation, withNoGesture);
testingDatastore = SpectrogramDatastore(dataDirTesting, withNoGesture);
%dataSample = preview(trainingDatastore);
clear dataDirTraining dataDirValidation

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.DataDimensions;

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
clear numTrainingSamples numValidationSamples numTestingSamples

%% THE NEURAL NETWORK ARCHITECTURE IS DEFINED
numClasses = trainingDatastore.NumClasses;
lgraph = setNeuralNetworkArchitecture(inputSize, numClasses);
analyzeNetwork(lgraph);
% Clean up helper variable
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
clear maxEpochs miniBatchSize

%% NETWORK TRAINING
net = trainNetwork(trainingDatastore, lgraph, options);
clear options lgraph

%% ACCURACY FOR EACH DATASET
% Get training-validation-tests accuracies
accTraining = calculateAccuracy(net, trainingDatastore);
accValidation = calculateAccuracy(net, validationDatastore);
accTesting = calculateAccuracy(net, testingDatastore);
% The total data for training-validation-tests is obtained
strAccTraining = ['Training accuracy: ', num2str(accTraining)];
strAccValidation = ['Validation accuracy: ', num2str(accValidation)];
strAccTesting = ['Testing accuracy: ', num2str(accTesting)];
% The amount of training-validation-tests data is printed
fprintf('\n%s\n%s\n%s\n', strAccTraining, strAccValidation, strAccTesting);
clear accTraining accValidation accTesting strAccTraining strAccValidation strAccTesting

%% CONFUSION MATRIX FOR EACH DATASET
calculateConfusionMatrix(net, trainingDatastore, 'training', withNoGesture);
calculateConfusionMatrix(net, validationDatastore, 'validation', withNoGesture);
calculateConfusionMatrix(net, testingDatastore, 'testing', withNoGesture);

%% SAVE MODEL
if ~exist("models", 'dir')
   mkdir("models");
end
save(['models/model_', datestr(now,'dd-mm-yyyy_HH-MM-ss')], 'net');

%% FUNCTION TO STABLISH THE NEURAL NETWORK ARCHITECTURE
function lgraph = setNeuralNetworkArchitecture(inputSize, numClasses)
    % Create layer graph
    lgraph = layerGraph();
    % Add layer branches
    tempLayers = imageInputLayer(inputSize,"Name","data");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1a-3x3_reduce")
        reluLayer("Name","Inception_1a-3x3_relu_reduce")
        convolution2dLayer([3 3],18,"Name","Inception_1a-3x3","Padding",[1 1 1 1])
        reluLayer("Name","Inception_1a-3x3_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1a-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],18,"Name","Inception_1a-pool_proj")
        reluLayer("Name","Inception_1a-relu-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],18,"Name","Inception_1a-1x1")
        reluLayer("Name","Inception_1a-1x1_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1a-5x5_reduce")
        reluLayer("Name","Inception_2a-5x5_relu_reduce_2")
        convolution2dLayer([5 5],18,"Name","Inception_1a-5x5","Padding",[2 2 2 2])
        reluLayer("Name","Inception_1a-5x5_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1")
        crossChannelNormalizationLayer(5,"Name","crossnorm_4")
        dropoutLayer(0.5,"Name","dropout")
        fullyConnectedLayer(numClasses,"Name","fc_1")
        softmaxLayer("Name","softmax")
        classificationLayer("Name","classoutput")];
    lgraph = addLayers(lgraph,tempLayers);
    % clean up helper variable
    lgraph = connectLayers(lgraph,"data","Inception_1a-3x3_reduce");
    lgraph = connectLayers(lgraph,"data","Inception_1a-pool");
    lgraph = connectLayers(lgraph,"data","Inception_1a-1x1");
    lgraph = connectLayers(lgraph,"data","Inception_1a-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1a-relu-pool_proj","depthcat_1/in4");
    lgraph = connectLayers(lgraph,"Inception_1a-1x1_relu","depthcat_1/in1");
    lgraph = connectLayers(lgraph,"Inception_1a-3x3_relu","depthcat_1/in2");
    lgraph = connectLayers(lgraph,"Inception_1a-5x5_relu","depthcat_1/in3");
end

%% FUNCTION TO CALCULATE ACCURACY OF A DATASTORE
function accuracy = calculateAccuracy(net, datastore)
    YPred = classify(net,datastore);
    YValidation = datastore.Labels;
    % Calculate accuracy
    accuracy = sum(YPred == YValidation)/numel(YValidation);
end

%% FUNCTION TO CALCULATE AD PLOT A CONFUSION MATRIX
function calculateConfusionMatrix(net, datastore, datasetName, withNoGesture)
    % Get predictions of each frame
    predLabels = classify(net, datastore);
    realLabels = datastore.Labels;
    % Stablish clases
    classes = categorical(Shared.setNoGestureUse(withNoGesture));
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

%% EXTRA THINGS
%{

    %% DIVIDE DATASTORE

%% FUNCTION TO DIVIDE DATASTORE IN TWO HALVES
function [firstDatstore, secondDatastore] = divideDatastore(dataStore, percentage)
    % First datstore(percentage%) && second datastore(1 - percentage%)
    [firstDatstore, secondDatastore] =  partition(dataStore, percentage);
end


    %% TSNE

%% ANALIZE CHARACTERISTIC EXTRACTOR USING T-SNE
% Inputs: datastore, net, layer, numSamples, numPCAComponents, perplexity
tsneAnalisis(trainingDatastore, net, 'depthcat_1', 50, 50, 20); % data % [data, acts]

%% FUNCTION TO PLOT T-SNE IN 2D
function tsne2D(acts, cats, numPCAComponents, perplexity)
    newPoints = tsne(acts, 'NumPCAComponents', numPCAComponents, 'Perplexity', perplexity);
    figure
    gscatter(newPoints(:,1),newPoints(:,2),cats);
end

%% FUNCTION TO PLOT T-SNE IN 3D
function tsne3D(acts, cats, numPCAComponents, perplexity)
    newPoints3D = tsne(acts,'Algorithm','barneshut', 'NumPCAComponents', numPCAComponents, ...
        'NumDimensions',3, 'Perplexity', perplexity);
    figure;
    scatter3(newPoints3D(:,1),newPoints3D(:,2),newPoints3D(:,3),15,cats,'filled');
    view(-93,14);
end

%% FUNCTION TO PLOT T-SNE IN 2D WITH DIFERENT DISTANCES
function tsneDistancesEval(acts, cats, numPCAComponents, perplexity)
    figure;
    % Cosine
    Y = tsne(acts,'Algorithm','exact','Distance','cosine', ...
        'NumPCAComponents', numPCAComponents, 'Perplexity', perplexity);
    subplot(1,3,1);
    gscatter(Y(:,1),Y(:,2),cats);
    title('Cosine');
    % Chebychev
    Y = tsne(acts,'Algorithm','exact','Distance','chebychev', ...
        'NumPCAComponents', numPCAComponents, 'Perplexity', perplexity);
    subplot(1,3,2)
    gscatter(Y(:,1),Y(:,2),cats)
    title('Chebychev')
    % Euclidean
    Y = tsne(acts,'Algorithm','exact','Distance','euclidean', ... 
        'NumPCAComponents', numPCAComponents, 'Perplexity', perplexity);
    subplot(1,3,3)
    gscatter(Y(:,1),Y(:,2),cats)
    title('Euclidean')
end

%% FUNCTION TO EVLUATE A DATASTORE SAMPLE USING T-SNE
function [data, acts] = tsneAnalisis(datastore, net, layer, numSamples, numPCAComponents, perplexity)
    rng default % for reproducibility
    % Get samples
    originalMinibatch = datastore.MiniBatchSize;
    reset(datastore);
    datastore.MiniBatchSize = numSamples;
    data = read(datastore);
    % Get labels
    labels = cellfun(@(label) label, data.responses);
    % Get activations from layer
    acts = activations(net, data, layer);
    % Reshape the activations
    actDims = size(acts);
    acts = reshape(acts, actDims(4), prod(actDims(1:3)));
    % Make t-sne analysis
    tsne2D(acts, labels, numPCAComponents, perplexity);
    tsne3D(acts, labels, numPCAComponents, perplexity);
    tsneDistancesEval(acts, labels, numPCAComponents, perplexity);
    % Reset the dataset
    datastore.MiniBatchSize = originalMinibatch;
    reset(datastore);
end


%}

