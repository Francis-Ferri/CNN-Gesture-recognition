%{
    MODEL CREATION
        1. Load datastores
        2. Define the amount of data
        3. Divide into training validation and testing
        4. Train the network
        5. Measure accuracy for datastores
        6. Calculate confusion matrices
        7. Evaluate data samples with the validation library
        8. Plot a sample (actual / predicted)
%}

%% SET DATASTORES PATHS
dataDirTraining = fullfile('Datastores', 'training');
dataDirValidation = fullfile('Datastores', 'validation');

%% THE DATASTORES RE CREATED
% The classes are defined
classes = ["fist", "noGesture", "open", "pinch", "waveIn", "waveOut"];
trainingDatastore = SpectrogramDatastore(dataDirTraining);
validationDatastore = SpectrogramDatastore(dataDirValidation);
%dataSample = preview(trainingDatastore);
clear dataDirTraining dataDirValidation

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.DataDimensions;

%% DEFINE THE AMOUNT OF DATA
% The amount of data to be used in the creation is specified ]0:1]
trainingDatastore = setDataAmount(trainingDatastore, 1);
validationDatastore = setDataAmount(validationDatastore, 1);

%% THE DATA IS DIVIDED IN TRAINING-VALIDATION-TESTING
% The training-validation-tests data is obtained
[validationDatastore, testingDatastore] = divideDatastore(validationDatastore);
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
maxEpochs = 2;%10
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
clear options

%% ANALIZE CHARACTERISTIC EXTRACTOR USING T-SNE
% Inputs: datastore, net, layer, numSamples, numPCAComponents, perplexity
tsneAnalisis(trainingDatastore, net, 'depthcat_4', 500, 576, 100);

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

%% CONFUSION MATRIX FOR EACH DATASET
calculateConfusionMatrix(net, trainingDatastore, 'training');
calculateConfusionMatrix(net, validationDatastore, 'validation');
calculateConfusionMatrix(net, testingDatastore, 'testing');

%% SAVE MODEL
save(['model_', datestr(now,'dd-mm-yyyy_HH-MM-ss')], 'net');

%% EVALUATING DATA
dataDirTrainingSeq = fullfile('Datastores', 'trainingSequence');
dataDirValidationSeq = fullfile('Datastores', 'validationSequence');
trainingSeqDatastore = SpectrogramDatastoreEval(dataDirTrainingSeq);
validationSeqDatastore = SpectrogramDatastoreEval(dataDirValidationSeq);
clear dataDirTrainingSeq dataDirValidationSeq

%% CALCULATE ACCURACIES
fprintf('\nRESULTS\n');
disp('Training results');
[clasifications, predictions, overlapings, procesingTimes] = evaluateDataStore(net, trainingSeqDatastore);
disp('Validation results');
evaluateDataStore(net, validationSeqDatastore);

%% PLOT PREDICCTION/REAL SAMPLE FROM DATASET
plotPredictionDatastore(net, trainingSeqDatastore, 30);

%% FUNCTION TO DIVIDE DATASTORE IN TWO HALVES
function [firstDatstore, secondDatastore] = divideDatastore(dataStore)
    % First datstore(50%) && second datastore(50%)
    [firstDatstore, secondDatastore] =  partition(dataStore, 0.5);
end

%% FUNCTION TO EVLUATE A DATASTORE SAMPLE USING T-SNE
function tsneAnalisis(datastore, net, layer, numSamples, numPCAComponents, perplexity)
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

%% FUNCTION TO CALCULATE ACCURACY OF A DATASTORE
function accuracy = calculateAccuracy(net, datastore)
    YPred = classify(net,datastore);
    YValidation = datastore.Labels;
    % Calculate accuracy
    accuracy = sum(YPred == YValidation)/numel(YValidation);
end

%% FUCTION TO EVALUATE ALL THE SAMPLES USING THE VALIDATION LIBRARY
function [classifications, predictions, overlapings, procesingTimes] = evaluateDataStore(net, ds)
    reset(ds)
    numObservations = ds.NumObservations;
    % Allocate space to save the results
    [classifications, predictions, overlapings, procesingTimes] = preallocateValidationResults(numObservations);
    % Index initialization
    idx = 1;
    [classifications, predictions, overlapings, procesingTimes, idx] = evaluateSequences(net, ds, ...
    classifications, predictions, overlapings, procesingTimes, idx);
    % Change the minibatch to process the remaining
    ds.MiniBatchSize = 1;
    [classifications, predictions, overlapings, procesingTimes, ~] = evaluateSequences(net, ds, ...
    classifications, predictions, overlapings, procesingTimes, idx);
    % Change NaN to 0 in the overlapping error
    overlapings(isnan(overlapings)) = 0;
    calculateValidationResults(classifications, predictions, overlapings, procesingTimes);
end

%% FUNCTION TO EVALUATE THE SAMPLES IN A SEQUENCE DATASTORE
function [classifications, predictions, overlapings, procesingTimes, idx] = evaluateSequences(net, ...
    ds, classifications, predictions, overlapings, procesingTimes, idx)
    while hasdata(ds)
        [data, info] = read(ds);
        labels = info.labels;
        groundTruths = info.groundTruths;
        timepointSequences = info.timePointSequences;
        for i = 1:length(labels)
            gestureName = labels{i};
            % Original information
            repInfo.gestureName = gestureName;
            if ~isequal(gestureName,'noGesture')
                repInfo.groundTruth = groundTruths{i};
            end
            % Allocate space for the results
            sequence = data.sequences{i};
            numFrames = length(sequence);
            % Predictions for each frame
            [vectorOfLabels, vectorOfProcessingTimes] = sequencePredictions(net, numFrames, sequence);
            % Classify prediction
            [class, vectorOfLabels] = classifyPredictions(vectorOfLabels);
            result = evaluateSample(vectorOfLabels, timepointSequences{i}, ...
                vectorOfProcessingTimes, class, repInfo);
            % Save results
            classifications(idx) = result.classResult;
            if ~isequal(gestureName,'noGesture')
                predictions(idx) = result.recogResult;
                overlapings(idx) = result.overlappingFactor;
            end
            procesingTimes(idx) = sum(vectorOfProcessingTimes); %time
            idx = idx + 1;
        end
    end
end

%% FUNCTION TO MAKE PREDICTIONS FOR A SEQUENCE OF FRAMES OF A SAMPLE
function [vectorOfLabels, vectorOfProcessingTimes] = sequencePredictions(net, numFrames, sequence)
    vectorOfLabels = cell (1, numFrames);
    vectorOfProcessingTimes = zeros(1,numFrames);
    % Predict and calculate time
    for j = 1:numFrames
        frame = sequence{j,1};
        timer = tic;
        yPred = classify(net,frame);
        time = toc(timer);
        vectorOfLabels{1, j} = char(yPred);
        vectorOfProcessingTimes(1, j) = time;
    end
end

%% FUNCTION TO EVALUATE A SAMPLE WITH THE VALIDATION LIBRARY
function result = evaluateSample(vectorOfLabels, timepointSequences, vectorOfProcessingTimes, class, repInfo)
    % Predicted information
    response.vectorOfLabels = vectorOfLabels;
    response.vectorOfTimePoints = timepointSequences;
    response.vectorOfProcessingTimes = vectorOfProcessingTimes;
    response.class = class;
    % Evaluate
    result = evalRecognition(repInfo, response);
end

%% FUCTION TO PREALLOCATE SPACE FOR VALIDATION LIBRARY RESULT
function [clasifications, predictions, overlapings, procesingTimes] = preallocateValidationResults(numObservations)
    % Allocate space to save the results
    clasifications = zeros(numObservations, 1);
    predictions = -1*ones(numObservations, 1);
    overlapings = -1*ones(numObservations, 1);
    procesingTimes = zeros(numObservations, 1);
end

%% FUNCTION TO CLASSIFY PREDICTIONS
function [class, yPred] = classifyPredictions(yPred)
    gestures = {'fist', 'noGesture', 'open', 'pinch', 'waveIn', 'waveOut'};
    yPred = categorical(yPred,gestures);
    categories = categorical(gestures);
    catCounts = countcats(yPred);
    [catCounts,indexes] = sort(catCounts,'descend');
    newCategories = categories(indexes);
    if catCounts(2) >= 4
       class = newCategories(2);
    else
       class = categories(2);
    end
end

%% FUNCTION TO CALCULATE THE RESULTS OF A SEQUENCE DATASTORE
function calculateValidationResults(clasifications, predictions, overlapings, procesingTimes)
    % Perform the calculation
    accClasification = sum(clasifications==1) / length(clasifications);
    % This is because noGesture is not predicted (-1)
    accPrediction = sum(predictions==1) / sum(predictions==1 | predictions==0);
    avgOverlapingFactor = mean(overlapings(overlapings ~= -1)); %& ~isnan(overlapings)
    avgProcesing_time = mean(procesingTimes);
    % Display the results
    fprintf('Classification accuracy: %f\n',accClasification);
    fprintf('Prediction accuracy: %f\n',accPrediction);
    fprintf('Avegage overlaping factor: %f\n',avgOverlapingFactor);
    fprintf('Avegage procesing time: %f\n',avgProcesing_time);  
end

%% FUNCTION TO PLOT A COMPARISON (REAL/PREDICTED) OF A SAMPLE FROM DATASTORE
function plotPredictionDatastore(net, datastore, idx)
    reset(datastore)
    % Validate number of sample
    idxMax = datastore.NumObservations;
    if idx<1, idx=1; elseif idx>idxMax, idx=idxMax; end
    % Recovering the sample
    batch_size = datastore.MiniBatchSize;
    count = 0;
    while idx > count
        data = read(datastore);
        count = count + batch_size;
    end
    idx = idx - (count - batch_size);
    sequence = data.sequences{idx};
    numFrames = length(sequence);
    vectorOfLabels = cell (1, numFrames);
    for i = 1:numFrames
        frame = sequence{i,1};
        yPred = classify(net,frame);
        vectorOfLabels{1, i} = char(yPred);
    end
    yReal = data.labelSequences{idx};
    plotPredictionComparison(yReal, vectorOfLabels);
    reset(datastore);
end

%% FUNCTION TO PLOT A COMPARISON (REAL/PREDICTED)
function plotPredictionComparison(YTest, YPred)
    % Make the lavels categorical
    gestures = {'fist', 'noGesture', 'open', 'pinch', 'waveIn', 'waveOut'};
    YPred = categorical(YPred,gestures);
    % Plot comparison
    figure
    plot(YPred,'.-')
    hold on
    plot(YTest)
    hold off
    xlabel("Frame")
    ylabel("Gesture")
    title("Predicted Gestures")
    legend(["Predicted" "Test Data"])
end

%% FUNCTION TO CALCULATE AD PLOT A CONFUSION MATRIX
function calculateConfusionMatrix(net, datastore, datasetName)
    % Get predictions of each frame
    predLabels = classify(net, datastore);
    realLabels = datastore.Labels;
    % Stablish clases
    clases = categorical({'fist', 'noGesture', 'open', 'pinch','waveIn', 'waveOut'});
    % Create the confusion matrix
    confusionMatrix = confusionmat(realLabels, predLabels,'Order',clases);
    figure('Name', ['Confusion Matrix - ' datasetName])
        matrixChart = confusionchart(confusionMatrix, clases);
        % Chart options
        matrixChart.ColumnSummary = 'column-normalized';
        matrixChart.RowSummary = 'row-normalized';
        matrixChart.Title = ['Hand gestures - ' datasetName];
        sortClasses(matrixChart,clases);
end

%% FUNCTION TO STABLISH THE NEURL NETWORK ARCHITECTURE
function lgraph = setNeuralNetworkArchitecture(inputSize, numClasses)
    % Create layer graph
    lgraph = layerGraph();
    % Add layer branches
    tempLayers = imageInputLayer(inputSize,"Name","data");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],32,"Name","Inception_1a-1x1")
        reluLayer("Name","Inception_1a-1x1_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1a-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],16,"Name","Inception_1a-pool_proj")
        reluLayer("Name","Inception_1a-relu-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],8,"Name","Inception_1a-5x5_reduce")
        reluLayer("Name","Inception_2a-5x5_relu_reduce_2")
        convolution2dLayer([5 5],16,"Name","Inception_1a-5x5","Padding",[2 2 2 2])
        reluLayer("Name","Inception_1a-5x5_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_1a-3x3_reduce")
        reluLayer("Name","Inception_1a-3x3_relu_reduce")
        convolution2dLayer([3 3],32,"Name","Inception_1a-3x3","Padding",[1 1 1 1])
        reluLayer("Name","Inception_1a-3x3_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1")
        maxPooling2dLayer([2 2],"Name","pool1-2x2_s2","Stride",[2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],32,"Name","Inception_2a-1x1")
        reluLayer("Name","Inception_2a-1x1_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_2a-3x3_reduce")
        reluLayer("Name","Inception_2a-3x3_relu_reduce")
        convolution2dLayer([3 3],32,"Name","Inception_2a-3x3","Padding",[1 1 1 1])
        reluLayer("Name","Inception_2a-3x3_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],8,"Name","Inception_2a-5x5_reduce")
        reluLayer("Name","Inception_2a-5x5_relu_reduce_1")
        convolution2dLayer([5 5],16,"Name","Inception_2a-5x5","Padding",[2 2 2 2])
        reluLayer("Name","Inception_2a-5x5_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_2a-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],16,"Name","Inception_2a-pool_proj")
        reluLayer("Name","Inception_2a-relu-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_2")
        maxPooling2dLayer([2 2],"Name","pool2-2x2_s2","Stride",[2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],8,"Name","Inception_3a-5x5_reduce")
        reluLayer("Name","Inception_3a-5x5_relu_reduce")
        convolution2dLayer([5 5],16,"Name","Inception_3a-5x5","Padding",[2 2 2 2])
        reluLayer("Name","Inception_3a-5x5_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_3a-3x3_reduce")
        reluLayer("Name","Inception_3a-3x3_relu_reduce")
        convolution2dLayer([3 3],32,"Name","Inception_3a-3x3","Padding",[1 1 1 1])
        reluLayer("Name","Inception_3a-3x3_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],32,"Name","Inception_3a-1x1")
        reluLayer("Name","Inception_3a-1x1_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_3a-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],16,"Name","Inception_3a-pool_proj")
        reluLayer("Name","Inception_3a-relu-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_3")
        maxPooling2dLayer([2 2],"Name","pool3-2x2_s2","Stride",[2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_4a-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],16,"Name","Inception_4a-pool_proj")
        reluLayer("Name","Inception_4a-relu-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],32,"Name","Inception_4a-1x1")
        reluLayer("Name","Inception_4a-1x1_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],8,"Name","Inception_4a-5x5_reduce")
        reluLayer("Name","Inception_4a-5x5_relu_reduce")
        convolution2dLayer([5 5],16,"Name","Inception_4a-5x5","Padding",[2 2 2 2])
        reluLayer("Name","Inception_4a-5x5_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],16,"Name","Inception_4a-3x3_reduce")
        reluLayer("Name","Inception_4a-3x3_relu_reduce")
        convolution2dLayer([3 3],32,"Name","Inception_4a-3x3","Padding",[1 1 1 1])
        reluLayer("Name","Inception_4a-3x3_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_4")
        dropoutLayer(0.5,"Name","dropout")
        fullyConnectedLayer(numClasses,"Name","fc")
        softmaxLayer("Name","softmax")
        classificationLayer("Name","classoutput")];
    lgraph = addLayers(lgraph,tempLayers);
    % Connect layer branches
    lgraph = connectLayers(lgraph,"data","Inception_1a-1x1");
    lgraph = connectLayers(lgraph,"data","Inception_1a-pool");
    lgraph = connectLayers(lgraph,"data","Inception_1a-5x5_reduce");
    lgraph = connectLayers(lgraph,"data","Inception_1a-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_1a-1x1_relu","depthcat_1/in1");
    lgraph = connectLayers(lgraph,"Inception_1a-relu-pool_proj","depthcat_1/in4");
    lgraph = connectLayers(lgraph,"Inception_1a-5x5_relu","depthcat_1/in3");
    lgraph = connectLayers(lgraph,"Inception_1a-3x3_relu","depthcat_1/in2");
    lgraph = connectLayers(lgraph,"pool1-2x2_s2","Inception_2a-1x1");
    lgraph = connectLayers(lgraph,"pool1-2x2_s2","Inception_2a-3x3_reduce");
    lgraph = connectLayers(lgraph,"pool1-2x2_s2","Inception_2a-5x5_reduce");
    lgraph = connectLayers(lgraph,"pool1-2x2_s2","Inception_2a-pool");
    lgraph = connectLayers(lgraph,"Inception_2a-1x1_relu","depthcat_2/in1");
    lgraph = connectLayers(lgraph,"Inception_2a-3x3_relu","depthcat_2/in2");
    lgraph = connectLayers(lgraph,"Inception_2a-relu-pool_proj","depthcat_2/in4");
    lgraph = connectLayers(lgraph,"Inception_2a-5x5_relu","depthcat_2/in3");
    lgraph = connectLayers(lgraph,"pool2-2x2_s2","Inception_3a-5x5_reduce");
    lgraph = connectLayers(lgraph,"pool2-2x2_s2","Inception_3a-3x3_reduce");
    lgraph = connectLayers(lgraph,"pool2-2x2_s2","Inception_3a-1x1");
    lgraph = connectLayers(lgraph,"pool2-2x2_s2","Inception_3a-pool");
    lgraph = connectLayers(lgraph,"Inception_3a-3x3_relu","depthcat_3/in2");
    lgraph = connectLayers(lgraph,"Inception_3a-1x1_relu","depthcat_3/in1");
    lgraph = connectLayers(lgraph,"Inception_3a-relu-pool_proj","depthcat_3/in4");
    lgraph = connectLayers(lgraph,"Inception_3a-5x5_relu","depthcat_3/in3");
    lgraph = connectLayers(lgraph,"pool3-2x2_s2","Inception_4a-pool");
    lgraph = connectLayers(lgraph,"pool3-2x2_s2","Inception_4a-1x1");
    lgraph = connectLayers(lgraph,"pool3-2x2_s2","Inception_4a-5x5_reduce");
    lgraph = connectLayers(lgraph,"pool3-2x2_s2","Inception_4a-3x3_reduce");
    lgraph = connectLayers(lgraph,"Inception_4a-relu-pool_proj","depthcat_4/in4");
    lgraph = connectLayers(lgraph,"Inception_4a-3x3_relu","depthcat_4/in2");
    lgraph = connectLayers(lgraph,"Inception_4a-5x5_relu","depthcat_4/in3");
    lgraph = connectLayers(lgraph,"Inception_4a-1x1_relu","depthcat_4/in1");
end
