%{
    MODEL CREATION
%}

%%
dataDirTraining = fullfile('Datastores', 'trainingDatastore');
dataDirValidation = fullfile('Datastores', 'validationDatastore');

%% THE DATASTORES RE CREATED
% The classes are defined
classes = ["fist","noGesture","open","pinch","waveIn","waveOut"];
trainingDatastore = SpectrogramDatastore(dataDirTraining);
validationDatastore = SpectrogramDatastore(dataDirValidation);
%dataSample = preview(training_datastore);
clear dataDirTraining dataDirValidation

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.SequenceDimension;

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
numHiddenUnits = 200;
filterSize = 2;
numFilters = 20;
numClasses = length(classes);
layers = [ ...
    sequenceInputLayer(inputSize,'Name','input')
    
    sequenceFoldingLayer('Name','fold')
    
    convolution2dLayer(filterSize,numFilters,'Name','conv') 
    batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    maxPooling2dLayer([2 2],"Name","maxpool")
    
    dropoutLayer(0.2,"Name","drop")
    
    convolution2dLayer(filterSize,2*numFilters,'Name','conv_1') 
    batchNormalizationLayer('Name','bn_1')
    reluLayer('Name','relu_1')
    
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm')
    
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
clear inputSize numHiddenUnits filterSize numFilters numClasses

%% LINK FOLD AND UNFOLD LAYERS
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');
clear layers

%% THE OPTIONS ARE DIFINED
maxEpochs = 1;
miniBatchSize = 8;
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'ValidationData', validationDatastore, ...
    'ValidationFrequency',50, ...
    'ValidationPatience',5, ...
    'Plots','training-progress');
clear maxEpochs miniBatchSize

%% NETWORK TRAINING
net = trainNetwork(trainingDatastore, lgraph, options);
clear options 

%% CALCULATE ACCURACIES
fprintf('\nRESULTS\n');
disp('Training results');
evaluateDataStore(net, validationDatastore);
disp('Validation results');
evaluateDataStore(net, validationDatastore);
disp('Testing results');
evaluateDataStore(net, testingDatastore);

%% PLOT PREDICCTION/REAL SAMPLE FROM DATASET
plotPredictionDatastore(net, testingDatastore, 2);

%% FUNCTION TO DIVIDE DATASTORE IN TWO HALVES
function [firstDatstore, secondDatastore] = divideDatastore(dataStore)
    % First datstore(50%) && second datastore(50%)
    firstDatstore =  partition(dataStore,2,1);
    secondDatastore = partition(dataStore,2,2);
end

%% FUNCTION TO SET THE AMOUNT OF DATA
function datastore = setDataAmount(datastore, dataAmount)
    % Obtain the limit value index
    idxLimit = floor(size(datastore.Labels,1) * dataAmount);
    % The data is split within the datastore
    datastore.Datastore.Files = datastore.Datastore.Files(1:idxLimit);
    datastore.Labels = datastore.Labels(1:idxLimit);
    % NumObservations must be counted again
    reset(datastore);
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
    yPred = classify(net,data.sequences{idx});
    yReal = data.labelSequences{idx};
    plotPredictionComparison(yReal, yPred);
    reset(datastore);
end

%% FUNCTION TO PLOT A COMPARISON (REAL/PREDICTED)
function plotPredictionComparison(YTest, YPred)
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

%% FUCTION TO EVALUATE DATA USING THE VALIDATION LIBRARY
function evaluateDataStore(net, datastore)
    reset(datastore)
    numObservations = datastore.NumObservations;
    % Allocate space to save the results
    [clasifications, predictions, overlapings, procesingTimes] = preallocateValidationResults(numObservations);
    % Index initialization
    idx = 1;
    while hasdata(datastore)
        [data, info] = read(datastore);
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
            % Predict and calculate time
            timer = tic;
            yPred = classify(net,data.sequences(i));
            time = toc(timer);
            % Classify prediction
            class = classifyPredictions(yPred{1});
            % Predicteed information
            response.vectorOfLabels = yPred{1};
            response.vectorOfTimePoints = timepointSequences{i};
            nFrames = length(yPred{1});
            time_frame = time / nFrames;
            response.vectorOfProcessingTimes = time_frame*ones(1,nFrames);
            response.class = class;
            % Evaluate
            result = evalRecognition(repInfo, response);
            % Save results
            clasifications(idx) = result.classResult;
            if ~isequal(gestureName,'noGesture')
                predictions(idx) = result.recogResult;
                overlapings(idx) = result.overlappingFactor;
            end
            procesingTimes(idx) = time;
            idx = idx + 1;
        end
    end
    calculateValidationResults(clasifications, predictions, procesingTimes, overlapings);
end

%% FUCTION TO PREALLOCATE SPACE FOR VALIDATION LIBRARY RESULT
function [clasifications, predictions, overlapings, procesingTimes] = preallocateValidationResults(numObservations)
    % Allocate space to save the results
    clasifications = zeros(numObservations, 1);
    predictions = -1*ones(numObservations, 1);
    overlapings = zeros(numObservations, 1);
    procesingTimes = zeros(numObservations, 1);
end

%% FUNCTION TO CLASSIFY PREDICTIONS
function class = classifyPredictions(yPred)
    gestures = categorical({'fist'; 'noGesture'; 'open'; 'pinch'; 'waveIn'; 'waveOut'});
    catCounts = countcats(yPred);
    [catCounts,indexes] = sort(catCounts,'descend');
    newCategories = gestures(indexes);
    if catCounts(2) >= 4
       class = newCategories(2);
    else
       class = gestures(2);
    end
end

%%
function calculateValidationResults(clasifications, predictions, procesingTimes, overlapings)
    % Perform the calculation
    accClasification = sum(clasifications==1) / length(clasifications);
    accPrediction = sum(predictions==1) / sum(predictions==1 | predictions==0);
    avgProcesing_time = mean(procesingTimes);
    avgOverlapingFactor = mean(overlapings(overlapings ~= 0 & ~isnan(overlapings)));
    % Display the results
    fprintf('Classification accuracy: %f\n',accClasification);
    fprintf('Prediction accuracy: %f\n',accPrediction);
    fprintf('Avegage procesing time: %f\n',avgProcesing_time);
    fprintf('Avegage overlaping factor: %f\n\n',avgOverlapingFactor);
end

%% ARQUITECTURAS PROVADAS
%{
layers = [ ...
    sequenceInputLayer(dimensions)
    flattenLayer
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
%}
%{
convolution2dLayer(filterSize,numFilters,'Name','conv_1') 
    batchNormalizationLayer('Name','bn_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer([2 2],"Name","maxpool_1")
%}
%{
layers = [ ...
    sequenceInputLayer(inputSize,'Name','input')
    
    sequenceFoldingLayer('Name','fold')
    
    convolution2dLayer(filterSize,numFilters,'Name','conv')
    batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    %
    maxPooling2dLayer([2 2],"Name","maxpool")
    
    dropoutLayer(0.2,"Name","drop")
    
    convolution2dLayer(filterSize,2*numFilters,'Name','conv_1') 
    batchNormalizationLayer('Name','bn_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer([2 2],"Name","maxpool_1")
    
    %
    
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm')
    
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
%}
%{
    function calculateClassificationAccuracy(datastore)
        while(hasdata(datastore))
            [data,info] = read(datastore);
            sequences_pred = classify(net,data.sequences);

            yReal = info.label;
            for i = 1:length(yReal)
                data_similarity(idx) = sum(sequences_pred{i} == yReal{i})./numel(yReal{i});
                idx = idx + 1;
            end
        end
    end
%}
%{
    %% CALCULATE ACCURACIES
    training_acc = calculateAccuracy(net,trainingDatastore);
    validation_acc = calculateAccuracy(net, validationDatastore);
    testing_acc = calculateAccuracy(net, testingDatastore);
    % Print accuracies
    text_training_acc = ['Training samples: ', num2str(training_acc)];
    text_validation_acc = ['Validation samples: ', num2str(validation_acc)];
    text_testing_acc = ['Testing samples: ', num2str(testing_acc)];
    % The amount of training-validation-tests data is printed
    fprintf('\n%s\n%s\n%s\n',text_training_acc, text_validation_acc, text_testing_acc);
    clear text_training_acc text_validation_acc text_testing_acc
%}
%{
    %% FUNCTION TO CALCULATE ACCURACY
    function acc = calculateAccuracy(net, datastore)
        data_similarity = zeros(datastore.NumObservations, 1);
        idx = 1;
        while(hasdata(datastore))
            data = read(datastore);
            yPred = classify(net,data.sequences);
            yReal = data.label_sequences;
            for i = 1:length(yReal)
                data_similarity(idx) = sum(yPred{i} == yReal{i})./numel(yReal{i});
                idx = idx + 1;
            end
        end
        acc = mean(data_similarity);
        reset(datastore);
    end
%}
%{
    layers = [ ...
        sequenceInputLayer(inputSize,'Name','input')

        sequenceFoldingLayer('Name','fold')

        convolution2dLayer(filterSize,numFilters,'Name','conv')
        batchNormalizationLayer('Name','bn')
        reluLayer('Name','relu')

        sequenceUnfoldingLayer('Name','unfold')
        flattenLayer('Name','flatten')

        lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm')

        fullyConnectedLayer(numClasses, 'Name','fc')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classification')];
%}