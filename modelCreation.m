%{
    MODEL CREATION
%}

%%
dataDirTraining = fullfile('Datastores', 'training');
dataDirValidation = fullfile('Datastores', 'validation');

%% THE DATASTORES RE CREATED
% The classes are defined
classes = ["fist","noGesture","open","pinch","waveIn","waveOut"];
trainingDatastore = SpectrogramDatastore(dataDirTraining);
validationDatastore = SpectrogramDatastore(dataDirValidation);
%dataSample = preview(trainingDatastore);
clear dataDirTraining dataDirValidation

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.DataDimensions;

%% DEFINE THE AMOUNT OF DATA
% The amount of data to be used in the creation is specified ]0:1]
trainingDatastore = setDataAmount(trainingDatastore, 1);
validationDatastore = setDataAmount(validationDatastore, 0.5);

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
layers = [ ...
    imageInputLayer([101, 40, 8])%40
    convolution2dLayer([3 3],8,"Padding","same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride',2)
    convolution2dLayer([3 3],8,"Padding","same")%20
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses,"Name","fc2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

clear numHiddenUnits filterSize numFilters numClasses

%% THE OPTIONS ARE DIFINED
maxEpochs = 2;
miniBatchSize = 32;
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',3, ...
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
net = trainNetwork(trainingDatastore, layers, options);
clear options 

%% CONFUSION MATRIX


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

%%
%{ 
layers = [ ...
    imageInputLayer([101, 40, 8])%40
    convolution2dLayer([3 3],8,"Padding","same")
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride',2)
    convolution2dLayer([3 3],8,"Padding","same")%20
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride',2)
    convolution2dLayer([3 3],8,"Padding","same")%10
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],8,"Padding","same")%5
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride',2)
    convolution2dLayer([3 3],8,"Padding","same")%5
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],8,"Padding","same")%5
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride',2)
    convolution2dLayer([3 3],16,"Padding","same")%2
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],16,"Padding","same")
    batchNormalizationLayer
    reluLayer
    convolution2dLayer([3 3],32,"Padding","same")
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(128,"Name","fc")
    fullyConnectedLayer(64,"Name","fc1")
    fullyConnectedLayer(numClasses,"Name","fc2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];


%}