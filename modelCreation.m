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
filterSize = 2;
numFilters = 20;
numClasses = length(classes);
layers = [ ...
    imageInputLayer(inputSize, 'Name','input')
    convolution2dLayer(filterSize,numFilters,'Name','conv', 'Padding','same') 
    batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    maxPooling2dLayer([2 2],"Name","maxpool")
    
    dropoutLayer(0.2,"Name","drop")
    
    convolution2dLayer(filterSize,2*numFilters,'Name','conv_1', 'Padding','same') 
    batchNormalizationLayer('Name','bn_1')
    reluLayer('Name','relu_1')
    
    
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
clear inputSize numHiddenUnits filterSize numFilters numClasses

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
net = trainNetwork(trainingDatastore, layers, options);
clear options 

%% EVALUATING DATA
dataDirTrainingEval = fullfile('Datastores', 'trainingEvalDatastore');
trainingEvalDatastore = SpectrogramDatastoreEval(dataDirTrainingEval);
dataDirValidationEval = fullfile('Datastores', 'validationEvalDatastore');
validationEvalDatastore = SpectrogramDatastoreEval(dataDirValidationEval);
clear dataDirEvaluation

%% CALCULATE ACCURACIES
fprintf('\nRESULTS\n');
disp('Training results');
evaluateDataStore(net, trainingEvalDatastore);
disp('Validation results');
evaluateDataStore(net, validationEvalDatastore);

%% PLOT PREDICCTION/REAL SAMPLE FROM DATASET
plotPredictionDatastore(net, trainingEvalDatastore, 2);

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
            % Allocate space for the results
            sequence = data.sequences{i};
            numFrames = length(sequence);
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
            % Classify prediction
            [class, vectorOfLabels] = classifyPredictions(vectorOfLabels);
            % Predicteed information
            response.vectorOfLabels = vectorOfLabels;
            response.vectorOfTimePoints = timepointSequences{i};
            response.vectorOfProcessingTimes = vectorOfProcessingTimes;
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
