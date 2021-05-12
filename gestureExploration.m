%{


%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset';
trainingDir = 'trainingJSON';

%% GET THE USERS DIRECTORIES
[users, trainingPath] = Shared.getUsers(dataDir, trainingDir);
clear dataDir trainingDir

%% CREATE MATRIX FOR SAVE GESTURE POINT AVERAGE RESULTS
gestures = {'fist'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
groundTruthsAvgs = zeros(length(users), length(gestures));

%% CALCULATE THE GROUNDTRUTHS AVERAGES OF EACH USERS FOR EACH GESTURE
numGestures = length(gestures);
parfor i = 1:length(users)
    % Get user samples
    [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, users(i));
    % Calculate the averages of each class
    userAverages = calculateAverageUserClasses(trainingSamples, validationSamples, gestures);
    % Put ground truths in the matrix
    for j = 1:numGestures
         groundTruthsAvgs(i, j) = userAverages.(gestures{j});
    end
end
clear averages i j samplesKeys trainingSamples validationSamples numGestures

%% CALCULATE THE FINAL GROUNDTRUTHS
fprintf('\n---Analisis per gesture---\n');
fprintf('Gesture - Mean - Std\n');
for i = 1:length(gestures)
    gestureName = gestures{i};
    gestureMean = mean(groundTruthsAvgs(:,i));
    gestureStd = std(groundTruthsAvgs(:,i));
    fprintf('%s - %.2f - %.2f\n', gestureName, gestureMean, gestureStd);
end
fprintf('--------------------------\n');
clear i gestureName gestureMean gestureStd groundTruthsAvgs

%% CALCULATE THE GLOBAL AVERAGE PER USER
groundTruthsAvgsGlobal = zeros(length(users), 1);
parfor i = 1:length(users)
    % Get user samples
    [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, users(i));
    % Calculate the average of each user
    userAverage = calculateAverageUserGlobal(trainingSamples, validationSamples, gestures);
    groundTruthsAvgsGlobal(i, 1) = userAverage;
end
clear gestures

%% CALCULATE THE FINAL GLOBAL AVERAGE
globalMean = mean(groundTruthsAvgsGlobal(:,1));
globalStd = std(groundTruthsAvgsGlobal(:,1));
fprintf('\n---Global analisis---\n');
fprintf('Mean - Std\n');
fprintf('%.2f - %.2f\n', globalMean, globalStd);
fprintf('---------------------\n');
clear groundTruthsAvgsGlobal globalMean globalStd

%% CALCULATE THE GLOBAL NOGESTURE AVERAGE PER USER
noGesturesAvgsGlobal = zeros(length(users), 1);
noGesturesAvgsChannels = zeros(length(users), Shared.numChannels);
parfor i = 1:length(users)
    % Get user samples
    [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, users(i));
     % Calculate the noGesture average of each user
    [userAvgNoGesture, userAvgNGChannels] = calculateAverageUserNoGesture(trainingSamples, validationSamples);
    noGesturesAvgsGlobal(i, 1) = userAvgNoGesture;
    noGesturesAvgsChannels(i,:) = userAvgNGChannels;
end
clear trainingSamples validationSamples gestures trainingPath users

%% CALCULATE THE GLOBAL NOGESTURE AVERAGE
noGestureMean = mean(noGesturesAvgsGlobal(:,1));
noGestureStd = std(noGesturesAvgsGlobal(:,1));
fprintf('\n---Global noGesture---\n');
fprintf('Mean - Std\n');
fprintf('%.2f - %.2f\n', noGestureMean, noGestureStd);
fprintf('-----------------------\n');
clear noGestureMean noGestureStd

%% CALCULATE NOGESTURE AVERAGE PER CHANNEL
noGestureChannelMean = mean(noGesturesAvgsChannels, 1);
noGestureChannelStd = std(noGesturesAvgsChannels, 0, 1);
fprintf('\n---Channel noGesture---\n');
fprintf('Channel - Mean - Std\n');
for i = 1:Shared.numChannels
  fprintf('%d - %.2f - %.2f\n',i, noGestureChannelMean(1,i), noGestureChannelStd(1,i));  
end
fprintf('-----------------------\n');
clear noGesturesAvgsChannels noGesturesAvgsGlobal noGestureChannelMean noGestureChannelStd i

%% FUNCTION TO TRANSFORM THE SAMPLES IN CELLS {GESTURE, GROUNDTRUTH}
function transformedSamples = getClassGroundTruth(samples, gestures)
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(length(gestures) * Shared.numGestureRepetitions, 2);
    for i = Shared.numGestureRepetitions + 1:length(samplesKeys) % Gestures start at 26
        sample = samples.(samplesKeys{i});
        gestureName = sample.gestureName;
        groundTruthIndex = sample.groundTruthIndex;
        groundTruth = groundTruthIndex(2) - groundTruthIndex(1);
        transformedSamples{i - Shared.numGestureRepetitions, 1} = gestureName;
        transformedSamples{i - Shared.numGestureRepetitions, 2} = groundTruth;
    end
end

%% FUNCTION TO CALCULATE THE AVERAGE GROUNDTRUTH FOR EACH CLASS OF A USER
function averages = calculateAverageUserClasses(samplesTrain, samplesValidation, gestures)
    % Get classes and groundTruths
    transformedSamplesTrain = getClassGroundTruth(samplesTrain, gestures);
    transformedSamplesValidation = getClassGroundTruth(samplesValidation, gestures);
    % concatenate train and validation transformations
    transformedSamples = [transformedSamplesTrain; transformedSamplesValidation];
    for j = 1:length(gestures)
        gestureName = gestures{j};
        idxs = cellfun(@(x) isequal(x,gestureName), transformedSamples(:,1));
        gestureSamples = transformedSamples(idxs,:);
        groundTruths = cell2mat(gestureSamples(:,2));
        averages.(gestureName) =  mean(groundTruths);
    end
end

%% FUNCTION TO CALCULATE THE AVERAGE GROUNDTRUTH FOR EACH CLASS OF A USER
function average = calculateAverageUserGlobal(samplesTrain, samplesValidation, gestures)
    % Get classes and groundTruths
    transformedSamplesTrain = getClassGroundTruth(samplesTrain, gestures);
    transformedSamplesValidation = getClassGroundTruth(samplesValidation, gestures);
    % Concatenate train and validation transformations
    transformedSamples = [transformedSamplesTrain; transformedSamplesValidation];
    groundTruths = cell2mat(transformedSamples(:,2));
    average = mean(groundTruths);
end

%% FUNCTION TO TRANSFORM THE SAMPLES IN CELLS {GESTURE, GROUNDTRUTH}
function transformedSamples = getNoGesturesAvgUser(samples)
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(Shared.numGestureRepetitions, 2);
    for i = 1:Shared.numGestureRepetitions % NoGestures end at 25
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        signal = Shared.getSignal(emg);
        % PODRIA SACAR PROMEDIO POR CNAL IGUAL
        transformedSamples{i,1} = mean(signal, 'all');
        numChannels = size(signal, 2);
        meanPerChannel = zeros(1, numChannels);
        for j = 1:numChannels
            meanPerChannel(1, j)= mean(signal(:, j));
        end
        transformedSamples{i,2} = meanPerChannel;
    end
end

%% FUNCTION TO  CALCULATE THE AVERAGE OF NOGESTURE OF A USER
function [userAvgNoGesture, userAvgNGChannels] = calculateAverageUserNoGesture(trainingSamples, validationSamples)
    % Get the mean of each noGesture sample
    avgNoGesturesTrain = getNoGesturesAvgUser(trainingSamples);
    avgNoGesturesValidation = getNoGesturesAvgUser(validationSamples);
    % Concatenate train and validation transformations
    noGesturesGlobal = cell2mat([avgNoGesturesTrain(:,1); avgNoGesturesValidation(:,1)]);
    noGesturesPerChannel = cell2mat([avgNoGesturesTrain(:,2); avgNoGesturesValidation(:,2)]);
    userAvgNoGesture = mean(noGesturesGlobal);
    userAvgNGChannels = mean(noGesturesPerChannel,1); % dimension 1 = vertical
end
