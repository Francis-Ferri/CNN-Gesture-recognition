%{


%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset';
trainingDir = 'trainingJSON';

%% GET THE USERS DIRECTORIES
[users, trainingPath] = getUsers(dataDir, trainingDir);
clear dataDir trainingDir

%% CREATE MATRIX FOR GESTURE POINT AVERAGE
gestures = {'fist'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
groundTruthsAvgs = zeros(length(users), length(gestures));

%% CALCULATE THE GROUNDTRUTHS AVERAGES OF EACH USERS FOR EACH GESTURE
numGestures = length(gestures);
parfor i = 1:length(users)
    % Get user samples
    [trainingSamples, validationSamples] = getTrainingTestingSamples(trainingPath, users(i));
    % Calculate the averages of each class
    userAverages = calculateAverageUserClasses(trainingSamples, validationSamples, gestures);
    % Put ground truths in the matrix
    for j = 1:numGestures
         groundTruthsAvgs(i, j) = userAverages.(gestures{j});
    end
end
clear averages i j samplesKeys trainingSamples validationSamples

%% CALCULATE THE FINAL GROUNDTRUTHS
fprintf('\n-----------------\n');
fprintf('Gesture - Mean - Std\n');
for i = 1:length(gestures)
    gestureName = gestures{i};
    gestureMean = mean(groundTruthsAvgs(:,i));
    gestureStd = std(groundTruthsAvgs(:,i));
    fprintf('%s - %.2f - %.2f\n', gestureName, gestureMean, gestureStd);
    resultAverages.(gestureName) = gestureMean;
end
fprintf('-----------------\n');
clear i gestureName gestureMean gestureStd

%% CALCULATE THE GLOBAL AVERAGE PER USER
groundTruthsAvgsGlobal = zeros(length(users), 1);
parfor i = 1:length(users)
    % Get user samples
    [trainingSamples, validationSamples] = getTrainingTestingSamples(trainingPath, users(i));
    % Calculate the averages of each class
    userAverage = calculateAverageUserGlobal(trainingSamples, validationSamples, gestures);
    groundTruthsAvgsGlobal(i, 1) = userAverage;
end
clear numGestures

%% CALCULATE THE FINAL GLOBAL AVERAGE
globalMean = mean(groundTruthsAvgsGlobal(:,1));
globalStd = std(groundTruthsAvgsGlobal(:,1));
fprintf('\n-----Global-----\n');
fprintf('Mean - Std\n');
fprintf('%.2f - %.2f\n', globalMean, globalStd);
fprintf('------------------\n');

%% GET THE USER LIST
function [users, dataPath] = getUsers(dataDir, subDir)
    dataPath = fullfile(dataDir, subDir);
    users = ls(dataPath);
    users = strtrim(string(users(3:length(users),:)));
    rng(9); % To shuffle the users
    users = users(randperm(length(users)));
end

%% GET TRAINING AND TESTING SAMPLES FOR AN USER
function [trainingSamples, testingSamples] = getTrainingTestingSamples(path, user)
    filePath = fullfile(path, user, strcat(user, '.json'));
    jsonFile = fileread(filePath);
    jsonData = jsondecode(jsonFile);
    % Extract samples
    trainingSamples = jsonData.trainingSamples;
    testingSamples = jsonData.testingSamples;
end

%% TRANSFORM THE SAMPLES IN CELLS {GESTURE, GROUNDTRUTH}
function transformedSamples = getClassGroundTruth(samples, gestures)
    numRepetitions = 25;
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(length(gestures)*numRepetitions, 2);
    for i = numRepetitions+1:length(samplesKeys)
        sample = samples.(samplesKeys{i});
        gestureName = sample.gestureName;
        groundTruthIndex = sample.groundTruthIndex;
        groundTruth = groundTruthIndex(2) - groundTruthIndex(1);
        transformedSamples{i-numRepetitions,1} = gestureName;
        transformedSamples{i-numRepetitions,2} = groundTruth;
    end
end

%% CALCULATE THE AVERAGE GROUNDTRUTH FOR EACH CLASS OF A USER
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

%% CALCULATE THE AVERAGE GROUNDTRUTH FOR EACH CLASS OF A USER
function average = calculateAverageUserGlobal(samplesTrain, samplesValidation, gestures)
    % Get classes and groundTruths
    transformedSamplesTrain = getClassGroundTruth(samplesTrain, gestures);
    transformedSamplesValidation = getClassGroundTruth(samplesValidation, gestures);
    % concatenate train and validation transformations
    transformedSamples = [transformedSamplesTrain; transformedSamplesValidation];
    groundTruths = cell2mat(transformedSamples(:,2));
    average = mean(groundTruths);
end
