%{

%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset'; %'CEPRA_2019_13_DATASET_FINAL'
trainingDir = 'trainingJSON'; %'training'

%% GET THE USERS DIRECTORIES
[users, trainingPath] = SharedFunctions.getUsers(dataDir, trainingDir);
clear dataDir trainingDir

%%                      BORRAR ESTO AL ACABAR SOLO ES PARA HACER PRUEBAS CON PORCIONES
users = users(1:1);

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
categories = {'fist'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
trainingDatastore = createDatastore('Datastores/training', categories);
validationDatastore = createDatastore('Datastores/validation', categories);
clear categories

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
parfor i = 1:length(users) % % parfor
    % Get user samples
    [trainingSamples, validationSamples] = SharedFunctions.getTrainingTestingSamples(trainingPath, users(i));
    % Training data
    transformedSamplesTraining = generateData(trainingSamples);
    saveSampleInDatastore(transformedSamplesTraining, users(i), trainingDatastore);
    % Validation data
    transformedSamplesValidation = generateData(validationSamples);
    saveSampleInDatastore(transformedSamplesValidation, users(i), validationDatastore);
end
clear i categories validationSamples transformedSamplesValidation

%% INCLUDE NOGESTURE
% Define the directories where the frames will be found
datastores = {trainingDatastore; validationDatastore};
noGesturePerUser = cell(2, 1);
clear trainingSamples transformedSamplesTraining trainingDatastore validationDatastore

%% CALCULATE THE MIN NUMBER OF FRAMES FOR EACH DATASTORE
parfor i = 1:length(datastores)
    % Create a file datastore.
    fds = fileDatastore(datastores{i,1}, ...
        'ReadFcn',@SharedFunctions.readFile, ...
        'IncludeSubfolders',true);
    % Create labels
    labels = SharedFunctions.createLabels(fds.Files, false);
    % Get the mininum number of frames for all category
    catCounts = sort(countcats(labels));
    minNumber = catCounts(1);
    % Generate noGesture frames
    noGesturePerUser{i,1} = ceil(minNumber / length(users));
end
clear i labels fds catCounts minNumber

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
categories = {'noGesture'};
trainingDatastore = createDatastore(datastores{1,1}, categories);
validationDatastore = createDatastore(datastores{2,1}, categories);
clear categories datastores

%% GENERATION OF NOGESTURE SPECTROGRAMS TO CREATE THE MODEL
noGestureTraining = noGesturePerUser{1,1};
noGestureValidation = noGesturePerUser{2,1};
parfor i = 1:length(users) % % parfor
    % Get user samples
    [trainingSamples, validationSamples] = SharedFunctions.getTrainingTestingSamples(trainingPath, users(i));
    % Training data
    transformedSamplesTraining = generateDataNoGesture(trainingSamples, noGestureTraining);
    saveSampleInDatastore(transformedSamplesTraining, users(i), trainingDatastore);
    % Validation data
    transformedSamplesValidation = generateDataNoGesture(validationSamples, noGestureValidation);
    saveSampleInDatastore(transformedSamplesValidation, users(i), validationDatastore);
end
clear i validationSamples transformedSamplesValidation noGesturePerUser trainingDatastore validationDatastore

%% FUNCTION TO CREATE DATASTORE
function datastore = createDatastore(datastore, labels)
    if ~exist(datastore, 'dir')
       mkdir(datastore)
    end
    % One folder is created for each class
    for i = 1:length(labels)
        path = fullfile(datastore, char(labels(i)));
        if ~exist(path, 'dir')
             mkdir(path);
        end
    end
end

%% FUNCTION TO GENERATE FRAMES
function [data] = generateFrames(signal, groundTruth, numGesturePoints)
    % Frame configurations
    FRAME_WINDOW = 300;
    WINDOW_STEP = 15;
    TOLERANCE_WINDOW = 0.75;
    TOLERNCE_GESTURE = 0.9;
    % Inicialization
    numWindows = floor((length(signal)-FRAME_WINDOW) /WINDOW_STEP)+1;
    % Allocate space for the results
    data = cell(numWindows, 2);
    % Creating frames
    for i = 1:numWindows
        traslation = ((i-1)*WINDOW_STEP);
        inicio = 1 + traslation;
        finish = FRAME_WINDOW + traslation;
        timestamp = inicio + floor(FRAME_WINDOW/2);
        frameGroundTruth = groundTruth(inicio: finish);
        totalOnes = sum(frameGroundTruth == 1);
        if totalOnes >= FRAME_WINDOW * TOLERANCE_WINDOW || totalOnes >= numGesturePoints * TOLERNCE_GESTURE
            % Get Spectrogram of the window
            frameSignal = signal(inicio:finish, :);
            spectrograms = SharedFunctions.generateSpectrograms(frameSignal);
            data{i,1} = spectrograms; % datum
            data{i,2} = timestamp; % time
        end
        % Filter to get the gesture frames and discard the noGestures
        idx = cellfun(@(x) ~isempty(x), data(:,1));
        data = data(idx,:);
    end  
end

%% FUCTION TO GENERATE THE DATA
function transformedSamples = generateData(samples)
    noGesturePerUser = 25; % Gestures start at 26 sample
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(length(samplesKeys)- noGesturePerUser, 3);
    for i = 26:length(samplesKeys) 
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        groundTruth = sample.groundTruth;
        numGesturePoints = sample.groundTruthIndex(2) - sample.groundTruthIndex(1);
        % Get signal from sample
        signal = SharedFunctions.getSignal(emg);
        signal = SharedFunctions.preprocessSignal(signal);
        % Generate spectrograms
        data = generateFrames(signal, groundTruth, numGesturePoints);
        % Adding the transformed data
        transformedSamples{i-noGesturePerUser,1} = data;
        transformedSamples{i-noGesturePerUser,2} = gestureName;
        transformedSamples{i-noGesturePerUser,3} = transpose(groundTruth);
    end
end

%% FUNCTION O GENERATE NO GESTURE FRAMES
function data = generateFramesNoGesture(signal, numWindows)
    % Frame configurations
    FRAME_WINDOW = 300;
    WINDOW_STEP = 15;
    % Allocate space for the results
    data = cell(numWindows, 2);
    for i = 1:numWindows
        traslation = ((i-1)*WINDOW_STEP) + 100; %displacement included
        inicio = 1 + traslation;
        finish = FRAME_WINDOW + traslation;
        timestamp = inicio + floor(FRAME_WINDOW/2);
        frameSignal = signal(inicio:finish, :);
        spectrograms = SharedFunctions.generateSpectrograms(frameSignal);
        data{i,1} = spectrograms; % datum
        data{i,2} = timestamp; % label
    end  
end

%% CREAR DATOS DE ESPECTROGRAMAS
function transformedSamples = generateDataNoGesture(samples, totalFrames)
    noGesturePerUser = 25; % Gestures start at 26 sample
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(noGesturePerUser, 2);
    framesPerSample = ceil(totalFrames/noGesturePerUser);
    for i = 1:25
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        % Get signal from sample
        signal = SharedFunctions.getSignal(emg);
        signal = SharedFunctions.preprocessSignal(signal);
        % Generate spectrograms
        data = generateFramesNoGesture(signal, framesPerSample);
        % Adding the transformed data
        transformedSamples{i,1} = data;
        transformedSamples{i,2} = gestureName;
    end
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveSampleInDatastore(samples, user, dataStore)
    FRAME_WINDOW = 300;
    for i = 1:length(samples) 
        frames = samples{i,1};
        class = samples{i,2};
        % Data in frames
        spectrograms = frames(:,1);
        timestamps = frames(:,2);
        for j = 1:length(spectrograms)
            data = spectrograms{j, 1};
            start = floor(timestamps{j,1} - FRAME_WINDOW/2);
            finish = floor(timestamps{j,1} + FRAME_WINDOW/2);
            fileName = strcat(strtrim(user),'-', int2str(i), '-', ...
                '[',int2str(start), '-', int2str(finish), ']');
            % The folder corresponds to the class 
            savePath = fullfile(dataStore, char(class), fileName);
            save(savePath,'data');
        end
    end
end
