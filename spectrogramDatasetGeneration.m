%{

%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset'; %'CEPRA_2019_13_DATASET_FINAL'
trainingDir = 'trainingJSON'; %'training'

%% GET THE USERS DIRECTORIES
[users, trainingPath] = Shared.getUsers(dataDir, trainingDir);
% Divide in two datasets
limit = length(users)- Shared.numTestUsers;
usersTrainVal = users(1:limit, 1);
usersTest = users(limit+1:length(users), 1);
clear dataDir trainingDir users numTestUsers limit

%%                      BORRAR ESTO AL ACABAR SOLO ES PARA HACER PRUEBAS CON PORCIONES
usersTrainVal = usersTrainVal(1:1);
usersTest = usersTest(1:1);

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
categories = {'fist'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
trainingDatastore = createDatastore('Datastores/training', categories);
validationDatastore = createDatastore('Datastores/validation', categories);
testingDatastore = createDatastore('Datastores/testing', categories);
clear categories

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
usersSets = {usersTrainVal, 'usersTrainVal'; usersTest, 'usersTest'};
% For each user set (trainVal and test)
for i = 1:length(usersSets)
    users = usersSets{i,1};
    usersSet = usersSets{i,2};
    parfor j = 1:length(users)
        % Get user samples
        [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, users(j));
        % Transform samples
        transformedSamplesTraining = generateData(trainingSamples);
        transformedSamplesValidation = generateData(validationSamples);
        % Save samples
        if isequal(usersSet, 'usersTrainVal')
            saveSampleInDatastore(transformedSamplesTraining, users(j), 'train', trainingDatastore);
            saveSampleInDatastore(transformedSamplesValidation, users(j), 'validation', validationDatastore);
        elseif isequal(usersSet, 'usersTest')
            saveSampleInDatastore(transformedSamplesTraining, users(j), 'train', testingDatastore);
            saveSampleInDatastore(transformedSamplesValidation, users(j), 'validation', testingDatastore);
        end 
    end
end
clear i j categories validationSamples transformedSamplesValidation users usersSet

%% INCLUDE NOGESTURE
% Define the directories where the frames will be found
datastores = {trainingDatastore; validationDatastore; testingDatastore};
usersInDatastore = {length(usersTrainVal); length(usersTrainVal); length(usersTest)};
noGesturePerUser = cell(length(datastores), 1);
clear trainingSamples transformedSamplesTraining trainingDatastore validationDatastore testingDatastore

%% CALCULATE THE MIN NUMBER OF FRAMES FOR EACH DATASTORE
parfor i = 1:length(datastores)
    % Create a file datastore.
    fds = fileDatastore(datastores{i,1}, ...
        'ReadFcn',@Shared.readFile, ...
        'IncludeSubfolders',true);
    % Create labels
    labels = Shared.createLabels(fds.Files, false);
    % Get the mininum number of frames for all category
    catCounts = sort(countcats(labels));
    minNumber = catCounts(1);
    % Generate noGesture frames
    noGesturePerUser{i,1} = ceil(minNumber / usersInDatastore{i,1});
end
clear i labels fds catCounts minNumber

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
categories = {'noGesture'};
trainingDatastore = createDatastore(datastores{1,1}, categories);
validationDatastore = createDatastore(datastores{2,1}, categories);
testingDatastore = createDatastore(datastores{3,1}, categories);
clear categories datastores

%% GENERATION OF NOGESTURE SPECTROGRAMS TO CREATE THE MODEL
noGestureTraining = noGesturePerUser{1,1};
noGestureValidation = noGesturePerUser{2,1};
noGestureTesting = ceil(noGesturePerUser{3,1} / 2);
for i = 1:length(usersSets)
    users = usersSets{i,1};
    usersSet = usersSets{i,2};
    parfor j = 1:length(users) % % parfor
        % Get user samples
        [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, users(j));
        % Transform samples
        if isequal(usersSet, 'usersTrainVal')
            transformedSamplesTraining = generateDataNoGesture(trainingSamples, noGestureTraining);
            transformedSamplesValidation = generateDataNoGesture(validationSamples, noGestureValidation);
            % Save samples
            saveSampleInDatastore(transformedSamplesTraining, users(j), 'train', trainingDatastore);
            saveSampleInDatastore(transformedSamplesValidation, users(j), 'validation', validationDatastore);
        elseif isequal(usersSet, 'usersTest')
            transformedSamplesTraining = generateDataNoGesture(trainingSamples, noGestureTesting);
            transformedSamplesValidation = generateDataNoGesture(validationSamples, noGestureTesting);
            % Save samples
            saveSampleInDatastore(transformedSamplesTraining, users(j), 'validation',testingDatastore);
            saveSampleInDatastore(transformedSamplesValidation, users(j), 'train', testingDatastore);
        end
    end
end
clear i j validationSamples transformedSamplesValidation trainingDatastore validationDatastore testingDatastore
clear users usersSet noGestureTraining noGestureValidation noGestureTesting

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
    % Inicialization
    numWindows = floor((length(signal)-Shared.FRAME_WINDOW) /Shared.WINDOW_STEP)+1;
    % Allocate space for the results
    data = cell(numWindows, 2);
    % Creating frames
    for i = 1:numWindows
        traslation = ((i-1)* Shared.WINDOW_STEP);
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        frameGroundTruth = groundTruth(inicio:finish);
        totalOnes = sum(frameGroundTruth == 1);
        if totalOnes >= Shared.FRAME_WINDOW * Shared.TOLERANCE_WINDOW || ...
                totalOnes >= numGesturePoints * Shared.TOLERNCE_GESTURE
            % Get Spectrogram of the window
            frameSignal = signal(inicio:finish, :);
            spectrograms = Shared.generateSpectrograms(frameSignal);
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
    noGesturePerUser = Shared.numGestureRepetitions;
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(length(samplesKeys)- noGesturePerUser, 3);
    for i = noGesturePerUser + 1:length(samplesKeys) 
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        groundTruth = sample.groundTruth;
        numGesturePoints = sample.groundTruthIndex(2) - sample.groundTruthIndex(1);
        % Get signal from sample
        signal = Shared.getSignal(emg);
        signal = Shared.preprocessSignal(signal);
        % Generate spectrograms
        data = generateFrames(signal, groundTruth, numGesturePoints);
        % Adding the transformed data
        transformedSamples{i - noGesturePerUser, 1} = data;
        transformedSamples{i - noGesturePerUser, 2} = gestureName;
        transformedSamples{i - noGesturePerUser, 3} = transpose(groundTruth);
    end
end

%% FUNCTION O GENERATE NO GESTURE FRAMES
function data = generateFramesNoGesture(signal, numWindows)
    % Allocate space for the results
    data = cell(numWindows, 2);
    for i = 1:numWindows
        traslation = ((i-1) * Shared.WINDOW_STEP) + 100; %displacement included
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        frameSignal = signal(inicio:finish, :);
        spectrograms = Shared.generateSpectrograms(frameSignal);
        data{i,1} = spectrograms; % datum
        data{i,2} = timestamp; % label
    end  
end

%% CREAR DATOS DE ESPECTROGRAMAS
function transformedSamples = generateDataNoGesture(samples, totalFrames)
    noGesturePerUser = Shared.numGestureRepetitions;
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(noGesturePerUser, 2);
    framesPerSample = ceil(totalFrames / noGesturePerUser);
    for i = 1:noGesturePerUser
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        % Get signal from sample
        signal = Shared.getSignal(emg);
        signal = Shared.preprocessSignal(signal);
        % Generate spectrograms
        data = generateFramesNoGesture(signal, framesPerSample);
        % Adding the transformed data
        transformedSamples{i,1} = data;
        transformedSamples{i,2} = gestureName;
    end
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveSampleInDatastore(samples, user, type, dataStore)
    % For each sample
    for i = 1:length(samples) 
        frames = samples{i,1};
        class = samples{i,2};
        % Data in frames
        spectrograms = frames(:,1);
        timestamps = frames(:,2);
        for j = 1:length(spectrograms)
            data = spectrograms{j, 1};
            start = floor(timestamps{j,1} - Shared.FRAME_WINDOW/2);
            finish = floor(timestamps{j,1} + Shared.FRAME_WINDOW/2);
            fileName = strcat(strtrim(user),'-', type, '-',int2str(i), '-', ...
                '[',int2str(start), '-', int2str(finish), ']');
            % The folder corresponds to the class 
            savePath = fullfile(dataStore, char(class), fileName);
            save(savePath,'data');
        end
    end
end
