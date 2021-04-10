%{

%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset'; %'CEPRA_2019_13_DATASET_FINAL'
trainingDir = 'trainingJSON'; %'training'

%% GET THE USERS DIRECTORIES
[users, trainingPath] = getUsers(dataDir, trainingDir);
clear dataDir trainingDir

%%                      BORRAR ESTO AL ACABAR SOLO ES PARA HACER PRUEBAS CON PORCIONES
users = users(1:1);

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
classes = {'fist'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
trainingDatastore = createDatastore('Datastores/training', classes, true);
validationDatastore = createDatastore('Datastores/validation', classes, true);
% To evaluate each sample as a sequence
%trainingEvalDatastore = createDatastore('Datastores/trainingSequence', labels);
%validationEvalDatastore = createDatastore('Datastores/validationSequence', labels);

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
for i = 1:length(users) % % parfor
    % Get user samples
    [trainingSamples, validationSamples] = getTrainingTestingSamples(trainingPath, users(i));
    % Training data
    transformedSamplesTraining = generateData(trainingSamples);
    saveSampleInDatastore(transformedSamplesTraining, users(i), trainingDatastore);
    %saveSampleSeqInDatastore(transformedSamplesTraining, users(i), trainingEvalDatastore);
    % Validation data
    transformedSamplesValidation = generateData(validationSamples);
    saveSampleInDatastore(transformedSamplesValidation, users(i), validationDatastore);
    %saveSampleSeqInDatastore(transformedSamplesValidation, users(i), validationEvalDatastore);
end
clear i validationSamples transformedSamplesValidation


%% INCLUDE NOGESTURE
% Define the directories where the frames will be found
dataDir = 'Datastores';
datastores = {'training'; 'validation'};

%% CREATE A FILEDATASTORE 
datastore = datastores{1};
folder = fullfile(dataDir, datastore);
% Create a file datastore.
fds = fileDatastore(folder, ...
    'ReadFcn',@readFile, ...
    'IncludeSubfolders',true);
clear dataDir datastores datastore folder

%%  CREATE LABELS
labels = createLabels(fds.Files, classes);
clear fds

%% GET THE MININUM NUMBER OF FRAMES FOR ALL CATEGORY
catCounts = sort(countcats(labels));
minNumber = catCounts(1);
clear catCounts

%% GENERATE NO GESTURE FRAMES
noGesturePerUser = ceil(minNumber/ length(users));

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
classes = {'noGesture'};
trainingDatastore = createDatastore('Datastores/training', classes, false);
validationDatastore = createDatastore('Datastores/validation', classes, false);

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
for i = 1:length(users) % % parfor
    % Get user samples
    [trainingSamples, validationSamples] = getTrainingTestingSamples(trainingPath, users(i));
    % Training data
    transformedSamplesTraining = generateDataNoGesture(trainingSamples, noGesturePerUser);
    saveSampleInDatastore(transformedSamplesTraining, users(i), trainingDatastore);
    %saveSampleSeqInDatastore(transformedSamplesTraining, users(i), trainingEvalDatastore);
    % Validation data
    transformedSamplesValidation = generateDataNoGesture(validationSamples, noGesturePerUser);
    saveSampleInDatastore(transformedSamplesValidation, users(i), validationDatastore);
    %saveSampleSeqInDatastore(transformedSamplesValidation, users(i), validationEvalDatastore);
end
clear i validationSamples transformedSamplesValidation


%% GET THE USER LIST
function [users, dataPath] = getUsers(dataDir, subDir)
    dataPath = fullfile(dataDir, subDir);
    users = ls(dataPath);
    users = strtrim(string(users(3:length(users),:)));
    rng(9); % seed
    users = users(randperm(length(users)));
end

%% FUNCTION TO CREATE DATASTORE
function datastore = createDatastore(datastore, labels, createRoot)
    if createRoot
        mkdir(datastore);
    end
    % One folder is created for each class
    for i = 1:length(labels)
        mkdir(fullfile(datastore, char(labels(i))));
    end
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

%% FUNCTION TO GET THE EMG SIGNAL
function signal = getSignal(emg)
    channels = fieldnames(emg); % get chanels
    signal = zeros(length(emg.(channels{1})), length(channels)); % ex: 1000 x 8
    for j = 1:length(channels)
        signal(:,j) = emg.(channels{j});
    end
end

%% FUNCTION TO RECTIFY EMG
function rectifiedEMG = rectifyEMG(rawEMG, rectFcn)
    switch rectFcn
        case 'square'
            rectifiedEMG = rawEMG.^2;
        case 'abs'
            rectifiedEMG = abs(rawEMG);
        case 'none'
            rectifiedEMG = rawEMG;
        otherwise
            fprintf('Wrong rectification function. Valid options are square, abs and none');
    end
end

%% FUNCTION TO APLY A FILTER TO EMG SIGNAL
function EMGsegment_out = preProcessEMGSegment(EMGsegment_in, Fa, Fb, rectFcn)
    % Normalization
    if max( abs(EMGsegment_in(:)) ) > 1
        drawnow;
        EMGnormalized = EMGsegment_in/128;
    else
        EMGnormalized = EMGsegment_in;
    end
    EMGrectified = rectifyEMG(EMGnormalized, rectFcn);
    EMGsegment_out = filtfilt(Fb, Fa, EMGrectified);
end

%% FUNCTION TO PREPROCESS A SIGNAL
function signal = preprocessSignal(signal)
    [Fb, Fa] = butter(5, 0.1, 'low');
    signal = preProcessEMGSegment(signal, Fa, Fb, 'abs');
end

%% FUNCTION TO GENERATE SPECTROGRAMS
function spectrograms = generateSpectrograms(signal)
    % Spectrogram parameters
    FRECUENCIES = (0:12);
    sampleFrecuency = 200;
    % Almost mandaory 200 to analize from 0 to 100 fecuencies
    WINDOW = 24;
    OVERLAPPING = floor(WINDOW*0.5); %floor(WINDOW*0.75); %floor(WINDOW*0.5); % WINDOW -1
    % Preallocate space for the spectrograms
    numCols = floor((length(signal)-OVERLAPPING)/(WINDOW-OVERLAPPING));
    spectrograms = zeros(length(FRECUENCIES), numCols, size(signal, 2));
    % Spectrograms generation
    for i = 1:size(signal, 2)
        [~,~,~,ps] = spectrogram(signal(:,i), WINDOW, OVERLAPPING, FRECUENCIES, sampleFrecuency, 'yaxis');
        spectrograms(:,:,i) = ps;
    end
end

%% generateSpectrograms
function [data] = generateFrames(signal, groundTruth, numGesturePoints)
    % Frame onfigurations
    FRAME_WINDOW = 300;
    WINDOW_STEP = 15;
    TOLERANCE_WINDOW = 0.75;
    TOLERNCE_GESTURE = 0.9;
    % Inicialization
    numWindows = floor((length(signal)-FRAME_WINDOW) /WINDOW_STEP)+1;
    
    % Creating spectrograms (ex: 101 x n x 8)
    %[spectrograms, timestamps, params] = generateSpectrograms(signal);
    % Creating frames
    % Allocate space for the results
    data = cell(numWindows, 2);
    for i = 1:numWindows
        traslation = ((i-1)*WINDOW_STEP);
        inicio = 1 + traslation;
        finish = FRAME_WINDOW + traslation;
        timestamp = inicio + floor(FRAME_WINDOW/2);
        frameGroundTruth = groundTruth(inicio: finish);
        totalOnes = sum(frameGroundTruth == 1);
        if totalOnes >= FRAME_WINDOW * TOLERANCE_WINDOW || totalOnes >= numGesturePoints * TOLERNCE_GESTURE
            %  PARA CADA CANAL hay que sacar el expectrograma
            frameSignal = signal(inicio:finish, :);
            spectrograms = generateSpectrograms(frameSignal);
            data{i,1} = spectrograms; % datum
            data{i,2} = timestamp; % label
        end
        % filter data
        idx = cellfun(@(x) ~isempty(x), data(:,1));
        data = data(idx,:);
    end  
end

%% CREAR DATOS DE ESPECTROGRAMAS
function transformedSamples = generateData(samples)
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(length(samplesKeys)- 25, 3);
    for i = 26:length(samplesKeys)
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        groundTruth = sample.groundTruth;
        numGesturePoints = sample.groundTruthIndex(2) - sample.groundTruthIndex(1);
        % Get signal from sample
        signal = getSignal(emg);
        signal = preprocessSignal(signal);
        % Generate spectrograms
        data = generateFrames(signal, groundTruth, numGesturePoints);
        % Adding the transformed data
        transformedSamples{i-25,1} = data;
        transformedSamples{i-25,2} = gestureName;
        transformedSamples{i-25,3} = transpose(groundTruth);
    end
end

%% FUNCTION TO READ A FILE
function data = readFile(filename)
    % Load a Matlab file
    data = load(filename).data;
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveSampleInDatastore(samples, user, data_store)
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
            savePath = fullfile(data_store, char(class),fileName);
            save(savePath,'data');
        end
    end
end

%% FUNCION PARA CREAR ETIQUETAS
function labels = createLabels(files, classes)
    % Get the number of files
    numObservations = numel(files);
    % Allocate spacce for labels
    labels = cell(numObservations,1);
    parfor i = 1:numObservations
        file = files{i};
        filepath = fileparts(file); % ../datastore/class
        % The last part of the path is the label
        [~,label] = fileparts(filepath); % [../datastore, class]
        labels{i,1} = label;
    end
    labels = categorical(labels,classes);
end

%% generateSpectrograms
function data = generateFramesNoGesture(signal, numWindows)
    % Frame onfigurations
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
        spectrograms = generateSpectrograms(frameSignal);
        data{i,1} = spectrograms; % datum
        data{i,2} = timestamp; % label
    end  
end


%% CREAR DATOS DE ESPECTROGRAMAS
function transformedSamples = generateDataNoGesture(samples, totalFrames)
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(25, 2);
    framesPerSample = ceil(totalFrames/25);
    for i = 1:25
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        % Get signal from sample
        signal = getSignal(emg);
        signal = preprocessSignal(signal);
        % Generate spectrograms
        data = generateFramesNoGesture(signal, framesPerSample);
        % Adding the transformed data
        transformedSamples{i,1} = data;
        transformedSamples{i,2} = gestureName;
    end
end