%{
SPECTROGRAM DATASET GENERATION
    1. The datastore folders are created
    2. For each user:
        a. Samples are obteined
        b. For each sample:
            i. The sample is diveded into frames
            ii. For each frame:
                I. The spectrogram is calculated
            iii. The frames are sabed with their corresponding labels
        c. Each sample with its frames and labels are saved in the
        datastore
%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset';
trainingDir = 'trainingJSON';

%% GET THE TRAINING DIRECTORIES
trainingPath = fullfile(dataDir, trainingDir);
users = ls(trainingPath);
users = strtrim(string(users(3:length(users),:)));
users = users(randperm(length(users)));
clear data_dir trainingDir

%%                      BORRAR ESTO AL ACABAR SOLO ES PARA HACER PRUEBAS CON PORCIONES
users = users(1:1);

%% THE STRUCTURE OF THE DATA STORE IS DEFINED
labels = {'fist'; 'noGesture'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
trainingDatastore = createDatastore('Datastores/trainingDatastore', labels);
validationDatastore = createDatastore('Datastores/validationDatastore', labels);

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
for i = 1:length(users) % % parfor
    % Get user samples
    [trainingSamples, validationSamples] = getTrainingTestingSamples(trainingPath, users(i));
    % Training data
    transformedSamplesTraining = generateData(trainingSamples);
    saveSampleInDatastore(transformedSamplesTraining, users(i), trainingDatastore);
    % Validation data
    transformedSamplesValidation = generateData(validationSamples);
    saveSampleInDatastore(transformedSamplesValidation, users(i), validationDatastore);
end
clear labels i validationSamples transformedSamplesValidation

%% GET TRAINING AND TESTING SAMPLES FOR AN USER
function [trainingSamples, testingSamples] = getTrainingTestingSamples(path, user)
    filePath = fullfile(path, user, strcat(user, '.json'));
    jsonFile = fileread(filePath);
    jsonData = jsondecode(jsonFile);
    % Extract samples
    trainingSamples = jsonData.trainingSamples;
    testingSamples = jsonData.testingSamples;
end

%% FUNCTION TO CREATE DATASTORE
function datastore = createDatastore(datastore, labels)
    mkdir(datastore);
    % One folder is created for each class
    for i = 1:length(labels)
        mkdir(fullfile(datastore, char(labels(i))));
    end
end

%% CREAR DATOS DE ESPECTROGRAMAS
function transformedSamples = generateData(samples)
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(length(samplesKeys), 3);
    for i = 1: length(samplesKeys)
        % Get sample
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        % Get signal from sample
        signal = getSignal(emg);
        signal = preprocessSignal(signal);
        % Generate spectrograms
        [data] = generateSpectrograms(signal, sample, gestureName);
        % Adding the transformed data
        transformedSamples{i,1} = data;
        transformedSamples{i,2} = gestureName;
        if ~isequal(gestureName,'noGesture')
            groundTruth = sample.groundTruth;
            transformedSamples{i,3} = transpose(groundTruth);
        end
    end
end

%% FUNCTION TO GET THE EMG SIGNAL
function signal = getSignal(emg)
    chanels = fieldnames(emg); % get chanels
    signal = zeros(length(emg.(chanels{1})), length(chanels)); % ex: 1000 x 8
    for j = 1:length(chanels)
        signal(:,j) = emg.(chanels{j});
    end
end

%%
function signal = preprocessSignal(signal)
    %{ 
        Consideraciones preprocesado
            filtro pasa bajos de 57 Hz - 50 - 10
            Myo -1 +1
            Normaliza por usuario
            Normaliza toda la señal Por gesto y reescalas maxima amplitud para la señal de Wave in debe eqivaler a + 1 y la minima a --1
            Promedio de los maximo  de los minimos y señales de testeo
            Maximo y minimo
    %}
end

%% generateSpectrograms
function [data] = generateSpectrograms(signal, sample,gestureName)
    %{
        Consideraciones espectrogramas
            ventanas de 20, ventanas de 50 ventanas de 100
            No sacar directamente espectrogras, si no el logaritmo de la 
            transformada de FOurier al cuadrado como en speech recognition
    %}
    % Spectrogram parameters
    frecuencies = 1:1:100;
    sampleFrecuency = 200;
    % Number of frames
    frameSize = 20;
    totalFrames = floor(length(signal) / frameSize);
    % Allocate space for the results
    data = cell(totalFrames,3);
    % Initialize windowing parameters
    frameIdx = 1;
    position = 1;
    % Allocate space for the spectrograms
    spectrograms = zeros(size(frecuencies,2), size(signal,2)); % 100 x 8
    while frameIdx <= totalFrames
        % Spectrogram for each channel
        for j = 1:size(signal,2)
            spectrograms(:,j) = spectrogram(signal(position:(position + frameSize - 1), j), ...
                frameSize, 0, frecuencies, sampleFrecuency, 'yaxis');
        end
        data{frameIdx,1} = abs(spectrograms); % datum
        data{frameIdx,2} = 'noGesture'; % label
        data{frameIdx,3} = round(position + (frameSize / 2)); % timepoint
        % Check no gesture and change label
        if ~isequal(gestureName,'noGesture')
            groundTruth = sample.groundTruth;
            frameGroundTruth = groundTruth(position : position + frameSize - 1);
            totalZeros = sum(frameGroundTruth == 0);
            % More than half of window -> no gesture
            if totalZeros <= frameSize/2
                data{frameIdx, 2} = gestureName; % label
            end
        end
        position = position + frameSize;
        frameIdx = frameIdx + 1;
    end
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveSampleInDatastore(samples, user, data_store)
    for i = 1:length(samples) 
        frames = samples{i,1};
        class = samples{i,2};
        % The name to save is (user_sample index)
        fileName = strcat(strtrim(user),'_', int2str(i));
        % The folder corresponds to the class 
        savePath = fullfile(data_store, char(class),fileName);
        % Savind data
        data.frames = frames;
        if ~isequal(class,'noGesture')
            data.groundTruth = samples{i,3};
        end
        save(savePath,'data');
    end
end
