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
dataDir = 'EMG_EPN612_Dataset'; %'CEPRA_2019_13_DATASET_FINAL'
trainingDir = 'trainingJSON'; %'training'

%% GET THE USERS DIRECTORIES
trainingPath = fullfile(dataDir, trainingDir);
users = ls(trainingPath);
users = strtrim(string(users(3:length(users),:)));
rng(9); % seed
users = users(randperm(length(users)));
clear data_dir trainingDir

%%                      BORRAR ESTO AL ACABAR SOLO ES PARA HACER PRUEBAS CON PORCIONES
users = users(1:1);

%% THE STRUCTURE OF THE DATA STORE IS DEFINED
labels = {'fist'; 'noGesture'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
trainingDatastore = createDatastore('Datastores/training', labels);
validationDatastore = createDatastore('Datastores/validation', labels);
trainingEvalDatastore = createDatastore('Datastores/trainingSequence', labels);
validationEvalDatastore = createDatastore('Datastores/validationSequence', labels);

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
for i = 1:length(users) % % parfor
    % Get user samples
    [trainingSamples, validationSamples] = getTrainingTestingSamples(trainingPath, users(i));
    % Training data
    transformedSamplesTraining = generateData(trainingSamples);
    saveSampleInDatastore(transformedSamplesTraining, users(i), trainingDatastore);
    saveSampleSeqInDatastore(transformedSamplesTraining, users(i), trainingEvalDatastore);
    % Validation data
    transformedSamplesValidation = generateData(validationSamples);
    saveSampleInDatastore(transformedSamplesValidation, users(i), validationDatastore);
    saveSampleSeqInDatastore(transformedSamplesValidation, users(i), validationEvalDatastore);
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
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        if isequal(gestureName,'noGesture')  
            groundTruth = [];
        else
            groundTruth = sample.groundTruth;
        end
        % Get signal from sample
        signal = getSignal(emg);
        signal = preprocessSignal(signal);
        % Generate spectrograms
        [data] = generateFrames(signal, groundTruth, gestureName);
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
function [data] = generateFrames(signal, groundTruth, gestureName)
    % Creating spectrograms
    [spectrograms, timestamps, params] = generateSpectrograms(signal); % 101 x n x 8
    % Creating frames
    NUMCOLSFRAME = 40;
    numFrames = floor(params.numCols / NUMCOLSFRAME);
    % Allocate space for the results
    data = cell(numFrames,3);
    for i = 1:numFrames
        % start and finish colums in spectrogram
        firstCol = ((i - 1) * NUMCOLSFRAME) + 1;
        lastCol = NUMCOLSFRAME * i;
         % Start and finish points in sample
        firstPoint = timestamps(firstCol) - (params.window/2) + 1;
        lastPoint = timestamps(lastCol) + (params.window/2);
        % Save data
        data{i,1} = spectrograms(:, firstCol:lastCol, :); % datum
        data{i,2} = 'noGesture'; % label
        data{i,3} = floor((firstPoint + lastPoint) / 2); % timepoint
        % Check no gesture and change label
        if ~isequal(gestureName,'noGesture')
            frameGroundTruth = groundTruth(firstPoint:lastPoint);
            totalZeros = sum(frameGroundTruth == 0);
            % More than half of window -> no gesture
            if totalZeros <= (lastPoint - firstPoint)/2 
                data{i, 2} = gestureName; % label
            end
        end    
    end  
end

%% FUNCTION TO GENERATE SPECTROGRAMS
function [spectrograms,timestamps, params] = generateSpectrograms(signal)
    % Spectrogram parameters
    frecuencies = (0:100);
    sampleFrecuency = 200;
    WINDOW = 200;
    OVERLAPPING = 199; %floor(window*0.5);
    % Allocate space for the spectrograms
    numCols = floor((length(signal)-OVERLAPPING)/(WINDOW-OVERLAPPING));
    spectrograms = zeros(101, numCols, 8);
    % Spectrograms generation
    for i = 1:8
        [s,f,t,ps] = spectrogram(signal(:,i), WINDOW, OVERLAPPING, frecuencies, sampleFrecuency, 'yaxis');
        spectrograms(:,:,i) = ps;
    end
    % Get times
    timestamps = round(t * 200);
    % Get the spectrogram real value
    %spectrograms = abs(spectrograms);
    % Get the spectrogram real part
    %spectrograms = real(spectrograms);
    %spectrograms = power(real(spectrograms), 2);
    %spectrograms = log(power(real(spectrograms), 2));
    % Put parameters in structure
    params.numCols = numCols;
    params.window = WINDOW;
    params.overlappimg = OVERLAPPING;
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveSampleInDatastore(samples, user, data_store)
    for i = 1:length(samples) 
        frames = samples{i,1};
        class = samples{i,2};
        % Data in frames
        spectrograms = frames(:,1);
        labels = frames(:,2);
        if isequal(class,'noGesture')
            rng(9); % seed
            gestureIdxs = randperm(length(labels),floor(length(labels)*0.4));
        else
            isGesture = cellfun(@(label) isequal(label, class),labels);
            gestureIdxs = find(isGesture);
        end
        for j = 1:length(gestureIdxs)
            data = spectrograms{gestureIdxs(j), 1};
            fileName = strcat(strtrim(user),'_', int2str(i), '_', ...
                int2str(gestureIdxs(j)), '-', int2str(length(spectrograms)));
            % The folder corresponds to the class 
            savePath = fullfile(data_store, char(class),fileName);
            save(savePath,'data');
        end
    end
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveSampleSeqInDatastore(samples, user, data_store)
    for i = 1:length(samples) 
        frames = samples{i,1};
        class = samples{i,2};
        % The name to save is (user_sample index)
        fileName = strcat(strtrim(user), '_', int2str(i));
        % The folder corresponds to the class 
        savePath = fullfile(data_store, char(class), fileName);
        % Savind data
        data.frames = frames;
        if ~isequal(class,'noGesture')
            data.groundTruth = samples{i,3};
        end
        save(savePath,'data');
    end
end
