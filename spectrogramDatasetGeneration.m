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
data_dir = 'EMG_EPN612_Dataset';
training_dir = 'trainingJSON';

%% GET THE TRAINING DIRECTORIES
training_path = fullfile(data_dir, training_dir);
users = ls(training_path);
users = strtrim(string(users(3:length(users),:)));
users = users(randperm(length(users)));
clear data_dir training_dir

%%                      BORRAR ESTO AL ACABAR SOLO ES PARA HACER PRUEBAS CON PORCIONES
users = users(1:3);

%% THE STRUCTURE OF THE DATA STORE IS DEFINED
labels = {'fist'; 'noGesture'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
training_datastore = createDatastore('Datastores/training_datastore', labels);
validation_datastore = createDatastore('Datastores/validation_datastore', labels);

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
for i = 1:length(users) % % parfor
    % Get user samples
    [training_samples, validation_samples] = getTrainingTestingSamples(training_path, users(i));
    % Training data
    transformed_samples_training = generateData(training_samples);
    saveSampleInDatastore(transformed_samples_training, users(i), training_datastore);
    % validation data
    transformed_samples_validation = generateData(validation_samples);
    saveSampleInDatastore(transformed_samples_validation, users(i), validation_datastore);
end
clear labels i training_samples validation_samples transformed_samples_validation

%% GET TRAINING AND TESTING SAMPLES FOR AN USER
function [trainingSamples, testingSamples] = getTrainingTestingSamples(path, user)
    file_path = fullfile(path, user, strcat(user, '.json'));
    json_file = fileread(file_path);
    json_data = jsondecode(json_file);
    % Extract samples
    trainingSamples = json_data.trainingSamples;
    testingSamples = json_data.testingSamples;
end

%% FUNCTION TO CREATE DATASTORE
function data_store = createDatastore(data_store, labels)
    mkdir(data_store);
    % One folder is created for each class
    for i = 1:length(labels)
        mkdir(fullfile(data_store, char(labels(i))));
    end
end

%% CREAR DATOS DE ESPECTROGRAMAS
function transformed_samples = generateData(samples)
    % Get sample keys
    samples_keys = fieldnames(samples);
    % Allocate space for the results
    transformed_samples = cell(length(samples_keys), 2);
    for i = 1: length(samples_keys)
        % Get sample
        sample = samples.(samples_keys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        % Get signal from sample
        signal = getSignal(emg);
        signal = preprocessSignal(signal);
        % Generate spectrograms
        data = generateSpectrograms(signal, sample, gestureName);
        % Adding the transformed data
        transformed_samples{i,1} = data;
        transformed_samples{i,2} = gestureName;
    end
end

%% FUNCTION TO GET THE EMG SIGNAL
function signal = getSignal(emg)
    chanels = fieldnames(emg); % get chanels
    signal = zeros(length(emg.(chanels{1})), length(chanels)); % Ex: 1000 x 8
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
function data = generateSpectrograms(signal, sample, gestureName)
    %{
        Consideraciones espectrogramas
            ventanas de 20, ventanas de 50 ventanas de 100
            No sacar directamente espectrogras, si no el logaritmo de la 
            transformada de FOurier al cuadrado como en speech recognition
    %}
    FRAME_SIZE = 20;
    % Spectrogram parameters
    frecuencies = 1:1:100;
    sample_frecuency = 200;
    % Number of frames
    total_frames = floor(length(signal) / FRAME_SIZE);
    % Allocate space for the results
    data = cell(total_frames,2);
    % Initialize windowing parameters
    frame_idx = 1;
    position = 1;
    % Allocate space for the spectrograms
    spectrograms = zeros(size(frecuencies,2), size(signal,2)); % 100 x 8
    while frame_idx <= total_frames
        % Spectrogram for each channel
        for j = 1:size(signal,2)
            spectrograms(:,j) = spectrogram(signal(position:(position + FRAME_SIZE - 1), j), ...
                FRAME_SIZE, 0, frecuencies, sample_frecuency, 'yaxis');
        end
        data{frame_idx,1} = abs(spectrograms); % datum
        data{frame_idx,2} = gestureName; % label
        % Check no gesture and change label
        if ~isequal(gestureName,'noGesture')
            groundTruth = sample.groundTruth;
            frame_groundTruth = groundTruth(position : position + FRAME_SIZE - 1);
            total_zeros = sum(frame_groundTruth == 0);
            % More than half of window -> no gesture
            if total_zeros > FRAME_SIZE/2
                data{frame_idx,2} = 'noGesture'; % label
            end
        end
        position = position + FRAME_SIZE;
        frame_idx = frame_idx + 1;
    end
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function saveSampleInDatastore(samples, user, data_store)
    for i = 1:length(samples) 
        frames = samples{i,1};
        class = samples{i,2};
        % The name to save is (user_sample index)
        file_name = strcat(strtrim(user),'_', int2str(i));
        % The folder corresponds to the class 
        save_path = fullfile(data_store, char(class),file_name);
        save(save_path,'frames');
    end
end
