

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
data_dir = 'EMG_EPN612_Dataset';
training_dir = 'trainingJSON';

%% GET THE TRAINING DIRECTORIES
training_path = fullfile(data_dir, training_dir);
users = ls(training_path);
users = strtrim(string(users(3:length(users),:)));
rng(9); % seed
users = users(randperm(length(users)));
clear data_dir training_dir

%%                      BORRAR ESTO AL ACABAR SOLO ES PARA HACER PRUEBAS CON PORCIONES
users = users(1:5);

%% THE STRUCTURE OF THE DATA STORE IS DEFINED
labels = {'fist'; 'noGesture';  'open'; 'pinch'; 'waveIn'; 'waveOut'};
training_datastore = create_data_store('Datastores/training_datastore', labels);
validation_datastore = create_data_store('Datastores/validation_datastore', labels);

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
% parfor
for i = 1:length(users)
    [training_samples, validation_samples] = getTrainingTestingSamples(training_path, users(i));
    % Training data
    transformed_samples_training = generate_data(training_samples);
    save_sample_data_store(transformed_samples_training, users(i), training_datastore);
    % validation data
    transformed_samples_validation = generate_data(validation_samples);
    save_sample_data_store(transformed_samples_validation, users(i), validation_datastore);
end

clear labels i training_samples validation_samples transformed_samples_validation


%% GET TRAINING AND TESTING SAMPLES FOR AN USER
function [trainingSamples, testingSamples] = getTrainingTestingSamples(path, user)
    file_path = fullfile(path, user, strcat(user, '.json'));
    json_file = fileread(file_path);
    json_data = jsondecode(json_file);
    trainingSamples = json_data.trainingSamples;
    testingSamples = json_data.testingSamples;
end

%% FUNCTION TO CREATE DATASTORE
function data_store = create_data_store(data_store, labels)
    mkdir(data_store);
    % The folder is created for each class
    for i = 1:length(labels)
        mkdir(fullfile(data_store, char(labels(i))));
    end
end

%% CREAR DATOS DE ESPECTROGRAMAS
function transformed_samples = generate_data(samples)
    %{
        COnsideraciones espectrogramas
            ventanas de 20, ventanas de 50 ventanas de 100
            No sacar directamente espectrogras, si no el logaritmo de la 
                transformada de FOuruer al cuadrado como en speech
                recognition
    %}
    % Spectrogram parameters
    frecuencies = 1:1:100;
    sample_frecuency = 200;
    % Frame size
    FRAME_SIZE =  20;
    % Get sample keys
    names_samples = fieldnames(samples);
    % Allocate space for the results
    transformed_samples = cell(length(names_samples), 1);
    for i = 1: length(names_samples)
        % Get sample data
        sample = samples.(names_samples{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        % Get signal
        chanels = fieldnames(emg); % get chanels
        signal = zeros(length(emg.(chanels{1})), length(chanels));
        for j = 1:length(chanels)
            signal(:,j) = emg.(chanels{j});
        end
         %{ 
          Consideraciones preprocesado
            filtro pasa bajos de 57 Hz - 50 - 10
            Myo -1 +1
            Normaliza por usuario
            Normaliza toda la señal Por gesto y reescalas maxima amplitud para la señal de Wave in debe eqivaler a + 1 y la minima a --1
                Promedio de los maximo  de los minimos y señales de testeo
            Maximo y minimo

        %}
        % Number of frames
        total_frames = floor(length(signal) / FRAME_SIZE);
        % Generate the frames
        data = cell(total_frames,2);
        frame_idx = 1;
        position = 1;
        spectrograms = zeros(size(frecuencies,2), size(signal,2)); % 100 x 8
        while frame_idx <= total_frames
            for j = 1:size(signal,2)
                spectrograms(:,j) = spectrogram(signal(position:(position+FRAME_SIZE-1), j), FRAME_SIZE, 0, frecuencies, sample_frecuency, 'yaxis');
            end
            data{frame_idx,1} = abs(spectrograms); % datum
            data{frame_idx,2} = gestureName; % label
            if ~isequal(gestureName,'noGesture')
                % Check no gesture 
                groundTruth = sample.groundTruth;
                frame_groundTruth = groundTruth(position : position + FRAME_SIZE - 1);
                ceros = sum(frame_groundTruth==0);
                % More than half of window -> no gesture
                if ceros > FRAME_SIZE/2
                    data{frame_idx,2} = 'noGesture'; % label
                end
            end
            position = position + FRAME_SIZE;
            frame_idx = frame_idx+1;
        end
        % Adding the transformed data
        transformed_samples{i,1} = data;
        transformed_samples{i,2} = gestureName;
    end
end

%% FUNCTION TO SAVE SPECTROGRAMS IN DATASTORE
function save_sample_data_store(samples, user, data_store)
    for i = 1:length(samples) 
        frames = samples{i,1};
        class = samples{i,2};
        % Se guada usuario-tipo-indice
        nombre_guardar = strcat(strtrim(user),'_', int2str(i));
        save_path = fullfile(data_store, char(class),nombre_guardar);
        save(save_path,'frames');
    end
end
