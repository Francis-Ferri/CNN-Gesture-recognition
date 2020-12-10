

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

%% AVERAGE GROUND TRUTH IS CHECKED
avg_groundThuth_length = getAvgGroundThuthLength(users,training_path);
% avg_groundThuth_length = 261; %obtenido con todas las muestras

%% THE STRUCTURE OF THE DATA STORE IS DEFINED
labels = {'fist'; 'noGesture';  'open'; 'pinch'; 'waveIn'; 'waveOut'};
training_datastore = create_data_store('Datastores/training_datastore', labels);
validation_datastore = create_data_store('Datastores/validation_datastore', labels);

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
% parfor
for i = 1:length(users)
    [training_samples, validation_samples] = getTrainingTestingSamples(training_path, users(i));
    % Training data
    transformed_samples_training = generate_data(training_samples, avg_groundThuth_length);
    save_sample_data_store(transformed_samples_training, users(i), training_datastore);
    % validation data
    transformed_samples_validation = generate_data(validation_samples, avg_groundThuth_length);
    save_sample_data_store(transformed_samples_validation, users(i), validation_datastore);
end

clear labels i training_samples validation_samples transformed_samples_validation

%% FUNCTION TO FIND THE MAXIMUM SIGNAL GROUNDTHUTH LENGTH
function max_groundThuth_length = getAvgGroundThuthLength(users, path)
    sum_samples = 0;
    num_samples = 0;
    % The highest signal is sought in each user
    for i = 1:length(users)
        [trainingSamples, testingSamples] = getTrainingTestingSamples(path, users(i));
        [sum_samples, num_samples] = getLengthGroundThuth(trainingSamples, sum_samples, num_samples);
        [sum_samples, num_samples] = getLengthGroundThuth(testingSamples, sum_samples, num_samples);
    end
    max_groundThuth_length = round(sum_samples / num_samples);
end

%% GET TRAINING AND TESTING SAMPLES FOR AN USER
function [trainingSamples, testingSamples] = getTrainingTestingSamples(path, user)
    file_path = fullfile(path, user, strcat(user, '.json'));
    json_file = fileread(file_path);
    json_data = jsondecode(json_file);
    trainingSamples = json_data.trainingSamples;
    testingSamples = json_data.testingSamples;
end

%% GET SIGNALS GROUNDTHUTH SUM AND SAMPLE NUMBER
function [sum_samples, num_samples] = getLengthGroundThuth(samples, sum_samples, num_samples)
    fields = fieldnames(samples);
    for i = 1:length(fields)
        gestureName = samples.(fields{i}).gestureName;
        if ~strcmp(gestureName, 'noGesture')
            groundTruthIndex = samples.(fields{i}).groundTruthIndex;
            gesture_duration = groundTruthIndex(2) - groundTruthIndex(1);
            sum_samples = sum_samples + gesture_duration;
            num_samples = num_samples + 1;
        end
    end
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
function transformed_samples = generate_data(samples, avg_groundThuth_length)
    names_samples = fieldnames(samples);
    % Allocate space for the results
    transformed_samples = cell(length(names_samples), 1);
    for i = 1: length(names_samples)
        % Get sample data
        sample = samples.(names_samples{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        % get gesture's boundaries
        if isequal(gestureName,'noGesture')
            % The noGesture signaldosn't have boundaries, so we put an
            % average length
            limits = [100, 100+avg_groundThuth_length];
        else
            limits = sample.groundTruthIndex;
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
        
        chanels = fieldnames(emg); % get the chanels
        % Get just the movement protion of the signal
        num_samples = limits(2)-limits(1)+1;
        signal = zeros(num_samples, length(chanels)); %[num_samples, 8]
        for j = 1:length(chanels)
            signal(:,j) = emg.(chanels{j})(limits(1):limits(2));
        end
        % Spectrogram parameters
        %{
            COnsideraciones espectrogramas
                ventanas de 20, ventanas de 50 ventanas de 100
                No sacar directamente espectrogras, si no el logaritmo de la 
                    transformada de FOuruer al cuadrado como en speech
                    recognition
        %}
        frame_size = 20;
        total_frames = floor((limits(2)-limits(1)) / frame_size);
        frecuencies = 1:1:100;
        sample_frecuency = 200;
        % Generate the frames
        frames = cell(total_frames, 1); % Allocate space
        frame_idx = 1;
        position = 1;
        while frame_idx <= total_frames
            % Allocate space for the spectrograms
            spectrograms = zeros(size(frecuencies,2), size(signal,2)); % 100 x 8
            % calculate the spectrogram for each chanel
            for j = 1:size(signal,2)
                spectrograms(:,j) = spectrogram(signal(position:(position+frame_size), j), frame_size, 0, frecuencies, sample_frecuency, 'yaxis');
            end
            spectrograms = abs(spectrograms);
            frames{frame_idx,1} = spectrograms;
            position = position + frame_size;
            frame_idx = frame_idx+1;
        end
        transformed_samples{i,1} = frames;
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
