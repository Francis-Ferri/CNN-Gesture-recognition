%{
CUSTOM MINIBATCH SPECTROGRAM DATASTORE
    1. A Datastore type fileDatastore is generated
    2. Spectrogram files are read
    3. A Labels field is created in the Datastore
    4. Add support to minibatch
    5. Add support to shuffle
    6. Add support to partitions
%}

classdef SpectrogramDatastore < matlab.io.Datastore & ...
                                matlab.io.datastore.MiniBatchable & ...
                                matlab.io.datastore.Shuffleable & ...
                                matlab.io.datastore.Partitionable
    properties
        Datastore
        Labels
        NumClasses
        SequenceDimension
        MiniBatchSize
        FrameSize
    end
    
    properties(SetAccess = protected)
        NumObservations
    end
    
    properties(Access = private)
        % This property is inherited from Datastore
        CurrentFileIndex
    end
    
    methods
        function ds = SpectrogramDatastore(folder)
            % Create a file datastore.
            fds = fileDatastore(folder, ...
                'ReadFcn',@readSequence, ...
                'IncludeSubfolders',true);
            ds.Datastore = fds;
            % Read labels from folder names
            labels = createLabels(fds.Files);
            ds.Labels = labels;
            ds.NumClasses = numel(unique(labels));
            numObservations = numel(fds.Files);
            % Determine sequence dimension
            filename = ds.Datastore.Files{1};
            sample = load(filename).data.frames{1};
            ds.SequenceDimension = [size(sample,1), size(sample,2), 1];
            % Initialize datastore properties
            ds.MiniBatchSize = 8;
            ds.FrameSize = 20;
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;
            % shuffle
            ds = shuffle(ds);
            % order
            ds = order(ds);
        end
        
        function tf = hasdata(ds)
            % Return true if more data is available
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end
        
        function [data,info] = read(ds)            
            % Funcion para leer datos
            miniBatchSize = ds.MiniBatchSize;
            predictors = cell(miniBatchSize, 1);
            responses = cell(miniBatchSize, 1);
            % Se leen los datos para el tamaño del minibatch
            for i = 1:miniBatchSize
                data = read(ds.Datastore).data;
                predictors{i,1} = data.frames;
                class = ds.Labels(ds.CurrentFileIndex);
                responses{i,1} = class;
                ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            end
            % Se preprocesan los datos
            info.labels = responses;
            [data, time_point_sequences, groundTruths] = preprocessData(ds, predictors, responses);
            info.time_point_sequences = time_point_sequences;
            info.groundTruths = groundTruths;
        end
        
        function [data, timepoint_sequences, groundTruths] = preprocessData(ds,predictors, responses)
            miniBatchSize = ds.MiniBatchSize;
            frameSize = ds.FrameSize;
            sequences = cell(miniBatchSize, 1);
            label_sequences = cell(miniBatchSize, 1);
            timepoint_sequences = cell(miniBatchSize, 1);
            groundTruths = cell(miniBatchSize, 1);
            % Calculate maximum length of sequences
            predictors_lengths = cellfun(@(predictor) length(predictor), predictors);
            max_length = max(predictors_lengths);
            % Creating data
            parfor i = 1:miniBatchSize
                num_frames = length(predictors{i});
                sequence = zeros(size(predictors{i}{1},1), size(predictors{i}{1},2), 1, max_length);
                sequence_labels  = cell(1, max_length);
                sequence_timepoints = zeros(1, max_length);
                groundTruth = zeros(1, max_length);
                for j = 1:num_frames
                    sequence(:,:,j) = predictors{i}{j,1};
                    sequence_labels{1, j} = predictors{i}{j,2};
                    sequence_timepoints(j) = predictors{i}{j,3};
                end
                if ~isequal(responses{i},'noGesture')
                    for j = 1:num_frames
                        groundTruth(j) = predictors{i}{j,4};
                    end
                end
                timepoint = predictors{i}{j,3};
                for j = 1:(max_length - num_frames)
                    sequence_labels{1, num_frames + j} = 'noGesture';
                    timepoint = timepoint + frameSize;
                    sequence_timepoints(num_frames + j) = timepoint;
                end
                if ~isequal(responses{i},'noGesture')
                    for j = 1:(max_length - num_frames)
                        groundTruth(num_frames + j) = 0;
                    end
                end
                sequences{i,1} = sequence;
                label_sequences{i,1} = categorical(sequence_labels, {'fist', 'noGesture', 'open', ... 
                'pinch','waveIn', 'waveOut'});
                timepoint_sequences{i,1} = sequence_timepoints;
                if ~isequal(responses{i},'noGesture')
                    groundTruths{i,1} = groundTruth;
                end
            end
            % En este caso enviams los datos en forma de tabla
            data = table(sequences,label_sequences);
        end
        
        function reset(ds)
            % Reset to the start of the data
            reset(ds.Datastore);
            ds.CurrentFileIndex = 1;
            ds.NumObservations = size(ds.Datastore.Files, 1);
        end
        
        function subds = partition(myds,n,ii)
            subds = copy(myds);
            subds.Datastore = partition(subds.Datastore,n,ii);
            numObservations = numel(subds.Datastore.Files);
            % Create the new labels
            subds.Labels = createLabels(subds.Datastore.Files);
            subds.NumObservations = numObservations;
            reset(subds);
            reset(myds);
        end
                
        function dsNew = shuffle(ds)
            % Create a copy of datastore
            dsNew = copy(ds);
            dsNew.Datastore = copy(ds.Datastore);
            fds = dsNew.Datastore;
            % Shuffle files and their corresponding labels
            numObservations = dsNew.NumObservations;
            rng(9); % seed
            idx = randperm(numObservations);
            fds.Files = fds.Files(idx);
            dsNew.Labels = dsNew.Labels(idx);
        end
        
        function ds = order(ds)
            numObservations = numel(ds.Labels);
            sequenceLengths = zeros(numObservations, 1);
            files =  ds.Datastore.Files;
            parfor i=1:numObservations
                filename = files{i};
                data = load(filename).data.frames;
                sequenceLengths(i) = size(data,1);
            end
            [~,idx] = sort(sequenceLengths);
            ds.Datastore.Files = ds.Datastore.Files(idx);
            ds.Labels = ds.Labels(idx);
        end
        
    end
    
    methods (Access = protected)
        function dscopy = copyElement(ds)
            dscopy = copyElement@matlab.mixin.Copyable(ds);
            dscopy.Datastore = copy(ds.Datastore);
        end
        
        function n = maxpartitions(myds) 
            n = maxpartitions(myds.FileSet); 
        end  
    end
    
    methods (Hidden = true)
        function frac = progress(ds)
            % Determine percentage of data read from datastore
            frac = (ds.CurrentFileIndex - 1) / ds.NumObservations;
        end
    end
    
end

%% FUNCION PARA CREAR ETIQUETAS
function labels = createLabels(files)
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
    labels = categorical(labels,{'fist', 'noGesture', 'open', ... 
                'pinch','waveIn', 'waveOut'});
end

function data = readSequence(filename)
    data = load(filename);
end

