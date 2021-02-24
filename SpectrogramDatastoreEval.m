%{
CUSTOM MINIBATCH SPECTROGRAM DATASTORE
    1. A Datastore type fileDatastore is generated
    2. Spectrogram files are read
    3. A Labels field is created in the Datastore
    4. Add support to minibatch
    5. Add support to shuffle
    6. Add support to partitions
%}

classdef SpectrogramDatastoreEval < matlab.io.Datastore & ...
                                matlab.io.datastore.MiniBatchable & ...
                                matlab.io.datastore.Shuffleable & ...
                                matlab.io.datastore.Partitionable
    properties
        Datastore
        Labels
        NumClasses
        DataDimensions
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
        function ds = SpectrogramDatastoreEval(folder)
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
            ds.DataDimensions = size(sample);
            % Initialize datastore properties
            ds.MiniBatchSize = 8;
            ds.FrameSize = size(sample, 2);
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;
            % shuffle
            ds = shuffle(ds);
        end
        
        function tf = hasdata(ds)
            % Return true if more data is available
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end
        
        function [data,info] = read(ds)            
            % Function to read data
            miniBatchSize = ds.MiniBatchSize;
            predictors = cell(miniBatchSize, 1);
            responses = cell(miniBatchSize, 1);
            groundTruths = cell(miniBatchSize, 1);
            % Data for minibatch size is read
            for i = 1:miniBatchSize
                data = read(ds.Datastore).data;
                predictors{i,1} = data.frames;
                class = ds.Labels(ds.CurrentFileIndex);
                responses{i,1} = class;
                if ~isequal(class,'noGesture')
                    groundTruths{i,1} = data.groundTruth;
                end
                ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            end
            % Data is preprocessed
            info.labels = responses;
            [data, timePointSequences] = preprocessData(ds, predictors);
            info.timePointSequences = timePointSequences;
            info.groundTruths = groundTruths;
        end
        
        function [data, timepointSequences] = preprocessData(ds,predictors)
            % Function to preprocess data
            miniBatchSize = ds.MiniBatchSize;
            sequences = cell(miniBatchSize, 1);
            labelSequences = cell(miniBatchSize, 1);
            timepointSequences = cell(miniBatchSize, 1);
            % Create data
            parfor i = 1:miniBatchSize
                numFrames = length(predictors{i});
                sequence = cell(numFrames, 1);
                sequenceLabels  = cell(1, numFrames);
                sequenceTimepoints = zeros(1, numFrames);
                for j = 1:numFrames
                    sequence{j, 1} = predictors{i}{j,1};
                    sequenceLabels{1, j} = predictors{i}{j,2};
                    sequenceTimepoints(j) = predictors{i}{j,3};
                end
                sequences{i,1} = sequence;
                labelSequences{i,1} = categorical(sequenceLabels, {'fist', 'noGesture', 'open', ... 
                'pinch','waveIn', 'waveOut'});
                timepointSequences{i,1} = sequenceTimepoints;
            end
            % Put the data in table form
            data = table(sequences,labelSequences);
        end
        
        function reset(ds)
            % Reset to the start of the data
            reset(ds.Datastore);
            ds.CurrentFileIndex = 1;
            ds.NumObservations = size(ds.Datastore.Files, 1);
        end
        
        function [ds1, ds2] = partition(ds, percentage)
            % Get the limit of the new division
            numObservations = ds.NumObservations;
            newLimit = floor(numObservations * percentage);
            % Create the first datastore
            ds1 = setNumberFiles(ds, 1, newLimit);
            % Create the second datastore
            ds2 = setNumberFiles(ds, newLimit+1, numObservations);
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
    % Load a Matlab file
    data = load(filename);
end

function dsNew = setNumberFiles(ds, first, last)
    % Create the first datastore
    dsNew = copy(ds);
    %dsNew.Datastore = copy(ds.Datastore);
    fds = dsNew.Datastore;
    fds.Files = fds.Files(first:last);
    dsNew.Labels = dsNew.Labels(first:last);
    reset(dsNew);
end
