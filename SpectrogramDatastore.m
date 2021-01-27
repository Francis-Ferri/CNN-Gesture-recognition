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
        %FrameSize
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
            [labels, numObservations] = createLabels(fds.Files);
            ds.Labels = labels;
            ds.NumClasses = numel(unique(labels));
            % Determine sequence dimension
            filename = ds.Datastore.Files{1};
            sample = load(filename).data;
            ds.SequenceDimension = size(sample);
            % Initialize datastore properties
            ds.MiniBatchSize = 8;
            %ds.FrameSize = 20;
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;
            % shuffle
            ds = balanceGestureSamples(ds);
            ds = shuffle(ds);
        end
        
        function tf = hasdata(ds)
            % Return true if more data is available
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end
        
        function [data,info] = read(ds)            
            % Function to read data
            info = struct;
            miniBatchSize = ds.MiniBatchSize;
            predictors = cell(miniBatchSize, 1);
            responses = cell(miniBatchSize, 1);
            % Data for minibatch size is read
            for i = 1:miniBatchSize
                %data = read(ds.Datastore).data;
                data = read(ds.Datastore);
                predictors{i,1} = data;
                class = ds.Labels(ds.CurrentFileIndex);
                responses{i,1} = class;
                ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            end
            % Data is preprocessed;
            data = preprocessData(ds, predictors, responses);
        end
        
        function data = preprocessData(ds, predictors, responses)
            % Function to preprocess data
            miniBatchSize = ds.MiniBatchSize;
            for i = 1:miniBatchSize
                predictors{i} = predictors{i}.data;
            end
            % Put the data in table form
            data = table(predictors,responses);
        end
        
        function reset(ds)
            % Reset to the start of the data
            reset(ds.Datastore);
            ds.CurrentFileIndex = 1;
            ds.NumObservations = size(ds.Datastore.Files, 1);
        end
        
        function subds = partition(myds,n,ii)
            % Particionate the datastore
            subds = copy(myds);
            subds.Datastore = partition(subds.Datastore,n,ii);
            % Create the new labels
            [labels, numObservations] = createLabels(subds.Datastore.Files);
            subds.Labels = labels;
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
        
        function ds = balanceGestureSamples(ds)
            labels = ds.Labels;
            files = ds.Datastore.Files;
            gestures = categorical(categories(labels));
            catCounts = sort(countcats(labels));
            minNumber = catCounts(1);
            % Allocate space for results
            newLabels = cell(minNumber * length(catCounts),1);
            newFiles = cell(minNumber * length(catCounts),1);
            % Get equal number of samples for each gesture
            for i = 1:length(gestures)
                isGesture = ismember(labels, gestures(i));
                gestureIdxs = find(isGesture);
                for j = 1:minNumber
                    newLabels{((i-1)*minNumber) + j, 1} = char(gestures(i));
                    newFiles{((i-1)*minNumber) + j, 1} = files{gestureIdxs(j)};
                end
            end
            newLabels = categorical(newLabels,categories(gestures));
            % Save the transformed data
            ds.Labels = newLabels;
            ds.NumObservations = length(newLabels);
            ds.Datastore.Files = newFiles;
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
function [labels, numObservations] = createLabels(files)
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

