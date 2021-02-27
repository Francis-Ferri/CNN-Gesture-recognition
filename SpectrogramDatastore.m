%{
CUSTOM MINIBATCH SPECTROGRAM DATASTORE
    1. A datastore of type fileDatastore is generated
    2. Spectrogram files are read
    3. A Label field is created in the Datastore
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
        DataDimensions
        MiniBatchSize
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
            ds.DataDimensions = size(sample);
            % Initialize datastore properties
            ds.MiniBatchSize = 32;
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
            parfor i = 1:miniBatchSize
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
        
        function [ds1, ds2] = partition(ds, percentage)
            ds1 = copy(ds);
            ds2 = copy(ds);
            % Get the limit of the new division
            numObservations = ds.NumObservations;
            numClassSamples = floor(numObservations / ds.NumClasses);
            limitOfSamples = floor(numClassSamples*percentage);
            dsNew = matchSampleNumber(ds, numClassSamples);
            labels = dsNew.Labels;
            files = dsNew.Datastore.Files;
            
            ds1Labels = {};
            ds1Files = {};
            ds2Labels = {};
            ds2Files = {};
            
            for i = 0:ds.NumClasses-1
                start = (i * numClassSamples) + 1;
                limit = start + limitOfSamples;
                last = (i+1) * numClassSamples;
                ds1Labels = [ds1Labels; labels(start:limit-1, 1)];
                ds1Files = [ds1Files; files(start:limit-1, 1)];
                ds2Labels = [ds2Labels; labels(limit:last, 1)];
                ds2Files = [ds2Files; files(limit:last, 1)];
                
            end
            % Create the first datastore
            ds1.NumObservations = length(ds1Labels);
            ds2.NumObservations = length(ds2Labels);
            
            ds1.Labels = ds1Labels;
            ds1.Datastore.Files = ds1Files;
            % Create the second datastore
            ds2.Labels = ds2Labels;
            ds2.Datastore.Files = ds2Files;
            ds1 = shuffle(ds1);
            ds2 = shuffle(ds2);
        end
                
        function dsNew = shuffle(ds)
            % Create a copy of datastore
            dsNew = copy(ds);
            fds = dsNew.Datastore;
            % Shuffle files and their corresponding labels
            numObservations = dsNew.NumObservations;
            rng(9); % seed
            idx = randperm(numObservations);
            fds.Files = fds.Files(idx);
            dsNew.Labels = dsNew.Labels(idx);
        end
        
        function dsNew = balanceGestureSamples(ds)
            labels = ds.Labels;
            catCounts = sort(countcats(labels));
            minNumber = catCounts(1);
            dsNew = matchSampleNumber(ds, minNumber);
            dsNew = shuffle(dsNew);
        end
        
        function dsNew = setDataAmount(ds, percentage)
            % Get the limit of the new division
            numObservations = ds.NumObservations;
            newLimit = floor(numObservations * percentage);
            numClassSamples = floor(newLimit/ds.NumClasses);
            % Set the new number of files
            dsNew = matchSampleNumber(ds, numClassSamples);
            dsNew = shuffle(dsNew);
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

function ds = matchSampleNumber(ds, repetitions)
    labels = ds.Labels;
    files = ds.Datastore.Files;
    gestures = categorical(categories(labels));
    % Allocate space for results
    newLabels = {};
    newFiles = {};
    % Get equal number of samples for each gesture
    parfor i = 1:length(gestures)
        gestureLabels = cell(repetitions, 1);
        gestureFiles = cell(repetitions, 1);
        % Put 1s where is the gesture and 0s where is not
        isGesture = ismember(labels, gestures(i));
        % Get indexes of ones
        gestureIdxs = find(isGesture);
        for j = 1:repetitions
            gestureLabels{j, 1} = char(gestures(i));
            gestureFiles{j, 1} = files{gestureIdxs(j)};
        end
        newLabels = [newLabels; gestureLabels];
        newFiles = [newFiles; gestureFiles];
    end
    newLabels = categorical(newLabels,categories(gestures));
    % Save the transformed data
    ds.Labels = newLabels;
    ds.NumObservations = length(newLabels);
    ds.Datastore.Files = newFiles;
end
