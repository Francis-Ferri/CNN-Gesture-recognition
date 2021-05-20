%{ 
    LSTM
%}

classdef SpectrogramDatastoreLSTM < matlab.io.Datastore & ...
                                matlab.io.datastore.MiniBatchable & ...
                                matlab.io.datastore.Shuffleable & ...
                                matlab.io.datastore.Partitionable
     
    properties
        Datastore
        Labels
        SequenceLength
        FrameDimensions %FrameSize DataDimensions
        %NumClasses
        MiniBatchSize
        
        
        HELPER
    end
    
    properties(SetAccess = protected)
        NumObservations
    end
    
    properties(Access = private)
        CurrentFileIndex
    end
    
    methods
        
        function ds = SpectrogramDatastoreLSTM(folder)
            
            % Create a file datastore.
            fds = fileDatastore(folder, ...
                'ReadFcn',@Shared.readFile, ...
                'IncludeSubfolders',true);
            ds.Datastore = fds;
            
            % Read labels from folder names
            [labels, numObservations] = Shared.createLabels(fds.Files, true);
            ds.Labels = labels;
            %ds.NumClasses = numel(unique(labels));
            
            % Initialize datastore properties
            ds.MiniBatchSize = 32;
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;
            
            % Shuffle
            ds = shuffle(ds);
            
            % Determine sequence and frame dimensions
            sample = load(ds.Datastore.Files{1}).data;
            ds.FrameDimensions = size(sample.sequenceData{1,1});
            if isequal(Shared.NOGESTURE_FILL, 'all')
                ds.SequenceLength = length(sample.sequenceData);
            elseif isequal(Shared.NOGESTURE_FILL, 'some')
                ds.SequenceLength = 'variable';
                ds = order(ds);
            end
            
        end
        
        function tf = hasdata(ds)
            % Return true if more data is available
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end
        
        function [data,info] = read(ds)            
            % Function to read data
            miniBatchSize = ds.MiniBatchSize;
            [sequencesData, responses, groundTruths]  = deal(cell(miniBatchSize, 1));
            % Data for minibatch size is read
            for i = 1:miniBatchSize
                %data = read(ds.Datastore).data;
                content = read(ds.Datastore);
                sequencesData{i,1} = content.sequenceData;
                class = ds.Labels(ds.CurrentFileIndex);
                responses{i,1} = class;
                if ~isequal(class, 'noGesture')
                    groundTruths{i, 1} = content.groundTruth;
                end
                ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            end
            
            % Data is preprocessed;
            [data, timestamps] = preprocessData(ds, sequencesData);
            % Set information
            info = struct('responses', responses, 'timestamps', timestamps, 'groundTruths', groundTruths);
        end
        
        
        
        function [data, timestamps] = preprocessData(ds, sequencesData)
            miniBatchSize = ds.MiniBatchSize;
            [sequences, labelsSequences, timestamps]  = deal(cell(miniBatchSize, 1));
            
            if isequal(Shared.NOGESTURE_FILL, 'all')
                % TODO ESTA IGUAL ASI QUE SOLO MANDAR
                % Set sequence dimensions
                numFrames = ds.SequenceLength;
                sequenceDimensions = [ds.FrameDimensions, numFrames];
                
                parfor i = 1:miniBatchSize
                    % Allocate space for new data with filling
                    newSequence = zeros(sequenceDimensions);
                    newLabels  = cell(1, numFrames);
                    newTimestamps = cell(1, numFrames);
                    sequenceData = sequencesData{i, 1};
                    for j = 1:numFrames
                        newSequence(:,:, :,j) = sequenceData{j,1};
                        newLabels{1, j} = sequenceData{j,2};
                        newTimestamps{1, j} = sequenceData{j,3};
                    end
                    sequences{i,1} = newSequence;
                    labelsSequences{i,1} = categorical(newLabels, Shared.setNoGestureUse(true));
                    timestamps{i,1} = newTimestamps;  
                end
                
            elseif isequal(Shared.NOGESTURE_FILL, 'some')
               
                % Get max frame number in sequences
                sequencesLengths = cellfun(@(sequence) length(sequence), sequencesData);
                maxLength = max(sequencesLengths);
            
                % Set sequence dimensions
                sequenceDimensions = [ds.FrameDimensions, maxLength];
        

                if Shared.CONSIDER_PREVIOUS
                    numRows = ds.FrameDimensions(1);
                    numCols = ds.FrameDimensions(2);
                    strideSequence = numCols - round((1 - (Shared.WINDOW_STEP_LSTM ... 
                        / Shared.FRAME_WINDOW)) * numCols);
                end
                parfor i = 1:miniBatchSize
                   
                    
                    % Allocate space for new data with filling 
                    newLabels  = cell(1, maxLength);
                    newTimestamps = cell(1, maxLength);
                    newSequence = zeros(sequenceDimensions);
                    
                    
                    % Initialize sequence space
                    if isequal(Shared.SEQUENCE_INIT, 'noGesture')
                        % ESTOY RELLENANDO DE LOS VALORES DE LA SEÑAL CUANDO DEBERIA RELLENAR DE LOS FRAMES
                        %newSequence = (2 * Shared.noGestureStd) * rand(sequenceDimensions) ... 
                            %+ (Shared.noGestureMean - Shared.noGestureStd);
                        newSequence = zeros(sequenceDimensions);
                    elseif isequal(Shared.SEQUENCE_INIT, 'zeros')
                        newSequence = zeros(sequenceDimensions);
                    end
                    
                    % Put original data 
                    sequenceData = sequencesData{i, 1};
                    numFrames = length(sequenceData);
                    for j = 1:numFrames
                        newSequence(:,:, :,j) = sequenceData{j,1};
                        newLabels{1, j} = sequenceData{j,2};
                        newTimestamps{1, j} = sequenceData{j,3};
                    end
                    
                    % Put filling at the end to match the max length
                    lastTimestamps = sequenceData{numFrames,3};
                    lastFrame = sequenceData{numFrames,1};
                    
                    % EN ESTE BLOQUE ESTA EL ERROR
                    for j = 1:(maxLength - numFrames)
                        if Shared.CONSIDER_PREVIOUS
                            % HAY QUE DETERMINAR CUANTO SE DESPLAZO EL
                            % FRAME EN COLUMNAS
                            frameRemain = lastFrame(:, 1+strideSequence:numCols, :);
                            % DEBE DE SER UN REELENO QUE SIMULE NG
                            filling = zeros(numRows, strideSequence, Shared.numChannels);
                            newFrame = [frameRemain, filling];
                            newSequence(:,:, :,j) = newFrame;
                            lastFrame = newFrame;
                        end
                        newLabels{1, numFrames + j} = 'noGesture';
                        newTimestamps{1, numFrames + j} = lastTimestamps + j * Shared.WINDOW_STEP_LSTM;
                    end
                    
                    sequences{i,1} = newSequence;
                    labelsSequences{i,1} = categorical(newLabels, Shared.setNoGestureUse(true));
                    timestamps{i,1} = newTimestamps;
                end
            end
            % Put the data in table form
            data = table(sequences, labelsSequences);
        end
        
        function reset(ds)
            % Reset to the start of the data
            reset(ds.Datastore);
            ds.CurrentFileIndex = 1;
            ds.NumObservations = size(ds.Datastore.Files, 1);
        end
        
        function [ds1, ds2] = partition(ds, percentage)
            % Create copys to set the result
            ds1 = copy(ds); ds2 = copy(ds);
            
            % Get the limit of the new division
            numObservations = ds.NumObservations;
            numClassSamples = floor(numObservations / ds.NumClasses);
            limitOfSamples = floor(numClassSamples*percentage);
            
            % Match the specidfied number of samples and order them
            dsNew = matchSampleNumberInOrder(ds, numClassSamples);
            dsLabels = dsNew.Labels;
            dsFiles = dsNew.Datastore.Files;
            
            % Create cell to set the new labels and files
            ds1Labels = {}; ds1Files = {}; ds2Labels = {}; ds2Files = {};
            
            % Divide data per gesture
            parfor i = 1:ds.NumClasses
                
                % Calculate the samples per partition
                labels = dsLabels;
                files = dsFiles;
                start = ((i-1) * numClassSamples) + 1;
                limit = start + limitOfSamples;
                last = i * numClassSamples;
                
                % Set new number of files and labels
                ds1Labels = [ds1Labels; labels(start:limit-1, 1)];
                ds1Files = [ds1Files; files(start:limit-1, 1)];
                ds2Labels = [ds2Labels; labels(limit:last, 1)];
                ds2Files = [ds2Files; files(limit:last, 1)];
            end
            
            % Set the data to new datastores
            ds1 = prepareNewDatastore(ds1, ds1Labels, ds1Files);
            ds2 = prepareNewDatastore(ds2, ds2Labels, ds2Files);
        end
  
        function dsNew = shuffle(ds)
            % Create a copy of datastore
            dsNew = copy(ds);
            fds = dsNew.Datastore;
            % Shuffle tthe filedatastore
            [fds, idxs] = Shared.shuffle(fds);
            % Save new order
            dsNew.Datastore = fds;
            dsNew.Labels = dsNew.Labels(idxs);
        end
        
        function dsNew = balanceGestureSamples(ds)
            % Calcule the class with less samples
            labels = ds.Labels;
            catCounts = sort(countcats(labels));
            minNumber = catCounts(1);
            % get a new balanced datastore and shuflle it
            dsNew = matchSampleNumberInOrder(ds, minNumber);
            dsNew = shuffle(dsNew);
        end
        
        function dsNew = setDataAmount(ds, percentage)
            if percentage == 1 
                dsNew = ds;
            else
                % Get the limit of the new division
                numObservations = ds.NumObservations;
                newLimit = floor(numObservations * percentage);
                numClassSamples = floor(newLimit/ds.NumClasses);
                % Set the new number of files
                dsNew = matchSampleNumberInOrder(ds, numClassSamples);
                dsNew = shuffle(dsNew);
            end
        end
        
        function ds = order(ds)
            % Order the datasroes by number of frames
            numObservations = ds.NumObservations;
            sequenceLengths = zeros(numObservations, 1);
            files =  ds.Datastore.Files;
            parfor i=1:numObservations
                filename = files{i};
                sequenceData = load(filename).data.sequenceData;
                sequenceLengths(i) = size(sequenceData, 1);
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

%% FUNCTION TO MATCH THE NUMBER OF SAMPLES OF EACH GESTURE
function ds = matchSampleNumberInOrder(ds, repetitions)
    labels = ds.Labels;
    gesturefiles = ds.Datastore.Files;
    gestures = categorical(categories(labels));
    
    % Allocate space for results
    newLabels = {}; newFiles = {};
    
    % Get equal number of samples for each gesture
    parfor i = 1:length(gestures)
        files = gesturefiles;
        gestureLabels = cell(repetitions, 1);
        gestureFiles = cell(repetitions, 1);
        
        % Put 1s where is the gesture and 0s where is not
        isGesture = ismember(labels, gestures(i));
        % Get indexes of ones
        gestureIdxs = find(isGesture);
        
        % Save data until the limit (repetitions)
        for j = 1:repetitions
            gestureLabels{j, 1} = char(gestures(i));
            gestureFiles{j, 1} = files{gestureIdxs(j)};
        end
        
        % Concatenate the labels and files
        newLabels = [newLabels; gestureLabels];
        newFiles = [newFiles; gestureFiles];
    end
    % Make data categorical
    newLabels = categorical(newLabels,categories(gestures));
    
    % Save the transformed data
    ds.Labels = newLabels;
    ds.NumObservations = length(newLabels);
    ds.Datastore.Files = newFiles;
end

%% FUNCTION TO PREPARE A NEW DATASTORE
function ds = prepareNewDatastore(ds, dsLabels, dsFiles)
    % Set the data to new datastores
    ds.NumObservations = length(dsLabels);
    ds.Labels = dsLabels;
    ds.Datastore.Files = dsFiles;
    % Shufle datastore
    ds = shuffle(ds);
end
