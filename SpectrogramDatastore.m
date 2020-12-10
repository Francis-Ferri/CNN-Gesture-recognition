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
            % Construct a MySequenceDatastore object

            % Create a file datastore. The readSequence function is
            % defined following the class definition.
            fds = fileDatastore(folder, ...
                'ReadFcn',@readSequence, ...
                'IncludeSubfolders',true);
            ds.Datastore = fds;
            % Read labels from folder names
            labels = createLabels(fds.Files);
            labels = categorical(labels);
            ds.Labels = setcats(labels,{'fist', 'noGesture', 'open', ... 
                'pinch','waveIn', 'waveOut'});
            ds.NumClasses = numel(unique(labels));
            numObservations = numel(fds.Files);
            % Determine sequence dimension. When you define the LSTM
            % network architecture, you can use this property to
            % specify the input size of the sequenceInputLayer.
            ds.SequenceDimension = [100,8];
            
            % Initialize datastore properties.
            ds.MiniBatchSize = 8;
            ds.NumObservations = numObservations;
            ds.CurrentFileIndex = 1;
        end
        
        function tf = hasdata(ds)
            % Return true if more data is available
            tf = ds.CurrentFileIndex + ds.MiniBatchSize - 1 ...
                <= ds.NumObservations;
        end
        
        function [data,info] = read(ds)            
            % Funcion para leer datos
            miniBatchSize = ds.MiniBatchSize;
            info = struct;
            predictors = cell(miniBatchSize, 1);
            responses = cell(miniBatchSize, 1);
            % Se leen los datos para el tamaño del minibatch
            for i = 1:miniBatchSize
                predictors{i,1} = read(ds.Datastore).frames;
                responses{i,1} = ds.Labels(ds.CurrentFileIndex);
                ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            end
            % Se preprocesan los datos
            data = preprocessData(ds,predictors,responses);
        end
        
        function data = preprocessData(ds,samples,responses)
            miniBatchSize = ds.MiniBatchSize;
            predictors = {};
            labels = {};
            idx = 1;
            for i = 1:miniBatchSize
                for j =  1:size(samples{i},1)
                    predictors{idx, 1} = samples{i}{j};
                    labels{idx, 1} = responses{i};
                    idx = idx + 1;
                end
            end
            
            % En este caso enviams los datos en forma de tabla
            data = table(predictors,labels);
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
            labels = createLabels(subds.Datastore.Files);
            subds.Labels = categorical(labels);
            labels = categorical(labels);
            subds.Labels = setcats(labels,{'fist', 'noGesture', 'open', ... 
                'pinch','waveIn', 'waveOut'});
            subds.NumObservations = numObservations;
            reset(subds);         
        end
                
        
        function dsNew = shuffle(ds)
            % dsNew = shuffle(ds) shuffles the files and the
            % corresponding labels in the datastore.
            
            % Create a copy of datastore
            dsNew = copy(ds);
            dsNew.Datastore = copy(ds.Datastore);
            fds = dsNew.Datastore;
            
            % Shuffle files and corresponding labels
            numObservations = dsNew.NumObservations;
            idx = randperm(numObservations);
            fds.Files = fds.Files(idx);
            dsNew.Labels = dsNew.Labels(idx);
        end
    end
    
    methods (Access = protected)
        % If you use the DsFileSet object as a property, then 
        % you must define the copyElement method. The copyElement
        % method allows methods such as readall and preview to 
        % remain stateless
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
    %Se obtiene el numero de archivos
    numObservations = numel(files);
    % Para cada archivo
    labels = cell(numObservations,1);
    for i = 1:numObservations
        file = files{i};
        filepath = fileparts(file);
        % La ultima parte dela ruta antes del nombre del archivo es la etiqueta
        [~,label] = fileparts(filepath);
        labels{i,1} = label;
    end
end

function data = readSequence(filename)
    data = load(filename);
end
