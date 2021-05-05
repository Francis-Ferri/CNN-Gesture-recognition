classdef SharedFunctions
    
    properties (Constant)
        FRAME_WINDOW = 300;
        % Recognition
        WINDOW_STEP = 30;
        FRAME_CLASS_THRESHOLD = 0.5;
        MIN_LABELS_SEQUENCE = 4;
    end
    
    methods(Static)
        
        % FUNCTION TO READ A FILE
        function data = readFile(filename)
            % Load a Matlab file
            data = load(filename).data;
        end
        
        % GET THE USER LIST
        function [users, dataPath] = getUsers(dataDir, subDir)
            dataPath = fullfile(dataDir, subDir);
            users = ls(dataPath);
            users = strtrim(string(users(3:length(users),:)));
            rng(9); % seed
            users = users(randperm(length(users)));
        end
        
        % GET TRAINING AND TESTING SAMPLES FOR AN USER
        function [trainingSamples, testingSamples] = getTrainingTestingSamples(path, user)
            filePath = fullfile(path, user, strcat(user, '.json'));
            jsonFile = fileread(filePath);
            jsonData = jsondecode(jsonFile);
            % Extract samples
            trainingSamples = jsonData.trainingSamples;
            testingSamples = jsonData.testingSamples;
        end
        
        % FUNCTION TO GET THE EMG SIGNAL
        function signal = getSignal(emg)
            % Get chanels
            channels = fieldnames(emg);
            % Signal dimensions (1000 x 8 aprox)
            signal = zeros(length(emg.(channels{1})), length(channels));
            for j = 1:length(channels)
                signal(:,j) = emg.(channels{j});
            end
        end
        
        % FUNCTION TO RECTIFY EMG
        function rectifiedEMG = rectifyEMG(rawEMG, rectFcn)
            switch rectFcn
                case 'square'
                    rectifiedEMG = rawEMG.^2;
                case 'abs'
                    rectifiedEMG = abs(rawEMG);
                case 'none'
                    rectifiedEMG = rawEMG;
                otherwise
                    fprintf('Wrong rectification function. Valid options are square, abs and none');
            end
        end
        
        % FUNCTION TO APLY A FILTER TO EMG
        function EMGsegment_out = preProcessEMGSegment(EMGsegment_in, Fa, Fb, rectFcn)
            % Normalization
            if max( abs(EMGsegment_in(:)) ) > 1
                drawnow;
                EMGnormalized = EMGsegment_in/128;
            else
                EMGnormalized = EMGsegment_in;
            end
            EMGrectified = SharedFunctions.rectifyEMG(EMGnormalized, rectFcn);
            EMGsegment_out = filtfilt(Fb, Fa, EMGrectified);
        end
        
        % FUNCTION TO PREPROCESS A SIGNAL
        function signal = preprocessSignal(signal)
            [Fb, Fa] = butter(5, 0.1, 'low');
            signal = SharedFunctions.preProcessEMGSegment(signal, Fa, Fb, 'abs');
        end
        
        % FUNCTION TO GENERATE SPECTROGRAMS
        function spectrograms = generateSpectrograms(signal)
            % Spectrogram parameters
            FRECUENCIES = (0:12);
            sampleFrecuency = 200;
            WINDOW = 24;
            OVERLAPPING = floor(WINDOW*0.5); %floor(WINDOW*0.75); %floor(WINDOW*0.5); % WINDOW -1
            % Preallocate space for the spectrograms
            numCols = floor((length(signal)-OVERLAPPING)/(WINDOW-OVERLAPPING));
            spectrograms = zeros(length(FRECUENCIES), numCols, size(signal, 2));
            % Spectrograms generation
            for i = 1:size(signal, 2)
                [~,~,~,ps] = spectrogram(signal(:,i), WINDOW, OVERLAPPING, FRECUENCIES, sampleFrecuency, 'yaxis');
                spectrograms(:,:,i) = ps;
            end
        end
        
        % FUNCTION TO SHUFFLE SAMPLES IN A FILE DATASTORE
        function [fds, idx] = shuffle(fds)
            % Get the number of files
            numObservations = numel(fds.Files);
            % Shuffle files and their corresponding labels
            rng(9); % seed
            idx = randperm(numObservations);
            fds.Files = fds.Files(idx);
        end

        % FUNCTION TO CREATE LABELS
        function [labels, numObservations] = createLabels(files, withNoGesture)
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
            if withNoGesture
                classes = ["fist", "noGesture", "open", "pinch", "waveIn", "waveOut"];
            else
                classes = ["fist", "open", "pinch", "waveIn", "waveOut"];
            end
            %categories = categorical{classes};
            labels = categorical(labels, classes);
        end
        
    end
end