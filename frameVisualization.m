%{

%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'Datastores';
datastores = {'training'; 'validation'};

%% CREATE A FILEDATASTORE
datastore = datastores{1};
folder = fullfile(dataDir, datastore);
% Create a file datastore.
fds = fileDatastore(folder, ...
    'ReadFcn',@readFile, ...
    'IncludeSubfolders',true);
fds = shuffle(fds);

%% READ THE DATASTORE
[frames, labels] = readFrames(fds);

%% VISUALIZE FRAMES
channel = 1;
visualizeFramesInDatstore(frames, labels, datastore, channel);

%% FUNCTION TO READ A FILE
function data = readFile(filename)
    % Load a Matlab file
    data = load(filename).data;
end

%% FUNCTION TO SHUFFLE THE SAMPLES
function fds = shuffle(fds)
    % Get the number of files
    numObservations = numel(fds.Files);
    % Shuffle files and their corresponding labels
    idx = randperm(numObservations);
    fds.Files = fds.Files(idx);
end

%% FUNCTION TO READ FRAMES
function [frames, labels] = readFrames(fds)            
    % Function to read data
    miniBatchSize = 15;
    files = fds.Files(1:miniBatchSize,1);
    frames = cell(miniBatchSize,1);
    labels = cell(miniBatchSize,1);
    for i = 1:miniBatchSize
        filepath = fileparts(files{i,1}); % ../datastore/class
        % The last part of the path is the label
        [~,label] = fileparts(filepath); % [../datastore, class]
        frame = readFile(files{i,1});
        frames{i,1} = frame;
        labels{i,1} = label;
    end
end

%% FUNCTION TO SUBPLOT A SPECTROGRAM FRAME
function subPlotSpectrogram(plotPosition, signal, label, channel)
    f = 1:size(signal, 1);
    t = 1:size(signal, 2);
    ps = signal(:,:,channel);
    subplot(3, 5, plotPosition)
        surf(t,f,ps,'EdgeColor','none');   
        axis xy; axis tight; colormap(jet); view(0,90);
        title(strcat('Gesture-', label))
end

%% FUNCTION TO VISUALIZE FRAMES
function visualizeFramesInDatstore(frames, labels, type, channel)
    figure('Name', strcat('Gestures-', type, '-channel-', int2str(channel)));
    for i = 1:length(frames)
        subPlotSpectrogram(i, frames{i}, labels{i}, channel);
    end
end
