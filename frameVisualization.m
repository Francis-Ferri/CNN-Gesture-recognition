%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
datastoreDir = 'Datastores';
datastores = {'training'; 'validation'; 'trainingSequence'; 'validationSequence'};
gestures = {'fist', 'noGesture', 'open', 'pinch', 'waveIn', 'waveOut'};

%% GET A SPECIFIC FRAME
datastore = datastores{1};
gestureClass = gestures{1};
% Get Frames
[frames, dataPath] = getFrames(datastoreDir, datastore, gestureClass);
frame = frames{4};
data = getFrame(dataPath, frame);

%% PLOT THE FRAME
[t, f, ps] = plotFrameChanels(datastore, gestureClass, frame, data);

%% FUNCTION TO PLOT THE SPECTOGRAMS OF EACH CHANNEL
function [t, f, ps] = plotFrameChanels(datastore, gestureClass, frame, data)
    figure('Name', strcat(datastore, '-', gestureClass, '-',  frame))
    for i = 1:size(data, 3)
        t = 1:size(data, 2);
        f = 0:size(data, 1)-1;
        ps = data(:,:, i);
        subplot(4, 2, i)
        surf(t,f,ps,'EdgeColor','none');   
            axis xy; axis tight; colormap(jet); view(0,90);
            title(strcat('Channel-', int2str(i)));
    end
end
    
%% GET THE FRAME LIST
function [frames, dataPath] = getFrames(dataDir, subDir, class)
    dataPath = fullfile(dataDir, subDir, class);
    disp(dataPath);
    frames = ls(dataPath);
    frames = strtrim(string(frames(3:length(frames),:)));
    frames = frames(randperm(length(frames)));
end

%% GET SPECIFIC FRAME 
function data = getFrame(dataPath, file)
    % Get user samples
    filePath = fullfile(dataPath, file);
    data = load(filePath).data;
end


