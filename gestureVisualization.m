%{

%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset';
trainingDir = 'trainingJSON';

%% GET THE USERS DIRECTORIES
[users, trainingPath] = getUsers(dataDir, trainingDir);
clear dataDir trainingDir

%% SELECT ONE USER AND SAMPLE
user = users(2);
type = 'training'; %validation
numSample = 40;
if numSample > 25
    sample = getSample(trainingPath, user, numSample, type);
else
    disp('No se pueden seleccionar muestras no gesture');
end

%% PREPORCESS THE SIGNAL
% Wn = Fc/(Fs/2)
[Fb, Fa] = butter(5, 0.1, 'low');
filteredSignal = preProcessEMGSegment(sample.signal, Fa, Fb, 'abs');
clear Fb Fa

%% PLOT SIGNAL AND SPECTROGRAMS OF EACH CHANNEL
plotSignalChanels(user, type, numSample, sample.gesture, filteredSignal);
plotSpectrogramChanels(user, type, numSample, sample.gesture, filteredSignal);

%% PLOT SIGNAL AND SPECTROGRAM OF CHANNEL
numChannel = 6;
ps = plotSignalSpectrogramChanel(user, type, numSample, sample.gesture, filteredSignal, numChannel);

%% PLOT A 3D SPECTROGRAM OF A CHANNEL
numChannel = 6;
plot3DSpectrogram(user, type, numSample, sample.gesture, filteredSignal, numChannel);

%% VISUALIZE FRAMES
channel = 3;
visualizeFrames(filteredSignal, sample, user, numSample,channel, 'signal');
visualizeFrames(filteredSignal, sample, user, numSample,channel, 'spectrogram');

%% VISUUALIZE EACH CHENNEL OF A FRAME
FRAME_WINDOW = 300;
groundTruthMid = floor((sample.groundTruthIdx(2) + sample.groundTruthIdx(1)) / 2);
% Calculate the start and end points
start = groundTruthMid - floor(FRAME_WINDOW/2);
finish = groundTruthMid + floor(FRAME_WINDOW/2);
signal = filteredSignal(start:finish-1, :);
% Plot the signal and spectrogram for each channel
plotSignalChanels(user, type, numSample, sample.gesture, signal);
plotSpectrogramChanels(user, type, numSample, sample.gesture, signal);

%% GET THE USER LIST
function [users, dataPath] = getUsers(dataDir, subDir)
    dataPath = fullfile(dataDir, subDir);
    users = ls(dataPath);
    users = strtrim(string(users(3:length(users),:)));
    rng(9); % seed
    users = users(randperm(length(users)));
end

%% GET TRAINING AND TESTING SAMPLES FOR AN USER
function [trainingSamples, testingSamples] = getTrainingTestingSamples(path, user)
    filePath = fullfile(path, user, strcat(user, '.json'));
    jsonFile = fileread(filePath);
    jsonData = jsondecode(jsonFile);
    % Extract samples
    trainingSamples = jsonData.trainingSamples;
    testingSamples = jsonData.testingSamples;
end

%% FUNCTION TO GET THE EMG SIGNAL
function signal = getSignal(emg)
    channels = fieldnames(emg); % get chanels
    signal = zeros(length(emg.(channels{1})), length(channels)); % ex: 1000 x 8
    for j = 1:length(channels)
        signal(:,j) = emg.(channels{j});
    end
end

%% GET SPECIFIC SAMPLE FROM A USER
function sampleData = getSample(dataPath, user, numSample, type)
    % Get user samples
    [trainingSamples, validationSamples] = getTrainingTestingSamples(dataPath, user);
    samplesKeys = fieldnames(trainingSamples); % Same for validation
    if isequal(type, 'validation')
        samples = validationSamples;
    else 
        samples = trainingSamples;
    end
    sample = samples.(samplesKeys{numSample});
    % Get signal data
    sampleData.gesture = sample.gestureName;
    sampleData.signal = getSignal(sample.emg);
    sampleData.groundTruth = sample.groundTruth;
    groundTruthIdx = sample.groundTruthIndex;
    sampleData.groundTruthLength = groundTruthIdx(2) - groundTruthIdx(1);
    sampleData.groundTruthIdx = sample.groundTruthIndex;
end

%% FUNCTION TO RECTIFY EMG
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

%% FUNCTION TO PREPROCESS EMG
function EMGsegment_out = preProcessEMGSegment(EMGsegment_in, Fa, Fb, rectFcn)
    % Normalization
    if max( abs(EMGsegment_in(:)) ) > 1
        drawnow;
        EMGnormalized = EMGsegment_in/128;
    else
        EMGnormalized = EMGsegment_in;
    end
    EMGrectified = rectifyEMG(EMGnormalized, rectFcn);
    EMGsegment_out = filtfilt(Fb, Fa, EMGrectified);
end

%% FUNCTION TO PLOT SIGNAL IN EACH CHANNEL
function plotSignalChanels(user, type, numSample, gestureName, signal)
    figure('Name', strcat(user, '-', type, '-',  int2str(numSample), '-', string(gestureName)))
    for i = 1:size(signal, 2)
        subplot(4, 2, i)
            plot(signal(:,i));
            title(strcat('Channel-', int2str(i)));
    end
end

%% FUNCTION TO CALCUTLATE A SPECTROGRAM
function [s, f, t, ps] = calculateSpectrogram(signal)
    % Spectrogram parameters
    FRECUENCIES = (0:12);
    sampleFrecuency = 200;
    % Almost mandaory 200 to analize from 0 to 100 fecuencies
    WINDOW = 24;
    OVERLAPPING = floor(WINDOW*0.5); %floor(WINDOW*0.75); %floor(WINDOW*0.5); % WINDOW -1
    % Plot the figure
    [s, f, t, ps] = spectrogram(signal, WINDOW, OVERLAPPING, FRECUENCIES, sampleFrecuency, 'yaxis');
end

%% FUNCTION TO PLOT THE SPECTOGRAMS OF EACH CHANNEL
function plotSpectrogramChanels(user, type, numSample, gestureName, signal)
    figure('Name', strcat(user, '-', type, '-',  int2str(numSample), '-', string(gestureName)))
    for i = 1:size(signal, 2)
        [~, f, t, ps] = calculateSpectrogram(signal(:,i));
        subplot(4, 2, i)
            surf(t,f,ps,'EdgeColor','none');   
            axis xy; axis tight; colormap(jet); view(0,90);
            title(strcat('Channel-', int2str(i)));
    end
end

%% FUNCTION TO PLOT SIGNAL AND SPECTROGRAM OF CHANNEL
function ps = plotSignalSpectrogramChanel(user, type, numSample, gestureName, signal, numChannel)
    channel = signal(:,numChannel);
    figure('Name', strcat(user, '-', type, '-',  int2str(numSample), '-', ...
        string(gestureName), '-', int2str(numChannel)))
    subplot(1, 2, 1)
        plot(channel)
        title('Signal')
    subplot(1, 2, 2)
        [~, f, t, ps] = calculateSpectrogram(channel);
        surf(t, f, ps,'EdgeColor','none');   
        axis xy; axis tight; colormap(jet); view(0,90);
        title('Spectrogram')
end

%% FUNCTION TO PLOT A 3D SPECTROGRAM OF A CHANNEL
function plot3DSpectrogram(user, type, numSample, gestureName, signal, numChannel)
    [~, f, t, ps] = calculateSpectrogram(signal(:,numChannel));
    figure('Name', strcat(user, '-', type, '-',  int2str(numSample), '-', ...
        string(gestureName), '-', int2str(numChannel)))
    surf(t,f,ps,'EdgeColor','none');
    colormap jet
    title('Spectrogram')
end

%% FUCTION TO SUBPLOT A SIGNAL FRAME
function subplotSignal(plotPosition, signal, timestamp)
    subplot(2, 5, plotPosition)
        plot(signal)
        title(strcat('Timestamp-', int2str(timestamp)))
end

%% FUCTION TO SUBPLOT A SPECTROGRAM FRAME
function subPlotSpectrogram(plotPosition, signal, timestamp)
    [~, f, t, ps] = calculateSpectrogram(signal(:));
    subplot(2, 5, plotPosition)
        surf(t,f,ps,'EdgeColor','none');   
        axis xy; axis tight; colormap(jet); view(0,90);
        title(strcat('Timestamp-', int2str(timestamp)))
end

%% FUNCTION TO VISUALIZE FRAMES
function visualizeFrames(signal, sample, user, numSample,channel, type)
    % Frame onfigurations
    FRAME_WINDOW = 300;
    WINDOW_STEP = 15;
    TOLERANCE_WINDOW = 0.75;
    TOLERNCE_GESTURE = 0.9;
    % Inicialization
    groundTruth = sample.groundTruth;
    numGesturePoints = sample.groundTruthLength;
    numWindows = floor((length(signal)-FRAME_WINDOW) /WINDOW_STEP)+1;
    
    plotPosition = 1;
    figure('Name', strcat(user, '-', type, '-',  int2str(numSample), '-', string(sample.gesture), '-', int2str(channel)));
    for i = 1:numWindows
        traslation = ((i-1)*WINDOW_STEP);
        inicio = 1 + traslation;
        finish = FRAME_WINDOW + traslation;
        
        timestamp = inicio + floor(FRAME_WINDOW/2);
        frameGroundTruth = groundTruth(inicio: finish);
        totalOnes = sum(frameGroundTruth == 1);
        if totalOnes >= FRAME_WINDOW * TOLERANCE_WINDOW || totalOnes >= numGesturePoints * TOLERNCE_GESTURE
            frameSignal = signal(inicio:finish, channel);
            if isequal(type, 'signal')
                subplotSignal(plotPosition, frameSignal, timestamp)
            elseif isequal(type, 'spectrogram')
                subPlotSpectrogram(plotPosition, frameSignal, timestamp);
            end
            plotPosition = plotPosition + 1;
        end
    end
end

