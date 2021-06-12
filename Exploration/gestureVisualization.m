%{
    EXPLORATION
%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset';
trainingDir = 'trainingJSON';

%% GET THE USERS DIRECTORIES
[users, trainingPath] = Shared.getUsers(dataDir, trainingDir);
clear dataDir trainingDir

%% SELECT ONE USER AND SAMPLE
user = users(2);
type = 'training'; %validation
numSample = 40;
if numSample > 25
    sample = getSample(trainingPath, user, numSample, type);
else
    disp('Not allowed to select noGesture samples and samples outside the bounderies [0:125]');
end

%% PREPORCESS THE SIGNAL
[Fb, Fa] = butter(5, 0.1, 'low'); % Wn = Fc/(Fs/2)
filteredSignal = Shared.preProcessEMGSegment(sample.signal, Fa, Fb, 'abs');
% Clean up variables
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
numChannel = 3;
visualizeFrames(filteredSignal, sample, user, numSample, numChannel, 'signal');
visualizeFrames(filteredSignal, sample, user, numSample, numChannel, 'spectrogram');

%% VISUALIZE EACH CHENNEL OF A FRAME
groundTruthMid = floor((sample.groundTruthIdx(2) + sample.groundTruthIdx(1)) / 2);
% Calculate the start and end points
start = groundTruthMid - floor(Shared.FRAME_WINDOW / 2);
finish = groundTruthMid + floor(Shared.FRAME_WINDOW / 2);
signal = filteredSignal(start:finish-1, :);

% Plot the signal and spectrogram for each channel
plotSignalChanels(user, type, numSample, sample.gesture, signal);
plotSpectrogramChanels(user, type, numSample, sample.gesture, signal);

% Clean up variables
clear groundTruthMid start finish signal

%% GET SPECIFIC SAMPLE FROM A USER
function sampleData = getSample(dataPath, user, numSample, type)
    
    % Get user samples
    [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(dataPath, user);
    samplesKeys = fieldnames(trainingSamples); % Same for validation
    if isequal(type, 'validation')
        samples = validationSamples;
    else 
        samples = trainingSamples;
    end
    sample = samples.(samplesKeys{numSample});
    
    % Get signal data
    sampleData.gesture = sample.gestureName;
    sampleData.signal = Shared.getSignal(sample.emg);
    sampleData.groundTruth = sample.groundTruth;
    groundTruthIdx = sample.groundTruthIndex;
    sampleData.groundTruthLength = groundTruthIdx(2) - groundTruthIdx(1);
    sampleData.groundTruthIdx = sample.groundTruthIndex;
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
    sampleFrecuency = 200;
    % Plot the figure
    [s, f, t, ps] = spectrogram(signal, Shared.WINDOW, Shared.OVERLAPPING, ...
        Shared.FRECUENCIES, sampleFrecuency, 'yaxis');
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
    % Inicialization
    groundTruth = sample.groundTruth;
    numGesturePoints = sample.groundTruthLength;
    numWindows = floor((length(signal) - Shared.FRAME_WINDOW) / Shared.WINDOW_STEP) + 1;
    
    % Figure creation
    plotPosition = 1;
    figure('Name', strcat(user, '-', type, '-',  int2str(numSample), '-', string(sample.gesture), '-', int2str(channel)));
    for i = 1:numWindows
        
        % Get window information
        traslation = ((i-1) * Shared.WINDOW_STEP);
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW/2);
        frameGroundTruth = groundTruth(inicio: finish);
        totalOnes = sum(frameGroundTruth == 1);
        
        % Check if the window is over the threasholds
        if totalOnes >= Shared.FRAME_WINDOW * Shared.TOLERANCE_WINDOW || ...
                totalOnes >= numGesturePoints * Shared.TOLERNCE_GESTURE
            frameSignal = signal(inicio:finish, channel);
            
            % Choose between signal or spectrogram
            if isequal(type, 'signal')
                subplotSignal(plotPosition, frameSignal, timestamp)
            elseif isequal(type, 'spectrogram')
                subPlotSpectrogram(plotPosition, frameSignal, timestamp);
            end
            
            % Slide the window
            plotPosition = plotPosition + 1;
        end
    end
end

