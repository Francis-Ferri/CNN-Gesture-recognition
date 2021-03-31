%{
VISUALIZATION OF DATA
    * Visualize the signal of each channel
    * Crete and visualize the spectrogram of each channel
    * Visualize the signal and spresctrogram of one channel
    * Visualize a 3D plot of onechannel of a spectrogram
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
[signal, gestureName] = getSample(trainingPath, user, numSample, type);

%% PREPORCESS THE SIGNAL
[Fb, Fa] = butter(5, 0.1, 'low');
signal = preProcessEMGSegment(signal, Fa, Fb, 'abs');

%% PLOT SIGNAL AND SPECTROGRAMS OF EACH CHANNEL
plotSignalChanels(user, type, numSample, gestureName, signal);
plotSpectrogramChanels(user, type, numSample, gestureName, signal);

%% SELECT A CHANEL TO PLOT THE SPECTROGRAM
numChannel = 6;

%% PLOT SIGNAL AND SPECTROGRAM OF CHANNEL
ps = plotSignalSpectrogramChanel(user, type, numSample, gestureName, signal, numChannel);

%% PLOT A 3D SPECTROGRAM OF A CHANNEL
plot3DSpectrogram(user, type, numSample, gestureName, signal, numChannel)

%% GET THE USER LIST
function [users, dataPath] = getUsers(dataDir, subDir)
    dataPath = fullfile(dataDir, subDir);
    users = ls(dataPath);
    users = strtrim(string(users(3:length(users),:)));
    rng(9); % seed
    users = users(randperm(length(users)));
end

%% GET SPECIFIC SAMPLE FROM A USER
function [signal, gestureName] = getSample(dataPath, user, numSample, type)
    % Get user samples
    [trainingSamples, validationSamples] = getTrainingTestingSamples(dataPath, user);
    if isequal(type, 'validation')
        samplesKeys = fieldnames(validationSamples);
        samples = validationSamples;
    else 
        samplesKeys = fieldnames(trainingSamples);
        samples = trainingSamples;
    end
    sample = samples.(samplesKeys{numSample});
    emg = sample.emg;
    gestureName = sample.gestureName;
    % Get signal from sample
    signal = getSignal(emg);
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

%% FUNCTION TO PLOT SIGNAL IN EACH CHANNEL
function plotSignalChanels(user, type, numSample, gestureName, signal)
    figure('Name', strcat(user, '-', type, '-',  int2str(numSample), '-', string(gestureName)))
    for i = 1:size(signal, 2)
        subplot(4, 2, i)
            plot(signal(:,i));
            title(strcat('Channel-', int2str(i)));
    end
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

%% FUNCTION TO CALCUTLATE A SPECTROGRAM
function [s, f, t, ps] = calculateSpectrogram(signal)
    % Spectrogram parameters
    FRECUENCIES = (0:10);
    sampleFrecuency = 200;
    % Almost mandaory 200 to analize from 0 to 100 fecuencies
    WINDOW = 20;
    OVERLAPPING = floor(WINDOW*0.5); %floor(WINDOW*0.75); %floor(WINDOW*0.5); % WINDOW -1
    % Plot the figure
    [s, f, t, ps] = spectrogram(signal, WINDOW, OVERLAPPING, FRECUENCIES, sampleFrecuency, 'yaxis');
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

%% EXTRA THINGS
%{
    % Short time Fourier transform
    s = stft(chanel,200, 'Window',hamming(200,'periodic'), ...
        'OverlapLength',199, 'Centered', false, 'FFTLength', 200);
    % Get the spectrogram's module values
    abs(s)
    % Get the data in decibels I think
    10*log10(abs(s))
%}
