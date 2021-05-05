%{

%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset'; %'CEPRA_2019_13_DATASET_FINAL'
trainingDir = 'trainingJSON'; %'training'

%% GET THE USERS DIRECTORIES
[users, trainingPath] = SharedFunctions.getUsers(dataDir, trainingDir);
clear dataDir trainingDir

%%                      BORRAR ESTO AL ACABAR SOLO ES PARA HACER PRUEBAS CON PORCIONES
users = users(1:1);

%% LOAD THE MODEL
modelFile = 'model_26-04-2021_09-43-42';
modelFileName = fullfile('models', modelFile);
model = load(modelFileName).net;
clear modelFile modelFileName

%% PREALLOCATE SPACE FOR RESULTS
numSamplesUser = 150;
% Allocate space to save the results
[classifications, recognitions, overlapings, procesingTimes] = preallocateResults(length(users), numSamplesUser);
[classificationsVal, recognitionsVal, overlapingsVal, procesingTimesVal] = preallocateResults(length(users), numSamplesUser);
clear numObservations

%% EVALUATE EACH USER
for i = 1:length(users) % % parfor
    % Get user samples
    [trainingSamples, validationSamples] = SharedFunctions.getTrainingTestingSamples(trainingPath, users(i));
    % Transform samples
    transformedSamplesTraining = transformSamples(trainingSamples);
    userResults = evaluateSamples(transformedSamplesTraining, model);
    % Set user's training results
    classifications(i, :) = userResults.classifications;
    recognitions(i, :) = userResults.recognitions;
    overlapings(i, :) = userResults.overlapings;
    procesingTimes(i, :) = userResults.procesingTimes;
    % Validation data
    transformedSamplesValidation = transformSamples(validationSamples);
    userResults = evaluateSamples(transformedSamplesValidation, model);
    % Set user's training results
    classificationsVal(i, :) = userResults.classifications;
    recognitionsVal(i, :) = userResults.recognitions;
    overlapingsVal(i, :) = userResults.overlapings;
    procesingTimesVal(i, :) = userResults.procesingTimes;
end
disp('Resultados de datos de entrenamiento');
calculateValidationResults(classifications, recognitions, overlapings, procesingTimes);
disp('Resultados de datos de validacion');
calculateValidationResults(classificationsVal, recognitionsVal, overlapingsVal, procesingTimesVal);
clear i trainingSamples validationSamples transformedSamplesValidation

%% FUCTION TO PREALLOCATE SPACE FOR VALIDATION LIBRARY RESULT
function [clasifications, recognitions, overlapings, procesingTimes] = preallocateResults(numUsers, numSamplesUser)
    % Allocate space to save the results
    clasifications = zeros(numUsers, numSamplesUser);
    recognitions = zeros(numUsers, numSamplesUser);
    overlapings = zeros(numUsers, numSamplesUser);
    procesingTimes = zeros(numUsers, numSamplesUser);
end

%% CREAR DATOS DE ESPECTROGRAMAS
function transformedSamples = transformSamples(samples)
    % Get sample keys
    samplesKeys = fieldnames(samples);
    % Allocate space for the results
    transformedSamples = cell(length(samplesKeys), 3);
    for i = 1:length(samplesKeys)
        % Get sample data
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        % Get signal from sample
        signal = SharedFunctions.getSignal(emg);
        % Adding the transformed data
        transformedSamples{i,1} = signal;
        transformedSamples{i,2} = gestureName;
        if ~isequal(gestureName,'noGesture')
            groundTruth = sample.groundTruth;
            transformedSamples{i,3} = transpose(groundTruth);
        end
    end
end

%% FUCTION TO PREALLOCATE SPACE FOR VALIDATION LIBRARY RESULT
function [clasifications, recognitions, overlapings, procesingTimes] = preallocateUserResults(numObservations)
    % Allocate space to save the results
    clasifications = zeros(numObservations, 1);
    recognitions = -1*ones(numObservations, 1);
    overlapings = -1*ones(numObservations, 1);
    procesingTimes = zeros(numObservations, 1);
end

%% FUNCTION TO EVALUETE SAMPLES OF A USER
function userResults = evaluateSamples(samples, model)
    % Preallocate space for results
    [classifications, recognitions, overlapings, procesingTimes] = preallocateUserResults(length(samples));
    % For each sample of a user
    for i = 1:length(samples)
        emg = samples{i, 1};
        gesture = samples{i, 2};
        groundTruth = samples{i, 3};
        % Prepare repetition information
        if ~isequal(gesture,'noGesture')
            repInfo.groundTruth = logical(groundTruth);
        end
        repInfo.gestureName = toCategoricalGesture({gesture});
        % Evaluate a sample with slidding window
        [labels, timestamps, processingTimes] = evaluateSampleFrames(emg, model);
        % Classify prediction
        class = classifyPredictions(labels);
        % Postprocess the sample
        labels = setWrongLabelsToNoGestute(labels, char(class));
        % Prepare response
        response.vectorOfLabels = labels;
        response.vectorOfTimePoints = timestamps;
        response.vectorOfProcessingTimes = processingTimes;
        response.class = class;
        % Send to validation toolbox
        result = evalRecognition(repInfo, response);
        % Save results
        classifications(i) = result.classResult;
        if ~isequal(gesture,'noGesture')
            size(result.recogResult)
            recognitions(i) = result.recogResult;
            overlapings(i) = result.overlappingFactor;
        end
        procesingTimes(i) = mean(processingTimes); %time
        % Set User Results
        userResults.classifications = classifications;
        userResults.recognitions = recognitions;
        userResults.overlapings = overlapings;
        userResults.procesingTimes = procesingTimes;
    end
end

%% FUNCTION TO CALCULATE THE RESULTS OF A SEQUENCE DATASTORE
function calculateValidationResults(classifications, recognitions, overlapings, procesingTimes)
% VERIFICAR QUE ESTA  ES UNA BUEBNA FORMA DE FLATTEN
    classifications = classifications(:);
    recognitions = recognitions(:);
    overlapings = overlapings(:);
    procesingTimes = procesingTimes(:);
    % Change NaN to 0 in the overlapping factor to prevent error
    overlapings(isnan(overlapings)) = 0;
    % Calculate accuracies
    accClasification = sum(classifications==1) / length(classifications);
    accRecognition = sum(recognitions==1) / sum(recognitions==1 | recognitions==0); %noGesture has (-1)
    avgOverlapingFactor = mean(overlapings(overlapings ~= -1)); %& ~isnan(overlapings)
    avgProcesingTime = mean(procesingTimes);
    % Display the results
    fprintf('Classification accuracy: %f\n',accClasification);
    fprintf('Recognition accuracy: %f\n',accRecognition);
    fprintf('Avegage overlaping factor: %f\n',avgOverlapingFactor);
    fprintf('Avegage procesing time: %f\n',avgProcesingTime);  
end

%% FUNCTION TO SET WRONG LABELS TO NOGESTURE
function labels = setWrongLabelsToNoGestute(labels, class)
    for i = 1:length(labels)
        if ~isequal(labels{1,i}, class)
            labels{1,i} = 'noGesture';
        end
    end
    labels = toCategoricalGesture(labels);
end

%% FUNCTION TO TRANSFORM TOCATEGORICAL
function yCat = toCategoricalGesture(yPred)
    gestures = {'fist', 'noGesture', 'open', 'pinch', 'waveIn', 'waveOut'};
    yCat = categorical(yPred,gestures);
end

%% FUNCTION TO CLASSIFY PREDICTIONS
function class = classifyPredictions(yPred)
    gestures = {'fist', 'noGesture', 'open', 'pinch', 'waveIn', 'waveOut'};
    categories = toCategoricalGesture(gestures);
    % Delete noGestures
    idxs = cellfun(@(label) ~isequal(label,'noGesture'), yPred);
    yPred = toCategoricalGesture(yPred(idxs));
    % Count the number of labels per gesture
    catCounts = countcats(yPred);
    [catCounts,indexes] = sort(catCounts,'descend');
    newCategories = categories(indexes);
    if catCounts(1) >= SharedFunctions.MIN_LABELS_SEQUENCE
       class = newCategories(1);
    else
       class = toCategoricalGesture({'noGesture'}); 
    end
end

%% FUNCTION TO EVALUATE SAMPLE FRAMES
function [labels, timestamps, processingTimes] = evaluateSampleFrames(signal, model)
    % Frame onfigurations
    FrameWindow = SharedFunctions.FRAME_WINDOW;
    windowsStep = SharedFunctions.WINDOW_STEP;
    % Signal Data
    numPoints = length(signal);
    numChannels = size(signal, 2);
    % Preallocate space for the spectrograms
    numWindows = floor((length(signal)-(FrameWindow/2)) /windowsStep) + 1;
    labels = cell(1, numWindows);
    timestamps = zeros(1,numWindows);
    processingTimes = zeros(1,numWindows);
    % TODO: RELLENAR AL INICIO ANTES DE LA PRUEBA
    % ACTUALMENTE: CON RELLENO EN LA VENTANA DESLIZANTE EN LOS ULTIMOS FRAMES AL FINAL
    % Start the frame classification
    idx = 1; inicio = 1;
    while inicio <= numPoints - floor(FrameWindow /2) +1
        timer = tic;
        finish = inicio + FrameWindow -1;
        timestamp = inicio + floor((FrameWindow - 1) / 2);
        if finish <= numPoints
            frameSignal = signal(inicio:finish, :);
        else
            frameSignal = zeros(FrameWindow, numChannels);
            frameSignal(1:numPoints-inicio+1, :) = signal(inicio:numPoints, :);
            extraPoints = finish - numPoints +1;
            % TODO: RELLENAR CON RANDOM DEL PROMEDIO DE NO GESTURE
            fill = zeros(extraPoints, numChannels); 
            frameSignal(numPoints-inicio+1:FrameWindow, :) =  fill;
        end
        frameSignal = SharedFunctions.preprocessSignal(frameSignal);
        spectrograms = SharedFunctions.generateSpectrograms(frameSignal);
        [predicction, predictedScores] = classify(model,spectrograms);
        % Check if the prediction is over the frame classification threshold
        if max(predictedScores) < SharedFunctions.FRAME_CLASS_THRESHOLD
            predicction = 'noGesture';
        else
            predicction = char(predicction);
        end
        processingTime = toc(timer);
        % Save sample results
        labels{1, idx} =  predicction; % datum
        timestamps(1, idx) = timestamp; % label
        processingTimes(1, idx) = processingTime; % processing time
        % Slide the window
        idx = idx + 1;
        inicio = inicio + windowsStep;
    end
end

%% FOR TESTING
%{
 evaluateSamples
    % ===== JUST FOR TESTING =====
    data = cell(length(samples), 7);  
    % ===== JUST FOR TESTING =====

    % ===== JUST FOR TESTING =====
    data{i, 1} = result.classResult;
    data{i, 2} = result.recogResult;
    data{i, 3} = result.overlappingFactor;
    data{i, 4} = gesture;
    data{i, 5} = char(class);
    data{i, 6} = groundTruth;
    timestamps = num2cell(timestamps);
    maxScore = num2cell(maxScore);
    data{i, 7} = [cellstr(labels); timestamps; maxScore; scores];
    % ===== JUST FOR TESTING =====

    % ===== JUST FOR TESTING =====
    userResults.data = data;  
    % ===== JUST FOR TESTING =====
            
evaluateSampleFrames
    % ===== JUST FOR TESTING =====
    maxScore = zeros(1,numWindows);
    scores =  cell(1,numWindows); % 6 = NumClases
    % ===== JUST FOR TESTING =====

    % ===== JUST FOR TESTING =====
    maxScore(1, idx) = max(predictedScores);
    scores{1, idx} = predictedScores;
    % ===== JUST FOR TESTING =====
%}
