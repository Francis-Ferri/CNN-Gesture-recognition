%{

%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
dataDir = 'EMG_EPN612_Dataset'; %'CEPRA_2019_13_DATASET_FINAL'
trainingDir = 'trainingJSON'; %'training'

%% GET THE USERS DIRECTORIES
[users, trainingPath] = Shared.getUsers(dataDir, trainingDir);
% Divide in two datasets
limit = length(users)- Shared.numTestUsers;
usersTrainVal = users(1:limit, 1);
usersTest = users(limit+1:length(users), 1);
clear dataDir trainingDir users numTestUsers limit

%%                      BORRAR ESTO AL ACABAR SOLO ES PARA HACER PRUEBAS CON PORCIONES
usersTrainVal = usersTrainVal(1:1);
usersTest = usersTest(1:1);

%% LOAD THE MODEL
modelFile = 'model_26-04-2021_09-43-42';
modelFileName = fullfile('models', modelFile);
model = load(modelFileName).net;
clear modelFile modelFileName

%% PREALLOCATE SPACE FOR RESULTS TRAINING AND VALIDATION
% Allocate space to save the results
% Training
[classifications, recognitions, overlapings, procesingTimes] =  ... 
    preallocateResults(length(usersTrainVal));
% Validation
[classificationsVal, recognitionsVal, overlapingsVal, procesingTimesVal] = ... 
    preallocateResults(length(usersTrainVal));

%% EVALUATE EACH USER FOR TRAINING AND VALIDATION
for i = 1:length(usersTrainVal) % % parfor
    % Get user samples
    [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, usersTrainVal(i));
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
disp('Training data results');
[dataTrain, resultsTrain] = calculateValidationResults(classifications, recognitions, ... 
    overlapings, procesingTimes);
disp('Validacion data results');
[dataValidation, resultsValidation] = calculateValidationResults(classificationsVal, ... 
    recognitionsVal, overlapingsVal, procesingTimesVal);
clear i trainingSamples validationSamples transformedSamplesValidation
clear classifications recognitions overlapings procesingTimes
clear classificationsVal recognitionsVal overlapingsVal procesingTimesVal

%% PREALLOCATE SPACE FOR RESULTS TESTING
% Testing - users training samples
[classificationsTest1, recognitionsTest1, overlapingsTest1, procesingTimesTest1] =  ...
    preallocateResults(length(usersTest));
% Testing - users validation samples
[classificationsTest2, recognitionsTest2, overlapingsTest2, procesingTimesTest2] =  ...
    preallocateResults(length(usersTest));

%% EVALUATE EACH USER FOR TESTING
for i = 1:length(usersTest) 
     % Get user samples
    [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, usersTest(i));
    % Transform samples
    transformedSamplesTraining = transformSamples(trainingSamples);
    userResults = evaluateSamples(transformedSamplesTraining, model);
    % Set user's training results
    classificationsTest1(i, :) = userResults.classifications;
    recognitionsTest1(i, :) = userResults.recognitions;
    overlapingsTest1(i, :) = userResults.overlapings;
    procesingTimesTest1(i, :) = userResults.procesingTimes;
    % Validation data
    transformedSamplesValidation = transformSamples(validationSamples);
    userResults = evaluateSamples(transformedSamplesValidation, model);
    % Set user's training results
    classificationsTest2(i, :) = userResults.classifications;
    recognitionsTest2(i, :) = userResults.recognitions;
    overlapingsTest2(i, :) = userResults.overlapings;
    procesingTimesTest2(i, :) = userResults.procesingTimes;
end
classificationsTest = [classificationsTest1; classificationsTest2];
recognitionsTest = [recognitionsTest1; recognitionsTest2];
overlapingsTest = [overlapingsTest1; overlapingsTest2];
procesingTimesTest = [procesingTimesTest1; procesingTimesTest2];
disp('Testing data results');
[dataTest, resultsTest] = calculateValidationResults(classificationsTest, ... 
    recognitionsTest, overlapingsTest, procesingTimesTest);
clear i trainingSamples validationSamples transformedSamplesValidation
clear classificationsTest1 recognitionsTest1 overlapingsTest1 procesingTimesTest1
clear classificationsTest2 recognitionsTest2 overlapingsTest2 procesingTimesTest2
clear classificationsTest recognitionsTest overlapingsTest procesingTimesTest

%% FUCTION TO PREALLOCATE SPACE FOR VALIDATION LIBRARY RESULT
function [clasifications, recognitions, overlapings, procesingTimes] = preallocateResults(numUsers)
    % Allocate space to save the results
    clasifications = zeros(numUsers, Shared.numSamplesUser);
    recognitions = zeros(numUsers, Shared.numSamplesUser);
    overlapings = zeros(numUsers, Shared.numSamplesUser);
    procesingTimes = zeros(numUsers, Shared.numSamplesUser);
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
        signal = Shared.getSignal(emg);
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
        response = struct('vectorOfLabels', labels, 'vectorOfTimePoints', timestamps, ... 
            'vectorOfProcessingTimes', processingTimes, 'class', class);
        % Send to validation toolbox
        result = evalRecognition(repInfo, response);
        % Save results
        classifications(i) = result.classResult;
        if ~isequal(gesture,'noGesture')
            recognitions(i) = result.recogResult;
            overlapings(i) = result.overlappingFactor;
        end
        procesingTimes(i) = mean(processingTimes); %time
        % Set User Results
        userResults = struct('classifications', classifications,  'recognitions', ... 
            recognitions, 'overlapings', overlapings, 'procesingTimes', procesingTimes);
    end
end

%% FUNCTION TO CALCULATE THE RESULTS OF A SEQUENCE DATASTORE
function [data, results] = calculateValidationResults(classifications, recognitions, overlapings, procesingTimes)
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
    
data = struct('classifications', classifications, 'recognitions', ... 
        recognitions, 'overlapings', overlapings, 'procesingTimes', procesingTimes);
    
    results = struct('clasification', accClasification, 'recognition', ... 
        accRecognition, 'overlapingFactor', avgOverlapingFactor, 'procesingTime', avgProcesingTime);

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
    if catCounts(1) >= Shared.MIN_LABELS_SEQUENCE
       class = newCategories(1);
    else
       class = toCategoricalGesture({'noGesture'}); 
    end
end

%% FUNCTION TO EVALUATE SAMPLE FRAMES
function [labels, timestamps, processingTimes] = evaluateSampleFrames(signal, model)
    % Calculate the number of windows
    numPoints = length(signal);
    if isequal(Shared.FILLING_TYPE, 'before') || isequal(Shared.FILLING_TYPE, 'during')
         numWindows = floor((numPoints - (Shared.FRAME_WINDOW / 2)) / Shared.WINDOW_STEP_RECOG) + 1;
         stepLimit = numPoints - floor(Shared.FRAME_WINDOW / 2) + 1;
    elseif isequal(Shared.FILLING_TYPE, 'none')
        numWindows = floor((numPoints - (Shared.FRAME_WINDOW)) / Shared.WINDOW_STEP_RECOG) + 1;
        stepLimit = numPoints - Shared.FRAME_WINDOW + 1;
    end
    
    % Preallocate space for the spectrograms
    labels = cell(1, numWindows);
    timestamps = zeros(1,numWindows);
    processingTimes = zeros(1,numWindows);
    
    % Fill before frame classification
    if isequal(Shared.FILLING_TYPE, 'before')
        filling = (2 * Shared.noGestureStd) * rand(Shared.FRAME_WINDOW / 2, Shared.numChannels) ... 
            + (Shared.noGestureMean - Shared.noGestureStd);
        signal = [signal; filling];
    end
    
    % Start the frame classification
    idx = 1; inicio = 1;
    while inicio <= stepLimit
        % Start timer
        timer = tic;
        
        finish = inicio + Shared.FRAME_WINDOW - 1;
        timestamp = inicio + floor((Shared.FRAME_WINDOW - 1) / 2);
        
        % Fill during frame classification
        if finish > numPoints  && isequal(Shared.FILLING_TYPE, 'during')
            extraPoints = finish - numPoints +1;
            fill = (2 * Shared.noGestureStd) * rand(extraPoints, Shared.numChannels) ... 
                + (Shared.noGestureMean - Shared.noGestureStd);
            frameSignal =  [signal(inicio:numPoints, :); fill];
        else
            frameSignal = signal(inicio:finish, :);
        end
        frameSignal = Shared.preprocessSignal(frameSignal);
        spectrograms = Shared.generateSpectrograms(frameSignal);
        [predicction, predictedScores] = classify(model,spectrograms);
        
        % Check if the prediction is over the frame classification threshold
        if max(predictedScores) < Shared.FRAME_CLASS_THRESHOLD
            predicction = 'noGesture';
        else
            predicction = char(predicction);
        end
        
        % Stop timer
        processingTime = toc(timer);
        
        % Save sample results
        labels{1, idx} =  predicction; % datum
        timestamps(1, idx) = timestamp; % label
        processingTimes(1, idx) = processingTime; % processing time
        
        % Slide the window
        idx = idx + 1;
        inicio = inicio + Shared.WINDOW_STEP_RECOG;
    end
end

%% EXTRA
%{
    Using "FrameWindow/2" and adding  the other "FrameWindow/2" to the end means that there is not lose of data 
    
    Original formula for the number of windows:
    num_win=floor((n-(frame_len))/frame_step)+1;
    Link to number of windows formula:
    https://stackoverflow.com/questions/53796545/number-of-overlapping-windows-of-a-given-length
%}


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
