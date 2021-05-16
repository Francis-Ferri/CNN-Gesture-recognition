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
usersTrainVal = usersTrainVal(1:3);
usersTest = usersTest(1:3);

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
    [classifications(i, :), recognitions(i, :), overlapings(i, :), procesingTimes(i, :)] = ... 
        deal(userResults.classifications, userResults.recognitions, ... 
        userResults.overlapings, userResults.procesingTimes);
    
    % Validation data
    transformedSamplesValidation = transformSamples(validationSamples);
    userResults = evaluateSamples(transformedSamplesValidation, model);
    
    % Set user's training results
    [classificationsVal(i, :), recognitionsVal(i, :), overlapingsVal(i, :), procesingTimesVal(i, :)] = ... 
        deal(userResults.classifications, userResults.recognitions, ... 
        userResults.overlapings, userResults.procesingTimes);
end

% Print trainig results
fprintf('\n\n\tTraining data results\n\n');
resultsTrain = calculateValidationResults(classifications, recognitions, overlapings, ... 
    procesingTimes, length(usersTrainVal));

% Print validation results
fprintf('\n\n\tValidation data results\n\n');
resultsValidation = calculateValidationResults(classificationsVal, recognitionsVal, ... 
    overlapingsVal, procesingTimesVal, length(usersTrainVal));

clear i trainingSamples validationSamples transformedSamplesValidation classifications recognitions overlapings procesingTimes classificationsVal recognitionsVal overlapingsVal procesingTimesVal

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
    classificationsTest1(i, :) = userResults.classifications; recognitionsTest1(i, :) = userResults.recognitions; overlapingsTest1(i, :) = userResults.overlapings; procesingTimesTest1(i, :) = userResults.procesingTimes;
    
    % Validation data
    transformedSamplesValidation = transformSamples(validationSamples);
    userResults = evaluateSamples(transformedSamplesValidation, model);
    
    % Set user's training results
    classificationsTest2(i, :) = userResults.classifications; recognitionsTest2(i, :) = userResults.recognitions; overlapingsTest2(i, :) = userResults.overlapings; procesingTimesTest2(i, :) = userResults.procesingTimes;
end

% Combine testing part (training and validation samples)
[classificationsTest, recognitionsTest, overlapingsTest, procesingTimesTest] = ... 
    deal([classificationsTest1; classificationsTest2], [recognitionsTest1; recognitionsTest2], ... 
    [overlapingsTest1; overlapingsTest2], [procesingTimesTest1; procesingTimesTest2]);

% Print the results
fprintf('\n\n\tTesting data results\n\n');
dataTest = calculateValidationResults(classificationsTest, recognitionsTest, ... 
    overlapingsTest, procesingTimesTest, length(usersTest));

clear i trainingSamples validationSamples transformedSamplesValidation classificationsTest1 recognitionsTest1 overlapingsTest1 procesingTimesTest1 classificationsTest2 recognitionsTest2 overlapingsTest2 procesingTimesTest2n classificationsTest recognitionsTest overlapingsTest procesingTimesTest

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
        
        % Get sample data
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
        
        % Set a class for the sample
        class = classifyPredictions(labels);
        
        % Postprocess the sample (labels)
        labels = postprocessSample(labels, char(class));
        
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
        procesingTimes(i) = mean(processingTimes); %time (frame)
        
        % Set User Results
        userResults = struct('classifications', classifications,  'recognitions', ... 
            recognitions, 'overlapings', overlapings, 'procesingTimes', procesingTimes);
    end
end

%% FUNCTION TO CALCULATE THE RESULTS OF A DATASTORE (MEAN USERS)
function [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ... 
            calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, numUsers)

    % Allocate space for results
    [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ... 
        deal(zeros(numUsers, 1), zeros(numUsers, 1), zeros(numUsers, 1), zeros(numUsers, 1));
    
    % Calculate results per user
    for i = 1:numUsers
        classificationPerUser(i, 1) = sum(classifications(i, :) == 1) / length(classifications(i, :));
        % NoGesture omitted it has value = -1 
        recognitionPerUser(i , 1) = sum(recognitions(i, :) == 1) / ... 
            sum(recognitions(i, :) == 1 | recognitions(i, :) == 0);
        overlapingsUser = overlapings(i, :);
        overlapingPerUser(i, 1) = mean(overlapingsUser(overlapingsUser ~= -1),'omitnan');
        processingTimePerUser(i, 1) = mean(procesingTimes(i, :));
    end
end

%% FUNCTION TO CALCULATE THE RESULTS OF A DATASTORE (GLOBAL)
function [globalResps, globalStds] = calculateResultsGlobalMean(all, perUser, numUsers)
    
    % Calculate accuracies 
    accClasification = sum(all.classifications==1) / length(all.classifications);
    % NoGesture omitted it has value = -1 
    accRecognition = sum(all.recognitions==1) / sum(all.recognitions == 1 | all.recognitions == 0); 
    avgOverlapingFactor = mean(all.overlapings(all.overlapings ~= -1), 'omitnan');
    avgProcesingTime = mean(all.procesingTimes);
    
    % Set results
    globalResps = struct('accClasification', accClasification, 'accRecognition', accRecognition, ... 
        'avgOverlapingFactor', avgOverlapingFactor, 'avgProcesingTime', avgProcesingTime);
    
    % Stract data per user
    [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = deal( ... 
        perUser.classifications, perUser.recognitions, perUser.overlapings, perUser.procesingTimes);
    [stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal(0,0,0,0);
    
    % Calculate standard deviations regarding users means
    for i = 1:numUsers
        stdClassification = stdClassification + (classificationPerUser(i,1) - accClasification) ^ 2;
        stdRecognition = stdRecognition + (recognitionPerUser(i, 1) - accRecognition) ^ 2;
        stdOverlaping = stdOverlaping + (overlapingPerUser(i, 1) - avgOverlapingFactor) ^ 2;
        stdProcessingTime = stdProcessingTime + (processingTimePerUser(i, 1) - avgProcesingTime) ^ 2;
    end
    
    % Check number of users
    if numUsers > 1
         [stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal( ... 
             stdClassification / (numUsers - 1), stdRecognition / (numUsers - 1), ... 
             stdOverlaping / (numUsers - 1), stdProcessingTime / (numUsers - 1));
    else 
        [stdClassification, stdRecognition, stdOverlaping, stdProcessingTime] = deal(0,0,0,0);
    end
    
    % Set standard deviations
    globalStds = struct('stdClassification', stdClassification, 'stdRecognition', stdRecognition, ... 
        'stdOverlaping', stdOverlaping, 'stdProcessingTime', stdProcessingTime);
end

%% FUNCTION TO CALCULATE THE RESULTS OF A DATASTORE
function results = calculateValidationResults(classifications, recognitions, overlapings, procesingTimes, numUsers)
    
    % Calculate results using the mean values of users results
    [classificationPerUser, recognitionPerUser, overlapingPerUser, processingTimePerUser] = ... 
    calculateMeanUsers(classifications, recognitions, overlapings, procesingTimes, numUsers);
    
    % Print results using mean values
    disp('Results (mean of user results)');
    fprintf('Classification | acc: %f | std: %f  \n', ... 
        mean(classificationPerUser), std(classificationPerUser));
    fprintf('Recognition | acc: %f | std: %f  \n', ... 
        mean(recognitionPerUser), std(recognitionPerUser));
    fprintf('Overlaping | avg: %f | std: %f  \n', ... 
        mean(overlapingPerUser), std(overlapingPerUser));
    fprintf('Processing time | avg: %f | std: %f  \n\n', ... 
        mean(processingTimePerUser), std(processingTimePerUser));
    
    % Flatten samples
    [classifications, recognitions, overlapings, procesingTimes] = ... 
    deal(classifications(:), recognitions(:), overlapings(:), procesingTimes(:));
    
    % Organize data in structs
    all = struct('classifications', classifications, 'recognitions', recognitions, ... 
        'overlapings', overlapings, 'procesingTimes', procesingTimes);
    perUser =  struct('classifications', classificationPerUser, 'recognitions', recognitionPerUser, ... 
        'overlapings', overlapingPerUser, 'procesingTimes', processingTimePerUser);
    
    % Calculate results using a global mean
    [globalResps, globalStds] = calculateResultsGlobalMean(all, perUser, numUsers);
    
    % Print results using global values
    disp('Results (Global results)');
    fprintf('Classification | acc: %f | std: %f  \n', ... 
        globalResps.accClasification, globalStds.stdClassification);
    fprintf('Recognition | acc: %f | std: %f  \n', ... 
        globalResps.accRecognition, globalStds.stdRecognition);
    fprintf('Overlaping | avg: %f | std: %f  \n', ... 
        globalResps.avgOverlapingFactor, globalStds.stdOverlaping);
    fprintf('Processing time | avg: %f | std: %f  \n\n', ... 
        globalResps.avgProcesingTime, globalStds.stdProcessingTime);
    
    % Set results
    results = struct('clasification',  globalResps.accClasification, 'recognition', ... 
        globalResps.accRecognition, 'overlapingFactor', globalResps.avgOverlapingFactor, ... 
        'procesingTime', globalResps.avgProcesingTime);
end

%% FUNCTION TO SET WRONG LABELS TO NOGESTURE
function labels = postprocessSample(labels, class)

    if ismember(Shared.POSTPROCESS, {'1-1', '2-1', '1-2'})
        
        % Set start and finish of postprocess
        start = 2; finish = length(labels) - 1; % 1-1 by default
        if isequal(Shared.POSTPROCESS, '2-1')
            start = 3;
        elseif isequal(Shared.POSTPROCESS, '1-2')
            finish = length(labels) - 2;
        end

        % Check for misclassified labels
        for i = start:finish
            
            % Check left-current-right classes
            left = isequal(labels{1,i-1}, class);
            right = isequal(labels{1,i+1}, class);
            if isequal(Shared.POSTPROCESS, '2-1')
                left = isequal(labels{1,i-1}, class) || isequal(labels{1,i-2}, class);
            elseif isequal(Shared.POSTPROCESS, '1-2')
                right = isequal(labels{1,i+1}, class) || isequal(labels{1,i+2}, class);
            end
            current = ~isequal(labels{1,i}, class);
           
            % Replace the class if matches the criterium
            if left && right && current
                labels{1,i} = class;
            end
        end
        
    end
        
    % Set wrong labels to noGestute
    for i = 1:length(labels)
        if ~isequal(labels{1,i}, class)
            labels{1,i} = 'noGesture';
        end
    end
    
    % Transform to categorical
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
    
    % Set the class if labels are over the theashold
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
        if isequal(Shared.FILLING_TYPE, 'during') && finish > numPoints
            
            extraPoints = finish - numPoints + 1;
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

calculateValidationResults
     % ===== JUST FOR TESTING =====
    data = struct('classifications', classifications, 'recognitions', ... 
            recognitions, 'overlapings', overlapings, 'procesingTimes', procesingTimes);
    % ===== JUST FOR TESTING =====
%}


    
