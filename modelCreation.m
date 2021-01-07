%% THE TRAINING AND VALIDATION PATHS ARE DEFINED
data_dir_training = fullfile('Datastores', 'training_datastore');
data_dir_validation = fullfile('Datastores', 'validation_datastore');

%% THE DATASTORES RE CREATED
% The classes are defined
classes = ["fist","noGesture","open","pinch","waveIn","waveOut"];
training_datastore = SpectrogramDatastore(data_dir_training);
validation_datastore = SpectrogramDatastore(data_dir_validation);
% data_sample = preview(training_datastore);
clear data_dir_training data_dir_validation

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = training_datastore.SequenceDimension;

%% DEFINE THE AMOUNT OF DATA
% The amount of data to be used in the creation is specified ]0:1]
training_datastore = setDataAmount(training_datastore, 0.75);
validation_datastore = setDataAmount(validation_datastore, 0.75);

%% THE DATA IS DIVIDED IN TRAINING-VALIDATION-TESTING
% The training-validation-tests data is obtained
[validation_datastore, testing_datastore] = divide_datastore(validation_datastore);
% The total data for training-validation-tests is obtained
num_training_samples = ['Training samples: ', num2str(training_datastore.NumObservations)];
num_validation_samples = ['Validation samples: ', num2str(validation_datastore.NumObservations)];
num_testing_samples = ['Testing samples: ', num2str(testing_datastore.NumObservations)];
% The amount of training-validation-tests data is printed
fprintf('\n%s\n%s\n%s\n',num_training_samples,num_validation_samples, num_testing_samples);
clear datastore num_training_samples num_validation_samples num_testing_samples

%% THE NEURAL NETWORK ARCHITECTURE IS DEFINED
numHiddenUnits = 200;
filterSize = 2;
numFilters = 20;
numClasses = length(classes);
layers = [ ...
    sequenceInputLayer(inputSize,'Name','input')
    
    sequenceFoldingLayer('Name','fold')
    
    convolution2dLayer(filterSize,numFilters,'Name','conv') 
    batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    maxPooling2dLayer([2 2],"Name","maxpool")
    
    dropoutLayer(0.2,"Name","drop")
    
    convolution2dLayer(filterSize,2*numFilters,'Name','conv_1') 
    batchNormalizationLayer('Name','bn_1')
    reluLayer('Name','relu_1')
    
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm')
    
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
clear inputSize numHiddenUnits filterSize numFilters numClasses

%% LINK FOLD AND UNFOLD LAYERS
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');
clear layers

%% THE OPTIONS ARE DIFINED
maxEpochs = 1;
miniBatchSize = 8;
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'ValidationData', validation_datastore, ...
    'ValidationFrequency',50, ...
    'ValidationPatience',5, ...
    'Plots','training-progress');
clear maxEpochs miniBatchSize

%% NETWORK TRAINING
net = trainNetwork(training_datastore, lgraph, options);
clear options 

%% CALCULATE ACCURACIES
training_acc = calculateAccuracy(net,training_datastore);
validation_acc = calculateAccuracy(net, validation_datastore);
testing_acc = calculateAccuracy(net, testing_datastore);
% Print accuracies
text_training_acc = ['Training samples: ', num2str(training_acc)];
text_validation_acc = ['Validation samples: ', num2str(validation_acc)];
text_testing_acc = ['Testing samples: ', num2str(testing_acc)];
% The amount of training-validation-tests data is printed
fprintf('\n%s\n%s\n%s\n',text_training_acc, text_validation_acc, text_testing_acc);
clear text_training_acc text_validation_acc text_testing_acc

%% PLOT PREDICCTION/REAL SAMPLE FROM DATASET
plotPredictionDatastore(net, testing_datastore, 100);

%% FUNCTION TO DIVIDE DATASTORE IN TWO HALVES
function [first_datstore, second_datastore] = divide_datastore(dataStore)
    % first_datstore(50%) && second_datastore(50%)
    first_datstore =  partition(dataStore,2,1);
    second_datastore = partition(dataStore,2,2);
end

%% FUNCTION TO SET THE AMOUNT OF DATA
function datastore = setDataAmount(datastore, data_amount)
    % Obtain the limit value index
    idx_limit = floor(size(datastore.Labels,1) * data_amount);
    % The data is split within the datastore
    datastore.Datastore.Files = datastore.Datastore.Files(1:idx_limit);
    datastore.Labels = datastore.Labels(1:idx_limit);
    % NumObservations must be counted again
    reset(datastore);
end

%% FUNCTION TO CALCULATE ACCURACY
function acc = calculateAccuracy(net, datastore)
    data_similarity = zeros(datastore.NumObservations, 1);
    idx = 1;
    while(hasdata(datastore))
        data = read(datastore);
        yPred = classify(net,data.sequences);
        yReal = data.label_sequences;
        for i = 1:length(yReal)
            data_similarity(idx) = sum(yPred{i} == yReal{i})./numel(yReal{i});
            idx = idx + 1;
        end
    end
    acc = mean(data_similarity);
    reset(datastore);
end

%% 
function plotPredictionDatastore(net, datastore, idx)
    reset(datastore)
    % Validate number of sample
    idx_max = datastore.NumObservations;
    if idx<1, idx=1; elseif idx>idx_max, idx=idx_max; end
    % Recovering the sample
    batch_size = datastore.MiniBatchSize;
    count = 0;
    while idx > count
        data = read(datastore);
        count = count + batch_size;
    end
    idx = idx - (count - batch_size);
    yPred = classify(net,data.sequences{idx});
    yReal = data.label_sequences{idx};
    plotPredictionComparison(yReal, yPred)
    reset(datastore)
end

%%  
function plotPredictionComparison(YTest, YPred)
    figure
    plot(YPred,'.-')
    hold on
    plot(YTest)
    hold off
    xlabel("Frame")
    ylabel("Gesture")
    title("Predicted Gestures")
    legend(["Predicted" "Test Data"])
end

%% ARQUITECTURAS PROVADAS
%{
layers = [ ...
    sequenceInputLayer(dimensions)
    flattenLayer
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
%}
%{
convolution2dLayer(filterSize,numFilters,'Name','conv_1') 
    batchNormalizationLayer('Name','bn_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer([2 2],"Name","maxpool_1")
%}
%{
layers = [ ...
    sequenceInputLayer(inputSize,'Name','input')
    
    sequenceFoldingLayer('Name','fold')
    
    convolution2dLayer(filterSize,numFilters,'Name','conv')
    batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    %
    maxPooling2dLayer([2 2],"Name","maxpool")
    
    dropoutLayer(0.2,"Name","drop")
    
    convolution2dLayer(filterSize,2*numFilters,'Name','conv_1') 
    batchNormalizationLayer('Name','bn_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer([2 2],"Name","maxpool_1")
    
    %
    
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm')
    
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
%}