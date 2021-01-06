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
    
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm')
    
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];
clear inputSize numHiddenUnits numClasses

%%
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');

%% THE OPTIONS ARE DIFINED
maxEpochs = 3;
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
    'Plots','training-progress');
clear maxEpochs miniBatchSize

%%
net = trainNetwork(training_datastore, lgraph, options);

%%
predict = read(testing_datastore);

%%
YPred = classify(net,predict.sequences);

%%

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
