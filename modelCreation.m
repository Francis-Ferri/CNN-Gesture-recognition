%% THE TRAINING PATH IS DEFINED
data_dir = fullfile('Datastores', 'training_datastore');

%% THE DATASTORE IS CREATED
% The classes are defined
classes = ["fist","noGesture","open","pinch","waveIn","waveOut"];
datastore = SpectrogramDatastore(data_dir);
clear data_dir

%% VISUALIZE THE DATA FORMAT
table = preview(datastore);

%% THE INPUT DIMENSIONS ARE DEFINED
dimensions = get_input_dimensons(datastore);

%% DEFINE THE AMOUNT OF DATA
% The amount of data to be used in the creation is specified] 0: 1]
data_amount = 0.75; %1
datastore = setDataAmount(datastore, data_amount);
clear data_amount

%% THE DATA IS DIVIDED IN TRAINING-VALIDATION-TESTING
% The training-validation-tests data is obtained
[training_datastore, validation_datastore, testing_datastore] = divide_datastore(datastore);
% The total data for training-validation-tests is obtained
num_training_samples = ['Training samples: ', num2str(training_datastore.NumObservations)];
num_validation_samples = ['Validation samples: ', num2str(validation_datastore.NumObservations)];
num_testing_samples = ['Testing samples: ', num2str(testing_datastore.NumObservations)];
% The amount of training-validation-tests data is printed
fprintf('\n%s\n%s\n%s\n',num_training_samples,num_validation_samples, num_testing_samples);
clear datastore num_training_samples num_validation_samples num_testing_samples

%%
numHiddenUnits = 200;
numClasses = length(classes);
layers = [ ...
    sequenceInputLayer(dimensions)
    flattenLayer
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
clear dimensions numHiddenUnits numClasses

%%
miniBatchSize = 1;
options = trainingOptions('adam', ...
    'MaxEpochs',1, ...
    'GradientThreshold',2, ...
    'MiniBatchSize',miniBatchSize, ...
    'Verbose',0, ...
    'Plots','training-progress');
clear miniBatchSize

%%
net = trainNetwork(training_datastore, layers, options);

%%
predict = read(testing_datastore);

%%
YPred = classify(net,predict.sequences);


%%

maxEpochs = 100;
miniBatchSize = 8;
options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');

%%
net = trainNetwork(training_datastore,layers,options);
%% THE NETWORK ARCHITECTURE AND THE TRAINING PARAMETERS ARE ESTABLISHED
layers = [
    imageInputLayer(dimensions,"Name","imageinput","Normalization","none")
    convolution2dLayer([2 2],10,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same","Stride",[10 10])
    convolution2dLayer([2 2],10,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same","Stride",[10 10])
    fullyConnectedLayer(6,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

%% CONFIGURACION DE LA RED
minibatchsize = 8;
options = trainingOptions('sgdm', ...
    'MiniBatchSize',minibatchsize, ...
    'InitialLearnRate',0.001, ... %0.0001
    'ExecutionEnvironment','cpu', ... %gpu %multi-gpu
    'MaxEpochs',5, ...
    'ValidationData', validation_datastore, ...  
    'Verbose',false, ...   
    'Plots','training-progress', ...
    'Shuffle','every-epoch');
% 'ValidationFrequency',300, ...
% 'DispatchInBackground',true, ...
%'Shuffle','never', ...
% Se limpian variables innecesarias
clear minibatchsize 

%%
% model = trainNetwork(training_datastore, layers, options);

%%

%% FUNCTION DIVIDE DATASTORE IN TRAINING, VALIDATION AND TESTING
function [training_datastore, validation_datastore, testing_datastore] = divide_datastore(dataStore)
    % Training = 50%
    training_datastore =  partition(dataStore,2,1);
    remainder_datastore = partition(dataStore,2,2);
    % Validation = 25%
    validation_datastore = partition(remainder_datastore,2,1);
    % Testing = 25%
    testing_datastore = partition(remainder_datastore,2,2);
end

%% FUNCTION TO GET THE INPUT DIMESIONS
function dimensions = get_input_dimensons(datastore)
    % Sample of data from the DataStore
    table = preview(datastore);
    structure = table{1,1};
    sample_data = structure{1,1};
    dimensions = [size(sample_data,1), size(sample_data,2),size(sample_data,3)];
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

%%
%{
%% Get the sequence lengths for each observation.
numObservations = training_datastore.NumObservations;
for i=1:numObservations
    sequence = load(training_datastore.Datastore.Files{i}).frames;
    sequenceLengths(i) = size(sequence,1);
end
%%
[sequenceLengths,idx] = sort(sequenceLengths);
training_datastore.Datastore.Files = training_datastore.Datastore.Files(idx);
training_datastore.Labels = training_datastore.Labels(idx);

%%
figure
bar(sequenceLengths)
ylim([0 55])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")
%%
%}

