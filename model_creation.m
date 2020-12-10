%% THE TRAINING PATH IS DEFINED
data_dir = fullfile('Datastores', 'training_datastore');

%% THE DATASTORE IS CREATED
% The classes are defined
classes = ["fist","noGesture","open","pinch","waveIn","waveOut"];
datastore = SpectrogramDatastore(data_dir);
clear data_dir

            %% Para pruebas
table = preview(datastore);

%% THE INPUT DIMENSIONS ARE DEFINED
% Get the input dimensions of the DataStore
dimensions = get_input_dimensons(datastore);

%% THE DATA IS MIXED
% The data is mixed to obtain a new order
datastore = shuffle(datastore);
% The amount of data to be used in the creation is specified] 0: 1]
data_amount = 0.75; %1
% Obtain the limit value index
idx_limit = floor(size(datastore.Labels,1) * data_amount);
% The data is split within the datastore
datastore.Datastore.Files = datastore.Datastore.Files(1:idx_limit);
datastore.Labels = datastore.Labels(1:idx_limit);
% NumObservations must be counted again
reset(datastore);
clear data_amount idx_limit

%% THE DATA IS DIVIDED IN TRAINING-VALIDATION-TESTING
% The training-validation-tests data is obtained
[training_datastore, validation_datastore, testing_datastore] = divide_datastore(datastore);
% The total data for training-validation-tests is obtained
num_training_samples = ['Training samples: ', num2str(training_datastore.NumObservations)];
num_validation_samples = ['Validation samples: ', num2str(validation_datastore.NumObservations)];
num_testing_samples = ['Testing samples: ', num2str(testing_datastore.NumObservations)];
% The amount of training-validation-tests data is printed
disp(num_training_samples);
disp(num_validation_samples);
disp(num_testing_samples);
% Se limpian variables innecesarias
clear datastore num_training_samples num_validation_samples num_testing_samples

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
    dimensions = size(sample_data);
end

