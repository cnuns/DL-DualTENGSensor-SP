clc; clear; rng('default');

%% Load sequential data
load('dataCyc_h2050.mat')

%% Label normalization 
Ymax = 410;
Ymin = -410;
data = XTrain;
Yrange = range([Ymax, Ymin]);
YTrainNorm = (YTrain - Ymin)/Yrange;
labels = YTrainNorm;
                      
%% Data splitting within trainSet 
numChannels = size(data{1},1);
numObservations = numel(data);
[idxTrain, idxValidation, idxTest] = trainingPartitions(numObservations,[0.8 0.1 0.1]);
XTrain = data(idxTrain);
TTrain = labels(idxTrain);
XValidation = data(idxValidation);
TValidation = labels(idxValidation);
XTest = data(idxTest);
TTest = labels(idxTest);

%% Define TCN model
filterSize = 6;
numFilters = 16;
dropoutFactor = 0.005;
numBlocks = 4;

net = dlnetwork;
layer = sequenceInputLayer(numChannels,Normalization="zerocenter",Name="input");
net = addLayers(net,layer);
outputName = layer.Name;
for i = 1:numBlocks
    dilationFactor = 2^(i-1);

    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        spatialDropoutLayer(Probability=dropoutFactor)

        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
        layerNormalizationLayer
        reluLayer
        spatialDropoutLayer(Probability=dropoutFactor)

        additionLayer(2,Name="add_"+i)];

    % Add and connect layers.
    net = addLayers(net,layers);
    net = connectLayers(net,outputName,"conv1_"+i);

    % Skip connection.
    if i == 1
        % Include convolution in first skip connection.
        layer = convolution1dLayer(1,numFilters,Name="convSkip");

        net = addLayers(net,layer);
        net = connectLayers(net,outputName,"convSkip");
        net = connectLayers(net,"convSkip","add_" + i + "/in2");
    else
        net = connectLayers(net,outputName,"add_" + i + "/in2");
    end

    % Update layer output name.
    outputName = "add_" + i;
end

layers = [
    globalAveragePooling1dLayer(Name="avg")
    fullyConnectedLayer(14, Name="fc1")  % % here
    dropoutLayer(0.1)
    fullyConnectedLayer(1, Name="fc2")
    sigmoidLayer('Name', 'sigmoid')]; % here

net = addLayers(net,layers);
net = connectLayers(net,outputName,"avg");

%% Train TCN model
options = trainingOptions('adam', ...
    MiniBatchSize = 128*4, ...
    MaxEpochs=150, ...
    InitialLearnRate=0.01*0.5,...
    InputDataFormats="CTB", ...
    ValidationData={XValidation,TValidation}, ...
    ValidationFrequency=20, ...
    Shuffle="every-epoch", ...
    OutputNetwork="best-validation", ...
    Metrics="rmse", ...       
    Verbose=false);

[net, info] = trainnet(XTrain, TTrain, net, "mse", options);

%% Test TCN model within trainSet
miniBatchSizeTest = 128*4;
rmse = testnet(net,XTest,TTest,"rmse", MiniBatchSize = miniBatchSizeTest, InputDataFormats="CTB"); % here
YPred = minibatchpredict(net,XTest, MiniBatchSize = miniBatchSizeTest, InputDataFormats="CTB"); % here
rmse

%%
%pathtosave = ['\trainedmodel\'];
%save([pwd pathtosave savefilename], 'rmse', 'net', 'miniBatchSizeTest', 'XTrain', 'TTrain', 'XTest', 'TTest', 'YPred', 'XValidation', 'TValidation', 'info', 'Ymax', 'Ymin');









