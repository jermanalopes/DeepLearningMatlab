close all
clear all
clc
% 
mfile=matlab.desktop.editor.getActiveFilename;%this m file folder(directory) with its m file name
[mfolder, ~, ~] = fileparts(mfile);% this m file directory
cd(mfolder)% change this m file directory
for i=1:201
d{1,i}= load([mfolder '\QTDataset\ecg' num2str(i) '.mat']);
end
%% Mostrar as ondas cores diferentes no plot
data = {};
data.ecgSignal= d{1, 1}.ecgSignal;
data.signalRegionLabels = d{1, 1}.signalRegionLabels;
data = struct2cell(data)
displayWaveLabels(data,true,1000);

% Divide data train and test
[data_train,data_test] = dividedata(d,0.7,0.3);

% Todos os dados com a mesma dimensão
[data_train, data_test] = same_dim(data_train, data_test);

% Transforma os labels em amostrais e nao em intervalos
 [Tdata_train] = transform_labels(data_train);
 [Tdata_test] = transform_labels(data_test);

% Transforma o sinal e os labels em estruturas de tamanho de 5000.
  trainDs = resizeDataML(5000, Tdata_train);
  testDs = resizeDataML(5000, Tdata_test);
  
%% Configurações da Deep Learning
  load('layers.mat');
  load('options.mat');

% tall - transforma o processamento dos dados em paralelo. 
  tallTrainSet = tall(trainDs);
  tallTestSet = tall(testDs);

  trainData = gather(tallTrainSet); 
  testData = gather(tallTestSet); 
%%

data = trainData(:,1);
labels = trainData(:,2);


net = trainDeep(data,labels,layers,options);

predTest = classify(net,testData(:,1),'MiniBatchSize',50);



