function [Tdata_train] = transform_labels(data_train)
% Aplica os lables (P,QRS e T para cada amostra)
% Os dados de entrada s�o o sinal de ECG e os labels. Mas os lables
% est�o relacionados com intervalo de tempo. A fun��o utiliza esses 
% intervalos e aplica os labels para cada amostra. Logo, a sa�da da fun��o � o
% sinal ECG e os labels por amostra. 

for n=1:length(data_train)
new_label = [categorical];
new_label(1, 1:length(data_train{1,n}.ecgSignal)) = 'n/a';

tam = size(data_train{1,n}.signalRegionLabels);
for i=1:tam(1)
interval(i,:) = data_train{1,n}.signalRegionLabels{i,1};
label(i) = data_train{1,n}.signalRegionLabels{i,2};
end

for t=1:length(interval)
new_label(1, interval(t,1):interval(t,2)) = label(t);
end
Tdata_train{n,1} = data_train{1, n}.ecgSignal;  
Tdata_train{n,2} = new_label;
clear new_label;
clear interval;
clear label;
end
end
