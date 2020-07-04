function [data_train, data_test] = same_dim(data_train, data_test)

% Tranformar todos os sinais de entrada (treinamento e teste)
% da rede neural do mesmo tamanho. 
% Seleciona o menor tamanho de ECG entre todos e deixa os demais 
% com esse tamanho

for t=1:length(data_train)
t_dim(t) = length(data_train{1, t}.ecgSignal) ;
end
[v, p] = min(t_dim);
for p=1:length(data_train)
data_train{1, p}.ecgSignal = data_train{1, p}.ecgSignal(1:v)';
end
for p=1:length(data_test)
data_test{1, p}.ecgSignal = data_test{1, p}.ecgSignal(1:v)';
end
end