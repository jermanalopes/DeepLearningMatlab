function outputCell = resizeDataML(N, inputCell)

% Transforma o sinal em estruturas de tamanho de N, o objetivo
% é reduzir o tamanho do sinal para análise de rede neural.
% A saída dessa função é uma cell (divisoesdeN x 2). Em que 2
% é sinal e lables.
    targetLength = N;
    sig = inputCell{:,1};
    mask = inputCell{:,2};
    
    % Get number of chunks
    numChunks = floor(size(sig,2)/targetLength);
    
    % Truncate signal and mask to integer number of chunks
    sig = sig(1:numChunks*targetLength);
    mask = mask(1:numChunks*targetLength);
    
    % Create a cell array containing signal chunks
    sigOut = reshape(sig,targetLength,numChunks)';
    sigOut = num2cell(sigOut,2);
    
    % Create a cell array containing mask chunks
    lblOut = reshape(mask,targetLength,numChunks)';
    lblOut = num2cell(lblOut,2);
    
    % Output a two-column cell array with all chunks
    outputCell = [sigOut, lblOut];

end