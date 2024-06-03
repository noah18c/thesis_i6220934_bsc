function [bestL, bestM] = bestLM(matrix)
    median1 = median(matrix(:,1));
    median2 = median(matrix(:,2));
    
    % choose the L value that is closest to the median of the first column
    [~,index] = min(abs(matrix(:,1) - median1));
    bestL = matrix(index);
    
    % get the options of the corresponding column
    M_options = matrix(matrix(:,1) == bestL,2);
    
    % choose the M value that is closest to the median of that column
    [~, index] = min(abs(M_options-median2));
    bestM = M_options(index);

end



