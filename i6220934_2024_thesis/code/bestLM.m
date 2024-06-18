function [bestL, bestM] = bestLM(matrix)
    mean1 = mean(matrix(:,1));
    mean2 = mean(matrix(:,2));
    
    % choose the L value that is closest to the mean of the first column
    [~,index] = min(abs(matrix(:,1) - mean1));
    bestL = matrix(index);
    
    % get the options of the corresponding column
    M_options = matrix(matrix(:,1) == bestL,2);
    
    % choose the M value that is closest to the mean of that column
    [~, index] = min(abs(M_options-mean2));
    bestM = M_options(index);

end



