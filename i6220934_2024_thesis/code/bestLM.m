function [bestL, bestM] = bestLM(matrix)
    % bestLM Function to determine the best L and M values from a matrix.
    % The function finds the L value closest to the mean of the first column
    % and then finds the corresponding M value that is closest to the mean
    % of the second column.
    %
    % Syntax:
    %   [bestL, bestM] = bestLM(matrix)
    %
    % Inputs:
    %   matrix - A two-column matrix where the first column contains L values
    %            and the second column contains M values.
    %
    % Outputs:
    %   bestL  - The L value closest to the mean of the first column.
    %   bestM  - The M value closest to the mean of the second column 
    %            for the selected L value.
    %
    % Example:
    %   matrix = [1 2; 3 4; 5 6];
    %   [bestL, bestM] = bestLM(matrix)

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



