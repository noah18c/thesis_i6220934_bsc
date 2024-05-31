function [bestL, bestM] = bestLM(matrix)
    % Get frequency and value with most occurences
    [~, F, C] = mode(matrix,1);
    
    % Check which column is best, if both then it will take the first column
    if F(1,1) == F(1,2)
        if length(C{1})>1
            bestL = split_tie(C{1},matrix(:,1));
        else
            bestL = C{1};
        end
        
        % get the corresponding M values of L
        M_options = matrix(matrix(:,1) == bestL,2);
    
        bestM = split_tie(M_options,matrix(:,2));
    else
        % Check best dimension
        [~,best_dim] = max(F,[],2);
    
        % If there are multiple most occurring values, use split_tie function
        if length(C{best_dim})>1
            bestX = split_tie(C{best_dim},matrix(:,best_dim));
        else
            bestX = C{best_dim};
        end
        
        % get the corresponding values of other dimension
        if best_dim == 1
            bestL = bestX;
            Y_options = matrix(matrix(:,1) == bestL,2);
            bestM = split_tie(Y_options,matrix(:,2));
        else
            bestM = bestX;
            Y_options = matrix(matrix(:,2) == bestM,1);
            bestL = split_tie(Y_options,matrix(:,1));
        end
    end

end



