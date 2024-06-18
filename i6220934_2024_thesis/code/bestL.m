function bestL = bestL(vector)
    if size(vector,2) == 1
        mean1 = mean(vector(:,1));
        
        % choose the L value that is closest to the median of the first column
        [~,index] = min(abs(vector(:,1) - mean1));
        bestL = vector(index);
    else
        error('Input needs to be nx1 vector');
    end

end