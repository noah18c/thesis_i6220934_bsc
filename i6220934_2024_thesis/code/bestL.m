function bestL = bestL(vector)
    if size(vector,2) == 1
        [GcountsL, GroupsL] = groupcounts(vector);
        
        most_occurring_value = GroupsL(GcountsL == max(GcountsL));
        
        if length(most_occurring_value) > 1
            bestL = split_tie(most_occurring_value,vector);
        else
            bestL = most_occurring_value;
        end
    else
        error('Input needs to be nx1 vector');
    end

end