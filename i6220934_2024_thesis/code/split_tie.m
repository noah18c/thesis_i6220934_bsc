function bestX = split_tie(tie_values, all_values)

    medianValue = median(all_values);

    %get index of most occurring value that is closest to median
    [~,index] = min(abs(tie_values - medianValue));
    bestX = tie_values(index);
end