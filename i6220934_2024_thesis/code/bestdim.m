function [L, M] = bestdim(X)
    % The algorithm selects the most common or median-closest values for our L, M pairs.
    % For single-column matrices, it straightforwardly identifies the most common value.    
    % For multi-column matrices, it considers both columns, accounts for ties, and ensures 
    % the final selected values are the most representative of the L, M pairs.

    if size(X, 2) == 1
        % Find the unique values and their counts
        [counts, edges] = histcounts(X, 'BinMethod', 'integers');
        unique_values = edges(1:end-1);
        
        max_count = max(counts);
        median_value = median(X);
        
        most_common_values = unique_values(counts == max_count);
        [~, index] = min(abs(most_common_values - median_value));
        L = most_common_values(index);
        M = NaN;
    else
        [L, M] = choose_l_m(X);
    end
end

function [chosen_L, chosen_M] = choose_l_m(data)
    % Function to get the most frequent value or closest to median in case of a tie
    
    % Nested function to find the most frequent or median-closest value
    function value = most_frequent_or_closest_to_median(values)
        unique_values = unique(values);
        [counts, ~] = histcounts(values, [unique_values; max(unique_values)+1] - 0.5);
        max_count = max(counts);
        most_frequent_values = unique_values(counts == max_count);
        if length(most_frequent_values) > 1
            median_value = median(values);
            [~, idx] = min(abs(most_frequent_values - median_value));
            value = most_frequent_values(idx);
        else
            value = most_frequent_values;
        end
    end

    % Get the most frequent values for each column
    most_frequent_L = most_frequent_or_closest_to_median(data(:, 1));
    most_frequent_M = most_frequent_or_closest_to_median(data(:, 2));

    % Count occurrences in each column
    count_L = sum(data(:, 1) == most_frequent_L);
    count_M = sum(data(:, 2) == most_frequent_M);

    if count_L > count_M
        % Focus on rows where L is most frequent
        filtered_data = data(data(:, 1) == most_frequent_L, :);
        corresponding_M = filtered_data(:, 2);
        most_frequent_or_median_M = most_frequent_or_closest_to_median(corresponding_M);
        chosen_L = most_frequent_L;
        chosen_M = most_frequent_or_median_M;
    elseif count_M > count_L
        % Focus on rows where M is most frequent
        filtered_data = data(data(:, 2) == most_frequent_M, :);
        corresponding_L = filtered_data(:, 1);
        most_frequent_or_median_L = most_frequent_or_closest_to_median(corresponding_L);
        chosen_L = most_frequent_or_median_L;
        chosen_M = most_frequent_M;
    else
        % If there's a tie in the counts, choose the L and M pair
        chosen_L = most_frequent_L;
        chosen_M = most_frequent_or_closest_to_median(data(data(:, 1) == most_frequent_L, 2));
    end
end
