function predictions = ar_mlsvd(training_series, num_predict, ar_order, reduction,plotCompare)
    % Function to predict future points in a time series using 
    % Multilinear Singular Value Decomposition (MLSVD) and 
    % Autoregressive (AR) modeling.
    %
    % Inputs:
    %   training_series - the input time series data for training
    %   reduction - the reduction factor to determine the size of the core
    %   tensor (between 0 and 1)
    %   num_predict - the number of future points to predict
    %
    % Outputs:
    %   predictions - the predicted future points
    % Set default value for plot parameter if not provided
    if nargin < 5
        plotCompare = false;
    end

    % Approximate the dimensions s.t. H3D is almost cubic
    approx = floor(length(training_series)/3);
    
    H3D = hankelize(training_series, 'Sizes', [approx, approx]);
    
    size_core = round([size(H3D,1), size(H3D,2), size(H3D,3)] * reduction);
    
    [U, S] = lmlra(H3D, size_core);

    % Future programmer may train AR on each interaction of the orthonormal
    % bases. We wont do that due to computational limits. If need be, below
    % is how one would compute the low-rank approximation with a 1x1x1 core
    % x = S(1,1,1)*outprod(U{1}(:,1),U{2}(:,1),U{3}(:,1))
    
    % Reconstruct the low-rank tensor from lmlra factors
    low_rank_tensor = lmlragen(U, S);
    
    serial3d = serialize3D(low_rank_tensor);   
    
    model = ar(serial3d, ar_order);
    predictions = forecast(model, serial3d, num_predict);

    if plotCompare
        figure;
        title("Original vs Low-rank approx")
        plot(training_series);
        hold on;
        plot(serial3d);
        legend({"Original", "Low-rank approx"});
        xlabel("Time");
        ylabel("Value");
        hold off;
    end
end