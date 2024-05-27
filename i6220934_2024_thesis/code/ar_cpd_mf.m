function predictions = ar_cpd_mf(training_series, num_predict, ar_order, cp_order, plotCompare)
    % Function to predict future points in a time series using 
    % Canonical Polyadic Decomposition (CPD) and 
    % Autoregressive (AR) modeling. Decomposes Hankel tensor using CPD,
    % trains AR on total reconstructed tensor.
    %
    % Inputs:
    %   training_series - the input time series data for training
    %   num_predict - the number of future points to predict
    %   ar_order - the order of the AR model
    %   cp_order - the number of components for CPD
    %   plotCompare - (optional) boolean flag to plot comparison of results; false by default
    %
    % Outputs:
    %   predictions - the predicted future points

    if nargin < 5
        plotCompare = false;
    end

    % Decomposition using Hankel Tensor (CPD)
    dim_approx = floor(length(training_series)/3);
    H3D = hankelize(training_series, 'Sizes', [dim_approx, dim_approx]);

    % Compute CPD
    R = cp_order; % Number of components
    [fac,~] = cpd(H3D, R);

    low_rank_tensor = cpdgen(fac);

    % Take average of the anti-diagonal planes
    tensor_series = serialize3D(low_rank_tensor);
   
    model_tcomp = ar(tensor_series, ar_order);
    predictions = forecast(model_tcomp, tensor_series, num_predict);
        

    if plotCompare
        figure;
        title("Original vs Low-Rank Approximation")
        plot(training_series);
        hold on;
        plot(tensor_series);
        legend({"Original", "Low-rank Approx"});
        xlabel("Time");
        ylabel("Value");
        hold off;
    end

end