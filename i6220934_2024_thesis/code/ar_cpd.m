function predictions = ar_cpd(training_series, num_predict, ar_order, cp_order, plotCompare)
    % Function to predict future points in a time series using 
    % Canonical Polyadic Decomposition (CPD) and 
    % Autoregressive (AR) modeling.
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

    pred_tcomponents = zeros(num_predict, R);

    if plotCompare
        figure;
        title("Original vs Serialized components")
        plot(training_series);
        hold on;
    end

    for tcomp = 1:R
        tensor_component = cpdgen({fac{1}(:,tcomp),fac{2}(:,tcomp),fac{3}(:,tcomp)});
        
        % Reconstruct time series data
        tensor_series = serialize3D(tensor_component);

        if plotCompare
            plot(tensor_series);
            legend({"Original"});
            xlabel("Time");
            ylabel("Value");
            hold off;
        end
        
        model_tcomp = ar(tensor_series, ar_order);
        pred_tcomponents(:,tcomp) = forecast(model_tcomp, tensor_series, num_predict);
    end
  
    predictions = sum(pred_tcomponents,2);

end