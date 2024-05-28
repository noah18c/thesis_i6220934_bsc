function predictions = ar_svd(training_series, num_predict, ar_order, svd_order, varargin)
    % Function to predict future points in a time series using 
    % Singular Value Decomposition (SVD) and Autoregressive (AR) modeling.
    %
    % Inputs:
    %   training_series - the input time series data for training
    %   num_predict - the number of future points to predict
    %   ar_order - the order of the AR model
    %   svd_order - the number of components to retain in SVD
    %   varargin - optional name-value pair arguments:
    %       'L' - the length of the first column of the Hankel matrix
    %       'plotCompare' - boolean flag to plot comparison of results; false by default
    %
    % Outputs:
    %   predictions - the predicted future points

    % Define default values for optional parameters
    defaultL = floor(length(training_series) / 2) + 1;
    defaultPlotCompare = false;

    % Create an input parser
    p = inputParser;

    % Add required parameters
    addRequired(p, 'training_series', @isnumeric);
    addRequired(p, 'num_predict', @isnumeric);
    addRequired(p, 'ar_order', @isnumeric);
    addRequired(p, 'svd_order', @isnumeric);

    % Add optional name-value pair parameters
    addParameter(p, 'L', defaultL, @isnumeric);
    addParameter(p, 'plotCompare', defaultPlotCompare, @islogical);

    % Parse inputs
    parse(p, training_series, num_predict, ar_order, svd_order, varargin{:});

    % Extract values from the input parser
    L = p.Results.L;
    plotCompare = p.Results.plotCompare;

    % Ensure L is within valid range
    if L > length(training_series) || L < 1
        error('L must be a positive integer less than or equal to the length of the training series.');
    end

    % Create Hankel matrix
    H2D = hankel(training_series(1:L), training_series(L:end))
    [U, S, V] = svd(H2D, 'econ');    

    num_components = svd_order;
    pred_components = zeros(num_predict, num_components);

    if plotCompare
        figure;
        title("Original vs Serialized components")
        plot(training_series);
        hold on;
    end

    for comp = 1:num_components
        % Tensor for holding all components (which are matrices)
        matrix_component = U(:,comp) * S(comp, comp) * V(:,comp)';

        % Convert component matrix back to time series by averaging along anti-diagonals
        component_series = serialize2D(matrix_component);

        if plotCompare
            plot(component_series);
            legend({"Original"});
            xlabel("Time");
            ylabel("Value");
            hold off;
        end
        
        model_comp = ar(component_series, ar_order);
        pred_components(:,comp) = forecast(model_comp, component_series, num_predict);
    end

    predictions = sum(pred_components, 2);

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
