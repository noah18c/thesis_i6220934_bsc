function predictions = ar_cpd_col(training_series, num_predict, ar_order, cp_order, varargin)
    addpath('./tensorlab/');
    % Function to predict future points in a time series using 
    % Canonical Polyadic Decomposition (CPD) and 
    % Autoregressive (AR) modeling. AR is trained on third dimension of the
    % tensor and predicts next point in that dimension, then outer product
    % of a, b, and predicted points compute the next num_predict points.
    %
    % Inputs:
    %   training_series - the input time series data for training
    %   num_predict - the number of future points to predict
    %   ar_order - the order of the AR model
    %   cp_order - the number of components for CPD
    %   varargin - optional name-value pair arguments:
    %       'L' - the first dimension of the Hankel tensor
    %       'M' - the second dimension of the Hankel tensor
    %       'plotCompare' - boolean flag to plot comparison of results; false by default
    %
    % Outputs:
    %   predictions - the predicted future points

    % Define default values for optional parameters
    defaultL = floor(length(training_series) / 3);
    defaultM = floor(length(training_series) / 3);
    defaultPlotCompare = false;

    % Create an input parser
    p = inputParser;

    % Add required parameters
    addRequired(p, 'training_series', @isnumeric);
    addRequired(p, 'num_predict', @isnumeric);
    addRequired(p, 'ar_order', @isnumeric);
    addRequired(p, 'cp_order', @isnumeric);

    % Add optional name-value pair parameters
    addParameter(p, 'L', defaultL, @isnumeric);
    addParameter(p, 'M', defaultM, @isnumeric);
    addParameter(p, 'plotCompare', defaultPlotCompare, @islogical);

    % Parse inputs
    parse(p, training_series, num_predict, ar_order, cp_order, varargin{:});

    % Extract values from the input parser
    L = p.Results.L;
    M = p.Results.M;
    plotCompare = p.Results.plotCompare;

    % Ensure L and M are within valid range
    if L > length(training_series) || L < 1
        error('L must be a positive integer less than or equal to the length of the training series.');
    end
    if M > length(training_series) || M < 1
        error('M must be a positive integer less than or equal to the length of the training series.');
    end

    % Decomposition using Hankel Tensor (CPD)
    H3D = hankelize(training_series, 'Sizes', [L, M]);
    %disp("Dimension of hankel: "+size(H3D));


    % Compute CPD
    R = cp_order; % Number of components
    [fac, ~] = cpd(H3D, R);

    % Initialize all the dimensions of the low-rank tensor approximation
    comp_pred = zeros(num_predict, R);

    % Add all the factors of the components together
    for tcomp = 1:R
        a = fac{1}(:, tcomp);
        b = fac{2}(:, tcomp);
        c = fac{3}(:, tcomp);

        model_tcomp = ar(c, ar_order);
        c_predict = forecast(model_tcomp, c, num_predict);

        % Compute the outer product of a, b, and the predicted c values
        final_comp_predict = zeros(num_predict, 1);
        for i = 1:num_predict
            final_comp_predict(i) = a(end) * b(end) * c_predict(i);
        end

        comp_pred(:, tcomp) = final_comp_predict;
    end

    predictions = sum(comp_pred, 2);

    if plotCompare
        figure;
        title('Original vs Serialized components')
        plot(training_series);
        hold on;
        for tcomp = 1:R
            tensor_component = cpdgen({fac{1}(:, tcomp), fac{2}(:, tcomp), fac{3}(:, tcomp)});
            tensor_series = serialize3D(tensor_component);
            plot(tensor_series);
        end
        legend({'Original', 'Component 1', 'Component 2', 'Component 3'});
        xlabel('Time');
        ylabel('Value');
        hold off;
    end
end
