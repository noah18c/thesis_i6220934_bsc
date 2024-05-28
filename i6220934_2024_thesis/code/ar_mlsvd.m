function predictions = ar_mlsvd(training_series, num_predict, ar_order, reduction, varargin)
    % Function to predict future points in a time series using 
    % Multilinear Singular Value Decomposition (MLSVD) and 
    % Autoregressive (AR) modeling.
    %
    % Inputs:
    %   training_series - the input time series data for training
    %   reduction - the reduction factor to determine the size of the core
    %   tensor (between 0 and 1)
    %   num_predict - the number of future points to predict
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
    addRequired(p, 'reduction', @(x) isnumeric(x) && x > 0 && x <= 1);

    % Add optional name-value pair parameters
    addParameter(p, 'L', defaultL, @isnumeric);
    addParameter(p, 'M', defaultM, @isnumeric);
    addParameter(p, 'plotCompare', defaultPlotCompare, @islogical);

    % Parse inputs
    parse(p, training_series, num_predict, ar_order, reduction, varargin{:});

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

    % Approximate the dimensions so that H3D is almost cubic
    H3D = hankelize(training_series, 'Sizes', [L, M]);
    disp("Dimension of hankel: "+size(H3D));

    size_core = round([size(H3D, 1), size(H3D, 2), size(H3D, 3)] * reduction);
    
    [U, S] = lmlra(H3D, size_core);

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