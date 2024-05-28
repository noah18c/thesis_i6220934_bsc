function predictions = ar_cpd_ms(training_series, num_predict, ar_order, cp_order, varargin)
    % Function to predict future points in a time series using 
    % Canonical Polyadic Decomposition (CPD) and 
    % Autoregressive (AR) modeling. Decomposes Hankel tensor using CPD,
    % trains AR on individual components and sums predictions for final
    % prediction
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
    %       'embedding' - the embedding method: 1 for Hankel, 2 for segmentation
    %
    % Outputs:
    %   predictions - the predicted future points

    % Define default values for optional parameters
    defaultEmbedding = 1; % 1 is Hankel, 2 is segmentation
    defaultPlotCompare = false;

    % Create an input parser
    p = inputParser;

    % Add required parameters
    addRequired(p, 'training_series', @isnumeric);
    addRequired(p, 'num_predict', @isnumeric);
    addRequired(p, 'ar_order', @isnumeric);
    addRequired(p, 'cp_order', @isnumeric);

    % Add optional name-value pair parameters
    addParameter(p, 'embedding', defaultEmbedding, @isnumeric);
    addParameter(p, 'plotCompare', defaultPlotCompare, @islogical);

    % Parse initial inputs to determine embedding
    parse(p, training_series, num_predict, ar_order, cp_order, varargin{:});
    embedding = p.Results.embedding;

    % Set default values for L and M based on embedding method

    switch embedding
        case 1
            defaultL = floor(length(training_series) / 3);
            defaultM = floor(length(training_series) / 3);
        case 2
            defaultL = floor(length(training_series)^(1/3));
            defaultM = floor(length(training_series)^(1/3));
        otherwise
            error('Invalid embedding value.');
    end

    % Add L and M parameters after determining the embedding method
    addParameter(p, 'L', defaultL, @isnumeric);
    addParameter(p, 'M', defaultM, @isnumeric);

    % Re-parse inputs to include L and M with correct defaults
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

    switch embedding
        case 1
            % Decomposition using Hankel Tensor (CPD)
            H3D = hankelize(training_series, 'Sizes', [L, M]);
        
            % Compute CPD
            R = cp_order; % Number of components
            [fac, ~] = cpd(H3D, R);
        
            pred_tcomponents = zeros(num_predict, R);
        
            if plotCompare
                figure;
                title("Original vs Serialized components")
                plot(training_series);
                hold on;
            end
        
            for tcomp = 1:R
                tensor_component = cpdgen({fac{1}(:, tcomp), fac{2}(:, tcomp), fac{3}(:, tcomp)});
                
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
                pred_tcomponents(:, tcomp) = forecast(model_tcomp, tensor_series, num_predict);
            end
          
            predictions = sum(pred_tcomponents, 2);

        case 2
            % Decomposition using Segmented Tensor (CPD)
            S3D = segmentize(training_series,3,'Segsize',[L M],'UseAllSamples',true);
        
            % Compute CPD
            R = cp_order; % Number of components
            [fac, ~] = cpd(S3D, R);
        
            pred_tcomponents = zeros(num_predict, R);
        
            if plotCompare
                figure;
                title("Original vs Serialized components")
                plot(training_series);
                hold on;
            end
        
            for tcomp = 1:R
                tensor_component = cpdgen({fac{1}(:, tcomp), fac{2}(:, tcomp), fac{3}(:, tcomp)});
                
                % Reconstruct time series data
                tensor_series = desegmentize(tensor_component,'Dims',1:3);
        
                if plotCompare
                    plot(tensor_series);
                    legend({"Original"});
                    xlabel("Time");
                    ylabel("Value");
                    hold off;
                end
                
                model_tcomp = ar(tensor_series, ar_order);
                pred_tcomponents(:, tcomp) = forecast(model_tcomp, tensor_series, num_predict);
            end
          
            predictions = sum(pred_tcomponents, 2);

    end

    
end
