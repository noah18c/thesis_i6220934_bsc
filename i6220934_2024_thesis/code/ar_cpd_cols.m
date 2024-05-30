function predictions = ar_cpd_cols(training_series, num_predict, ar_order, cp_order, varargin)
    addpath('./tensorlab/');
    % Function to predict future points in a time series using 
    % Canonical Polyadic Decomposition (CPD) and 
    % Autoregressive (AR) modeling. AR is trained on individual components
    % third dimension of the tensor and predicts next point in that dimension, 
    % then the final prediction is made. All predictions of individual
    % components are added to create final prediction.
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
    defaultEmbedding = 1; % 1 is Hankel, 2 is segmentation
    defaultPlotCompare = false;
    defaultEven = false; % true = remove first value of uneven sequences for embedding 2
    defaultMethod = @mean;


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
    addParameter(p, 'L', [], @isnumeric);
    addParameter(p, 'M', [], @isnumeric);
    addParameter(p, 'even', defaultEven, @islogical);
    addParameter(p, 'Method', defaultMethod);

    % Parse inputs
    parse(p, training_series, num_predict, ar_order, cp_order, varargin{:});

    % Extract values from the input parser
    embedding = p.Results.embedding;
    plotCompare = p.Results.plotCompare;
    evenSequence = p.Results.even;
    method = p.Results.Method;

    % Set default values for L and M based on embedding method
    if embedding == 1
        defaultL = floor(length(training_series) / 3);
        defaultM = floor(length(training_series) / 3);
    elseif embedding == 2 || embedding == 3
        defaultL = floor(length(training_series)^(1/3));
        defaultM = floor(length(training_series)^(1/3));
    else
        error('Invalid embedding value.');
    end

    % Override defaults if L and M are provided
    L = defaultL;
    if ~isempty(p.Results.L)
        L = p.Results.L;
    end
    M = defaultM;
    if ~isempty(p.Results.M)
        M = p.Results.M;
    end

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
                title('Original vs Serialized Hankel tensor components')
                plot(training_series);
                hold on;
                for tcomp = 1:R
                    tensor_component = cpdgen({fac{1}(:, tcomp), fac{2}(:, tcomp), fac{3}(:, tcomp)});
                    tensor_series = dehankelize(tensor_component,'Dims', 1:3, 'Method',method);
                    plot(tensor_series);
                end
                legend({'Original', 'Component 1', 'Component 2', 'Component 3'});
                xlabel('Time');
                ylabel('Value');
                hold off;
            end

        case 2

            % Decomposition using Hankel Tensor (CPD)
            S3D = segmentize(training_series, 3, 'Segsize', [L M], 'UseAllSamples', true);
            %disp("Dimension of hankel: "+size(H3D));
        
        
            % Compute CPD
            R = cp_order; % Number of components
            [fac, ~] = cpd(S3D, R);
        
            % Initialize all the dimensions of the low-rank tensor approximation
            comp_pred = zeros(num_predict, R);
        
            % Add all the factors of the components together
            for tcomp = 1:R
                a = fac{1}(:, tcomp);
                b = fac{2}(:, tcomp);
                c = fac{3}(:, tcomp);
        
                model_tcomp = ar(c, ar_order);

                % due to the way segmentation works, AR needs to predict way less
                % points than with Hankel. More dependent on already
                % existing a and b
                a_predict = a(1:min(num_predict,length(a)));
                b_predict = b(1:min(ceil(num_predict/length(a)),length(b)));
                c_predict = forecast(model_tcomp, c, ceil(num_predict/(length(a)*length(b))));    

                % Compute the outer product of a, b, and the predicted c values
                predicted_segments = outprod(a_predict, b_predict, c_predict);

                % Desegmentize in order to get the predictions linearly
                desegmented_predictions = desegmentize(predicted_segments,'Method', method);

                % Desegmentation might include too many predictions
                final_comp_predict = desegmented_predictions(1:num_predict);

                comp_pred(:, tcomp) = final_comp_predict;
            end
        
            predictions = sum(comp_pred, 2);
        
            if plotCompare
                figure;
                title('Original vs Serialized segmented tensor components')
                plot(training_series);
                hold on;
                for tcomp = 1:R
                    tensor_component = cpdgen({fac{1}(:, tcomp), fac{2}(:, tcomp), fac{3}(:, tcomp)});
                    tensor_series = desegmentize(tensor_component,'Dims',1:3);
                    plot(tensor_series);
                end
                legend({'Original', 'Component 1', 'Component 2', 'Component 3'});
                xlabel('Time');
                ylabel('Value');
                hold off;
            end
        case 3
            
            % Decomposition using Hankel Tensor (CPD)
            D3D = segmentize(training_series, 3, 'Segsize', [L M], 'UseAllSamples', true);
            %disp("Dimension of hankel: "+size(H3D));
        
        
            % Compute CPD
            R = cp_order; % Number of components
            [fac, ~] = cpd(D3D, R);
        
            % Initialize all the dimensions of the low-rank tensor approximation
            comp_pred = zeros(num_predict, R);
        
            % Add all the factors of the components together
            for tcomp = 1:R
                a = fac{1}(:, tcomp);
                b = fac{2}(:, tcomp);
                c = fac{3}(:, tcomp);
        
                model_tcomp = ar(c, ar_order);

                % due to the way segmentation works, AR needs to predict way less
                % points than with Hankel. More dependent on already
                % existing c and b
                a_predict = forecast(model_tcomp, a, ceil(num_predict/(length(b)*length(c))));
                b_predict = b(1:min(ceil(num_predict/length(c)),length(b)));
                c_predict = c(1:min(num_predict,length(c)));

                % Compute the outer product of c, b, and the predicted a values
                predicted_decimates = outprod(a_predict, b_predict, c_predict);

                % Desegmentize in order to get the predictions linearly
                dedecimated_predictions = dedecimate(predicted_decimates,'Method', method);

                % Desegmentation might include too many predictions
                final_comp_predict = dedecimated_predictions(1:num_predict);

                comp_pred(:, tcomp) = final_comp_predict;
            end
        
            predictions = sum(comp_pred, 2);
        
            if plotCompare
                figure;
                title('Original vs Serialized segmented tensor components')
                plot(training_series);
                hold on;
                for tcomp = 1:R
                    tensor_component = cpdgen({fac{1}(:, tcomp), fac{2}(:, tcomp), fac{3}(:, tcomp)});
                    tensor_series = desegmentize(tensor_component,'Dims',1:3);
                    plot(tensor_series);
                end
                legend({'Original', 'Component 1', 'Component 2', 'Component 3'});
                xlabel('Time');
                ylabel('Value');
                hold off;
            end
    end
end
