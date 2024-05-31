function [all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex_rmse(signal_params, max_signals, num_experiments, LM_params, num_predict, varargin)
    addpath('./tensorlab/');

     % parameters for period, amplitude, sampling frequency (Hz), and interval to be tested
    % example signal parameter setup (needs to have exactly 4 columns)
    %{
    signal_params = [
        1, 1, 1, 100;
        10, 1, 1, 100;
        1, 100, 1, 100;
        10, 100, 1, 100;
        1, 1, 1, 200;
        10, 1, 1, 200;
        1, 100, 1, 200;
        10, 100, 1, 200
    ];
    %}

    % Add default values for optional parameters
    defaultOptimalOrder = 10;
    defaultNumComponents = 1;
    defaultThreshold = 0.001;
    defaultEmbedding = 1;
    defaultMethod = @mean;
    
    % Create an input parser
    p = inputParser;
    
    % Add required parameters
    addRequired(p, 'signal_params');
    addRequired(p, 'max_signals');
    addRequired(p, 'num_experiments');
    addRequired(p, 'LM_params');
    addRequired(p, 'num_predict');
    
    % Add optional name-value pair parameters
    addParameter(p, 'optimal_order', defaultOptimalOrder);
    addParameter(p, 'num_components', defaultNumComponents);
    addParameter(p, 'threshold', defaultThreshold);
    addParameter(p, 'embedding', defaultEmbedding);
    addParameter(p, 'method', defaultMethod);
    
    % Parse the inputs
    parse(p, signal_params, max_signals, num_experiments, LM_params, num_predict, varargin{:});
    
    % Assign parsed values to variables
    optimal_order = p.Results.optimal_order;
    num_components = p.Results.num_components;
    threshold = p.Results.threshold;
    embedding = p.Results.embedding;
    method = p.Results.method;
    
    max_signals_param = size(signal_params, 1);

    % num_predict, 7 models, number of different parameter setups
    all_errors_gt1_mean = zeros(num_predict, 7, max_signals_param);
    all_errors_gt2_mean = zeros(num_predict, 7, max_signals_param);
    
    for sim_param = 1:max_signals_param
        
        % Number of predictions by number of simulations of different signals for
        % different simulation parameters
        all_errors_gt1 = zeros(num_predict, 7, max_signals);
        all_errors_gt2 = zeros(num_predict, 7, max_signals);
    
        for sim = 1:max_signals
            disp("Parameter simulation " + sim_param + "/" + max_signals_param);
            disp("Generated signal " + sim + "/" + max_signals);
            
            %this parameter exist such that the learning part is always
            %of constant length no matter how many predictions you
            %want.
            lookout = num_predict - 1;
            time_series = rsignal(signal_params(sim_param, 1), signal_params(sim_param, 2), signal_params(sim_param, 3), signal_params(sim_param, 4)+lookout);

            N = length(time_series); % Number of sampling points in the time series
            
            % Ground truths for final point
            ground_truths1 = time_series(end - num_predict + 1:end); 

            % create noise on series proportionate to its range
            noise_param = signal_params(sim_param,5);
            noisy_series = noisify(noise_param,N,time_series);

            ground_truths2 = noisy_series(end - num_predict + 1:end);
            
            ground_truths = {ground_truths1, ground_truths2};
            training_series = noisy_series(1:end - num_predict);   
                
            % Prepare storage for predictions
            predictions_AR = zeros(num_predict, num_experiments);
            predictions_SVD = zeros(num_predict, num_experiments);
            predictions_cpd_s = zeros(num_predict, num_experiments);
            predictions_cpd_f = zeros(num_predict, num_experiments);
            predictions_cpd_cols = zeros(num_predict, num_experiments);
            predictions_cpd_colf = zeros(num_predict, num_experiments);
            predictions_mlsvd = zeros(num_predict, num_experiments); 

            % Prepare storage for real-time duration
            duration_AR = zeros(num_experiments, 1);
            duration_SVD = zeros(num_experiments, 1);
            duration_cpd_s = zeros(num_experiments, 1);
            duration_cpd_f = zeros(num_experiments, 1);
            duration_cpd_cols = zeros(num_experiments, 1);
            duration_cpd_colf = zeros(num_experiments, 1);
            duration_mlsvd = zeros(num_experiments, 1);

            % Perform the experiment
            for experiment = 1:num_experiments 
                if mod(experiment, round(num_experiments / 4)) == 0 && mod(experiment, 2) == 0
                    disp("iter " + experiment);
                elseif mod(experiment, round(num_experiments / 4)) == 0 || experiment == num_experiments
                    disp("iter " + experiment);
                end

                % Generate predictions once

                tic;
                model_AR = ar(training_series, optimal_order);  % AR(optimal_order) model
                predictions_AR(:, experiment) = forecast(model_AR, training_series, num_predict);
                duration_AR(experiment) = toc;
                
                tic;
                predictions_SVD(:, experiment) = ar_svd(training_series, num_predict, optimal_order, threshold(1), 'L', LM_params(1,1), 'embedding', embedding, 'Method', method);
                duration_SVD(experiment) = toc;

                tic;
                predictions_cpd_s(:, experiment) = ar_cpd_s(training_series, num_predict, optimal_order, num_components(1),'L', LM_params(2,1), 'M', LM_params(2,2), 'embedding', embedding, 'Method', method);
                duration_cpd_s(experiment) = toc;

                tic;
                predictions_cpd_f(:, experiment) = ar_cpd_f(training_series, num_predict, optimal_order, num_components(2), 'L', LM_params(3,1), 'M', LM_params(3,2),'embedding', embedding, 'Method', method);
                duration_cpd_f(experiment) = toc;
                
                tic;
                predictions_cpd_cols(:, experiment) = ar_cpd_cols(training_series, num_predict, optimal_order, num_components(3), 'L', LM_params(4,1), 'M', LM_params(4,2),'embedding', embedding, 'Method', method);
                duration_cpd_cols(experiment) = toc;

                tic;
                predictions_cpd_colf(:, experiment) = ar_cpd_colf(training_series, num_predict, optimal_order, num_components(4),'L', LM_params(5,1), 'M', LM_params(5,2), 'embedding', embedding, 'Method', method);
                duration_cpd_colf(experiment) = toc;

                tic;
                predictions_mlsvd(:, experiment) = ar_mlsvd(training_series, num_predict, optimal_order, threshold(2),'L', LM_params(6,1), 'M', LM_params(6,2), 'embedding', embedding, 'Method', method);
                duration_mlsvd(experiment) = toc;
            end   
                   
            % Calculate statistics over number of experiments done
            for gt = 1:2
                errors_AR = [rmse(predictions_AR, ground_truths{gt}, 2)];
                errors_SVD = [rmse(predictions_SVD, ground_truths{gt}, 2)];
                errors_cpd_s = [rmse(predictions_cpd_s, ground_truths{gt}, 2)];
                errors_cpd_f = [rmse(predictions_cpd_f, ground_truths{gt}, 2)];
                errors_cpd_cols = [rmse(predictions_cpd_cols, ground_truths{gt}, 2)];
                errors_cpd_colf = [rmse(predictions_cpd_colf, ground_truths{gt}, 2)];
                errors_mlsvd = [rmse(predictions_mlsvd, ground_truths{gt}, 2)];
                
                % Depending on ground truth, set stats to either errors_concat1 or 2
                if gt == 1
                    errors_concat1 = [errors_AR, errors_SVD, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf, errors_mlsvd];
                else
                    errors_concat2 = [errors_AR, errors_SVD, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf, errors_mlsvd];
                end
            end
    
            % Add matrix of error calculations as a slice to 3D matrix
            all_errors_gt1(:, :, sim) = errors_concat1;
            all_errors_gt2(:, :, sim) = errors_concat2;
        end
        
        % Below are the average evaluation scores (mean of matrix slices)
        % rows: mean; sd; RMSE; MRE; MRSE; duration
        % columns: AR, SVD, CPD_S, CPD_F, CPD_Cols, CPD_ColF, MLSVD
        % slices: results per signal parameter set
        all_errors_gt1_mean(:, :, sim_param) = mean(all_errors_gt1, 3);
        all_errors_gt2_mean(:, :, sim_param) = mean(all_errors_gt2, 3);
    
    end
    disp("Done!");
end
