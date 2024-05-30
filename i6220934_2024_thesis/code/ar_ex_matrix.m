function [best_L, all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex_matrix(signal_params, L_range, max_signals, num_experiments, varargin)
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
    defaultEmbedding = 1;
    defaultMethod = @mean;
    
    % Create an input parser
    p = inputParser;
    
    % Add required parameters
    addRequired(p, 'signal_params');
    addRequired(p, 'L_range');
    addRequired(p, 'max_signals');
    addRequired(p, 'num_experiments');
    
    % Add optional name-value pair parameters
    addParameter(p, 'optimal_order', defaultOptimalOrder);
    addParameter(p, 'num_components', defaultNumComponents);
    addParameter(p, 'embedding', defaultEmbedding);
    addParameter(p, 'method', defaultMethod);
    
    % Parse the inputs
    parse(p, signal_params, L_range, max_signals, num_experiments, varargin{:});
    
    % Assign parsed values to variables
    optimal_order = p.Results.optimal_order;
    num_components = p.Results.num_components;
    embedding = p.Results.embedding;
    method = p.Results.method;
    
    
    
    % Max signals parameter
    max_signals_param = size(signal_params, 1);  % You can adjust this as needed

    noise_option = 1;
    num_predict = 1;
    
    % 5 metrics, 1 model (including actual), number of different parameter setups
    all_errors_gt1_mean = zeros(5, 2, max_signals_param, length(L_range));
    all_errors_gt2_mean = zeros(5, 2, max_signals_param, length(L_range));
    
    for sim_param = 1:max_signals_param
        % Iterate over different values of L
        for L_idx = 1:length(L_range)
            fprintf("Testing L value %d\n", L_range(L_idx));
            L = L_range(L_idx);

            % Number of metrics by number of simulations of different signals for
            % different simulation parameters
            all_errors_gt1 = zeros(5, 2, max_signals);
            all_errors_gt2 = zeros(5, 2, max_signals);
        
            for sim = 1:max_signals
                disp("Parameter simulation " + sim_param + "/" + max_signals_param);
                disp("Generated signal " + sim + "/" + max_signals);
                
                time_series = rsignal(signal_params(sim_param, 1), signal_params(sim_param, 2), signal_params(sim_param, 3), signal_params(sim_param, 4));

                N = length(time_series); % Number of sampling points in the time series

                % Ground truth for final point
                ground_truths1 = time_series(end - num_predict + 1:end); 
                noisy_series = time_series + noise_option * (randn(N, 1) * 0.4);
                ground_truths2 = noisy_series(end - num_predict + 1:end);
                
                ground_truths = {ground_truths1, ground_truths2};
                training_series = noisy_series(1:end - num_predict);   
                    
                % Prepare storage for predictions
                predictions_SVD = zeros(num_predict, num_experiments);
                
                % Prepare storage for errors
                rel_error_SVD = zeros(num_predict, num_experiments, 2);
                norm_error_SVD = zeros(num_predict, num_experiments, 2);
                
                % Perform the experiment
                for experiment = 1:num_experiments 
                    if mod(experiment, round(num_experiments / 4)) == 0 && mod(experiment, 2) == 0
                        disp("iter " + experiment);
                    elseif mod(experiment, round(num_experiments / 4)) == 0 || experiment == num_experiments
                        disp("iter " + experiment);
                    end

                    % Generate predictions once
                    predictions_SVD(:, experiment) = ar_svd(training_series, num_predict, optimal_order, num_components, 'L', L,'embedding', embedding,'method', method);

                    % Calculate errors for both ground truths
                    for gt = 1:2
                        % Error calculations
                        rel_error_SVD(:, experiment, gt) = abs(predictions_SVD(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_SVD(:, experiment, gt) = norm(predictions_SVD(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                    end
                end   
                       
                % Calculate statistics over number of experiments done
                for gt = 1:2
                    errors_SVD = [mean(predictions_SVD, 2), std(predictions_SVD, 0, 2), rmse(predictions_SVD, ground_truths{gt}, 2), mean(rel_error_SVD(:, :, gt), 2), mean(norm_error_SVD(:, :, gt), 2)]';
                    
                    if gt == 1
                        actual_stats = [mean(ground_truths{gt}, 2), 0, 0, 0, 0]';
                        errors_concat1 = [actual_stats, errors_SVD];
                    else
                        actual_stats = [mean(ground_truths{gt}, 2), std(ground_truths{gt}, 0, 2), 0, 0, 0]';
                        errors_concat2 = [actual_stats, errors_SVD];
                    end
                end

                % Add matrix of error calculations as a slice to 3d matrix
                all_errors_gt1(:, :, sim) = errors_concat1;
                all_errors_gt2(:, :, sim) = errors_concat2;
            end
            
            % Below are the average evaluation scores (mean of matrix slices)
            % rows: mean; sd; RMSE; MRE; MRSE
            % columns: SVD
            % slices: results per signal parameter set
            all_errors_gt1_mean(:, :, sim_param, L_idx) = mean(all_errors_gt1, 3);
            all_errors_gt2_mean(:, :, sim_param, L_idx) = mean(all_errors_gt2, 3);
        end
    end
    disp("Done!");

    % Select best L based on average errors for SVD
    % Here, assuming lower RMSE means better performance.
    % Rows: mean; sd; RMSE; MRE; MRSE
    % Columns: SVD
    best_svd_L_idx = zeros(1, max_signals_param);
    
    for sim_param = 1:max_signals_param
        [~, best_svd_L_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 2, sim_param, :)));
    end

    best_L = struct('SVD', L_range(best_svd_L_idx));
end
