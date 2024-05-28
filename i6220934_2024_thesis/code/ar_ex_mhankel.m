function [best_L, all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex_mhankel(signal_params, max_signals, num_experiments, optimal_order, num_components, L_range)
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
    
    % Max signals parameter
    max_signals_param = size(signal_params, 1);  % You can adjust this as needed

    noise_option = 1;
    num_predict = 1;
    
    % 5 metrics, 4 models (including actual), number of different parameter setups
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
                
                time_series = rsignal(signal_params(sim_param, 1),signal_params(sim_param, 2),signal_params(sim_param, 3),signal_params(sim_param, 4));

                N = length(time_series); % Number of sampling points in the time series
                
                for gt = 1:2    
                    disp("GT " + gt);
                    % Ground truth for final point
                    ground_truths = zeros(num_predict, num_experiments);
                    
                    % Prepare storage for predictions
                    predictions_SVD = zeros(num_predict, num_experiments);
                    rel_error_SVD = zeros(num_predict, num_experiments);
                    norm_error_SVD = zeros(num_predict, num_experiments);
                    
                    % Perform the experiment
                    for experiment = 1:num_experiments 
                        if mod(experiment, round(num_experiments / 4)) == 0 && mod(experiment, 2) == 0
                            disp("iter " + experiment);
                        elseif mod(experiment, round(num_experiments / 4)) == 0 || experiment == num_experiments
                            disp("iter " + experiment);
                        end
        
                        if gt == 1
                            ground_truths(:, experiment) = time_series(end - num_predict + 1:end); 
                        end
                        
                        noisy_series = time_series + noise_option * (randn(N, 1) * 0.4);
                        
                        if gt == 2
                            ground_truths(:, experiment) = noisy_series(end - num_predict + 1:end);
                        end
                        
                        training_series = noisy_series(1:end - num_predict);   
                    
                        % 1. Decomposition using Hankel Matrix (SVD)
                        predictions_SVD(:, experiment) = ar_svd(training_series, num_predict, optimal_order, num_components, 'L', L);
                    
                        % 2. Error calculations
                        rel_error_SVD(:, experiment) = abs(predictions_SVD(:, experiment) - ground_truths(:, experiment)) ./ abs(ground_truths(:, experiment));
                        norm_error_SVD(:, experiment) = norm(predictions_SVD(:, experiment), 2) ./ norm(ground_truths(:, experiment), 2);
                    end   
                           
                    % Calculate statistics over number of experiments done
                    errors_SVD = [mean(predictions_SVD, 2), std(predictions_SVD, 0, 2), rmse(predictions_SVD, ground_truths, 2), mean(rel_error_SVD, 2), mean(norm_error_SVD, 2)]';
                    
                    % depending on ground truth, set stats to either errors_concat1 or 2
                    if gt == 1
                        actual_stats = [mean(ground_truths, 2), 0, 0, 0, 0]';
                        errors_concat1 = [actual_stats, errors_SVD];
                    else
                        actual_stats = [mean(ground_truths, 2), std(ground_truths, 0, 2), 0, 0, 0]';
                        errors_concat2 = [actual_stats, errors_SVD];
                    end
                end
        
                % add matrix of error calculations as a slice to 3d matrix
                all_errors_gt1(:, :, sim) = errors_concat1;
                all_errors_gt2(:, :, sim) = errors_concat2;
            end
            
            % below are the average evaluation scores (mean of matrix slices)
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
