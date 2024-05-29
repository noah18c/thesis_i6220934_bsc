function [best_L, all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex_thankel(signal_params, max_signals, num_experiments, optimal_order, num_components, reduction, L_range)
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
    
    % 5 metrics, 5 models (including actual), number of different parameter setups
    all_errors_gt1_mean = zeros(5, 6, max_signals_param, length(L_range));
    all_errors_gt2_mean = zeros(5, 6, max_signals_param, length(L_range));
    
    for sim_param = 1:max_signals_param
        % Iterate over different values of L
        for L_idx = 1:length(L_range)
            fprintf("Testing L value %d\n", L_range(L_idx));
            L = L_range(L_idx);

            % Number of metrics by number of simulations of different signals for
            % different simulation parameters
            all_errors_gt1 = zeros(5, 6, max_signals);
            all_errors_gt2 = zeros(5, 6, max_signals);
        
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
                predictions_cpd_s = zeros(num_predict, num_experiments);
                predictions_cpd_f = zeros(num_predict, num_experiments);
                predictions_cpd_cols = zeros(num_predict, num_experiments);
                predictions_cpd_colf = zeros(num_predict, num_experiments);
                predictions_mlsvd = zeros(num_predict, num_experiments);
                
                % Prepare storage for errors
                rel_error_cpd_s = zeros(num_predict, num_experiments, 2);
                norm_error_cpd_s = zeros(num_predict, num_experiments, 2);
                rel_error_cpd_f = zeros(num_predict, num_experiments, 2);
                norm_error_cpd_f = zeros(num_predict, num_experiments, 2);
                rel_error_cpd_cols = zeros(num_predict, num_experiments, 2);
                norm_error_cpd_cols = zeros(num_predict, num_experiments, 2);
                rel_error_cpd_colf = zeros(num_predict, num_experiments, 2);
                norm_error_cpd_colf = zeros(num_predict, num_experiments, 2);
                rel_error_mlsvd = zeros(num_predict, num_experiments, 2);
                norm_error_mlsvd = zeros(num_predict, num_experiments, 2);
                
                % Perform the experiment
                for experiment = 1:num_experiments                        
                    if mod(experiment, round(num_experiments / 4)) == 0 && mod(experiment, 2) == 0
                        disp("iter " + experiment);
                    elseif mod(experiment, round(num_experiments / 4)) == 0 || experiment == num_experiments
                        disp("iter " + experiment);
                    end

                    % Generate predictions once
                    predictions_cpd_s(:, experiment) = ar_cpd_s(training_series, num_predict, optimal_order, num_components, 'L', L);
                    predictions_cpd_f(:, experiment) = ar_cpd_f(training_series, num_predict, optimal_order, num_components, 'L', L);
                    predictions_cpd_cols(:, experiment) = ar_cpd_cols(training_series, num_predict, optimal_order, num_components, 'L', L);
                    predictions_cpd_colf(:, experiment) = ar_cpd_colf(training_series, num_predict, 10, num_components, 'L', L);
                    predictions_mlsvd(:, experiment) = ar_mlsvd(training_series, num_predict, optimal_order, reduction, 'L', L);

                    % Calculate errors for both ground truths
                    for gt = 1:2
                        % Error calculations
                        rel_error_cpd_s(:, experiment, gt) = abs(predictions_cpd_s(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_cpd_s(:, experiment, gt) = norm(predictions_cpd_s(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                        rel_error_cpd_f(:, experiment, gt) = abs(predictions_cpd_f(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_cpd_f(:, experiment, gt) = norm(predictions_cpd_f(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                        
                        rel_error_cpd_cols(:, experiment, gt) = abs(predictions_cpd_cols(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_cpd_cols(:, experiment, gt) = norm(predictions_cpd_cols(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                        rel_error_cpd_colf(:, experiment, gt) = abs(predictions_cpd_colf(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_cpd_colf(:, experiment, gt) = norm(predictions_cpd_colf(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                        
                        rel_error_mlsvd(:, experiment, gt) = abs(predictions_mlsvd(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_mlsvd(:, experiment, gt) = norm(predictions_mlsvd(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                    end
                end   
                       
                % Calculate statistics over number of experiments done
                for gt = 1:2
                    errors_cpd_s = [mean(predictions_cpd_s, 2), std(predictions_cpd_s, 0, 2), rmse(predictions_cpd_s, ground_truths{gt}, 2), mean(rel_error_cpd_s(:, :, gt), 2), mean(norm_error_cpd_s(:, :, gt), 2)]';
                    errors_cpd_f = [mean(predictions_cpd_f, 2), std(predictions_cpd_f, 0, 2), rmse(predictions_cpd_f, ground_truths{gt}, 2), mean(rel_error_cpd_f(:, :, gt), 2), mean(norm_error_cpd_f(:, :, gt), 2)]';
                    errors_cpd_cols = [mean(predictions_cpd_cols, 2), std(predictions_cpd_cols, 0, 2), rmse(predictions_cpd_cols, ground_truths{gt}, 2), mean(rel_error_cpd_cols(:, :, gt), 2), mean(norm_error_cpd_cols(:, :, gt), 2)]';
                    errors_cpd_colf = [mean(predictions_cpd_colf, 2), std(predictions_cpd_colf, 0, 2), rmse(predictions_cpd_colf, ground_truths{gt}, 2), mean(rel_error_cpd_colf(:, :, gt), 2), mean(norm_error_cpd_colf(:, :, gt), 2)]';
                    errors_mlsvd = [mean(predictions_mlsvd, 2), std(predictions_mlsvd, 0, 2), rmse(predictions_mlsvd, ground_truths{gt}, 2), mean(rel_error_mlsvd(:, :, gt), 2), mean(norm_error_mlsvd(:, :, gt), 2)]';
                    
                    if gt == 1
                        actual_stats = [mean(ground_truths{gt}, 2), 0, 0, 0, 0]';
                        errors_concat1 = [actual_stats, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf, errors_mlsvd];
                    else
                        actual_stats = [mean(ground_truths{gt}, 2), std(ground_truths{gt}, 0, 2), 0, 0, 0]';
                        errors_concat2 = [actual_stats, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf, errors_mlsvd];
                    end
                end

                % Add matrix of error calculations as a slice to 3d matrix
                all_errors_gt1(:, :, sim) = errors_concat1;
                all_errors_gt2(:, :, sim) = errors_concat2;
            end
            
            % Below are the average evaluation scores (mean of matrix slices)
            % rows: mean; sd; RMSE; MRE; MRSE
            % columns: cpd_s, cpd_f, cpd_cols, cpd_colf, mlsvd
            % slices: results per signal parameter set
            all_errors_gt1_mean(:, :, sim_param, L_idx) = mean(all_errors_gt1, 3);
            all_errors_gt2_mean(:, :, sim_param, L_idx) = mean(all_errors_gt2, 3);
        end
    end
    disp("Done!");

    % Select best L based on average errors for CPD-S, CPD-F, CPD-cols, CPD-ColF, and MLSVD
    % Here, assuming lower RMSE means better performance.
    % Rows: mean; sd; RMSE; MRE; MRSE
    % Columns: cpd_s, cpd_f, cpd_cols, cpd_colf, mlsvd
    best_cpd_s_L_idx = zeros(1, max_signals_param);
    best_cpd_f_L_idx = zeros(1, max_signals_param);
    best_cpd_cols_L_idx = zeros(1, max_signals_param);
    best_cpd_colf_L_idx = zeros(1, max_signals_param);
    best_mlsvd_L_idx = zeros(1, max_signals_param);
    
    for sim_param = 1:max_signals_param
        [~, best_cpd_s_L_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 1, sim_param, :)));
        [~, best_cpd_f_L_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 2, sim_param, :)));
        [~, best_cpd_cols_L_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 3, sim_param, :)));
        [~, best_cpd_colf_L_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 4, sim_param, :)));
        [~, best_mlsvd_L_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 5, sim_param, :)));
    end

    best_L = struct('cpd_s', L_range(best_cpd_s_L_idx), 'cpd_f', L_range(best_cpd_f_L_idx), 'cpd_cols', L_range(best_cpd_cols_L_idx), 'cpd_colf', L_range(best_cpd_colf_L_idx), 'mlsvd', L_range(best_mlsvd_L_idx));
end
