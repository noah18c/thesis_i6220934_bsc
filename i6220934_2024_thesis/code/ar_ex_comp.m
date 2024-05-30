function [best_components, all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex_comp(signal_params, max_signals, num_experiments, L, LM_params, optimal_order, components_range)
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
    all_errors_gt1_mean = zeros(5, 6, max_signals_param, length(components_range));
    all_errors_gt2_mean = zeros(5, 6, max_signals_param, length(components_range));
    
    for sim_param = 1:max_signals_param
        % Iterate over different values of num_components
        for comp_idx = 1:length(components_range)
            fprintf("Testing %d number of components\n", components_range(comp_idx));
            num_components = components_range(comp_idx);

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
                predictions_SVD = zeros(num_predict, num_experiments);
                predictions_cpd_s = zeros(num_predict, num_experiments);
                predictions_cpd_f = zeros(num_predict, num_experiments);
                predictions_cpd_cols = zeros(num_predict, num_experiments);
                predictions_cpd_colf = zeros(num_predict, num_experiments);
                
                % Prepare storage for errors
                rel_error_SVD = zeros(num_predict, num_experiments, 2);
                norm_error_SVD = zeros(num_predict, num_experiments, 2);
                rel_error_cpd_s = zeros(num_predict, num_experiments, 2);
                norm_error_cpd_s = zeros(num_predict, num_experiments, 2);
                rel_error_cpd_f = zeros(num_predict, num_experiments, 2);
                norm_error_cpd_f = zeros(num_predict, num_experiments, 2);
                rel_error_cpd_cols = zeros(num_predict, num_experiments, 2);
                norm_error_cpd_cols = zeros(num_predict, num_experiments, 2);
                rel_error_cpd_colf = zeros(num_predict, num_experiments, 2);
                norm_error_cpd_colf = zeros(num_predict, num_experiments, 2);
                
                % Perform the experiment
                for experiment = 1:num_experiments 
                    if mod(experiment, round(num_experiments / 4)) == 0 && mod(experiment, 2) == 0
                        disp("iter " + experiment);
                    elseif mod(experiment, round(num_experiments / 4)) == 0 || experiment == num_experiments
                        disp("iter " + experiment);
                    end

                    try
                        predictions_SVD(:, experiment) = ar_svd(training_series, num_predict, optimal_order, num_components, 'L', L);
                    catch
                        predictions_SVD(:, experiment) = NaN;
                    end
                
                    try
                        predictions_cpd_s(:, experiment) = ar_cpd_s(training_series, num_predict, optimal_order, num_components,'L', LM_params(1), 'M', LM_params(2));
                    catch
                        predictions_cpd_s(:, experiment) = NaN;
                    end
                
                    try
                        predictions_cpd_f(:, experiment) = ar_cpd_f(training_series, num_predict, optimal_order, num_components,'L', LM_params(1), 'M', LM_params(2));
                    catch
                        predictions_cpd_f(:, experiment) = NaN;
                    end
                
                    try
                        predictions_cpd_cols(:, experiment) = ar_cpd_cols(training_series, num_predict, optimal_order, num_components,'L', LM_params(1), 'M', LM_params(2));
                    catch
                        predictions_cpd_cols(:, experiment) = NaN;
                    end
                
                    try
                        predictions_cpd_colf(:, experiment) = ar_cpd_colf(training_series, num_predict, 10, num_components,'L', LM_params(1), 'M', LM_params(2));
                    catch
                        predictions_cpd_colf(:, experiment) = NaN;
                    end
                    % Calculate errors for both ground truths
                    for gt = 1:2
                        % Error calculations
                        rel_error_SVD(:, experiment, gt) = abs(predictions_SVD(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_SVD(:, experiment, gt) = norm(predictions_SVD(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                        rel_error_cpd_s(:, experiment, gt) = abs(predictions_cpd_s(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_cpd_s(:, experiment, gt) = norm(predictions_cpd_s(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                        rel_error_cpd_f(:, experiment, gt) = abs(predictions_cpd_f(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_cpd_f(:, experiment, gt) = norm(predictions_cpd_f(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                        
                        rel_error_cpd_cols(:, experiment, gt) = abs(predictions_cpd_cols(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_cpd_cols(:, experiment, gt) = norm(predictions_cpd_cols(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                        rel_error_cpd_colf(:, experiment, gt) = abs(predictions_cpd_colf(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_cpd_colf(:, experiment, gt) = norm(predictions_cpd_colf(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                    end
                end   
                       
                % Calculate statistics over number of experiments done
                for gt = 1:2
                    errors_SVD = [mean(predictions_SVD, 2), std(predictions_SVD, 0, 2), rmse(predictions_SVD, ground_truths{gt}, 2), mean(rel_error_SVD(:, :, gt), 2), mean(norm_error_SVD(:, :, gt), 2)]';
                    errors_cpd_s = [mean(predictions_cpd_s, 2), std(predictions_cpd_s, 0, 2), rmse(predictions_cpd_s, ground_truths{gt}, 2), mean(rel_error_cpd_s(:, :, gt), 2), mean(norm_error_cpd_s(:, :, gt), 2)]';
                    errors_cpd_f = [mean(predictions_cpd_f, 2), std(predictions_cpd_f, 0, 2), rmse(predictions_cpd_f, ground_truths{gt}, 2), mean(rel_error_cpd_f(:, :, gt), 2), mean(norm_error_cpd_f(:, :, gt), 2)]';
                    errors_cpd_cols = [mean(predictions_cpd_cols, 2), std(predictions_cpd_cols, 0, 2), rmse(predictions_cpd_cols, ground_truths{gt}, 2), mean(rel_error_cpd_cols(:, :, gt), 2), mean(norm_error_cpd_cols(:, :, gt), 2)]';
                    errors_cpd_colf = [mean(predictions_cpd_colf, 2), std(predictions_cpd_colf, 0, 2), rmse(predictions_cpd_colf, ground_truths{gt}, 2), mean(rel_error_cpd_colf(:, :, gt), 2), mean(norm_error_cpd_colf(:, :, gt), 2)]';
                    
                    if gt == 1
                        actual_stats = [mean(ground_truths{gt}, 2), 0, 0, 0, 0]';
                        errors_concat1 = [actual_stats, errors_SVD, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf];
                    else
                        actual_stats = [mean(ground_truths{gt}, 2), std(ground_truths{gt}, 0, 2), 0, 0, 0]';
                        errors_concat2 = [actual_stats, errors_SVD, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf];
                    end
                end

                % Add matrix of error calculations as a slice to 3d matrix
                all_errors_gt1(:, :, sim) = errors_concat1;
                all_errors_gt2(:, :, sim) = errors_concat2;
            end
            
            % Below are the average evaluation scores (mean of matrix slices)
            % rows: mean; sd; RMSE; MRE; MRSE
            % columns: SVD, CPD_S, CPD_F, CPD_Cols, CPD_ColF
            % slices: results per signal parameter set
            all_errors_gt1_mean(:, :, sim_param, comp_idx) = mean(all_errors_gt1, 3);
            all_errors_gt2_mean(:, :, sim_param, comp_idx) = mean(all_errors_gt2, 3);
        end
    end
    disp("Done!");

    % Select best component based on average errors for SVD, CPD_S, CPD_F, CPD_Cols, and CPD_ColF
    % Here, assuming lower RMSE means better performance.
    % Rows: mean; sd; RMSE; MRE; MRSE
    % Columns: SVD, CPD_S, CPD_F, CPD_Cols, CPD_ColF
    best_svd_idx = zeros(1, max_signals_param);
    best_cpd_s_idx = zeros(1, max_signals_param);
    best_cpd_f_idx = zeros(1, max_signals_param);
    best_cpd_cols_idx = zeros(1, max_signals_param);
    best_cpd_colf_idx = zeros(1, max_signals_param);
    
    for sim_param = 1:max_signals_param
        [~, best_svd_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 1, sim_param, :)));
        [~, best_cpd_s_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 2, sim_param, :)));
        [~, best_cpd_f_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 3, sim_param, :)));
        [~, best_cpd_cols_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 4, sim_param, :)));
        [~, best_cpd_colf_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 5, sim_param, :)));
    end

    best_components = struct('SVD', components_range(best_svd_idx), 'CPD_S', components_range(best_cpd_s_idx), 'CPD_F', components_range(best_cpd_f_idx), 'CPD_Cols', components_range(best_cpd_cols_idx), 'CPD_ColF', components_range(best_cpd_colf_idx));
end
