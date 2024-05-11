function yf = dataDrivenSimulation(u_tilde, y_tilde, uf, n_max, l_max, delta)
    % Extract sizes
    T = size(u_tilde, 2);  % Length of the historical data
    m = size(u_tilde, 1);  % Number of inputs
    p = size(y_tilde, 1);  % Number of outputs

    % Generate block-Hankel matrices for input and output
    U = blockHankel(u_tilde, l_max + delta, T - l_max - delta + 1);
    Y = blockHankel(y_tilde, l_max + delta, T - l_max - delta + 1);

    % Split into past and future blocks
    Up = U(1:l_max*m, :);
    Uf = U(l_max*m+1:end, :);
    Yp = Y(1:l_max*p, :);
    Yf = Y(l_max*p+1:end, :);

    % Initialize simulation
    t = size(uf, 2);  % Length of the input for which response is needed
    k = 0;
    yf = [];

    % Iterate until all input is processed
    while k * delta < t
        % Define current segment of uf handling edge cases
        currentSegment = uf(:, max(1, k*delta+1):min(t, (k+1)*delta));
        %currentSegmentLength = size(currentSegment, 2);

        % Current blocks of input and zero initial past output
        f_u = [zeros(l_max*m, 1); currentSegment(:)];
        f_y_p = zeros(l_max*p, 1);

        % Solve for g(k) using least squares
        G = [Up; Uf; Yp]\[f_u; f_y_p];

        % Compute the output for this segment
        y_segment = Yf * G;
        yf = [yf; y_segment];  % Concatenate to the output

        % Update k for the next iteration
        k = k + 1;
    end
end

function H = blockHankel(data, rows, cols)
    % Helper function to generate a block-Hankel matrix
    H = [];
    for i = 1:cols
        H = [H; data(:, i:i+rows-1)];
    end
    H = reshape(H, size(data, 1)*rows, cols);
end
