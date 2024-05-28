function full_signal = rsignal(period, amp, fs, interval)
% Generate sinusoidal signals
    dt = 1/fs;
    t = (0:dt:interval)';

    % Define base signals without decay and random addition
    base_signal1 = amp * sin(2 * pi * period * round((rand() * 0.1 + 0.01), 2) * t);
    base_signal2 = amp * sin(2 * pi * period * round((rand() * 0.1 + 0.01), 2) * t);
    base_signal3 = amp * sin(2 * pi * period * round((rand() * 0.1 + 0.01), 2) * t);
    
    % Apply signal option 3 (decay and random addition)
    signal1 = base_signal1 .* exp(-0.01 * t) + rand(size(t)) * 10;
    signal2 = base_signal2 .* exp(-0.01 * t) - rand(size(t)) * 10;
    signal3 = base_signal3 .* exp(-0.01 * t) + rand(size(t)) * 10;

    full_signal = signal1+signal2+signal3;
end