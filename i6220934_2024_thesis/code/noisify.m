function noisy_series = noisify(noise_param,N,time_series)
    rt = range(time_series);
    noise = (1-2.*round(rand(N,1))).*(noise_param*rand(N,1)*rt);
    noisy_series = time_series + noise;
end