function phi = tapered_window(filter_duration, window_duration, sampling_interval)

%% Function header.
% This function constructs the tapered window defined in Eq. (5) of [1].
%
%-- Inputs:
%      filter_duration   : A scalar that indicates the duration of the Kalman filter, in seconds.
%      window_duration   : A scalar that indicates the total length of the window, in seconds.
%                          'window_duration' must be greater than 'filter_duration'.
%      sampling_interval : A scalar indicating the time interval between adjacent noise samples, in seconds.
%
%-- Outputs:
%      phi: The window function, evaluated at the discrete times (0, dt, 2 * dt, ..., window_duration), where
%           dt = sampling_interval.
%
%-- Reference:
%      [1] S. Langel, O. Garcia Crespillo, and M. Joerger, "Frequency-domain modeling of correlated Gaussian
%          noise in Kalman filtering," IEEE Trans. Aerosp. Electron. Syst., vol. xx, no. xx, pp. xx-xx,
%          November, 2024, doi: 10.1109/TAES.2024.3442775.

%% Error checking.
if nargin < 3
    error('Not enough input arguments.');
elseif nargin > 3
    error('Too many input arguments.');
end

if ~isscalar(filter_duration)
    error('A positive scalar was expected for filter_duration, but a different input was provided.');
elseif ~isscalar(window_duration)
    error('A positive scalar was expected for window_duration, but a different input was provided.');
elseif ~isscalar(sampling_interval)
    error('A positive scalar was expected for sampling_interval, but a different input was provided.');
end

if filter_duration < 0
    error('A positive scalar was expected for filter_duration, but a different input was provided.');
elseif window_duration < 0
    error('A positive scalar was expected for window_duration, but a different input was provided.');
elseif sampling_interval < 0
    error('A positive scalar was expected for sampling_interval, but a different input was provided.');
end

if mod(filter_duration, sampling_interval) ~= 0
    error('The ratio of filter_duration to sampling_interval must be an integer.');
elseif mod(window_duration, sampling_interval) ~= 0
    error('The ratio of window_duration to sampling_interval must be an integer.');
elseif window_duration < filter_duration
    error('The window duration must be greater than or equal to the filter duration.');
end

%% Algorithm code.
% Compute lag numbers n and n_w that dictate where the window transitions from a value of 1 to 0, and the lag
% vector 'ell'. 
n = round(filter_duration / sampling_interval);
n_w = round(window_duration / sampling_interval);
ell = (0 : n_w)';

% The tapered window is flat for lag numbers less than or equal to n.
phi_flat_section(1 : n + 1, 1) = 1;

% Compute the parameter 'eta' which determines the argument to the exponential function in the tapered window.
eta = 1 + 2 * (n_w - ell(ell > n)) / (n - n_w);

% When eta is close to -1 or 1, the exponential argument approaches +/- infinity and an overflow warning is
% issued by MATLAB. To avoid this warning and eliminate the risk of numerical complications, a threshold of
% |eta| > 0.99 is used, which corresponds to a maximum exponential argument of 200. When |eta| > 0.99, we set
% the window value equal to its appropriate endpoint value (i.e., 0 or 1).
phi_tapered_section = zeros(n_w - n, 1);
phi_tapered_section(eta < -0.99) = 1;
phi_tapered_section(eta > 0.99) = 0;

index = eta >= -0.99 & eta <= 0.99;
argument = 4 * eta(index) ./ (1 - eta(index) .^ 2);
phi_tapered_section(index) = 1 ./ (exp(argument) + 1);

phi = [phi_flat_section; phi_tapered_section];

end
