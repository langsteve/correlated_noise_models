function model_parameters = autoregressive_model_determination(autocorrelation_sequence, lags, ...
    filter_duration, window_duration, model, alpha_1, alpha_2, num_elements)

%% Function header.
% This function determines either a white noise model, a first order Gauss-Markov model, or a second order
% autoregressive model for correlated noise using the method in [1].

%-- Inputs:
%   autocorrelation_sequence : An (nx1) vector containing the raw, one-sided ACS, i.e., the ACS corresponding
%                              to lags greater than or equal to zero. The ACS must be a positive definite
%                              sequence, meaning that if it is estimated from a time series of data, one must
%                              form a biased estimate of the ACS (e.g., using the 'biased' option in the
%                              MATLAB xcorr() function).
%
%   lags                     : An (nx1) vector containing the lags, in units of the correlation variable,
%                              associated with each value in the autocorrelation sequence. The lags must be
%                              equally spaced and must start from a lag of 0. Most often, the correlation
%                              variable is time but it could be some other variable. For example, one may be
%                              interested in modeling the correlation of gravity anomalies over distance, in
%                              which case the correlation variable would be distance.
%
%   filter_duration          : A positive scalar indicating the duration of the Kalman filter in seconds.
%
%   window_duration          : A positive scalar indicating the duration of the tapered window, in seconds.
%                              The window duration must be greater than or equal to the filter duration.
%
%   model                    : A case-insensitive string indicating the desired model. Valid inputs are:
%                                  'white noise'  : Noise is modeled as a white Gaussian noise (WGN) process.
%                                  'gauss markov' : Noise is modeled as the sum of a first-order Gauss-Markov
%                                                   process (GMP) and WGN.
%                                  'ar2'          : Noise is modeled as the sum of a second-order
%                                                   autoregressive (AR) process and WGN.
%
%   alpha_1 (optional)       : An (nx1) vector of AR model parameters 'alpha_1'. If model == 'white noise',
%                              this vector is ignored. If model == 'gauss markov', then alpha_1 must be in the
%                              interval [0, 1]. If model = 'ar2', alpha_1 must be in the interval [-2, 2].
%
%   alpha_2 (optional)       : An (nx1) vector of AR model parameters 'alpha_2'. If model == 'white noise' or
%                              model == 'gauss markov', this vector is ignored. If model == 'ar2', alpha_2
%                              must be the same length as alpha_1, and the pair of parameters must satisfy the
%                              inequalities: |alpha_2| <= 1 and (1 - alpha_2) ^ 2 - alpha_1 ^ 2 >= 0.
%
%   num_elements (optional)  : A positive integer. If alpha_1 and alpha_2 are not provided, this function
%                              considers multiple points evenly distributed throughout the feasible region
%                              when determining a Gauss-Markov model or a second-order AR model. Specifically,
%                              the input 'num_elements' is the number of triangles in Fig. 5 of [1] that will
%                              be considered when determining the model. To use this input, make alpha_1 = []
%                              and alpha_2 = []. The default value is 1E+06.
%
%-- Outputs:
%   model_parameters : For the WGN model, the (1x1) output is 'wgn_variance.' For the GMP + WGN model, the
%                      (3x1) output is [alpha_1_star; gmp_variance; wgn_variance]. For the second-order AR +
%                      WGN model, the (4x1) output is [alpha_1_star; alpha_2_star; ar_variance; wgn_variance].
%
%-- Reference:
%      [1] S. Langel, O. Garcia Crespillo, and M. Joerger, "Frequency-domain modeling of correlated Gaussian
%          noise in Kalman filtering," IEEE Trans. Aerosp. Electron. Syst., vol. xx, no. xx, pp. xx-xx,
%          November, 2024, doi: 10.1109/TAES.2024.3442775.

%% Error checking.
if nargin < 5
    error('Not enough input arguments.');
elseif nargin > 8
    error('Too many input arguments.');
end

if ~isvector(autocorrelation_sequence)
    error('The input autocorrelation_sequence must be a vector.');
elseif ~isvector(lags)
    error('The input lags must be a vector.');
elseif numel(autocorrelation_sequence) ~= numel(lags)
    error('The ACS vector and lag vector must have the same length.');
end

if ~isscalar(filter_duration) || ~isscalar(window_duration) || filter_duration <= 0 || window_duration <= 0
    error('The filter duration and window duration must be positive scalars.');
elseif window_duration < filter_duration
    error('The window duration must be greater than or equal to the filter duration.');
end

ind_lag_zero = find(lags == 0, 1);
if isempty(ind_lag_zero)
    error('The lag vector must include a lag of zero.');
end

if ~strcmpi(model, 'white noise') && ~strcmpi(model, 'gauss markov') && ~strcmpi(model, 'ar2')
    error('An invalid model was provided. Valid options are: "white noise", "gauss markov", or "ar2".');
end

if nargin == 6
    if ~isempty(alpha_1) && ~isvector(alpha_1)
        error('The input alpha_1 must be a vector.');
    end
    if strcmpi(model, 'ar2')
        error(['For the second order autoregressive model, alpha_1 and alpha_2 must both be specified. ', ...
               'Only alpha_1 was provided.']);
    elseif strcmpi(model, 'gauss markov')
        alpha_2 = zeros(numel(alpha_1), 1);
        infeasible_points = alpha_1 < 0 | alpha_1 > 1;
        if any(infeasible_points)
            alpha_1(infeasible_points) = [];
            if ~isempty(alpha_1)
                warning(['For the Gauss-Markov model, alpha_1 must be in the interval [0, 1]. ', ... 
                         'Values outside this interval were found and removed.']);
            else
                error('All values provided for alpha_1 are infeasible.');
            end
        end
    end
elseif nargin == 7
    if (~isempty(alpha_1) && ~isvector(alpha_1)) || (~isempty(alpha_2) && ~isvector(alpha_2))
        error('The inputs alpha_1 and alpha_2 must be vectors.');
    end
    if strcmpi(model, 'ar2')
        if numel(alpha_1) ~= numel(alpha_2)
            error('The inputs alpha_1 and alpha_2 must have the same number of elements.');
        else
            constraint = (1 - alpha_2(:)) .^ 2 - alpha_1(:) .^ 2;
            infeasible_points = abs(alpha_1) > 2 | abs(alpha_2) > 1 | constraint < 0;
            if any(infeasible_points)
                alpha_1(infeasible_points) = [];
                alpha_2(infeasible_points) = [];
                if ~isempty(alpha_1) && ~isempty(alpha_2)
                    warning(['Values for alpha_1 and alpha_2 outside the feasible region were found and ', ...
                             'removed.']);
                else
                    error('All values provided for alpha_1 and alpha_2 are infeasible.');
                end
            end
        end
    elseif strcmpi(model, 'gauss markov')
        infeasible_points = alpha_1 < 0 | alpha_1 > 1;
        if any(infeasible_points)
            alpha_1(infeasible_points) = [];
            if ~isempty(alpha_1)
                warning(['For the Gauss-Markov model, alpha_1 must be in the interval [0, 1]. ', ... 
                         'Values outside this interval were found and removed.']);
            else
                error('All values provided for alpha_1 are infeasible.');
            end
        end
    end
elseif nargin == 8
    if isempty(alpha_1) && isempty(alpha_2)
        if ~isscalar(num_elements) || mod(num_elements, 1) ~= 0 || num_elements <= 0
            error('The input num_elements must be a single, positive integer.');
        end
    else
        error('To use the input num_elements, alpha_1 and alpha_2 must be empty.');
    end
end

%% Algorithm code.
% Set the default value for 'num_elements' if a value was not provided.
if nargin ~= 8
    num_elements = 1E+06;
end

% Ensure that the ACS and lag vectors are column vectors.
autocorrelation_sequence = autocorrelation_sequence(:);
lags = lags(:);

% For the GMP and second-order AR models, construct a matrix containing the AR parameters alpha_1 and alpha_2.
% This matrix is constructed from the input arguments or generated automatically.
if strcmpi(model, 'gauss markov') || strcmpi(model, 'ar2')
    if nargin == 5 || nargin == 8 || isempty(alpha_1)
        parameter_matrix = get_feasible_points(num_elements);
    else
        parameter_matrix = [alpha_1, alpha_2];
    end
end

% Extract the sampling interval from the lag vector.
sampling_interval = lags(2) - lags(1);

% Trim down the ACS to coincide with the window length.
n_w = round(window_duration / sampling_interval);
autocorrelation_sequence(n_w + 2 : end) = [];

% Construct the tapered window and apply it to the raw ACS. Then normalize the resulting windowed ACS by the
% noise variance. 
noise_variance = autocorrelation_sequence(1);
phi = tapered_window(filter_duration, window_duration, sampling_interval);
normalized_and_windowed_ACS = phi .* autocorrelation_sequence / noise_variance;

% Determine all local extrema of g(Omega) by computing the roots of the generalized polynomial:
% p(x) = sum_{ell=1}^{n_w} ell * rho_{ell} U_{ell}(x), where rho is the normalized autocorrelation sequence
% for lag numbers greater than or equal to one and U_i(x) is the ith order Chebyshev polynomial of the second
% kind. We use the variable 'ell' instead of the letter 'l' to avoid difficulty distinguishing 'l' from the
% number 1. The roots of p(x) are provided in the x-domain, where x = cos(Omega). To translate the roots to
% the Omega-domain, take the inverse cosine of the roots in the x-domain.

clc; fprintf(['Finding all local maxima of g(', char(937), ')...\n']);

ell_vector = (1 : n_w)';
rho = normalized_and_windowed_ACS(2 : end);
poly_coeff = ell_vector .* rho;
poly_roots_x_domain = compute_generalized_polynomial_roots(poly_coeff);
critical_omega_of_g = sort(acos(poly_roots_x_domain));

% Extract the local maxima of g(Omega) using the second derivative test.
ell_squared_times_rho = ell_vector .^ 2 .* rho;
second_derivative_of_g = -2 * ell_squared_times_rho' * cos(ell_vector * critical_omega_of_g');
omega_at_maxima_of_g = [0; critical_omega_of_g(second_derivative_of_g < 0); pi];
maxima_of_g = (1 + 2 * rho' * cos(ell_vector * omega_at_maxima_of_g'))';

% If a white Gaussian noise model is sought, then the nondimensional variance of the model must be greater
% than or equal to the absolute maximum of g(Omega). To get the dimensional variance, multiply the
% nondimensional variance by the actual noise variance.
if strcmpi(model, 'white noise')
    absolute_maximum_of_g = max(maxima_of_g);
    model_parameters = noise_variance * absolute_maximum_of_g;
else
    % If a GMP or second-order AR model is sought, first determine an initial set of parameters for a minimum
    % variance model by solving the linear programming problem in Eq. (21) of [1].
    model_parameters = get_parameters(parameter_matrix, model, omega_at_maxima_of_g, maxima_of_g);
    
    % Perform the refinement step described in Theorems 3 and 4 of [1].
    refined_wgn_variance = get_refined_wgn_variance(model_parameters, normalized_and_windowed_ACS);
    refined_ar_variance = get_refined_ar_variance(model_parameters, normalized_and_windowed_ACS);
    
    % Find the combination of model variances with the minimum sum. 
    ar_variance = model_parameters(3);
    wgn_variance = model_parameters(4);
    if (ar_variance + refined_wgn_variance) < (refined_ar_variance + wgn_variance)
        model_parameters(4) = refined_wgn_variance;
    else
        model_parameters(3) = refined_ar_variance;
    end
    
    % Multiply the nondimensional model variances by the noise variance.
    model_parameters(3 : 4) = model_parameters(3 : 4) * noise_variance;
end

% Provide a print-out summarizing the noise model.
if strcmpi(model, 'white noise')
    output_string_wgn_variance = get_output_string(model_parameters);
    fprintf(['A white Gaussian noise model for the input noise component \x03c8_k is:\n\n', ...
             '\x03c8_k = q_k, where \x03c3_q^2 = ', output_string_wgn_variance, '.\n']);
else
    alpha_1_output_string = get_output_string(model_parameters(1));
    alpha_2_output_string = get_output_string(model_parameters(2));
    ar_variance_output_string = get_output_string(model_parameters(3));
    wgn_variance_output_string = get_output_string(model_parameters(4));
    if strcmpi(model, 'gauss markov')
        fprintf(['A first-order Gauss-Markov model for the input noise component \x03c8_k is:\n\n', ...
                 'y_k = ' num2str(alpha_1_output_string), '*y_(k-1) + u_k\n', ...
                 '\x03c8_k = y_k + q_k\n\nwhere \x03c3_y^2 = ', num2str(ar_variance_output_string), ...
                 ' and \x03c3_q^2 = ', num2str(wgn_variance_output_string), '.\n']);
    else
        fprintf(['A second-order autoregressive model for the input noise component \x03c8_k is:', ...
                 '\n\ny_k = ' num2str(alpha_1_output_string), '*y_(k-1) + ', ...
                 num2str(alpha_2_output_string), '*y_(k-2) + u_k\n \x03c8_k = y_k + q_k', ...
                 '\n\nwhere \x03c3_y^2 = ', num2str(ar_variance_output_string), ' and \x03c3_q^2 = ', ...
                 num2str(wgn_variance_output_string), '.\n']);
    end
end

% If a Gauss-Markov model is desired, then remove the 'alpha_2' entry from 'model_parameters'.
if strcmpi(model, 'gauss markov')
    model_parameters(2) = [];
end

end
