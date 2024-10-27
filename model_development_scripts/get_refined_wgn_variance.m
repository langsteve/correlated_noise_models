function refined_wgn_variance = get_refined_wgn_variance(model_parameters, normalized_and_windowed_ACS)

%% Function header.
% This function determines a refined variance for the WGN component of the noise model based on Theorem 3 in
% [1].
%
%-- Inputs:
%   model_parameters           : A (1x4) vector of initial parameters for a minimum variance model. The format
%                                of this vecor is: [alpha_1_star, alpha_2_star, ar_variance, wgn_variance].
%
%   normalized_and_windowed_ACS: An ((n_w + 1) x 1) vector of normalized and windowed ACS values.
%
%-- Output:
%   refined_wgn_variance: Refined variance of the WGN component of the noise model required to ensure that the
%                         model exceeds the true PSD at all frequencies.
%
%-- Reference:
%      [1] S. Langel, O. Garcia Crespillo, and M. Joerger, "Frequency-domain modeling of correlated Gaussian
%          noise in Kalman filtering," IEEE Trans. Aerosp. Electron. Syst., vol. xx, no. xx, pp. xx-xx,
%          November, 2024, doi: 10.1109/TAES.2024.3442775.

%% Error checking.
if ~isvector(model_parameters) || size(model_parameters, 2) ~= 4
    error('The input "model_parameters" must be a 1x4 vector. A different input was provided.');
end

if ~isvector(normalized_and_windowed_ACS)
    error('The input "normalized_and_windowed_ACS" must be a vector. A different input was provided.');
end

%% Algorithm code.

fprintf('Computing the refined WGN variance...\n');

alpha_1_star = model_parameters(1);
alpha_2_star = model_parameters(2);
ar_variance = model_parameters(3);

% Extract the lag number where the window function becomes zero and form a vector of summation indices.
n_w = numel(normalized_and_windowed_ACS) - 1;
ell_vector = (1 : n_w)';

% Parameters defining the AR(2) PSD.
gamma = (1 + alpha_2_star) / (1 - alpha_2_star) * ((1 - alpha_2_star) ^ 2 - alpha_1_star ^ 2);
b0 = 1 + alpha_1_star ^ 2 + alpha_2_star ^ 2;
b1 = 2 * alpha_1_star * (1 - alpha_2_star);
b2 = 2 * alpha_2_star;

% Form constants d.
d0 = b0 ^ 2 + b1 ^ 2 / 2 + b2 ^ 2 / 2;
d1 = b1 * b2 - 2 * b0 * b1;
d2 = b1 ^ 2 / 2 - 2 * b0 * b2;
d3 = b1 * b2;
d4 = b2 ^ 2 / 2;
    
poly_coeff = zeros(n_w + 3, 1);
for ell = 0 : n_w + 2
    % Get the indices to ACS values needed to compute the generalized polynomial coefficients in Theorem 3,
    % leveraging the fact that the windowed ACS is symmetric (i.e., ACS(lag == tau) = ACS(lag == -tau)) and
    % that the windowed ACS is zero for lag numbers >= n_w.
    indices = [ell - 3 : ell + 1, ell + 1 : ell + 5];
    ACS_indices = abs(indices);
    ACS_indices(ACS_indices > n_w) = n_w;
        
    % Compute the term c_ell.
    ACS_values = normalized_and_windowed_ACS(ACS_indices + 1);
    vector = [d4; d3; d2; d1; d0; d0; d1; d2; d3; d4];
    c_ell = indices * (ACS_values .* vector);

    % Add additional terms when ell == 0 and ell == 1 and form the vector of polynomial coefficients.
    poly_coeff(ell + 1, 1) = -c_ell;
    if ell == 0
        poly_coeff(ell + 1, 1) = poly_coeff(ell + 1, 1) + gamma * b1 * ar_variance;
    elseif ell == 1
        poly_coeff(ell + 1, 1) = poly_coeff(ell + 1, 1) + 2 * gamma * b2 * ar_variance;
    end
end
    
% Compute the roots of the generalized polynomial and tranform them to the Omega-domain. 
poly_roots_x_domain = compute_generalized_polynomial_roots(poly_coeff);
critical_omega_of_g = [0; sort(acos(poly_roots_x_domain)); pi];

% Determine the refined WGN variance.
g = 1 + 2 * normalized_and_windowed_ACS(2 : end)' * cos(ell_vector * critical_omega_of_g');
f = gamma ./ (b0 - b1 * cos(critical_omega_of_g) - b2 * cos(2 * critical_omega_of_g));
refined_wgn_variance = max([0; g' - f * ar_variance]);

end
