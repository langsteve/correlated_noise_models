function model_parameters = get_parameters(parameter_matrix, model, omega_at_maxima_of_g, maxima_of_g)

%% Function header.
% This function determines an initial set of model parameters by solving the LP problem in Eq. (21) of [1].
%
%-- Inputs:
%   parameter_matrix    : An (nx2) matrix of AR model parameters. The first column is alpha_1 and the second
%                         column is alpha_2.
%
%   model               : A case-insensitive string indicating the desired model. Valid inputs are:
%                         'gauss markov' or 'ar2'
%
%   omega_at_maxima_of_g: An (mx1) vector of frequencies where all local maxima of g(Omega) occur.
%
%   maxima_of_g         : An (mx1) vector of all local maximum values of g(Omega).
%
%-- Output:
%   model_parameters: A (1x4) vector containing the initial set of parameters for a minimum variance model.
%                     The format of this vector is: [alpha_1_star, alpha_2_star, ar_variance, wgn_variance].
%
%-- Reference:
%      [1] S. Langel, O. Garcia Crespillo, and M. Joerger, "Frequency-domain modeling of correlated Gaussian
%          noise in Kalman filtering," IEEE Trans. Aerosp. Electron. Syst., vol. xx, no. xx, pp. xx-xx,
%          November, 2024, doi: 10.1109/TAES.2024.3442775.

%% Error checking.
if ~strcmpi(model, 'gauss markov') && ~strcmpi(model, 'ar2')
    error('An invalid model was provided. Valid options are: "gauss markov" or "ar2".');
end

if ~isvector(omega_at_maxima_of_g) || ~isvector(maxima_of_g) || ...
        numel(omega_at_maxima_of_g) ~= numel(maxima_of_g)
    error('The inputs "omega_at_maxima_of_g" and "maxima_of_g" must be vectors of the same length.');
end

if ~ismatrix(parameter_matrix) || size(parameter_matrix, 2) ~= 2
    error('The input "parameter_matrix" must be an (nx2) matrix. A different input was provided.');
end

%% Algorithm code.
% Initialize the output.
model_parameters = [0, 0, Inf, Inf];

% If model == 'gauss markov', trim down the AR parameter matrix to only include those points on the alpha_1
% axis in the interval (0, 1). If model == 'ar2', remove any points on the boundary of the feasible region
% since these points lead to a degenerate power spectral density function.
if strcmpi(model, 'gauss markov')    
    alpha_2_not_equal_to_zero = abs(parameter_matrix(:, 2)) > 1E-12;
    parameter_matrix(alpha_2_not_equal_to_zero, :) = [];    
    
    alpha_1_less_than_zero = parameter_matrix(:, 1) < 0;
    parameter_matrix(alpha_1_less_than_zero, :) = [];

    alpha_1_equals_zero = abs(parameter_matrix(:, 1)) < 1E-12;
    parameter_matrix(alpha_1_equals_zero, :) = [];

    alpha_1_equals_one = abs(parameter_matrix(:, 1) - 1) < 1E-12;
    parameter_matrix(alpha_1_equals_one, :) = [];
else
    alpha_2_equals_minus_one = abs(parameter_matrix(:, 2) + 1) < 1E-12;
    parameter_matrix(alpha_2_equals_minus_one, :) = [];

    alpha_1 = parameter_matrix(:, 1);
    alpha_2 = parameter_matrix(:, 2);
    on_triangle_sides = abs((1 - alpha_2) .^ 2 - alpha_1 .^ 2) < 1E-12;
    parameter_matrix(on_triangle_sides, :) = [];
end

% Loop over all pairs of parameters (alpha_1, alpha_2).
num_parameter_pairs = size(parameter_matrix, 1);
previous_progress = 0;
for idx = 1 : num_parameter_pairs
    current_progress = idx / num_parameter_pairs * 100;
    if current_progress - previous_progress > 1
        clc; fprintf(['Determining initial set of model parameters... ', ...
                     num2str(current_progress, '%2.2f'), '%% complete.\n']);
        previous_progress = current_progress;
    end
    alpha_1 = parameter_matrix(idx, 1);
    alpha_2 = parameter_matrix(idx, 2);
    
    % Determine the discrete set of constraints for the LP problem.
    [constraint_frequencies, g_at_constraint_frequencies] = ...
        get_constraint_frequencies(omega_at_maxima_of_g, maxima_of_g, alpha_1, alpha_2);
    N = numel(constraint_frequencies);
            
    % Form the function 'f' that defines the PSD for a second-order AR process.
    gamma = (1 + alpha_2) / (1 - alpha_2) * ((1 - alpha_2) ^ 2 - alpha_1 ^ 2);
    b0 = 1 + alpha_1 ^ 2 + alpha_2 ^ 2;
    b1 = 2 * alpha_1 * (1 - alpha_2);
    b2 = 2 * alpha_2;
    beta = b0 - b1 * cos(constraint_frequencies) - b2 * cos(2 * constraint_frequencies);
    f = gamma ./ beta;
        
    % The nondimensional model variances are determined by solving the LP problem in Eq. (21) of [1]. It was
    % determined that MATLAB's linprog() solver was significantly slower than exhaustively checking all of the
    % intersection points for the constraints. This is likely only true for the particular case considered
    % here where just two variables are involved. One should reconsider using linprog() if more than two
    % variables are involved in the linear program.
    
    % Determine vertices corresponding to the cases where the AR variance is zero or the WGN variance is zero.
    maximum_wgn_variance = max(g_at_constraint_frequencies);
    maximum_ar_variance = max(g_at_constraint_frequencies ./ f);

    % Construct two index vectors that allow the constraint intersection points to be computed efficiently.
    index_1 = tril(ones(N - 1, 1) * (1 : N - 1));
    index_1(index_1 == 0) = [];

    index_2 = tril((2 : N)' * ones(1, N - 1));
    index_2(index_2 == 0) = [];

    % For two inequality constraints f1*x + y > g1 and f2*x + y > g2, the intersection point occurs at
    % x = (g2 - g1) / (f2 - f1) and y = g1 - f1 * x. In our case, x is the variance of the AR process and y is
    % the WGN variance.
    ar_variances = (g_at_constraint_frequencies(index_2) - g_at_constraint_frequencies(index_1)) ./ ...
                   (f(index_2) - f(index_1));
    wgn_variances = g_at_constraint_frequencies(index_1) - f(index_1) .* ar_variances;
    vertices = [[ar_variances, wgn_variances]; [maximum_ar_variance, 0]; [0, maximum_wgn_variance]];
        
    % Remove any vertices that have negative coordinates (and are thus infeasible).
    vertices(vertices(:, 1) < 0, :) = [];
    vertices(vertices(:, 2) < 0, :) = [];
        
    % Construct a feasibility matrix, where each column is the set of constraints evaluated at a given vertex.
    % If all of the constraints are satisfied for a given vertex, we should find no instances where an element
    % in the column corresponding to the vertex is less than zero. Only keep vertices that satisfy this
    % condition.
    constraint_matrix = [f, ones(N, 1)];
    feasibility_matrix = constraint_matrix * vertices' - g_at_constraint_frequencies;
    constraints_violated = sum(feasibility_matrix < -1E-12);
    vertices = vertices(constraints_violated == 0, :);
    
    % Find the vertex whose sum of coordinates is minimal.
    [~, index_to_minimum_sum] = min(sum(vertices, 2));
    new_variances = vertices(index_to_minimum_sum, :);
        
    % If the new variances have a smaller sum than those for the current model, update the current model.
    ar_variance = new_variances(1);
    wgn_variance = new_variances(2);
    if (ar_variance + wgn_variance) < (model_parameters(3) + model_parameters(4))
        model_parameters = [alpha_1, alpha_2, ar_variance, wgn_variance];
    end
end

end
