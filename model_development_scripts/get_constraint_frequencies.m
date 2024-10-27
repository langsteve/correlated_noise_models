function [constraint_frequencies, g_at_constraint_frequencies] = ...
    get_constraint_frequencies(omega_at_maxima_of_g, maxima_of_g, alpha_1, alpha_2)

%% Function header.
% This function determines what frequencies should be considered when defining the discrete set of constraints
% in the linear programming problem in Eq. (21) of [1]. The idea is to use the monotonicity of the model PSD
% S(Omega) as a way to sift out local maxima of the true PSD g(Omega) that do not have any influence over the
% criteria that S(Omega) > g(Omega). For example, if S(Omega) monotonically increases over a certain frequency
% range, then only the monotonically increasing set of local maxima of g(Omega) are relevant.
%
%-- Inputs:
%   omega_at_maxima_of_g : (nx1) column vector containing the frequencies where g(Omega) has a local maximum.
%   maxima_of_g          : (nx1) column vector containing the local maximum values of g(Omega).
%   alpha_1              : Scalar parameter for AR(2) process.
%   alpha_2              : Scalar parameter for AR(2) process.
%
%-- Outputs:
%   constraint_frequencies      : (mx1) vector of frequencies to be considered in the LP problem.
%   g_at_constraint_frequencies : (mx1) vector containing values of g(Omega) at the constraint frequencies.
%
% NOTE: alpha_1 and alpha_2 must satisfy the inequalities:
%       |alpha_2| <= 1 and (1 - alpha_2) ^ 2 - alpha_1 ^ 2 >= 0
%
%-- Reference:
%      [1] S. Langel, O. Garcia Crespillo, and M. Joerger, "Frequency-domain modeling of correlated Gaussian
%          noise in Kalman filtering," IEEE Trans. Aerosp. Electron. Syst., vol. xx, no. xx, pp. xx-xx,
%          November, 2024, doi: 10.1109/TAES.2024.3442775.

%% Error checking.
if nargin < 4
    error('Not enough input arguments.');
elseif nargin > 4
    error('Too many input arguments.');
end

if ~isvector(omega_at_maxima_of_g)
    error('A vector input for omega_at_maxima_of_g was expected, but a matrix was provided.');
elseif ~isvector(maxima_of_g)
    error('A vector input for maxima_of_g was expected, but a matrix was provided.');
elseif ~isscalar(alpha_1)
    error('A scalar was expected for alpha_1, but a different input was provided.');
elseif ~isscalar(alpha_2)
    error('A scalar was expected for alpha_2, but a different input was provided.');
end

if numel(omega_at_maxima_of_g) ~= numel(maxima_of_g)
    error('The input vectors omega_at_maxima_of_g and maxima_of_g must have the same length.');
end

if abs(alpha_2) > 1 || (1 - alpha_2) ^ 2 - alpha_1 ^ 2 < 0
    error('Autoregressive model parameters were provided that are outside the feasible region.');
end

%% Algorithm code.
% Ensure the inputs 'omega_at_maxima_of_g' and 'maxima_of_g' are column vectors.
omega_at_maxima_of_g = omega_at_maxima_of_g(:);
maxima_of_g = maxima_of_g(:);

% Let beta = (b0 - b1 * cos(w) - b2 * cos(2 * w)), so that the PSD for an AR(2) process y can be written as
% S = gamma * var_y / beta. The parameters b0, b1 and b2 are defined below.
b0 = 1 + alpha_1 ^ 2 + alpha_2 ^ 2;
b1 = 2 * alpha_1 * (1 - alpha_2);
b2 = 2 * alpha_2;

% The AR(2) PSD has a maximum or minimum at Omega_star = acos(arg) when 'arg' is in the interval (-1, 1). 
arg = alpha_1 * (alpha_2 - 1) / (4 * alpha_2);
if abs(arg) <= 1
    Omega_star = acos(arg);

    % The second derivative test is used to determine whether Omega_star corresponds to a maximum or minimum.
    % The second derivative of S is S'' = (gamma * var_y / beta ^ 3) * (2 * beta' ^ 2 - beta * beta''). Since
    % (gamma * var_y / beta ^ 3) is always positive, the sign of S'' is dictated by the sign of
    % xi = (2 * beta' ^ 2 - beta * beta'').
    beta = b0 - b1 * cos(Omega_star) - b2 * cos(2 * Omega_star);
    beta_prime = b1 * sin(Omega_star) + 2 * b2 * sin(2 * Omega_star);
    beta_double_prime = b1 * cos(Omega_star) + 4 * b2 * cos(2 * Omega_star);
    xi = 2 * beta_prime ^ 2 - beta * beta_double_prime;
    
    % Get indices to local maxima of g to the left and right of the maximum (or minimum) of S.
    less_than_Omega_star = omega_at_maxima_of_g <= Omega_star;
    greater_than_Omega_star = omega_at_maxima_of_g > Omega_star;

    % If xi > 0, the PSD has a local minimum at Omega_star. This implies that S monotonically decreases for 
    % Omega <= Omega_star, and monotonically increases for Omega > Omega_star. The exact opposite is true when
    % xi <= 0. We use this fact to sift out local maxima of g that follow the same pattern. To simplify
    % determination of the monotonic sequences, we use the fact that if a sequence of values
    % [a_1, a_2, ..., a_n] is monotonically increasing, then [a_n, a_(n-1), ..., a_1] is monotonically
    % decreasing.    
    if xi > 0
        decreasing_Omega = omega_at_maxima_of_g(less_than_Omega_star);
        decreasing_maxima_of_g = maxima_of_g(less_than_Omega_star);
        
        increasing_Omega = flipud(omega_at_maxima_of_g(greater_than_Omega_star));
        increasing_maxima_of_g = flipud(maxima_of_g(greater_than_Omega_star));
    else
        decreasing_Omega = omega_at_maxima_of_g(greater_than_Omega_star);
        decreasing_maxima_of_g = maxima_of_g(greater_than_Omega_star);
        
        increasing_Omega = flipud(omega_at_maxima_of_g(less_than_Omega_star));
        increasing_maxima_of_g = flipud(maxima_of_g(less_than_Omega_star));
    end
    decreasing_maxima_to_remove = obtain_monotonic_sequence(decreasing_maxima_of_g);
    decreasing_Omega(decreasing_maxima_to_remove) = [];
    decreasing_maxima_of_g(decreasing_maxima_to_remove) = [];

    increasing_maxima_to_remove = obtain_monotonic_sequence(increasing_maxima_of_g);
    increasing_Omega(increasing_maxima_to_remove) = [];
    increasing_maxima_of_g(increasing_maxima_to_remove) = [];

    constraint_frequencies = [decreasing_Omega; increasing_Omega];
    g_at_constraint_frequencies = [decreasing_maxima_of_g; increasing_maxima_of_g];

% If there is not a local maximum (or minimum) in the AR(2) PSD, then it must be either monotonically
% increasing or decreasing over the entire frequency range. To determine which, check the difference between
% the PSD values at Omega = 0 and Omega = pi. If S(0) - S(pi) > 0, then the model PSD is monotonically
% deceasing. Otherwise, it is monotonically increasing. From the definition for S(w) above, S(0) = gamma *
% var_y / (b0 - b1 - b2) and S(pi) = gamma * var_y / (b0 + b1 - b2). Thus, S(0) - S(pi) = gamma * var_y *
% (2 * b1 / ((b0 - b2) ^ 2 - b1 ^ 2)). Since (gamma * var_y) is always positive, we only need to check the
% sign of xi = (2 * b1 / ((b0 - b2) ^ 2 - b1 ^ 2)) to determine monotonicity.
else
    xi = 2 * b1 / ((b0 - b2) ^ 2 - b1 ^ 2);
    constraint_frequencies = omega_at_maxima_of_g;
    g_at_constraint_frequencies = maxima_of_g;
    if xi < 0
        constraint_frequencies = flipud(constraint_frequencies);
        g_at_constraint_frequencies = flipud(g_at_constraint_frequencies);
    end
    maxima_to_remove = obtain_monotonic_sequence(g_at_constraint_frequencies);
    constraint_frequencies(maxima_to_remove) = [];
    g_at_constraint_frequencies(maxima_to_remove) = [];
end

end


function points_to_remove = obtain_monotonic_sequence(sequence)

%% Function header.
% This function extracts a monotonically decreasing sequence from a given input sequence. To illustrate the
% approach, consider the following sequence of values, represented as a lollipop plot.
%
%             (1)
%              |                (4)
%              |    (2)          |
%              |     |    (3)    |
%              |     |     |     |
%              |     |     |     |
%______________|_____|_____|_____|___________________
%
% Start from the first point (1). Check if (2) is less than (1). It is, so continue. Now check if (3) is less
% than (2). It is, so continue. Check if (4) is less than (3). It is not in this case. So we look back over
% previous values in the sequence until we find a value that is greater than (4). If we find a
% value (which we would; it is point (1)), then we can remove any points in between ((2) and (3) in our case).
% If we do not find a point, then it must be that (4) is greater than (1), (2) and (3), and we can remove all
% of these previous values.
%
%-- Input:
%   sequence: An (n x 1) or (1 x n) sequence of real values.
%
%-- Output:
%   points_to_remove: An (n x 1) logical vector, with elements of 1 indicating the points to remove.

%% Error checking.
if nargin < 1
    error('Not enough input arguments.');
elseif nargin > 1
    error('Too many input arguments.');
end

if ~isvector(sequence)
    error('A vector input was expected, but a matrix was provided.');
end

%% Algorithm code.
n = numel(sequence);
points_to_remove = false(n, 1);
for idx = 2 : n
    if sequence(idx) > sequence(idx - 1)
        previous_larger_value = find(sequence(1 : idx - 1) > sequence(idx), 1, 'last');
        if isempty(previous_larger_value)
            points_to_remove(1 : idx - 1) = true;
        else
            points_to_remove(previous_larger_value + 1 : idx - 1) = true;
        end
    end
end

end
