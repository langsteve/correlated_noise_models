function parameter_matrix = get_feasible_points(num_elements)

%% Function header.
% This function determines a set of points evenly distributed throughout the feasible region S of a second
% order autoregressive process by partitioning S into triangular elements and extracting the coordinates of
% the vertices of each triangle.
%
%-- Input:
%   num_elements: Positive integer indicating how many triangular elements to use when partitioning S. See
%                 Fig. 5 in [1].
%
%-- Output:
%   parameter_matrix: An (nx2) matrix with each row describing the coordinates of a single point in S. The
%                     first column is alpha_1 and the second column is alpha_2.
%
%-- Reference:
%      [1] S. Langel, O. Garcia Crespillo, and M. Joerger, "Frequency-domain modeling of correlated Gaussian
%          noise in Kalman filtering," IEEE Trans. Aerosp. Electron. Syst., vol. xx, no. xx, pp. xx-xx,
%          November, 2024, doi: 10.1109/TAES.2024.3442775.

%% Error checking.
if ~isscalar(num_elements) || num_elements <= 0 || mod(num_elements, 1) ~= 0
    error('The input num_elements must be a positive integer.')
end

%% Algorithm code.
% To break up S evenly into triangular elements, the number of elements must be equal to 1, (1 + 3), (1 + 3 +
% 5), etcetera. In general, the rule is that num_elements must equal the sum of a sequence of odd integers.
% The sum of the first N odd integers is N^2, so first we determine N = sqrt(num_elements) and round up, if
% necessary, to ensure that N is an integer. Then, an adjustment is made to N (if necessary) to ensure that a
% sequence of points exists on the alpha_1 axis in case a Gauss-Markov model is sought. This requires (2/N) to
% be equal to 2 raised to a negative integer power, i.e., (2/N) = 2^m for some negative integer m. If this
% condition is not met, then m is rounded down to the nearest negative integer for which the condition is
% true, and N is recalculated.
N = ceil(sqrt(num_elements));
m = floor(log(2/N) / log(2));
N = 2 ^ (1 - m);

% To generate the appropriate number of triangular elements, a step size of 2/N must be used.
step_size = 2/N;
delta = (0 : step_size : 2.0)';

total_num_points = (N + 1) * (N + 2) / 2;
parameter_matrix = zeros(total_num_points, 2);
count = 0;
for idx = 1 : numel(delta)
    delta_cur = delta(idx);
    alpha_1 = (-2 + delta_cur : step_size : delta_cur)';
    alpha_2 = (-1 - delta_cur : step_size : 1 - delta_cur)';
    alpha_1(alpha_2 < -1) = [];
    alpha_2(alpha_2 < -1) = [];
    parameter_matrix(count + 1 : count + numel(alpha_1), :) = [alpha_1, alpha_2];
    count = count + numel(alpha_1);
end

end
