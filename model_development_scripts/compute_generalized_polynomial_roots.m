function poly_roots = compute_generalized_polynomial_roots(poly_coeff)

%% Function header.
% This function computes the real roots in the interval (-1, 1) of the generalized polynomial
% p(x) = sum_{i=0}^n c_i*U_i(x), where U_i(x) is the ith order Chebyshev polynomial of the second kind. The
% algorithm is based on Lemma 2 in [1].
%
%-- Input:
%      poly_coeff: (n + 1) x 1 column vector containing the generalized polynomial coefficients c_i.
%
%-- Output:
%      poly_roots: m x 1 column vector containing the real roots of p(x) in the interval (-1, 1).
%
%-- Reference:
%      [1] S. Langel, O. Garcia Crespillo, and M. Joerger, "Frequency-domain modeling of correlated Gaussian
%          noise in Kalman filtering," IEEE Trans. Aerosp. Electron. Syst., vol. xx, no. xx, pp. xx-xx,
%          November, 2024, doi: 10.1109/TAES.2024.3442775.
%-------------------------------------------------------------------------------------------------------------

%% Error checking.
if nargin < 1
    error('Not enough input arguments.');
elseif nargin > 1
    error('Too many input arguments.');
end

if ~isvector(poly_coeff)
    error('A vector input was expected, but a matrix was provided.');
end

%% Algorithm code.
% Ensure the input poly_coeff is a column vector.
poly_coeff = poly_coeff(:);

% Set the upper limit for the summation.
n = numel(poly_coeff) - 1;

% Construct the A matrix defined in Lemma 2 of [1].
A = diag([poly_coeff(end); ones(n - 1, 1)]);

% Construct the B matrix in Eq. (20) of [1].
first_row = [1, 0, 1, zeros(1, n - 3)];
first_col = [1; zeros(n - 2, 1)];
M = toeplitz(first_col, first_row);

d_vector = flipud(poly_coeff);
d_vector(1) = [];
d_vector(2) = d_vector(2) - poly_coeff(end);
B = (1/2) * [-d_vector'; M];

% Compute all roots of p(x) through generalized eigenvalues.
poly_roots = eig(B, A);

% Remove any roots that are outside the interval (-1, 1). This includes ALL roots that are imaginary.
poly_roots(isinf(poly_roots)) = [];
idx = abs(imag(poly_roots)) < eps;
poly_roots = poly_roots(idx);
poly_roots(abs(poly_roots) > 1) = [];

end
