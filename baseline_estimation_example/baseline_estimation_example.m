function baseline_estimation_example()

%% Function header.
% This function generates the results for the baseline estimation problem in Section V of [1]. 
% An input file directly provides the following inputs:

%-- sample_ACSs    : A 7x1801 matrix, where each row is the sample autocorrelation sequence (ACS) for a given
%                    satellite's single difference carrier phase measurement error. Each ACS corresponds to a
%                    particular curve in Fig. 7 of [1] for the lag vector [0 : 0.5 : 900] seconds.

%-- LOS_matrices   : An nx3 matrix that is a stack of the line-of-sight matrices 'E' in Eq. (28) of [1]. That
%                    is, this matrix is equal to: [E_0; E_1; E_2; ...].

%-- PRNs           : A 7x1 vector with the PRN numbers for each satellite.

% NOTE: To generate the true covariance matrix, we used the results from [2] (see Appendix A).

%-- References:
%      [1] S. Langel, O. Garcia Crespillo, and M. Joerger, "Frequency-domain modeling of correlated Gaussian
%          noise in Kalman filtering," IEEE Trans. Aerosp. Electron. Syst., vol. xx, no. xx, pp. xx-xx,
%          November, 2024, doi: 10.1109/TAES.2024.3442775.

%      [2] S. Langel, S. Khanafseh, and B. Pervan, "Bounding integrity risk for sequential state estimators
%          with stochastic modeling uncertainty," AIAA J. Guid. Control Dyn., vol. 37, no. 1, pp. 36-46. 

%% Algorithm code.

% Load and unpack the data file.
data_structure = load('baseline_estimation_data_file.mat');
sample_ACSs = data_structure.sample_ACSs;
LOS_matrices = data_structure.LOS_matrices;
PRNs = data_structure.PRNs;

% Simulation control parameters.
sampling_interval = 0.5;           % In seconds.
filter_duration = 600;             % In seconds.
window_duration = 800;             % In seconds.
maximum_lag = 900;                 % seconds.
t = (0: 0.02 : 1) * 2 * pi;        % Parametric variable for plotting covariance ellipses.

% Construct the lag vector, which for this example ranges from 0 to 900 seconds at 0.5 second intervals. Also
% create size variables and the double-difference transformation matrix (see Eq. (28) in [1]).
lags = (0 : sampling_interval : maximum_lag)';
num_sv = length(PRNs);
num_KF_states = 3 + 3 * num_sv - 1;
D = [eye(num_sv - 1), -ones(num_sv - 1, 1)];

%-------------------------------------------------------------------------------------------------------------
%------------- RUN A KALMAN FILTER USING SECOND-ORDER AUTOREGRESSIVE MODELS FOR CORRELATED NOISE -------------
%-------------------------------------------------------------------------------------------------------------

% Determine second-order autoregressive models for each satellite's single difference carrier phase
% measurement error.
AR2_noise_models = zeros(num_sv, 4);
for idx = 1 : num_sv
    model_parameters = autoregressive_model_determination(sample_ACSs(idx, :), lags, filter_duration, ...
                                                          window_duration, 'ar2');
    AR2_noise_models(idx, :) = model_parameters;
end

% Form matrices needed for the KF covariance propagation. The matrices Phi_a, U, L and Q are defined in Eq.
% (31) of [1], and the matrix Lambda is defined in Eq. (34) of [1].
Phi_a = [];
U = [];
L = [];
Q = [];
Lambda = [];
for idx = 1 : num_sv
    % Extract the noise model parameters for the current SV.
    alpha_1 = AR2_noise_models(idx, 1);
    alpha_2 = AR2_noise_models(idx, 2);
    ar_variance = AR2_noise_models(idx, 3);
    wgn_variance = AR2_noise_models(idx, 4);
    
    % Form the Phi, U, L, and Lambda matrices for the current SV.
    gam = (1 + alpha_2) / (1 - alpha_2) * ((1 - alpha_2) ^ 2 - alpha_1 ^ 2);
    Phi_idx = [alpha_1, alpha_2; 1, 0];
    U_idx = [gam * ar_variance, 0; 0, 0];
    L_idx = [1, 0];
    Lambda_idx = ar_variance * [1, alpha_1 / (1 - alpha_2); alpha_1 / (1 - alpha_2), 1];

    % Incorporate the matrices for the current SV into the overall KF matrices.
    Phi_a = blkdiag(Phi_a, Phi_idx);
    U = blkdiag(U, U_idx);
    L = blkdiag(L, L_idx);
    Q = blkdiag(Q, wgn_variance);
    Lambda = blkdiag(Lambda, Lambda_idx);
end

% Initial covariance matrices. 'Sigma_using_AR2_models' is the usual covariance matrix computed by the KF, and
% we use P to denote the true covariance matrix.
Sigma_using_AR2_models = blkdiag(eye(3) * 1E+04, eye(num_sv - 1) * 1E+04, Lambda);
P0 = blkdiag(Sigma_using_AR2_models(1 : 3 + num_sv - 1, 1 : 3 + num_sv - 1), zeros(2 * num_sv));

% State transition matrix F, process noise mapping matrix G, and measurement noise covariance matrix V.
F = blkdiag(eye(3 + num_sv - 1), Phi_a);
G = [zeros(3, 2 * num_sv); zeros(num_sv - 1, 2 * num_sv); eye(2 * num_sv)];
V = D * Q * D';

% Initialize a separate transition matrix 'scriptF' and noise mapping matrices 'scriptJ' for each SV. These
% are needed to propagate the true covariance matrix using the results from Appendix A in [2].
scriptF = eye(num_KF_states);
for i = 1 : num_sv
    sv{i}.scriptJ = zeros(num_KF_states, 0);
end

% Preallocate data storage matrices.
total_num_epochs = filter_duration / sampling_interval + 1;
KF_covariance_using_AR2_models = zeros(total_num_epochs, 1 + (3 + num_sv - 1) ^ 2);
true_covariance_using_AR2_models = zeros(total_num_epochs, 1 + (3 + num_sv - 1) ^ 2);
minimum_eig_val = zeros(total_num_epochs, 1);

% Initialize counters and index variables.
counter = 0;
ind_start = 0;

% Covariance propagation loop.
for TimeIndex = 1 : total_num_epochs
    
    % Line-of-sight vectors.
    ind_start = ind_start + 1;
    ind_end = TimeIndex * num_sv;
    E = LOS_matrices(ind_start : ind_end, :);
    
    % Time update.
    Sigma_using_AR2_models = F * Sigma_using_AR2_models * F' + G * U * G';
    scriptF = F * scriptF;
    for idx = 1 : num_sv
        sv{idx}.scriptJ = F * sv{idx}.scriptJ;
    end

    % Measurement update.
    H = [-D * E, eye(num_sv - 1), D * L];
    K = (Sigma_using_AR2_models * H') / (H * Sigma_using_AR2_models * H' + V);
    S = eye(num_KF_states) - K * H;
    Sigma_using_AR2_models = S * Sigma_using_AR2_models * S' + K * V * K';
    
    % Update the matrices used to propagate the true covariance matrix.
    scriptF = S * scriptF;
    matrix = K * D;
    for idx = 1 : num_sv
        current_vect = matrix(:, idx);
        sv{idx}.scriptJ = [S * sv{idx}.scriptJ, current_vect];
    end
    
    % Update the true covariance matrix.
    maximum_lag_number = (TimeIndex - 1) * sampling_interval;
    P = scriptF * P0 * scriptF';
    for idx = 1 : num_sv
        current_ACS = sample_ACSs(idx, :);
        lag_index = (lags >= 0 & lags <= maximum_lag_number)';
        R_matrix = toeplitz(current_ACS(lag_index));
        Gamma = sv{idx}.scriptJ; 
        P = P + Gamma * R_matrix * Gamma';
    end
    
    % Save results.
    counter = counter + 1;
    time = (TimeIndex - 1) * sampling_interval;
    Sigma_pos_amb = Sigma_using_AR2_models(1 : 3 + num_sv - 1, 1 : 3 + num_sv - 1);
    P_pos_amb = P(1 : 3 + num_sv - 1, 1 : 3 + num_sv - 1);
    KF_covariance_using_AR2_models(counter, :) = [time, Sigma_pos_amb(:)'];
    true_covariance_using_AR2_models(counter, :) = [time, P_pos_amb(:)'];
    
    new_eig_vals = eig(Sigma_pos_amb - P_pos_amb);
    minimum_eig_val(counter, 1) = min(new_eig_vals);

    %-- Update the starting index for satellite geometry.
    ind_start = ind_end;
end

%-------------------------------------------------------------------------------------------------------------
%---------------- RUN A KALMAN FILTER USING WHITE GAUSSIAN NOISE MODELS FOR CORRELATED NOISE -----------------
%-------------------------------------------------------------------------------------------------------------

% Size variables.
num_KF_states = 3 + num_sv - 1;

% Determine white Gaussian noise for each satellite's single difference carrier phase measurement error.
WGN_noise_models = zeros(num_sv, 1);
for idx = 1 : num_sv
    model_parameters = autoregressive_model_determination(sample_ACSs(idx, :), lags, filter_duration, ...
                                                          window_duration, 'white noise', 1, 1);
    WGN_noise_models(idx, :) = model_parameters;
end

% Initial KF covariance matrix.
Sigma_using_WGN_models = blkdiag(eye(3) * 1E+04, eye(num_sv - 1) * 1E+04);

% State transition matrix and measurement noise covariance matrix.
F = eye(3 + num_sv - 1);
V = D * diag(WGN_noise_models) * D';

% Preallocate data storage matrix.
KF_covariance_using_WGN_models = zeros(total_num_epochs, 1 + (3 + num_sv - 1) ^ 2);

% Initialize counters and index variables.
counter = 0;
ind_start = 0;

% Covariance propagation loop.
for TimeIndex = 1 : total_num_epochs
    
    % Line-of-sight vectors.
    ind_start = ind_start + 1;
    ind_end = TimeIndex * num_sv;
    E = LOS_matrices(ind_start : ind_end, :);
    
    % Time update.
    Sigma_using_WGN_models = F * Sigma_using_WGN_models * F';

    % Measurement update.
    H = [-D * E, eye(num_sv - 1)];
    K = (Sigma_using_WGN_models * H') / (H * Sigma_using_WGN_models * H' + V);
    S = eye(num_KF_states) - K * H;
    Sigma_using_WGN_models = S * Sigma_using_WGN_models * S' + K * V * K';
    
    % Save results.
    counter = counter + 1;
    time = (TimeIndex - 1) * sampling_interval;
    Sigma_pos_amb = Sigma_using_WGN_models(1 : 3 + num_sv - 1, 1 : 3 + num_sv - 1);
    KF_covariance_using_WGN_models(counter, :) = [time, Sigma_pos_amb(:)'];

    % Update the starting index for satellite geometry.
    ind_start = ind_end;
end

% Make Figs. 10 through 13 in [1].
make_plots(sampling_interval, filter_duration, num_sv, t, minimum_eig_val, KF_covariance_using_AR2_models, ...
           KF_covariance_using_WGN_models, true_covariance_using_AR2_models);

end
