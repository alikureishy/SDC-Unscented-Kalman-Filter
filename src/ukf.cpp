#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Chi-square distribution parameters for NIS check
#define CHI_SQ_3  7.8
#define CHI_SQ_2  5.991

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF(bool use_laser, bool use_radar) {
  use_laser_ = use_laser; // if this is false, laser measurements will be ignored (except during init)
  use_radar_ = use_radar; // if this is false, radar measurements will be ignored (except during init)

  /*****************************************************************************
   *  Initialize covariance matrix
   ****************************************************************************/
  this->P_ = MatrixXd::Identity(N_X, N_X);
  this->x_ = VectorXd::Zero(N_X);

  /*****************************************************************************
  *  Initialize the process noise covariance
  ****************************************************************************/
  this->Q_ = MatrixXd(N_AUG_DIFF, N_AUG_DIFF);
  this->Q_ << pow(STD_ACC, 2),      0.0,
                    0.0,    pow(STD_YAW_ACC, 2);

  /*****************************************************************************
  *  Initialize the measurement function for Lidar
  ****************************************************************************/
  this->H_lidar = MatrixXd(N_Z_LIDAR, N_X);
	this->H_lidar << 1, 0, 0, 0, 0,
				           0, 1, 0, 0, 0;

  /*****************************************************************************
   *  Initialize the sigma_weights
   ****************************************************************************/
  this->sigma_weights = VectorXd::Zero(N_SIGMA_PTS);
  this->sigma_weights.fill(0.5 / float(LAMBDA + N_AUG));
  this->sigma_weights[0] = LAMBDA / float(LAMBDA + N_AUG);

  /*****************************************************************************
  *  Initialize the measurement noise covariance matrix for lidar
  ****************************************************************************/
  this->R_lidar_ = MatrixXd(N_Z_LIDAR, N_Z_LIDAR);
  this->R_lidar_ << pow(STD_LAS_PX, 2),              0,
                                  0,              pow(STD_LAS_PY, 2);

  /*****************************************************************************
  *  Initialize the measurement noise covariance matrix for radar
  ****************************************************************************/
  this->R_radar_ = MatrixXd(N_Z_RADAR, N_Z_RADAR);
  this->R_radar_ << pow(STD_RAD_RHO, 2),             0,                          0,
                                  0,              pow(STD_RAD_PHI, 2),           0,
                                  0,                            0,            pow(STD_RAD_RHO_DOT, 2);
}

/**
 *
 */
const VectorXd& UKF::get_state() const {
  return this->x_;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::filter_cycle(MeasurementPackage& measurement) {
  float dt = (measurement.timestamp_ - previous_timestamp) / 1000000.0;	//dt - expressed in seconds

  if ((measurement.sensor_type_ == MeasurementPackage::LASER) && (use_laser_ == false)) {
    return;
  }
  if ((measurement.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_ == false)) {
    return;
  }
  cout << "-----------------------------------------" << endl;
  cout << "\t\t~~ " << (measurement.sensor_type_ == MeasurementPackage::LASER ? "Laser ~~" : "Radar ~~") << endl << measurement.raw_measurements_ << endl;

  if (!is_initialized_) {
    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    initialize(measurement, dt);
  }
  else  {
    /*****************************************************************************
     *  Predict + Update cycle
     ****************************************************************************/
    MatrixXd x_sigma_post_points(N_X, N_SIGMA_PTS);
    predict_state(this->x_,
                  this->P_,
                  this->sigma_weights,
                  this->Q_,
                  dt,
                  x_sigma_post_points,
                  this->x_,
                  this->P_);
    cout << "\t\tSigma-Post = " << endl << x_sigma_post_points << endl;
    cout << "\t\tx (Predicted) = " << endl << this->x_ << endl;
    cout << "\t\tP (Predicted) = " << endl << this->P_ << endl;

    if (measurement.sensor_type_ == MeasurementPackage::LASER) {
      regular_update(measurement,
                     this->H_lidar,
                     this->R_lidar_,
                     this->x_,
                     this->P_,
                     this->NIS_lidar_,
                     this->NIS_lidar_counter);
    } else {  // Radar
      unscented_update(measurement,
                       x_sigma_post_points,
                       this->sigma_weights,
                       this->R_radar_,
                       this->x_,
                       this->P_,
                       this->NIS_radar_,
                       this->NIS_radar_counter);
    }
  }

  // print the latest state and co-variance matrices
  cout << "\t\tx (Updated) = " << endl << this->x_ << endl;
  cout << "\t\tP (Updated) = " << endl << this->P_ << endl;
  cout << "\t\tNIS (Radar) ------> " << endl << this->NIS_radar_ << endl;
  cout << "\t\tNIS (Lidar) ------> " << endl << this->NIS_lidar_ << endl;
  previous_timestamp = measurement.timestamp_;
}

/**
 * Initialize the state using the first measurement
 */
void UKF::initialize(MeasurementPackage measurement, double dt) {
  if (measurement.sensor_type_ == MeasurementPackage::RADAR) {
    Tools::from_polar_to_ctrv(measurement.raw_measurements_, this->x_);
  }
  else if (measurement.sensor_type_ == MeasurementPackage::LASER) {
    Tools::from_cartesian_to_ctrv(measurement.raw_measurements_, this->x_);
  }
  is_initialized_ = true;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 */
void UKF::predict_state(const VectorXd& state, const MatrixXd& covariance, const VectorXd& sigma_weights, const MatrixXd& process_noise, double dt, MatrixXd& x_sigma_post_points, VectorXd& predicted_state, MatrixXd& predicted_covariance) const {
  MatrixXd x_sigma_pre_points(N_AUG, N_SIGMA_PTS);
  generate_sigma_points(this->x_, this->P_, x_sigma_pre_points);
  cout << "\t\tSigma-Pre = " << endl << x_sigma_pre_points << endl;

  transform_sigma_points(x_sigma_pre_points, dt, x_sigma_post_points);
  cout << "\t\tSigma-Post = " << endl << x_sigma_post_points << endl;

  extract_mean(x_sigma_post_points, sigma_weights, predicted_state);
  extract_state_covariance(x_sigma_post_points, sigma_weights, predicted_state, predicted_covariance);
}

/**
 *
 */
void UKF::unscented_update(MeasurementPackage measurement, const MatrixXd& x_sigma_post_points, const VectorXd& sigma_weights, const MatrixXd& measurement_noise, VectorXd& state, MatrixXd& covariance, double& nis, int& nis_counter) const {
  assert (measurement.sensor_type_ == MeasurementPackage::RADAR);

  MatrixXd z_sigma_post_points = MatrixXd(N_Z_RADAR, N_SIGMA_PTS);
  VectorXd z_sigma_mean = VectorXd(N_Z_RADAR);
  MatrixXd z_sigma_covariance = MatrixXd::Zero(N_Z_RADAR, N_Z_RADAR);
  MatrixXd x_z_cross_correlation = MatrixXd::Zero(N_X, N_Z_RADAR);
  MatrixXd kalman_gain(N_X, N_Z_RADAR);

  Tools::from_ctrvs_to_polars(x_sigma_post_points, z_sigma_post_points);
  extract_mean(z_sigma_post_points, sigma_weights, z_sigma_mean);
  extract_radar_covariance(z_sigma_post_points, sigma_weights, z_sigma_mean, measurement_noise, z_sigma_covariance);
  unscented_kalmanize(measurement, z_sigma_post_points, x_sigma_post_points, sigma_weights, z_sigma_mean, z_sigma_covariance, state, covariance, x_z_cross_correlation, kalman_gain, nis, nis_counter);
}

/**
 *
 */
void UKF::generate_sigma_points(const MatrixXd& state, const MatrixXd& covariance, MatrixXd& sigma_pre_points) const {
  // Create initial Augmented state vector x_aug
	VectorXd x_aug = VectorXd::Zero(N_AUG); // Mean of process noise (id#5 and id#6) is always zero
	// x_aug.segment(0, N_X) = state;
  x_aug << state, 0.0, 0.0;

  // Create initial Augmented covariance matrix P_aug, from P
	MatrixXd P_aug = MatrixXd::Zero(N_AUG, N_AUG);
	P_aug.topLeftCorner(N_X, N_X) = covariance;
  P_aug(5,5) = STD_ACC*STD_ACC;
  P_aug(6,6) = STD_YAW_ACC*STD_YAW_ACC;
  cout << "\t\tP_Aug = " << endl << P_aug << endl;

  // Append the noise covariance (Q_) to the bottom-right corner of P_aug
  // TODO: P_aug.bottomRightCorner(2, 2) << this->Q_; ???^^^

  // Create sq-rt matrix A for use with sigma point calculations
	MatrixXd A = P_aug.llt().matrixL();
	if (P_aug.llt().info() == Eigen::NumericalIssue) {
	    std::cout << "LLT failed!" << std::endl; // if decomposition fails, we have numerical issues
      Eigen::EigenSolver<MatrixXd> es(P_aug);
      std::cout << "Eigenvalues of P_aug:" << std::endl << es.eigenvalues() << endl;
	    // throw std::range_error("LLT failed");
	}

  // Calculate sigma points and set them into x_sigma_pre_points
	sigma_pre_points.col(0) = x_aug;
	// MatrixXd term = sqrt(LAMBDA + N_AUG) * A;
	for (int i = 0; i < N_AUG; ++i) {
		sigma_pre_points.col(i + 1) = x_aug + (sqrt(LAMBDA+N_AUG) * A.col(i)); //term.col(i);
		sigma_pre_points.col(i + N_AUG + 1) = x_aug - (sqrt(LAMBDA + N_AUG) * A.col(i)); //term.col(i);
	}
}

/**
 *
 */
void UKF::transform_sigma_points(const MatrixXd& sigma_pre_points, double dt, MatrixXd& sigma_post_points) const {
  assert (sigma_pre_points.cols() == N_SIGMA_PTS);
  for (int i = 0; i < N_SIGMA_PTS; i++) {
    VectorXd new_point = transform_sigma_point(sigma_pre_points.col(i), dt);
    assert (new_point.rows() == N_X); // Should be [N_X]
    sigma_post_points.col(i) << new_point;
  }
}

/**
 *
 */
VectorXd UKF::transform_sigma_point(const VectorXd& aug_point, double dt) const {
  assert(aug_point.rows() == N_AUG);
  // double px = aug_point(0);
	// double py = aug_point(1);
	double v = aug_point(2);
	double yaw = aug_point(3);
	double yaw_dot = aug_point(4);
	double noise_acc = aug_point(5);
	double noise_yaw_rate = aug_point(6);

  // F-term:
	VectorXd F_x_term(N_X);
	F_x_term.fill(0.0);
	if(yaw_dot == 0) {
		F_x_term(0) = (v * cos(yaw) * dt);
		F_x_term(1) = (v * sin(yaw) * dt);
  } else {
		F_x_term(0) = (v/(float)yaw_dot * (sin(yaw + yaw_dot * dt) - sin(yaw)));
		F_x_term(1) = (v/(float)yaw_dot * (-cos(yaw + yaw_dot * dt) + cos(yaw)));
	}
  F_x_term(2) = 0;
  F_x_term(3) = (yaw_dot * dt);
  F_x_term(4) = 0;

  // Q-term:
	VectorXd Q_term(N_X);
	Q_term(0) = (1/2.0 * dt * dt * cos(yaw) * noise_acc);
	Q_term(1) = (1/2.0 * dt * dt * sin(yaw) * noise_acc);
	Q_term(2) = (noise_acc * dt);
	Q_term(3) = (1/2.0 * dt * dt * noise_yaw_rate);
	Q_term(4) = (noise_yaw_rate * dt);

  // X_aug_next = X_aug(aug_point[0:5]) + F_x_term + Q_term
  return aug_point.head(N_X) + F_x_term + Q_term;
}

/**
 * Used to determine the mean and covariance of the given sigma points, regardless of
 * their dimensionality.
 */
void UKF::extract_mean(const MatrixXd& sigma_points, const VectorXd& sigma_weights, VectorXd& mean) const {
  assert(sigma_points.cols() == N_SIGMA_PTS);
  assert(sigma_points.col(0).rows() == mean.rows());
  assert (sigma_weights.rows() == N_SIGMA_PTS);

  // Mean:
  mean.fill(0.0);
  for(int i=0;i < sigma_points.cols(); i++) {
	  mean += sigma_weights(i) * sigma_points.col(i);
	}
}

void UKF::extract_state_covariance(const MatrixXd &sigma_post_points, const VectorXd& sigma_weights, const VectorXd &mean, MatrixXd &covariance) const {
  extract_covariance(sigma_post_points, sigma_weights, 3, mean, covariance);
}

void UKF::extract_radar_covariance(const MatrixXd &sigma_post_points, const VectorXd& sigma_weights, const VectorXd &mean, const MatrixXd& measurement_noise, MatrixXd &covariance) const {
  extract_covariance(sigma_post_points, sigma_weights, 1, mean, covariance);
  covariance+=measurement_noise;
}

void UKF::extract_covariance(const MatrixXd &sigma_post_points, const VectorXd& sigma_weights, int idx_of_angle, const VectorXd &mean, MatrixXd &covariance) const {
  assert(sigma_post_points.cols() == N_SIGMA_PTS);
  assert(sigma_post_points.col(0).rows() == mean.rows());
  assert (covariance.rows() == sigma_post_points.col(0).rows());
  assert (covariance.cols() == sigma_post_points.col(0).rows());
  assert (sigma_weights.rows() == N_SIGMA_PTS);

 	covariance.fill(0.0);
	for(int i=0; i < sigma_post_points.cols(); i++) {
		VectorXd diff_from_mean = sigma_post_points.col(i) - mean;
    diff_from_mean(idx_of_angle) = fmod(diff_from_mean(idx_of_angle), 2.0*M_PI); // Tools::normalize_angle(diff_from_mean(idx_of_angle));
    covariance += sigma_weights(i) * diff_from_mean * diff_from_mean.transpose() ;
	}
}

/**
 *
 */
void UKF::unscented_kalmanize(const MeasurementPackage& measurement, const MatrixXd& z_sigma_post_points, const MatrixXd& x_sigma_post_points, const VectorXd& sigma_weights, const VectorXd& z_sigma_mean, const MatrixXd& z_sigma_covariance, VectorXd& x_mean, MatrixXd& x_covariance, MatrixXd& x_z_cross_correlation, MatrixXd& kalman_gain, double& nis, int& nis_counter) const {
	// Calculate the cross-correlation matrix

	x_z_cross_correlation.fill(0.0);
	for(int i=0;i<N_SIGMA_PTS; ++i) {

	  VectorXd x_sigma_diff = (x_sigma_post_points.col(i) - x_mean);
    x_sigma_diff(3) = Tools::normalize_angle(x_sigma_diff(3));

    VectorXd z_sigma_diff = z_sigma_post_points.col(i) - z_sigma_mean;
    z_sigma_diff(1) = Tools::normalize_angle(z_sigma_diff(1));

		x_z_cross_correlation += (sigma_weights(i) * (x_sigma_diff * z_sigma_diff.transpose()));
	}
	// Use these to update the state x_ and P_ using the Kalman Gain
	kalman_gain = x_z_cross_correlation * z_sigma_covariance.inverse();

	//residual
	VectorXd z = measurement.raw_measurements_;
	VectorXd z_diff = z - z_sigma_mean;
  z_diff(1) = Tools::normalize_angle(z_diff(1));

  //update state mean and covariance matrix
	x_mean = x_mean + kalman_gain * (z_diff);
	x_covariance = x_covariance - kalman_gain * z_sigma_covariance * kalman_gain.transpose();

	nis = z_diff.transpose() * z_sigma_covariance.transpose() * z_diff;
  if (nis > CHI_SQ_3)
    nis_counter++;

  cout << "\t\tz (Predicted) = " << endl << z_sigma_mean << endl;
  cout << "\t\tS (Predicted) = " << endl << z_sigma_covariance << endl;
  cout << "\t\tSigma x-z correlation = " << endl << x_z_cross_correlation << endl;
  cout << "\t\ty = " << endl << z_diff << endl;
  cout << "\t\tKalman Gain = " << endl << kalman_gain << endl;
}

void UKF::regular_update(const MeasurementPackage& measurement, const MatrixXd& measurement_function, const MatrixXd& measurement_noise, VectorXd& state, MatrixXd& covariance, double& nis, int& nis_counter) const {
  VectorXd z_pred = measurement_function * state;

	VectorXd z = measurement.raw_measurements_;

	VectorXd y = z - z_pred;
	MatrixXd Ht = measurement_function.transpose();
	MatrixXd S = measurement_function * covariance * Ht + measurement_noise;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = covariance * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	state = state + (K * y);
	MatrixXd I = MatrixXd::Identity(state.rows(), state.rows());
	covariance = (I - K * measurement_function) * covariance;

	nis = y.transpose() * Si * y;
  if (nis > CHI_SQ_2)
    nis_counter++;

  cout << "\t\tz (Predicted) = " << endl << z_pred << endl;
  cout << "\t\tS (Predicted) = " << endl << S << endl;
  cout << "\t\ty = " << endl << y << endl;
  cout << "\t\tKalman Gain = " << endl << K << endl;
}