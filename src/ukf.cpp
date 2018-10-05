#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF(bool use_laser, bool use_radar) {
  use_laser_ = use_laser; // if this is false, laser measurements will be ignored (except during init)
  use_radar_ = use_radar; // if this is false, radar measurements will be ignored (except during init)

  std::cout << "3.0" << std::endl;
  /*****************************************************************************
   *  Initialize covariance matrix
   ****************************************************************************/
  this->P_ = MatrixXd(N_X, N_X);
          // px, py, v, phi, phi_dot
  this->P_ << 1, 0, 0, 0, 0, // px
              0, 1, 0, 0, 0, // py
              0, 0, 1, 0, 0, // v
              0, 0, 0, 1, 0, // phi
              0, 0, 0, 0, 1; // phi_dot
  std::cout << "3.0.1" << std::endl;
  this->x_ = VectorXd(N_X);
  this->x_.fill(0.0);

  std::cout << "3.1" << std::endl;
  /*****************************************************************************
  *  Initialize the process noise covariance
  ****************************************************************************/
  this->Q_ = MatrixXd(N_AUG_DIFF, N_AUG_DIFF);
  this->Q_ << pow(STD_ACC, 2), 0.0,
      0.0, pow(STD_YAW_ACC, 2);
  std::cout << "3.2" << std::endl;

  /*****************************************************************************
   *  Initialize the sigma_weights
   ****************************************************************************/
  this->sigma_weights = VectorXd(N_SIGMA_PTS);
  this->sigma_weights.fill(0.5 / (LAMBDA + N_AUG));
  this->sigma_weights[0] = LAMBDA / (LAMBDA + N_AUG);
  std::cout << "3.3" << std::endl;

  /*****************************************************************************
  *  Initialize the measurement noise covariance matrix for lidar
  ****************************************************************************/
  this->R_lidar_ = MatrixXd(N_Z_LIDAR, N_Z_LIDAR);
  this->R_lidar_ << pow(STD_LAS_PX, 2),              0,
                                  0,              pow(STD_LAS_PY, 2);
  std::cout << "3.4" << std::endl;

  /*****************************************************************************
  *  Initialize the measurement noise covariance matrix for radar
  ****************************************************************************/
  this->R_radar_ = MatrixXd(N_Z_RADAR, N_Z_RADAR);
  this->R_radar_ << pow(STD_RAD_RHO, 2),             0,                          0,
                                  0,              pow(STD_RAD_PHI, 2),           0,
                                  0,                            0,            pow(STD_RAD_RHO_DOT, 2);
  std::cout << "3.5" << std::endl;
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
void UKF::filter_cycle(MeasurementPackage measurement) {
  float dt = (measurement.timestamp_ - previous_timestamp) / 1000000.0;	//dt - expressed in seconds

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
    if (measurement.sensor_type_ == MeasurementPackage::LASER) {
      predict_state(this->x_, this->P_, this->sigma_weights, this->R_lidar_, dt, x_sigma_post_points, this->x_, this->P_);
      update_state(measurement, x_sigma_post_points, this->x_, this->P_, this->NIS_lidar_);
    } else {
      predict_state(this->x_, this->P_, this->sigma_weights, this->R_radar_, dt, x_sigma_post_points, this->x_, this->P_);
      update_state(measurement, x_sigma_post_points, this->x_, this->P_, this->NIS_radar_);
    }
  }

  // print the latest state and co-variance matrices
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
  previous_timestamp = measurement.timestamp_;
}

/**
 * Initialize the state using the first measurement
 */
void UKF::initialize(MeasurementPackage measurement, double dt) {
  /*****************************************************************************
   *  Read the first measurement
   ****************************************************************************/
  this->x_.fill(0.0);
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
  /*****************************************************************************
   *  Generate Sigma points
   ****************************************************************************/
  MatrixXd x_sigma_pre_points(N_AUG, N_SIGMA_PTS);
  generate_sigma_points(this->x_, this->P_, x_sigma_pre_points);

  /*****************************************************************************
   *  Extrapolate sigma points for process model at time k+1
   ****************************************************************************/
  transform_sigma_points(x_sigma_pre_points, dt, x_sigma_post_points);

  /*****************************************************************************
   *  Calculate predicted state and covariance
   ****************************************************************************/
  extract_mean_and_covariance(x_sigma_post_points, sigma_weights, predicted_state, predicted_covariance);

  /*****************************************************************************
   *  Add process noise:
   ****************************************************************************/
  predicted_state = predicted_state + process_noise;
}

/**
 *
 */
void UKF::update_state(MeasurementPackage measurement, const MatrixXd& x_sigma_post_points, VectorXd& state, MatrixXd& covariance, double& nis) const {
  /*****************************************************************************
  *  Create placeholder variables
  ****************************************************************************/
    MatrixXd z_sigma_post_points;
    VectorXd z_sigma_mean;
    MatrixXd z_sigma_covariance;
    MatrixXd x_z_cross_correlation;
    MatrixXd kalman_gain;

  /*****************************************************************************
  *  Type-specific: Initialize matrices according to the measurement type, and translate
  ****************************************************************************/
  if (use_radar_ && (measurement.sensor_type_ == MeasurementPackage::RADAR)) {
    z_sigma_post_points = MatrixXd(N_Z_RADAR, N_SIGMA_PTS);
    z_sigma_mean = VectorXd(N_Z_RADAR);
    z_sigma_covariance = MatrixXd(N_Z_RADAR, N_Z_RADAR);
    x_z_cross_correlation = MatrixXd(N_X, N_Z_RADAR);
    kalman_gain = MatrixXd(N_X, N_Z_RADAR);
    Tools::from_ctrvs_to_polars(x_sigma_post_points, z_sigma_post_points);
  }
  else if (use_laser_ && (measurement.sensor_type_ == MeasurementPackage::LASER)) {
    z_sigma_post_points = MatrixXd(N_Z_LIDAR, N_SIGMA_PTS);
    z_sigma_mean = VectorXd(N_Z_LIDAR);
    z_sigma_covariance = MatrixXd(N_Z_LIDAR, N_Z_LIDAR);
    x_z_cross_correlation = MatrixXd(N_X, N_Z_LIDAR);
    kalman_gain = MatrixXd(N_X, N_Z_LIDAR);
    Tools::from_ctrvs_to_cartesians(x_sigma_post_points, z_sigma_post_points);
  }

  /*****************************************************************************
  *  Determine the measurement and covariance that the translated sigma points yield,
  *  as the predicted measurement
  ****************************************************************************/
  extract_mean_and_covariance(z_sigma_post_points, this->sigma_weights, z_sigma_mean, z_sigma_covariance);

  /*****************************************************************************
  *  Utilize the predicted and actual measurements to update the state
  ****************************************************************************/
  kalmanize(measurement, z_sigma_post_points, x_sigma_post_points, this->sigma_weights, z_sigma_mean, z_sigma_covariance, state, covariance, x_z_cross_correlation, kalman_gain, nis);
}

/**
 *
 */
void UKF::generate_sigma_points(const MatrixXd& state, const MatrixXd& covariance, MatrixXd& sigma_pre_points) const {
  /*****************************************************************************
   *  Create initial Augmented state vector x_aug
   ****************************************************************************/
	VectorXd x_aug = VectorXd(N_AUG);
	x_aug.fill(0.0);
	x_aug.segment(0, N_X) = state;
  x_aug.segment(N_X, N_AUG).fill(0.0); // Mean of process noise is always zero

  /*****************************************************************************
   *  Create initial Augmented covariance matrix P_aug, from P
   ****************************************************************************/
	MatrixXd P_aug = MatrixXd(N_AUG, N_AUG);
	P_aug.fill(0.0);
	P_aug.topLeftCorner(N_X, N_X) = covariance;

  /*****************************************************************************
  *  Append the noise covariance (Q_) to the bottom-right corner of P_aug
  ****************************************************************************/
  P_aug.bottomRightCorner(2, 2) << this->Q_;

  /*****************************************************************************
   *  Create sq-rt matrix A for use with sigma point calculations
   ****************************************************************************/
	MatrixXd A = P_aug.llt().matrixL();
	if (P_aug.llt().info() == Eigen::NumericalIssue) {
	    std::cout << "LLT failed!" << std::endl; // if decomposition fails, we have numerical issues
      Eigen::EigenSolver<MatrixXd> es(P_aug);
      std::cout << "Eigenvalues of P_aug:" << std::endl << es.eigenvalues() << endl;
	    throw std::range_error("LLT failed");
	}

  /*****************************************************************************
  *  Calculate sigma points and set them into x_sigma_pre_points
  ****************************************************************************/
	sigma_pre_points.col(0) = x_aug;
	MatrixXd term = sqrt(LAMBDA + N_AUG) * A;
	for (int i = 0; i < N_AUG; ++i) {
		sigma_pre_points.col(i + 1) = x_aug + term.col(i);
		sigma_pre_points.col(i + N_AUG + 1) = x_aug - term.col(i);
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
  /*****************************************************************************
  * For easy reference
  ****************************************************************************/
  assert(aug_point.rows() == N_AUG);
  // double px = aug_point(0);
	// double py = aug_point(1);
	double v = aug_point(2);
	double psi = aug_point(3);
	double psi_dot = aug_point(4);
	double noise_acc = aug_point(5);
	double noise_yaw_rate = aug_point(6);

  /*****************************************************************************
  * Calculate the increment of x over dt (integral of x' over dt)
  *   Because X_next = X(sigma_points) + integral(X'dt) + Process_noise(Q)
  ****************************************************************************/
	VectorXd F_x_term(N_X);
	F_x_term.fill(0.0);
	if(psi_dot == 0) {
		F_x_term(0) = (v * cos(psi) * dt);
		F_x_term(1) = (v * sin(psi) * dt);
		F_x_term(2) = 0;
		F_x_term(3) = 0;
		F_x_term(4) = 0;
  } else {
		F_x_term(0) = (v/(float)psi_dot * (sin(psi + psi_dot * dt) - sin(psi)));
		F_x_term(1) = (v/(float)psi_dot * (-cos(psi + psi_dot * dt) + cos(psi)));
		F_x_term(2) = 0;
		F_x_term(3) = (psi_dot * dt);
		F_x_term(4) = 0;
	}

  /*****************************************************************************
  * Calculate the mean of the process noise, over dt
  ****************************************************************************/
	VectorXd Q_term(N_X);
	Q_term(0) = (1/2.0 * dt * dt * cos(psi) * noise_acc);
	Q_term(1) = (1/2.0 * dt * dt * sin(psi) * noise_acc);
	Q_term(2) = (noise_acc * dt);
	Q_term(3) = (1/2.0 * dt * dt * noise_yaw_rate);
	Q_term(4) = (noise_yaw_rate * dt);

  /*****************************************************************************
  * X_aug_next = X_aug(aug_point[0:5]) + F_x_term + Q_term
  ****************************************************************************/
  return aug_point.head(N_X) + F_x_term + Q_term;
}

/**
 * Used to determine the mean and covariance of the given sigma points, regardless of
 * their dimensionality.
 */
void UKF::extract_mean_and_covariance(const MatrixXd& sigma_points, const VectorXd& sigma_weights, VectorXd& mean, MatrixXd& covariance) const {
  assert(sigma_points.cols() == N_SIGMA_PTS);
  assert((covariance.rows() == sigma_points.col(0).rows()) && (covariance.cols() == sigma_points.col(0).rows()));
  assert ((sigma_weights.rows() == N_SIGMA_PTS) && (sigma_weights.cols() == 1));

  /*****************************************************************************
   *  Calculate state (mean) of sigma points
   ****************************************************************************/
  mean.fill(0.0);
  for(int i=0;i < sigma_points.cols(); i++) {
	  mean += sigma_weights(i) * sigma_points.col(i);
	}

  /*****************************************************************************
   *  Calculate covariance of sigma points
   ****************************************************************************/
	covariance.fill(0.0);
	for(int i=0;i < sigma_points.cols(); ++i) {
		VectorXd diff_from_mean = sigma_points.col(i) - mean;
    diff_from_mean(3) = Tools::normalize_angle(diff_from_mean(3));
    covariance += sigma_weights(i) * diff_from_mean * diff_from_mean.transpose() ;
	}
}

//-----------------------------------------------------

/**
 *
 */
void UKF::kalmanize(const MeasurementPackage& measurement, const MatrixXd& z_sigma_post_points, const MatrixXd& x_sigma_post_points, const VectorXd& sigma_weights, const VectorXd& z_sigma_mean, const MatrixXd& z_sigma_covariance, VectorXd& x_mean, MatrixXd& x_covariance, MatrixXd& x_z_cross_correlation, MatrixXd& kalman_gain, double& nis) const {
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
}