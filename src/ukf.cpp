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

  /*****************************************************************************
   *  Initialize covariance matrix
   ****************************************************************************/
            // px, py, v, phi, phi_dot
  this->P_  <<  1,  0,  0,  0,   0,   // px
                0,  1,  0,  0,   0,   // py
                0,  0,  1,  0,   0,   // v
                0,  0,  0,  1,   0,   // phi
                0,  0,  0,  0,   1;   // phi_dot

  /*****************************************************************************
  *  Initialize the process noise covariance
  ****************************************************************************/
  this->Q_ << pow(STD_ACC, 2),       0.0,
                 0.0,         pow(STD_YAW_ACC, 2);

  /*****************************************************************************
   *  Initialize the weights
   ****************************************************************************/
  this->weights_.fill(0.5/(LAMBDA + N_AUG));
  this->weights_[0] = LAMBDA / (LAMBDA + N_AUG);
}

const VectorXd& UKF::get_state() const {
  return this->x_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement) {
  float dt = (measurement.timestamp_ - previous_timestamp) / 1000000.0;	//dt - expressed in seconds

  if (!is_initialized_) {
    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    Initialize(measurement, dt);
  }
  else  {
    /*****************************************************************************
     *  Predict + Update cycle
     ****************************************************************************/
    PredictState(dt);
    SenseAndUpdate(measurement, dt);
  }

  // print the latest state and co-variance matrices
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
  previous_timestamp = measurement.timestamp_;
}

/**
 * Initialize the state using the first measurement
 */
void UKF::Initialize(MeasurementPackage measurement, double dt) {
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
void UKF::PredictState(double dt) {
  /*****************************************************************************
   *  Generate Sigma points
   ****************************************************************************/
	MatrixXd X_aug_sig = SelectSigmaPoints(this->x_, this->P_);

  /*****************************************************************************
   *  Extrapolate sigma points for process model at time k+1
   ****************************************************************************/
	MatrixXd X_aug_sig_pred = ProjectSigmaPoints(X_aug_sig, dt);

  /*****************************************************************************
   *  Calculate predicted state and covariance
   ****************************************************************************/
  ReverseSigmaPoints(X_aug_sig_pred, this->x_, this->P_);
}

MatrixXd UKF::SelectSigmaPoints(const MatrixXd& state, const MatrixXd& covariance) const {
	MatrixXd X_aug_sig = MatrixXd(N_AUG, N_SIGMA_PTS);  // Augmented sigma points

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
	    throw std::range_error("LLT failed");
	}

  /*****************************************************************************
  *  Calculate sigma points and set them into X_aug_sig
  ****************************************************************************/
	X_aug_sig.col(0) = x_aug;
	MatrixXd term = sqrt(LAMBDA + N_AUG) * A;
	for (int i = 0; i < N_AUG; ++i) {
		X_aug_sig.col(i + 1) = x_aug + term.col(i);
		X_aug_sig.col(i + N_AUG + 1) = x_aug - term.col(i);
	}

	return X_aug_sig;
}

MatrixXd UKF::ProjectSigmaPoints(const MatrixXd& aug_sigma_points, double dt) const {
  assert (aug_sigma_points.cols() == N_SIGMA_PTS);
  MatrixXd projected_sigma_points(N_X, N_SIGMA_PTS);
  for (int i = 0; i < N_SIGMA_PTS; i++) {
    VectorXd new_point = ProjectSinglePoint(aug_sigma_points.col(i), dt);
    assert (new_point.rows() == N_X); // Should be [N_X]
    projected_sigma_points.col(i) << new_point;
  }
  return projected_sigma_points;
}

VectorXd UKF::ProjectSinglePoint(const VectorXd& aug_point, double dt) const {
  /*****************************************************************************
  * For easy reference
  ****************************************************************************/
  assert(aug_point.rows() == N_AUG);
  double px = aug_point(0);
	double py = aug_point(1);
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

void UKF::ReverseSigmaPoints(const MatrixXd& sigma_points, VectorXd& state, MatrixXd& covariance) {
  assert(sigma_points.rows() == N_X);
  assert(sigma_points.cols() == N_SIGMA_PTS);

  /*****************************************************************************
   *  Calculate state (mean) of sigma points
   ****************************************************************************/
  state.fill(0.0);
  for(int i=0;i < sigma_points.cols(); i++) {
	  state += (this->weights_(i) * sigma_points.col(i));
	}

  /*****************************************************************************
   *  Calculate covariance of sigma points
   ****************************************************************************/
	covariance.fill(0.0);
	for(int i=0;i < sigma_points.cols(); ++i) {
		VectorXd diff_from_mean = sigma_points.col(i) - state;
    diff_from_mean(3) = Tools::normalize_angle(diff_from_mean(3));
    covariance += weights_(i) * diff_from_mean * diff_from_mean.transpose() ;
	}
}

void UKF::SenseAndUpdate(MeasurementPackage measurement, double dt) {
  if (use_radar_ && (measurement.sensor_type_ == MeasurementPackage::RADAR)) {
    /*****************************************************************************
     *  Update RADAR measurement
     ****************************************************************************/
    ekf_.R_ = R_radar_;
    ekf_.H_ = Hj;
    ProcessRadarMeasurement(measurement, dt);
  } else if (use_laser_ && (measurement.sensor_type_ == MeasurementPackage::LASER)) {
    /*****************************************************************************
     *  Update LIDAR measurement
     ****************************************************************************/
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ProcessRadarMeasurement(measurement, dt);
  }

  // Update Q with elapsed time
  int noise_ax = 9;
  int noise_ay = 9;
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;
  Q_ << (dt_4*noise_ax/4), 0, (dt_3*noise_ax/2), 0,
              0, (dt_4*noise_ay/4), 0, (dt_3*noise_ay/2),
            (dt_3*noise_ax/2), 0, (dt_2*noise_ax), 0,
              0, (dt_3*noise_ay/2), 0, (dt_2*noise_ay);

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::ProcessLidarMeasurement(MeasurementPackage meas_package, double dt) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::ProcessRadarMeasurement(MeasurementPackage meas_package, double dt) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
