#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Constants
 */
const int N_X                = 5;               // State dimension
const int N_AUG              = 7;               // Augmented state dimension
const int N_AUG_DIFF         = N_AUG - N_X;     // Difference in size between the augmented and original state
const int N_SIGMA_PTS        = 2 * N_AUG + 1;   // # of sigma points to be created
const double LAMBDA          = 3 - (2 * N_X);   // Sigma point spreading parameter
const double STD_LAS_PX      = 0.15;            // Laser measurement noise standard deviation position1 in m
const double STD_LAS_PY      = 0.15;            // Laser measurement noise standard deviation position2 in m
const double STD_RAD_RHO     = 0.3;             // Radar measurement noise standard deviation radius in m
const double STD_RAD_PHI     = 0.03;            // Radar measurement noise standard deviation angle in rad
const double STD_RAD_RHO_DOT = 0.3;             // Radar measurement noise standard deviation radius change in m/s
const double STD_ACC         = 30;              // Process noise standard deviation longitudinal acceleration in m/s^2
const double STD_YAW_ACC     = 30;              // Process noise standard deviation yaw acceleration in rad/s^2

class UKF
{
public:
  /**
     * Constructor
     */
  UKF(bool, bool);

  /**
     * ProcessMeasurement
     * @param meas_package The latest measurement data of either radar or laser
     */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
     * State accessor
     */
  const VectorXd &get_state() const;

  /**
     * Destructor
     */
  virtual ~UKF();

private:
  /**
     * Initialize initializes the state and covariance matrices
     */
  void Initialize(MeasurementPackage measurement, double dt);

  /**
     * Predict Predicts sigma points, the state, and the state covariance
     * matrix
     * @param dt Time between k and k+1 in s
     */
  void PredictState(double dt);
  MatrixXd SelectSigmaPoints(const MatrixXd& state, const MatrixXd& covariance) const;
  MatrixXd ProjectSigmaPoints(const MatrixXd& sigma_points, double dt) const;
  VectorXd ProjectSinglePoint(const VectorXd &aug_point, double dt) const;
  void ReverseSigmaPoints(const MatrixXd &sigma_points, VectorXd &state, MatrixXd &covariance);

  /**
   *  Process the update step, which includes:
   * - Predicting the measurement
   * - Comparing with measurement
   * - Updating the state, covariance and noise matrices
   */
  void SenseAndUpdate(MeasurementPackage measurement, double dt);

  /**
     * Updates the state and the state covariance matrix using a laser measurement
     * @param meas_package The measurement at k+1
     */
  void ProcessLidarMeasurement(MeasurementPackage meas_package, double dt);
  void PredictLidarMeasurement(double dt);
  void UpdateFromLidarMeasurement(MeasurementPackage meas_package, double dt);

  /**
     * Updates the state and the state covariance matrix using a radar measurement
     * @param meas_package The measurement at k+1
     */
  void ProcessRadarMeasurement(MeasurementPackage measurement, double dt);
  void PredictRadarMeasurement(double dt);
  void UpdateFromRadarMeasurement(MeasurementPackage meas_package, double dt);

private:
  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* Previous seen timestamp
  long long previous_timestamp;

  ///* Weights of sigma points
  VectorXd weights_(2 * N_AUG + 1);

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_(N_X);

  ///* state covariance matrix
  MatrixXd P_(N_X, N_X);

  ///* Process noise covariance
  MatrixXd Q_(N_AUG_DIFF, N_AUG_DIFF);

  ///* the current NIS for radar
  double NIS_radar_ = 0.0;

  ///* the current NIS for laser
  double NIS_laser_ = 0.0;

};

#endif /* UKF_H */
