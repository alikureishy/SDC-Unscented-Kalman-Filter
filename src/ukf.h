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

class UKF
{
  public:
    const int N_X                = 5;               // State dimension
    const int N_AUG              = 7;               // Augmented state dimension
    const int N_AUG_DIFF         = N_AUG - N_X;     // Difference in size between the augmented and original state
    const int N_SIGMA_PTS        = 2 * N_AUG + 1;   // # of sigma points to be created
    const double LAMBDA          = 3 - (2 * N_X);   // Sigma point spreading parameter
    const int N_Z_LIDAR          = 2;               // Dimensionality of LIDAR measurement noise
    const double STD_LAS_PX      = 0.15;            // Laser measurement noise standard deviation position1 in m
    const double STD_LAS_PY      = 0.15;            // Laser measurement noise standard deviation position2 in m
    const int N_Z_RADAR          = 3;               // Dimensionality of RADAR measurement noise
    const double STD_RAD_RHO     = 0.3;             // Radar measurement noise standard deviation radius in m
    const double STD_RAD_PHI     = 0.03;            // Radar measurement noise standard deviation angle in rad
    const double STD_RAD_RHO_DOT = 0.3;             // Radar measurement noise standard deviation radius change in m/s
    const double STD_ACC         = 30;              // Process noise standard deviation longitudinal acceleration in m/s^2
    const double STD_YAW_ACC     = 30;              // Process noise standard deviation yaw acceleration in rad/s^2

  public:
    /**
       * Constructor
       */
    UKF(bool, bool);

    /**
       * ProcessMeasurement
       * @param meas_package The latest measurement data of either radar or laser
       */
    void filter_cycle(MeasurementPackage meas_package);

    /**
       * State accessor
       */
    const VectorXd &get_state() const;

  private:
    /**
       * Initialize initializes the state and covariance matrices
       */
    void initialize(MeasurementPackage measurement, double dt);

    /**
       * Predict Predicts sigma points, the state, and the state covariance
       * matrix
       * @param dt Time between k and k+1 in s
       */
    void predict_state(const VectorXd& state, const MatrixXd& covariance, const VectorXd& sigma_weights, const MatrixXd& process_noise, double dt, MatrixXd& x_sigma_post_points, VectorXd& predicted_state, MatrixXd& predict_covariance) const;
    void generate_sigma_points(const MatrixXd& state, const MatrixXd& covariance, MatrixXd& sigma_pre_points) const;
    void transform_sigma_points(const MatrixXd& sigma_pre_points, double dt, MatrixXd& sigma_post_points) const;
    VectorXd transform_sigma_point(const VectorXd& sigma_pre_point, double dt) const;
    void extract_mean(const MatrixXd &sigma_post_points, const VectorXd& sigma_weights, VectorXd &mean) const;
    void extract_state_covariance(const MatrixXd &sigma_post_points, const VectorXd& sigma_weights, const VectorXd &mean, MatrixXd &covariance) const;
    void extract_radar_covariance(const MatrixXd &sigma_post_points, const VectorXd& sigma_weights, const VectorXd &mean, MatrixXd &covariance) const;
    void extract_covariance(const MatrixXd &sigma_post_points, const VectorXd& sigma_weights, int idx_of_angle, const VectorXd &mean, MatrixXd &covariance) const;
    /**
     *  Process the update step, which includes:
     * - Predicting the measurement
     * - Comparing with measurement
     * - Updating the state, covariance and noise matrices
     */

    void unscented_update(MeasurementPackage measurement, const MatrixXd& x_sigma_post_points, VectorXd& state, MatrixXd& covariance, double& nis) const;
    void predict_measurement(const MatrixXd& projected_sigma_points, const VectorXd& weights, VectorXd& predicted_reading, MatrixXd& predicted_reading_covariance) const;
    void unscented_kalmanize(const MeasurementPackage& measurement, const MatrixXd& z_sigma_post_points, const MatrixXd& x_sigma_post_points, const VectorXd& sigma_weights, const VectorXd& z_sigma_mean, const MatrixXd& z_sigma_covariance, VectorXd& x_mean, MatrixXd& x_covariance, MatrixXd& x_z_cross_correlation, MatrixXd& kalman_gain, double& nis) const;

    void regular_update(const MeasurementPackage& measurement, const MatrixXd& measurement_function, const MatrixXd& measurement_noise, VectorXd& state, MatrixXd& covariance, double& nis) const;

    /**
       * Updates the state and the state covariance matrix using a radar measurement
       * @param meas_package The measurement at k+1
       */

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
    VectorXd sigma_weights;

    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorXd x_;

    ///* state covariance matrix
    MatrixXd P_;

    ///* Process noise covariance
    MatrixXd Q_;

    ///* Measurement function (for linear translation - i.e, Lidar)
    MatrixXd H_lidar;

    ///* Measurement noise covariance matrices for radar and lidar:
    MatrixXd R_radar_;
    MatrixXd R_lidar_;

    ///* the current NIS for radar
    double NIS_radar_ = 0.0;

    ///* the current NIS for laser
    double NIS_lidar_ = 0.0;

};

#endif /* UKF_H */
