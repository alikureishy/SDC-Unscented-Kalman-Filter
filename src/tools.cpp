#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

VectorXd Tools::calculate_rmse(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
	  cout << "Invalid estimation or ground_truth data" << endl;
	  return rmse;
	}

	//accumulate squared residuals
	for(unsigned int i=0; i < estimations.size(); ++i){

		VectorXd residual = estimations[i] - ground_truth[i];

		//coefficient-wise multiplication
		residual = residual.array()*residual.array();
		rmse += residual;
	}

	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

    //return the result
	return rmse;
}

double Tools::determine_nis (VectorXd error, MatrixXd covariance) {
	return (error.transpose() * covariance.inverse() * error)(0, 0);
}

void Tools::from_polar_to_ctrv(const VectorXd &from_polar, VectorXd &to_ctrv) {
	assert(to_ctrv.rows() == 5);
	assert(from_polar.rows() == 3);

	float px, py, v, yaw, yaw_dot;

	float rho = from_polar(0);
	float phi = from_polar(1);
	// float rho_dot = from_polar(2); // rho_dot is the projection of the object's velocity v on the rho direction.

	// Translation:
	px = rho * cos(phi);
	py = rho * sin(phi);
	v = 0; // rho_dot is dropped because this is CTRV
	yaw = 0;	// CTRV
	yaw_dot = 0;	// CTRV

	to_ctrv << px, py, v, yaw, yaw_dot;

	if(to_ctrv(0) == 0 && to_ctrv(1) == 0) {
		to_ctrv(0) = 0.01;
		to_ctrv(1) = 0.01;
	}
}

void Tools::from_cartesian_to_ctrv(const VectorXd &from_cartesian, VectorXd &to_ctrv) {
	assert(to_ctrv.rows() == 5);
	assert(from_cartesian.rows() == 2);

	double px, py, v, yaw, yaw_dot;
	px = from_cartesian(0);
	py = from_cartesian(1);
	v = 0; // to_ctrv(2);				// We retain the existing value
	yaw = 0; // to_ctrv(3);			// We retain the existing value
	yaw_dot = 0; // to_ctrv(4);		// We retain the existing value

	to_ctrv << px, py, v, yaw, yaw_dot;

	if(to_ctrv(0) == 0 && to_ctrv(1) == 0) {
		to_ctrv(0) = 0.01;
		to_ctrv(1) = 0.01;
	}
}

void Tools::from_ctrvs_to_polars(const MatrixXd &from_ctrvs, MatrixXd& to_polars) {
	assert(from_ctrvs.rows() == 5);
	assert(to_polars.rows() == 3);
	for(int i=0; i<from_ctrvs.cols(); ++i) {
		double px = from_ctrvs.col(i)(0);
		double py = from_ctrvs.col(i)(1);
		double v = from_ctrvs.col(i)(2);
		double psi = from_ctrvs.col(i)(3);
		//double psi_dot = from_ctrv(4); // Not needed

		if(px == 0 && py == 0) {
			to_polars.col(i) << 0, 0, 0;
		} else {
			// Rho
			to_polars.col(i)(0) = sqrt(pow(px, 2) + pow(py, 2));

			// Phi
			to_polars.col(i)(1) = atan(py/float(px));

			// Rho - dot
			to_polars.col(i)(2) = ((px * v * cos(psi) + py * v * sin(psi))/float(to_polars.col(i)(0)));
		}
	}
}

void Tools::from_ctrvs_to_cartesians(const MatrixXd &from_ctrvs, MatrixXd& to_cartesians) {
	assert(from_ctrvs.rows() == 5);
	assert(to_cartesians.rows() == 2);

	for(int i=0; i<from_ctrvs.cols(); ++i) {

		double px = from_ctrvs.col(i)(0);
		double py = from_ctrvs.col(i)(1);

		// x position
		to_cartesians.col(i)(0) = px;

		// y position
		to_cartesians.col(i)(1) = py;
	}
}

double Tools::normalize_angle(double angle) {
	double normalized = angle;
	while (normalized > M_PI)
		normalized -= 2. * M_PI;      	// Normalize angle
	while (normalized < -M_PI)
		normalized += 2. * M_PI; 		// Normalize angle
	return normalized;
}

void Tools::calculate_mean(const MatrixXd& points, const VectorXd& weights, VectorXd& weighted_mean) {
	assert(points.col(0).rows() == weighted_mean.rows());
	assert(points.cols() == weights.rows());
	weighted_mean.fill(0.0);
	for (int i=0; i < points.cols(); i++) {
    	weighted_mean = weighted_mean + weights(i) * points.col(i);
  	}
}