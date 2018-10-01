#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
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

  void Tools::from_polar_to_ctrv(const VectorXd &from_polar, VectorXd &to_ctrv) {
    float px, py, v, yaw, yaw_dot;

	float rho = from_polar(0);
	float phi = from_polar(1);
	float rho_dot = from_polar(2); // rho_dot is the projection of the object's velocity v on the rho direction.

	// Translation:
	px = rho * cos(phi);
	py = rho * sin(phi);
	v = rho_dot;
	yaw = 0;
	yaw_dot = 0;

	to_ctrv << px, py, v, yaw, yaw_dot;
  }

  void Tools::from_cartesian_to_ctrv(const VectorXd &from_cartesian, VectorXd &to_ctrv) {
    float px, py, v, yaw, yaw_dot;
	px = from_cartesian(0);
	py = from_cartesian(1);
	v = to_ctrv(2);				// We retain the existing value
	yaw = to_ctrv(3);			// We retain the existing value
	yaw_dot = to_ctrv(4);		// We retain the existing value

	to_ctrv << px, py, v, yaw, yaw_dot;
  }

  void Tools::from_ctrv_to_polar(const VectorXd &from_ctrv, VectorXd& to_polar) {
	double px = from_ctrv(0);
	double py = from_ctrv(1);
	double v = from_ctrv(2);
	double psi = from_ctrv(3);
	double psi_dot = from_ctrv(4);

	if(px == 0 && py == 0) {
		to_polar << 0, 0, 0;
	} else {
		// Rho
		to_polar(0) = sqrt(pow(px, 2) + pow(py, 2));

		// Phi
		to_polar(1) = atan(py/float(px));

		// Rho - dot
		to_polar(2) = ((px * v * cos(psi) + py * v * sin(psi))/float(to_polar(0)));
	}
  }

  void Tools::from_ctrv_to_cartesian(const VectorXd &from_ctrv, VectorXd& to_cartesian) {
    float px, py, v, yaw, yaw_dot;

  }

  double Tools::normalize_angle(double angle) {
		double normalized = angle;
		while (normalized > M_PI)
			normalized -= 2. * M_PI;      	// Normalize angle
    while (normalized<-M_PI)
      normalized += 2. * M_PI; 		// Normalize angle
		return normalized;
	}