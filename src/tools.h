#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  Tools();
  /**
  * A helper method to calculate RMSE.
  */
  static VectorXd calculate_rmse(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
   * Conversion functions from/to the coordinate systems to/from the CTRV state
   */
  static void from_polar_to_ctrv(const VectorXd &from_polar, VectorXd &to_ctrv);
  static void from_cartesian_to_ctrv(const VectorXd &from_cartesian, VectorXd &to_ctrv);
  static void from_ctrvs_to_polars(const MatrixXd &from_ctrvs, MatrixXd& to_polars);
  static void from_ctrvs_to_cartesians(const MatrixXd &from_ctrvs, MatrixXd& to_cartesians);
  static double determine_nis (const VectorXd error, const MatrixXd covariance);
  static double normalize_angle(double angle);
  static void calculate_mean(const MatrixXd &points, const VectorXd &weights, VectorXd &weighted_mean);

private:
  /**
  * Disable the Constructor.
  */
};

#endif /* TOOLS_H_ */