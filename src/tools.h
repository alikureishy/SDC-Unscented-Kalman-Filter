#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * A helper method to calculate RMSE.
  */
  static VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
   * Conversion functions from/to the coordinate systems to/from the CTRV state
   */
  static void from_polar_to_ctrv(const VectorXd &from_polar, VectorXd &to_ctrv);
  static void from_cartesian_to_ctrv(const VectorXd &from_cartesian, VectorXd &to_ctrv);
  static void from_ctrv_to_polar(const VectorXd &from_ctrv, VectorXd& to_polar);
  static void from_ctrv_to_cartesian(const VectorXd &from_ctrv, VectorXd& to_cartesian);

  static double normalize_angle(double angle);

private:
  /**
  * Disable the Constructor.
  */
  Tools();
};

#endif /* TOOLS_H_ */