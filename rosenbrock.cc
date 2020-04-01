#include "ceres/ceres.h"
#include "glog/logging.h"
#include "numericdiff_first_order_function.h"


class RosenbrockFunctor
{
public:
  explicit RosenbrockFunctor(){}
  // template <typename T>
  // bool operator()(const T *const x, T *cost) const
  bool operator()(const double *const x, double *cost) const
  {
    *cost = (1.0 - x[0]) * (1.0 - x[0]) + 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);;
    return true;
  }
};


int main(int argc, char **argv)
{
  google::InitGoogleLogging(argv[0]);

  double parameters[2] = {10.0, 10.0};

  ceres::GradientProblemSolver::Options options;
//  options.line_search_direction_type = ceres::NONLINEAR_CONJUGATE_GRADIENT;
  options.max_num_iterations = 1000;
  options.minimizer_progress_to_stdout = true;

  ceres::GradientProblemSolver::Summary summary;
  
   ceres::FirstOrderFunction* function =
     new ceres::NumericDiffFirstOrderFunction<RosenbrockFunctor, 2, ceres::CENTRAL>(
         new RosenbrockFunctor());

  ceres::GradientProblem problem(function);
  ceres::Solve(options, problem, parameters, &summary);

  std::cout << summary.FullReport() << "\n";
  std::cout << "Final   x: " << parameters[0]
            << " y: " << parameters[1] << "\n";

   return 0;
}
