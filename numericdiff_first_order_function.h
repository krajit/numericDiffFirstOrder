#ifndef CERES_PUBLIC_NUMERICDIFF_FIRST_ORDER_FUNCTION_H_
#define CERES_PUBLIC_NUMERICDIFF_FIRST_ORDER_FUNCTION_H_

#include <memory>

#include "ceres/first_order_function.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/fixed_array.h"
#include "ceres/jet.h"
#include "ceres/types.h"

#include <array>

#include "Eigen/Dense"
#include "ceres/cost_function.h"
#include "ceres/internal/numeric_diff.h"
#include "ceres/internal/parameter_dims.h"
#include "ceres/numeric_diff_options.h"
#include "ceres/sized_cost_function.h"
#include "ceres/types.h"
#include "glog/logging.h"

namespace ceres
{

template <typename FirstOrderFunctor, int kNumParameters, NumericDiffMethodType method = CENTRAL>
class NumericDiffFirstOrderFunction : public FirstOrderFunction
{
public:
  // Takes ownership of functor.
  explicit NumericDiffFirstOrderFunction(FirstOrderFunctor *functor,
                                         Ownership ownership = TAKE_OWNERSHIP,
                                         const NumericDiffOptions &options = NumericDiffOptions())
      : functor_(functor),
        ownership_(ownership),
        options_(options)

  {
    static_assert(kNumParameters > 0, "kNumParameters must be positive");
  }

  virtual ~NumericDiffFirstOrderFunction()
  {
    if (ownership_ != TAKE_OWNERSHIP)
    {
      functor_.release();
    }
  }


  // Evaluate method in firstOrderFunction signature
  // This simply calls the other Evaluate method
  bool Evaluate(const double *const parameters,
                double *cost,
                double *gradient) const override
  {
    // your code here

    Evaluate(&parameters,cost,&gradient);

    return true;
  }


  // AK: Evaluate method copied from NumericDiffCostFunction
  bool Evaluate(double const *const *parameters,
                double *residuals,
                double **jacobians) const //override
  {
    using internal::FixedArray;
    using internal::NumericDiff;

    constexpr int kNumResiduals = 1;

    using ParameterDims =
      typename SizedCostFunction<kNumResiduals,kNumParameters >::ParameterDims;

    //constexpr int kNumParameters = ParameterDims::kNumParameters;
    constexpr int kNumParameterBlocks = 1; //ParameterDims::kNumParameterBlocks;

    // Get the function value (residuals) at the the point to evaluate.
    if (!internal::VariadicEvaluate<ParameterDims>(
            *functor_, parameters, residuals))
    {
      return false;
    }

    if (jacobians == NULL)
    {
      return true;
    }

    // Create a copy of the parameters which will get mutated.
    FixedArray<double> parameters_copy(kNumParameters);
    std::array<double *, kNumParameterBlocks> parameters_reference_copy =
        ParameterDims::GetUnpackedParameters(parameters_copy.data());

    for (int block = 0; block < kNumParameterBlocks; ++block)
    {
      memcpy(parameters_reference_copy[block],
             parameters[block],
             sizeof(double) * ParameterDims::GetDim(block));
    }

    internal::EvaluateJacobianForParameterBlocks<ParameterDims>::
        template Apply<method, kNumResiduals>(
            functor_.get(),
            residuals,
            options_,
            1, //   SizedCostFunction<kNumResiduals, kNumParameters>::num_residuals(),
            // AK: this was giving error. I don;t know why
            parameters_reference_copy.data(),
            jacobians);

    return true;
  }

  // evaluate method copied from numericDiffCostFunction

  int NumParameters() const override { return kNumParameters; }

private:
  std::unique_ptr<FirstOrderFunctor> functor_;
  Ownership ownership_;
  NumericDiffOptions options_;
};

} // namespace ceres

#endif // CERES_PUBLIC_NUMERICDIFF_FIRST_ORDER_FUNCTION_H_
