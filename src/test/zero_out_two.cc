/*
* @Author: Tom Roussel
* @Date:   2017-03-29 13:39:46
* @Last Modified by:   Tom Roussel
* @Last Modified time: 2017-03-29 14:15:21
*/

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut2")
    .Input("to_zero: float")
    .Input("not_used: float")
    .Output("zeroed: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class ZeroOutOp2 : public OpKernel {
 public:
  explicit ZeroOutOp2(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = input.size();
    for (int i = 1; i < N; i++) {
      output(i) = 0;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut2").Device(DEVICE_CPU), ZeroOutOp2);
