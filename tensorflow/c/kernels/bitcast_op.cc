#include <sstream>
#include "tensorflow/c/kernels.h"
#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/selective_registration.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/macros.h"

// BitcastOp implements a bitcast kernel, creating an output tensor that share
//