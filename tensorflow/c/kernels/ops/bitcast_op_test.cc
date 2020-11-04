#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value.util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/test.h"

namespace tensorflow{
namespace {
class DummyDevice: public DeviceBase {
public:
    explicit DummyDevice(ENV* env) DeviceBase(env){}
    Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
    return cpu_allocator();
    }
};


}

}