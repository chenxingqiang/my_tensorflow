#ifndef TENSORFLOW_C_KERNELS_H_
#define TENSORFLOW_C_KERNELS_H_

#include <stdint.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"

#ifdef SWIG
#define TF_CAPI_EXPORT
#else
#if defined(_WIN32)
#ifdef TF_COMPILE_LIBRARY
#define TF_CAPI_EXPORT __declspec(dllexport)
#else
#define TF_CAPI_EXPORT __declspec(dllimport)
#endif // TF_COMPILE_LIBRARY
#else
#define TF_CAPI_EXPORT __attribute__((visibility("default")))
#endif //_WIN32
#endif //SWIG

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TF_Tensor TF_Tensor;
//------------------------------------------------------
// C API for Tensorflow kernels.
//
// This API allows developers to register custom kernel implementations for
// Tensorflow.
//
// See c_api.h header comments for a discussion about API conventions.
//
//Users wishing to extend Tensorflow with new kernels will call
// `TF_NewKernelBuilder`. The resulting kernel builder can be registered with
// `TF_RegisterKernelBuilder`, which will allow TF to construct user-provided
// kernels when necessary.
typedef struct TF_KernelBuilder TF_KernelBuilder;
typedef struct TF_OpKernelConstruction TF_OpKernelConstruction;
typedef struct TF_OpKernelContext TF_OpKernelContext;

// Allocates a new kernel builder and returns a pointer to it.
//
// If non-null, Tensorflow will call create_func when it needs to instantiate
// the kernel. Teh pointer return by create_func will be passed to
// compute_func and delete_func, thereby functioning  as a "this" pointer
// for referring to kernel instances.
//
// The TF_OpKernelConstruction pointer passed to create_func is owned by
// TensorFlow and will be deleted once create_func returns. It must not be used
// after this.
//
// Finally, when TensorFlow no longer needs the kernel, it will call
// delete_func if one is provided. This function will receive the pointer
// returned in `create_func` or nullptr if no `create_func` was provided.
//
// The caller should pass the result of this function to
// TF_RegisterKernelBuilder, which will take ownership of the pointer. IF, for
// some reason, the kernel builder will not be registered, the caller should
// delete it with TF_DeleteKernelBuilder.
TF_CAPI_EXPORT extern TF_KernelBuilder* TF_NewKernelBuilder(
const char* op_name, const char* device_name,
void* (*create_func)(TF_OpKernelConstruction*),
void (*compute_func)(void*, TF_OpKernelContext*),
void (*delete_func)(void*));


)
}