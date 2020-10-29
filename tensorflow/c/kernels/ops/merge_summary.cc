#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/selective_registration.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

static void merge_summary_shape_inference_fn(TF_ShapeInferenceContext* ctx, TF_Status* status){
    TF_SetStatus(status, TF_OK, "");
    TF_ShapeHandle* result = TF_ShapeInferenceContextScalar(ctx);
    TF_ShapeInferenceContextSetOutput(ctx, 0, result, status);
    TF_DeleteShapeHandle(result);
}

void Register_MergeSummaryOp() {
    TF_Status* status = TF_NewStatus();
    TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("MergeSummary");
    TF_OpDefinitionBuilderAddInput(op_builder, "inputs: N * String");
    TF_OpDefinitionBuilderAddOutput(op_builder, "summary: string");
    TF_OpDefinitionBuilderAddAttr(op_builder, "N: int>=1");
    TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder, &merge_summary_shape_inference_fn);

    TF_Register_TF_OpDefinition(op_builder, status);
    CHECK_EQ(TF_GetCode(status),TF_OK)
        << "MergeSummary op is failed: " << TF_Message(status);
    TF_DeleteStatus(status);
}

TF_ATTRIBUTE_UNUSED static bool MergeSummaryOpRegistered = []() {
    if(SHOULD_REGISTER_OP("MergeSummary")){
        Register_MergeSummaryOp();
    }
 return true;
}();