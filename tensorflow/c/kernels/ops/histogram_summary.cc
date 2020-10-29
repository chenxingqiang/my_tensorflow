#include "tensorflow/c/ops.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/framework/selective_registration.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

static void histogram_summary_shape_inference_fn(TF_ShapeInferenceContext* ctx,
                                                    TF_Status* status) {

        TF_SetStatus(status, TF_OK, "");
        TF_ShapeHandle* result = TF_ShapeInferenceContextScalar(ctx);
        TF_ShapeInferenceContextSetOutput(ctx, 0, result, status);
        TF_DeleteShapeHandle(result);
        }

void Register_HistogramSummaryOp() {
        TF_Status* status = TF_NewStatus();

        TF_OpDefinitionBuilder* op_builder = TF_NewOpDefinitionBuilder("HistogramSummary");
        TF_OpDefinitionBuilderAddInput(op_builder, "tag: string");
        TF_OpDefinitionBuilderAddInput(op_builder, "values: T");
        TF_OpDefinitionBuilderAddOutput(op_builder, "summary: string");
        TF_OpDefinitionBuilderAddAttr(op_builder, "T: realnumbertype = DT_FLOAT");
        TF_OpDefinitionBuilderSetShapeInferenceFunction(op_builder, &histogram_summary_shape_inference_fn);

        TF_RegisterOpDefinition(op_builder, status);
        CHECK_EQ(TF_GetCode(status), TF_OK)
            << "HistogramSummary op registration failed: " << TF_Message(status);
        TF_DeleteStatus(status);
}

TF_ATTRIBUTE_UNUSED static bool HistogramSummaryOpRegistered =[]() {
   if (SHOULD_REGISTER_OP("HistogramSummary")){
     Register_HistogramSummaryOp();
   }
   return true;
}();