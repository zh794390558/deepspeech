// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/cinn/cinn_instruction_run_op.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/operators/cinn/cinn_launch_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::operators {

class CinnInstructionRunOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // The cinn-graph may hasn't input for CINN now support fill_constant,
    // and its all inputs may generated by fill_constant instead of by fetch.
    // OP_INOUT_CHECK(ctx->HasInputs(kX), "Input", kX, "CinnInstructionRun");
    OP_INOUT_CHECK(ctx->HasOutputs(kOutputs), "Output", kOutputs,
                   "CinnInstructionRun");
    const CinnCompiledObject& compiled_object =
        CinnCompiler::GetInstance()->GetCompiledObject(
            ctx->Attrs().Get<int64_t>(kCachedIndex));

    details::CinnLaunchContext* launch_context =
        compiled_object.launch_context.get();
    std::vector<std::string> output_args = ctx->Outputs(kOutputs);
    std::vector<framework::DDim> output_dims(output_args.size());
    std::transform(output_args.begin(), output_args.end(), output_dims.begin(),
                   [launch_context](const std::string& var_name) {
                     cinn_buffer_t* buffer =
                         launch_context->GetCinnBufferOfVar(var_name);
                     return framework::DDim(buffer->dims, buffer->dimensions);
                   });
    ctx->SetOutputsDim(kOutputs, output_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // Why we need override GetExpectedKernelType?
    // A cinn-graph may has no inpute var, if we use the base function,
    // it will check wheter input tensors is initialized. Here we rewrite
    // the function so that we can infer kernel type by output date type.
    if (ctx.InputSize(kX)) {
      // if the instruction has input, infer kernel type by input date type:
      return OperatorWithKernel::GetExpectedKernelType(ctx);
    }

    // Else infer kernel type by output date type:
    // The `OutputVar` will check wheter the kOutputs iff has one output var
    const framework::Variable* var = ctx.OutputVar(kOutputs);
    PADDLE_ENFORCE_NE(
        var, nullptr,
        platform::errors::InvalidArgument(
            "The cinn_instruction_run Op's Output Variable should not empty."));

    const framework::Tensor* tensor = nullptr;
    if (var->IsType<framework::Tensor>()) {
      tensor = &var->Get<framework::Tensor>();
    } else if (var->IsType<framework::LoDTensor>()) {
      tensor = &var->Get<framework::LoDTensor>();
    } else if (var->IsType<phi::SelectedRows>()) {
      tensor = &(var->Get<phi::SelectedRows>().value());
    } else if (var->IsType<framework::LoDTensorArray>()) {
      auto t_arr = &var->Get<framework::LoDTensorArray>();
      PADDLE_ENFORCE_EQ(t_arr->size(), 1UL,
                        platform::errors::InvalidArgument(
                            "The cinn_instruction_run Op should just has One "
                            "Output when Input empty."));
      tensor = &(t_arr->front());
    }

    PADDLE_ENFORCE_NE(
        tensor, nullptr,
        platform::errors::InvalidArgument(
            "The cinn_instruction_run Op's Output Tensor should not empty."));

    VLOG(4) << "The tensor [" << ctx.OutputName(kOutputs) << "]'s dtype is "
            << paddle::framework::DataType2String(tensor->dtype());
    auto output_type = paddle::framework::TransToProtoVarType(tensor->dtype());
    return framework::OpKernelType(output_type, ctx.device_context());
  }
};

class CinnInstructionRunOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kX,
             "(vector<LoDTensor>)"
             "which are the input arguments of this cinn instruction")
        .AsDuplicable();
    AddOutput(kOutputs,
              "(vector<LoDTensor>)"
              "which are the output arguments of this cinn instruction")
        .AsDuplicable();
    AddAttr<int64_t>(
        kCachedIndex,
        "(int64_t)"
        "the stored index of the cached compilation result in CinnCompiler,"
        "which is used to fetch the CinnCompiledObject where this cinn "
        "instruction is included");
    AddAttr<int64_t>(
        kInstructionIndex,
        "(int64_t)"
        "the index of this instruction to the cinn runtime program");
    AddComment(R"DOC(
CinnInstructionRun Operator.

This operator is used to launch a
CINN(https://github.com/PaddlePaddle/CINN/blob/develop/README.md) instruction execution

Both the input and output of this operator are a set of variables
which are the input and output arguments of the bound cinn instruction respectively.
In addition, there is an attribute named 'cached_index' should be
set necessarily to get the CinnCompiledObject where the instruction is included 
and 'instruction_index' is fetch the instruction object from complied runtime prograrm.

It accomplishes the execution of the instruction according to the following steps:
  0. Set the shapes ot the output variables at InferShape function with
     compilation result.
  1. Fetch the cinn instruction bound to this operator by 'cached_index'
     and 'instruction_index' from CinnCompiler.
  2. Prepare the input and output variables of the instruction in Paddle and share
     their buffers to CINN by setting 'memory' of according cinn_buffer_t.
  3. Launch CINN runtime to execute the instruction.

)DOC");
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;
using CPUDeviceContext = paddle::platform::CPUDeviceContext;
REGISTER_OPERATOR(
    cinn_instruction_run, ops::CinnInstructionRunOp,
    ops::CinnInstructionRunOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    cinn_instruction_run,
    ops::CinnInstructionRunOpKernel<CPUDeviceContext, bool>,
    ops::CinnInstructionRunOpKernel<CPUDeviceContext, int>,
    ops::CinnInstructionRunOpKernel<CPUDeviceContext, int64_t>,
    ops::CinnInstructionRunOpKernel<CPUDeviceContext, float>,
    ops::CinnInstructionRunOpKernel<CPUDeviceContext, double>);