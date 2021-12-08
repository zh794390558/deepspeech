// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/pybind/op_function_generator.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {

static std::unordered_map<std::string, paddle::framework::AttributeMap>
    operators_with_attrs = {};

static std::unordered_set<std::string> operators_to_skip = {
    "minus",
};

static std::unordered_set<std::string> operators_to_codegen = {};
static std::unordered_set<std::string> skipped_operators = {};

static std::string LegalizeVariableName(const std::string& var_name) {
  std::string ret = var_name;
  std::replace(ret.begin(), ret.end(), '-', '_');  // replace all '-' to '_'
  return ret;
}

static std::string AttrTypeToString(const proto::AttrType& type) {
  std::string ret;
  switch (type) {
    case (proto::AttrType::INT): {
      ret = "int";
      break;
    }
    case (proto::AttrType::FLOAT): {
      ret = "float";
      break;
    }
    case (proto::AttrType::STRING): {
      ret = "std::string&";
      break;
    }
    case (proto::AttrType::INTS): {
      ret = "std::vector<int>&";
      break;
    }
    case (proto::AttrType::FLOATS): {
      ret = "std::vector<float>&";
      break;
    }
    case (proto::AttrType::STRINGS): {
      ret = "std::vector<std::string>&";
      break;
    }
    case (proto::AttrType::BOOLEAN): {
      ret = "bool";
      break;
    }
    case (proto::AttrType::BOOLEANS): {
      ret = "std::vector<bool>&";
      break;
    }
    case (proto::AttrType::LONG): {
      ret = "int64_t";
      break;
    }
    case (proto::AttrType::LONGS): {
      ret = "std::vector<int64_t>&";
      break;
    }
    case (proto::AttrType::BLOCK): {
      ret = "paddle::framework::BlockDesc*";
      break;
    }
    case (proto::AttrType::BLOCKS): {
      ret = "std::vector<paddle::framework::BlockDesc*>&";
      break;
    }
    case (proto::AttrType::FLOAT64S): {
      ret = "std::vector<double>&";
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::Fatal(
          "AttrType of type boost::variant only supports specific data types."
          "However, detected unrecognized AttrType: %d",
          type));
    }
  }
  return ret;
}

template <typename T>
static std::string GetAttrValue(const framework::Attribute& attr,
                                bool is_vector) {
  std::string val = "";
  if (is_vector) {
    val += "{";
    for (auto x : BOOST_GET_CONST(std::vector<T>, attr)) {
      val += std::to_string(x) + ",";
    }
    if (val.size() > 1) val.pop_back();
    val += "}";
  } else {
    val = std::to_string(BOOST_GET_CONST(T, attr));
  }
  return val;
}

static std::pair<std::string, std::string> GetAttrType(
    const framework::Attribute& attr, bool is_arg) {
  std::string ret = "";
  std::string val = "";
  size_t variant_pos = attr.which();
  switch (variant_pos) {
    case (1): {
      ret = "int";
      val = GetAttrValue<int>(attr, false);
      break;
    }
    case (2): {
      ret = "float";
      val = GetAttrValue<float>(attr, false);
      break;
    }
    case (3): {
      ret = "std::string";
      if (is_arg) ret += "&";
      val = "\"" + BOOST_GET_CONST(std::string, attr) + "\"";
      break;
    }
    case (4): {
      ret = "std::vector<int>";
      if (is_arg) ret += "&";
      val = GetAttrValue<int>(attr, true);
      break;
    }
    case (5): {
      ret = "std::vector<float>";
      if (is_arg) ret += "&";
      val = GetAttrValue<float>(attr, true);
      break;
    }
    case (6): {
      ret = "std::vector<std::string>";
      if (is_arg) ret += "&";
      val += "{";
      for (auto x : BOOST_GET_CONST(std::vector<std::string>, attr)) {
        val += "\"" + x + "\"" + ",";
      }
      if (val.size() > 1) val.pop_back();
      val += "};";
      break;
    }
    case (7): {
      ret = "bool";
      val = GetAttrValue<bool>(attr, false);
      break;
    }
    case (8): {
      ret = "std::vector<bool>";
      if (is_arg) ret += "&";
      val = GetAttrValue<bool>(attr, true);
      break;
    }
    case (9): {
      ret = "BlockDesc*";
      break;
    }
    case (10): {
      ret = "int64_t";
      val = GetAttrValue<int64_t>(attr, false);
      break;
    }
    case (11): {
      ret = "std::vector<BlockDesc*>";
      if (is_arg) ret += "&";
      break;
    }
    case (12): {
      ret = "std::vector<int64_t>";
      if (is_arg) ret += "&";
      val = GetAttrValue<int64_t>(attr, true);
      break;
    }
    case (13): {
      ret = "std::vector<double>";
      if (is_arg) ret += "&";
      val = GetAttrValue<double>(attr, true);
      break;
    }
    default: {
      PADDLE_THROW(platform::errors::Fatal(
          "AttrType of type boost::variant only supports specific data types."
          "However, detected unrecognized AttrType: %d",
          variant_pos));
    }
  }
  return {ret, val};
}

static void SlotNameMatching(
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        grad_map,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        fwd_ins,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        fwd_outs,
    std::map<std::string, std::string>* grad_fwd_slotname_map_ptr,
    std::map<std::string, std::string>* grad_grad_slotname_map_ptr) {
  std::map<std::string, std::string>& grad_fwd_slotname_map =
      *grad_fwd_slotname_map_ptr;
  std::map<std::string, std::string>& grad_grad_slotname_map =
      *grad_grad_slotname_map_ptr;
  for (const auto& iter : grad_map) {
    const std::string& grad_slot_name = iter.first;
    const std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>&
        grad_vars = iter.second;

    // Find matching fwd_slot_name
    bool found_matching = false;
    for (const std::shared_ptr<paddle::imperative::VariableWrapper>& grad_var :
         grad_vars) {
      for (const auto& fwd_iter : fwd_ins) {
        const std::string& fwd_slot_name = fwd_iter.first;
        const std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>&
            fwd_vars = fwd_iter.second;
        for (const std::shared_ptr<paddle::imperative::VariableWrapper>&
                 fwd_var : fwd_vars) {
          if (grad_var == fwd_var) {
            if (grad_fwd_slotname_map.count(grad_slot_name) &&
                grad_fwd_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(platform::errors::Fatal(
                  "Detected mismatched slot names."
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name, grad_fwd_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_fwd_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }

          if (fwd_var->GetGradVar() && grad_var == fwd_var->GetGradVar()) {
            if (grad_grad_slotname_map.count(grad_slot_name) &&
                grad_grad_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(platform::errors::Fatal(
                  "Detected mismatched slot names."
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name, grad_grad_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_grad_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }
        }
      }
      for (const auto& fwd_iter : fwd_outs) {
        const std::string& fwd_slot_name = fwd_iter.first;
        const std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>&
            fwd_vars = fwd_iter.second;
        for (const std::shared_ptr<paddle::imperative::VariableWrapper>&
                 fwd_var : fwd_vars) {
          if (grad_var == fwd_var) {
            if (grad_fwd_slotname_map.count(grad_slot_name) &&
                grad_fwd_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(platform::errors::Fatal(
                  "Detected mismatched slot names"
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name, grad_fwd_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_fwd_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }

          if (fwd_var->GetGradVar() && grad_var == fwd_var->GetGradVar()) {
            if (grad_grad_slotname_map.count(grad_slot_name) &&
                grad_grad_slotname_map[grad_slot_name] != fwd_slot_name) {
              PADDLE_THROW(platform::errors::Fatal(
                  "Detected mismatched slot names."
                  "grad_slot_name %s matches both %s and %s fwd_slot_name",
                  grad_slot_name, grad_grad_slotname_map[grad_slot_name],
                  fwd_slot_name));
            }
            grad_grad_slotname_map[grad_slot_name] = fwd_slot_name;
            found_matching = true;
          }
        }
      }
    }

    if (!found_matching) {
      PADDLE_THROW(platform::errors::Fatal(
          "Detected mismatched slot names."
          "Found no matching fwd_slot_name for grad_slot_name: %s",
          grad_slot_name));

    } else {
      std::string fwd_slot_name = grad_grad_slotname_map.count(grad_slot_name)
                                      ? grad_grad_slotname_map[grad_slot_name]
                                      : grad_fwd_slotname_map[grad_slot_name];
      VLOG(6) << "Found matching fwd_slot_name: " << fwd_slot_name
              << " for grad_slot_name: " << grad_slot_name;
    }
  }
}

static bool CheckOpProto(proto::OpProto* op_proto) {
  if (op_proto == nullptr) {
    return false;
  }
  const std::string& op_type = op_proto->type();

  // Skip ooerator which is not inherit form OperatorWithKernel, like while,
  // since only OperatorWithKernel can run in dygraph mode.
  auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();
  if (!all_kernels.count(op_type)) {
    return false;
  }

  // Only handle matmul_v2 for now
  VLOG(1) << "------ Analyzing Op ------: " << op_type;

  if (!operators_to_codegen.count(op_type)) return false;
  if (operators_to_skip.count(op_type)) return false;

  return true;
}

/* --------------------------------------- */
/* --------- Preprocess Ins/Outs --------- */
/* --------------------------------------- */
static void PurifyForwardOpProto(
    const proto::OpProto& op_proto,
    std::unordered_map<std::string, size_t>* fwd_inputs_name_pos_map,
    std::unordered_map<std::string, size_t>* fwd_outputs_name_pos_map,
    std::vector<proto::OpProto::Var>* in_vars,
    std::vector<proto::OpProto::Var>* out_vars) {
  // Op Name
  const std::string op_name = op_proto.type();

  // Handle dispensable inputs
  for (const proto::OpProto::Var& input : op_proto.inputs()) {
    std::string input_name = input.name();

    // Delete dispensable tensor unless specified in op_ins_map
    if (input.dispensable()) {
      if (!op_ins_map.count(op_name) ||
          !op_ins_map[op_name].count(input_name)) {
        VLOG(6) << "Removing Dispensable Input: " << input_name;

        // in_vars
        auto iter = in_vars->begin();
        for (iter = in_vars->begin(); iter != in_vars->end(); iter++) {
          if (iter->name() == input_name) {
            break;
          }
        }
        in_vars->erase(iter);
      }
    }
  }

  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    std::string output_name = output.name();

    // Delete dispensable tensor unless specified in op_outs_map
    if (output.dispensable()) {
      if (!op_outs_map.count(op_name) ||
          !op_outs_map[op_name].count(output_name)) {
        VLOG(6) << "Removing Dispensable Output: " << output_name;

        // out_vars
        auto iter = out_vars->begin();
        for (iter = out_vars->begin(); iter != out_vars->end(); iter++) {
          if (iter->name() == output_name) {
            break;
          }
        }
        out_vars->erase(iter);
      }
    }
  }

  /* ------ Maping forward slot name to fwd position ------ */
  size_t in_pos = 0;
  for (const auto& var : *in_vars) {
    VLOG(6) << "Mapping input tensor: " << var.name()
            << " To position: " << in_pos;
    (*fwd_inputs_name_pos_map)[var.name()] = in_pos;
    in_pos++;
  }

  size_t out_pos = 0;
  for (const auto& var : *out_vars) {
    VLOG(6) << "Mapping output tensor: " << var.name()
            << " To position: " << out_pos;
    (*fwd_outputs_name_pos_map)[var.name()] = out_pos;
    out_pos++;
  }
}

static void PurifyGradOpProto(
    const proto::OpProto& op_proto,
    std::map<std::string, std::string>* grad_outs_slotname_map,
    std::map<std::string, std::string>* grad_ins_fwd_slotname_map,
    std::map<std::string, std::string>* grad_ins_grad_slotname_map,
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
        grad_ins,
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
        grad_outs) {
  // Op Name
  const std::string op_name = op_proto.type();

  // Handle dispensable inputs
  for (const proto::OpProto::Var& input : op_proto.inputs()) {
    std::string input_name = input.name();

    // Delete dispensable tensor unless specified in op_ins_map
    if (input.dispensable()) {
      if (!op_ins_map.count(op_name) ||
          !op_ins_map[op_name].count(input_name)) {
        VLOG(6) << "Removing Dispensable Input: " << input_name;

        // grad_outs_slotname_map
        auto grad_outs_slotname_map_purified = *grad_outs_slotname_map;
        for (const auto& iter : *grad_outs_slotname_map) {
          const std::string& grad_output_name = iter.first;
          const std::string& matched_input_name = iter.second;
          if (matched_input_name == input_name) {
            grad_outs_slotname_map_purified.erase(grad_output_name);

            PADDLE_ENFORCE(
                grad_outs->count(grad_output_name) > 0,
                paddle::platform::errors::Fatal(
                    "Unable to find gradient output name in grad_outs."));
            // grad_outs
            grad_outs->erase(grad_output_name);
          }
        }
        *grad_outs_slotname_map = grad_outs_slotname_map_purified;

        // grad_ins_fwd_slotname_map: output as tensorwrapper
        if (grad_ins_fwd_slotname_map->count(input_name))
          grad_ins_fwd_slotname_map->erase(input_name);

        // grad_ins: output as tensorwrapper
        if (grad_ins->count(input_name)) grad_ins->erase(input_name);
      }
    }
  }

  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    std::string output_name = output.name();

    // Delete dispensable tensor unless specified in op_outs_map
    if (output.dispensable()) {
      if (!op_outs_map.count(op_name) ||
          !op_outs_map[op_name].count(output_name)) {
        VLOG(6) << "Removing Dispensable Output: " << output_name;

        // grad_ins_grad_slotname_map
        auto grad_ins_grad_slotname_map_purified = *grad_ins_grad_slotname_map;
        for (const auto& iter : *grad_ins_grad_slotname_map) {
          const std::string& grad_input_name = iter.first;
          const std::string& matched_output_name = iter.second;
          if (matched_output_name == output_name) {
            grad_ins_grad_slotname_map_purified.erase(grad_input_name);

            PADDLE_ENFORCE(
                grad_ins->count(grad_input_name) > 0,
                paddle::platform::errors::Fatal(
                    "Unable to find gradient input name in grad_ins."));
            // grad_ins
            grad_ins->erase(grad_input_name);
          }
        }
        *grad_ins_grad_slotname_map = grad_ins_grad_slotname_map_purified;

        // grad_ins_fwd_slotname_map: output as tensorwrapper
        if (grad_ins_fwd_slotname_map->count(output_name))
          grad_ins_fwd_slotname_map->erase(output_name);

        // grad_ins: output as tensorwrapper
        if (grad_ins->count(output_name)) grad_ins->erase(output_name);
      }
    }
  }
}

/* -------------------------------- */
/* --------- Collect Info --------- */
/* -------------------------------- */
static void CollectForwardInformationFromOpInfo(
    const paddle::framework::OpInfo& op_info,
    std::vector<proto::OpProto::Var>* in_vars,
    std::vector<proto::OpProto::Var>* out_vars) {
  const proto::OpProto& op_proto = *op_info.proto_;
  for (const proto::OpProto::Var& input : op_proto.inputs()) {
    in_vars->push_back(input);
  }
  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    out_vars->push_back(output);
  }
}

static bool CollectGradInformationFromOpInfo(
    const paddle::framework::OpInfo& op_info, bool* generate_forward_only,
    std::vector<std::string>* grad_op_types,                         // grad
    std::map<std::string, std::string>* grad_outs_slotname_map,      // grad
    std::map<std::string, std::string>* grad_ins_fwd_slotname_map,   // grad
    std::map<std::string, std::string>* grad_ins_grad_slotname_map,  // grad
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
        grad_ins,  // grad
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>*
        grad_outs  // grad
    ) {
  const proto::OpProto& op_proto = *op_info.proto_;
  const std::string& op_type = op_proto.type();
  std::vector<int64_t> dims = {1, 1, 1, 1};

  /* ------ Prepare "ins" ------ */
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VarBase>>>
      ins;
  for (const proto::OpProto::Var& input : op_proto.inputs()) {
    const std::string& in_name = input.name();

    // Handle dispensable input:
    // 1. At python level, dispensable input will be detected at Python-C
    // interface and filled with an empty vector
    // 2. At C++ level, customers should always pass an empty vector for any
    // dispensable input
    // 3. During further lowering, there will always be a placeholder VarBase
    // in ins/outs no matter whether it's dispensable or not
    // As a result, we always create input VarBase regardless of its
    // dispensability.

    // Handle duplicable input: list(VarBase) or VarBase
    // We dont know the exact number of inputs expected,
    // but we only need to identify the slot name order,
    // therefore fill in 1 single input VarBase is enough in this scenario
    ins[in_name] = {std::shared_ptr<paddle::imperative::VarBase>(
        new paddle::imperative::VarBase("auto_" + in_name))};
    ins[in_name][0]->SetOverridedStopGradient(false);
    ins[in_name][0]->MutableVar()->GetMutable<framework::LoDTensor>();
  }
  VLOG(6) << "Prepared Forward Ins Map, size = " << ins.size();

  /* ------ Prepare "outs" ------ */
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VarBase>>>
      outs;
  for (const proto::OpProto::Var& output : op_proto.outputs()) {
    const std::string& out_name = output.name();

    // We always create output VarBase regardless of its dispensability.
    // We dont know the exact number of outputs during code generation,
    // however, simply identifying the slot name order would be enough
    outs[out_name] = {std::shared_ptr<paddle::imperative::VarBase>(
        new paddle::imperative::VarBase("auto_" + out_name))};
    outs[out_name][0]->SetOverridedStopGradient(false);
    outs[out_name][0]->MutableVar()->GetMutable<framework::LoDTensor>();
  }
  VLOG(6) << "Prepared Forward Outs Map, size = " << outs.size();

  framework::AttributeMap attrs;
  paddle::framework::AttributeMap default_attrs;
  auto* attr_checker = op_info.Checker();
  if (attr_checker) {
    VLOG(6) << "Checking AttributeMap Settings";
    attr_checker->Check(&attrs, true, /*only_check_exist_value=*/true);
    default_attrs = attr_checker->GetDefaultAttrMap();
    VLOG(6) << "AttributeMap Checking Passed";
  } else {
    VLOG(6) << "Detected Null Attribute Checker, use empty default_attrs";
  }

  if (operators_with_attrs.count(op_type)) {
    VLOG(6) << "Found operator " << op_type << " using special AttributeMap";
    attrs = operators_with_attrs[op_type];
  }

  VLOG(6) << "Prepared Default Attributes Map, size = " << default_attrs.size();
  for (const auto& iter : default_attrs) {
    VLOG(6) << iter.first;
  }

  /* ---------------------------- */
  /* --------- Backward --------- */
  /* ---------------------------- */
  /* ------ Fwd paddle::imperative::VariableWrapper Map ------ */
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
      fwd_ins;
  std::map<std::string,
           std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
      fwd_outs;
  for (const auto& iter : ins) {
    fwd_ins[iter.first] = {};
    for (const std::shared_ptr<paddle::imperative::VarBase>& var_base :
         iter.second) {
      fwd_ins[iter.first].push_back(var_base->SharedVar());
    }
  }
  for (const auto& iter : outs) {
    fwd_outs[iter.first] = {};
    for (const std::shared_ptr<paddle::imperative::VarBase>& var_base :
         iter.second) {
      fwd_outs[iter.first].push_back(var_base->SharedVar());
    }
  }
  VLOG(6) << "Constructed Forward paddle::imperative::VariableWrapper Map";

  /* ------ Run GradOpMaker ------ */
  if (!op_info.dygraph_grad_op_maker_) {
    VLOG(6) << op_type << " has no GradOpMaker";
    *generate_forward_only = true;
    return false;
  }

  std::shared_ptr<paddle::imperative::GradOpNode> grad_node =
      op_info.dygraph_grad_op_maker_(op_type, ins, outs, attrs, default_attrs,
                                     {});

  if (!grad_node) {
    VLOG(6) << "Got nullptr GradOpNode for " << op_type
            << " likely registered EmptyGradOpMaker";
    *generate_forward_only = true;
    return false;
  }

  /*
  if (grad_node->size() > 1) {
    // Backward attributes can be super complicated
    VLOG(6) << "Skip GradOpNode with multiple OpBases for now: " << op_type;
    skipped_operators.insert(op_type);
    return false;
  }
  */

  VLOG(6) << "Prepared GradOpNode";

  /* ---- Collect Default Attr Map ---- */
  for (auto iter = grad_node->begin(); iter < grad_node->end(); iter++) {
    // Each OpBase
    paddle::imperative::OpBase& op_base = *iter;
    grad_op_types->push_back(op_base.Type());
  }

  /* ------ Get Grad ins/outs ---- */
  // In case of multiple OpBase, stitch all the respective ins/outs into one
  VLOG(6) << "In function size: " << grad_node->size();
  for (auto iter = grad_node->begin(); iter < grad_node->end(); iter++) {
    const paddle::imperative::OpBase& op_base = *iter;
    const std::map<std::string, paddle::imperative::SavedVariableWrapperList>&
        g_ins = op_base.GetInsMap();
    const std::map<std::string, paddle::imperative::SavedVariableWrapperList>&
        g_outs = op_base.GetOutsMap();

    for (const auto& it : g_ins) {
      if (!grad_ins->count(it.first)) (*grad_ins)[it.first] = {};
      for (auto vw_iter = it.second.begin(); vw_iter != it.second.end();
           vw_iter++) {
        std::shared_ptr<paddle::imperative::VariableWrapper> vw = *vw_iter;
        (*grad_ins)[it.first].push_back(vw);
      }
    }

    for (const auto& it : g_outs) {
      if (!grad_outs->count(it.first)) (*grad_outs)[it.first] = {};
      for (auto vw_iter = it.second.begin(); vw_iter != it.second.end();
           vw_iter++) {
        std::shared_ptr<paddle::imperative::VariableWrapper> vw = *vw_iter;
        (*grad_outs)[it.first].push_back(vw);
      }
    }
  }

  /* ------ Slot Name Matching ---- */
  // grad_ins -> fwd_ins, fwd_outs
  SlotNameMatching(*grad_ins, fwd_ins, fwd_outs, grad_ins_fwd_slotname_map,
                   grad_ins_grad_slotname_map);
  VLOG(6) << "Finished Slotname Matching for Grad_Ins";

  // grad_outs -> fwd_ins, fwd_outs
  SlotNameMatching(*grad_outs, fwd_ins, fwd_outs, grad_outs_slotname_map,
                   grad_outs_slotname_map);
  VLOG(6) << "Finished Slotname Matching for Grad_Outs";

  return true;
}

/* --------------------------------------------------- */
/* --------- CodeGen: Forward GradNode Creation ------ */
/* --------------------------------------------------- */
static std::string GenerateGradNodeCreationContent(
    const std::unordered_map<std::string, size_t>& fwd_inputs_name_pos_map,
    const std::unordered_map<std::string, size_t>& fwd_outputs_name_pos_map,
    const std::map<std::string, std::string>& grad_ins_fwd_slotname_map,
    const std::string& op_type, const std::vector<proto::OpProto::Var>& in_vars,
    const std::vector<proto::OpProto::Var>& out_vars) {
  VLOG(6) << "Generating GradNode Creation codes";

  // [Generation] Construct GradOpNode
  // Run ComputeRequiredGrad

  // If single output slotname and not duplicable,
  // then generate: "egr::AutogradMeta* p_autograd_out =
  // egr::EagerUtils::autograd_meta("op_proto->outputs()[0].name()")"
  std::string get_autograd_meta_str = "  // Prepare Autograd Meta \n";
  for (const proto::OpProto::Var& input : in_vars) {
    const std::string& input_name = input.name();
    const std::string& input_autograd_name = "p_autograd_" + input_name;

    if (input.duplicable()) {
      const char* GET_MULTI_AUTOGRAD_META_TEMPLATE =
          "  std::vector<egr::AutogradMeta*> %s = "
          "egr::EagerUtils::unsafe_autograd_meta(%s);\n";
      get_autograd_meta_str += paddle::string::Sprintf(
          GET_MULTI_AUTOGRAD_META_TEMPLATE, input_autograd_name, input_name);

    } else {
      const char* GET_SINGLE_AUTOGRAD_META_TEMPLATE =
          "  egr::AutogradMeta& %s = "
          "*egr::EagerUtils::unsafe_autograd_meta(%s);\n";
      get_autograd_meta_str += paddle::string::Sprintf(
          GET_SINGLE_AUTOGRAD_META_TEMPLATE, input_autograd_name, input_name);
    }
  }
  VLOG(6) << "Generated inputs autograd_meta";

  // If single output slotname and not duplicable,
  // then generate: "egr::AutogradMeta* p_autograd_out =
  // egr::EagerUtils::autograd_meta("op_proto.outputs()[0].name()")"
  for (const proto::OpProto::Var& output : out_vars) {
    const std::string& output_name = output.name();
    const std::string& output_autograd_name = "p_autograd_" + output_name;

    if (output.duplicable()) {
      const char* GET_MULTI_AUTOGRAD_META_TEMPLATE =
          "  std::vector<egr::AutogradMeta*> %s = "
          "egr::EagerUtils::multi_autograd_meta(&%s);\n";
      get_autograd_meta_str += paddle::string::Sprintf(
          GET_MULTI_AUTOGRAD_META_TEMPLATE, output_autograd_name, output_name);

    } else {
      const char* GET_SINGLE_AUTOGRAD_META_TEMPLATE =
          "  egr::AutogradMeta& %s = "
          "*egr::EagerUtils::autograd_meta(&%s);\n";
      get_autograd_meta_str += paddle::string::Sprintf(
          GET_SINGLE_AUTOGRAD_META_TEMPLATE, output_autograd_name, output_name);
    }
  }
  VLOG(6) << "Generated outputs autograd_meta";

  std::string prepare_autograd_meta_str = "";
  prepare_autograd_meta_str += get_autograd_meta_str;
  prepare_autograd_meta_str += "\n";

  // [GradOpNode] GetTraceBackward
  std::string trace_backward_str =
      "  bool trace_backward = egr::Controller::Instance().HasGrad();\n";
  prepare_autograd_meta_str += trace_backward_str;
  prepare_autograd_meta_str += "\n";

  // [GradOpNode] Generation
  std::string grad_node_creation_str = "";

  size_t bwd_in_slot_num = out_vars.size();
  size_t bwd_out_slot_num = in_vars.size();
  const char* GRAD_OP_NODE_TEMPLATE =
      "    auto grad_node = std::make_shared<GradNode%s>(%d, %d);\n";
  grad_node_creation_str += "    // Create GradOpNode\n";
  grad_node_creation_str += paddle::string::Sprintf(
      GRAD_OP_NODE_TEMPLATE, op_type, bwd_in_slot_num, bwd_out_slot_num);
  grad_node_creation_str += "\n";

  VLOG(6) << "Generated GradOpNode construction";

  // [GradOpNode] Set Attrs
  grad_node_creation_str += "    // Set Attributes\n";
  grad_node_creation_str += "    grad_node->SetAttrMap(std::move(attrs));\n";
  grad_node_creation_str +=
      "    grad_node->SetDefaultAttrMap(std::move(default_attrs));\n";
  grad_node_creation_str += "\n";

  // [GradOpNode] Set TensorWrappers
  grad_node_creation_str += "    // Set Tensor Wrappers\n";
  for (auto& kv : grad_ins_fwd_slotname_map) {
    const std::string& tensor_wrapper_name = kv.second;
    const char* SET_TENSOR_WRAPPER_TEMPLATE =
        "    grad_node->SetTensorWrapper%s(%s);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        SET_TENSOR_WRAPPER_TEMPLATE, tensor_wrapper_name, tensor_wrapper_name);
  }
  grad_node_creation_str += "\n";
  VLOG(6) << "Generated SetTensorWrapper";

  // [GradOpNode] SetGradOutMeta
  // [GradOpNode] Add Edges
  std::string compute_require_grad_args = "trace_backward";
  for (const proto::OpProto::Var& input : in_vars) {
    const std::string& input_name = input.name();
    const std::string& input_autograd_name = "p_autograd_" + input_name;
    compute_require_grad_args += ", &" + input_autograd_name;
    size_t input_position = fwd_inputs_name_pos_map.at(input_name);

    const char* SET_GRAD_OUT_META_TEMPLATE =
        "    grad_node->SetGradOutMeta(%s, %d);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        SET_GRAD_OUT_META_TEMPLATE, input_autograd_name, input_position);

    const char* ADD_EDGES_TEMPLATE = "    grad_node->AddEdges(%s, %d);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        ADD_EDGES_TEMPLATE, input_autograd_name, input_position);
  }

  // [GradOpNode] SetGradInMeta
  // [AutogradMeta] SetOutRank
  // [AutogradMeta] SetHistory
  std::string pass_stop_gradient_args = "false";
  for (const proto::OpProto::Var& output : out_vars) {
    const std::string& output_name = output.name();
    const std::string& output_autograd_name = "p_autograd_" + output_name;
    pass_stop_gradient_args += ", &" + output_autograd_name;
    size_t output_position = fwd_outputs_name_pos_map.at(output_name);

    const char* SET_GRAD_IN_META_TEMPLATE =
        "    grad_node->SetGradInMeta(%s, %d);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        SET_GRAD_IN_META_TEMPLATE, output_autograd_name, output_position);

    const char* SET_OUT_RANK_TEMPLATE =
        "    egr::EagerUtils::SetOutRankWithSlot(&%s, %d);\n";
    grad_node_creation_str += paddle::string::Sprintf(
        SET_OUT_RANK_TEMPLATE, output_autograd_name, output_position);

    const char* SET_HISTORY_TEMPLATE =
        "    egr::EagerUtils::SetHistory(&%s, grad_node);\n";
    grad_node_creation_str +=
        paddle::string::Sprintf(SET_HISTORY_TEMPLATE, output_autograd_name);
  }
  VLOG(6) << "Generated SetGradIn/OutMeta";

  // [Generation] GradNode Creation
  const char* GRAD_NODE_CREATION_TEMPLATE =
      "  %s"
      "  bool require_any_grad = egr::EagerUtils::ComputeRequireGrad(%s);\n"
      "  if(require_any_grad) {\n"
      "    egr::EagerUtils::PassStopGradient(%s);\n"
      "%s\n  }";
  std::string grad_node_creation_body_str = paddle::string::Sprintf(
      GRAD_NODE_CREATION_TEMPLATE, prepare_autograd_meta_str,
      compute_require_grad_args, pass_stop_gradient_args,
      grad_node_creation_str);

  return grad_node_creation_body_str;
}

/* -------------------------------- */
/* --------- CodeGen: Forward ----- */
/* -------------------------------- */
static std::pair<std::string, std::string> GenerateForwardFunctionContents(
    bool generate_forward_only,
    const std::unordered_map<std::string, size_t>& fwd_inputs_name_pos_map,
    const std::unordered_map<std::string, size_t>& fwd_outputs_name_pos_map,
    const std::map<std::string, std::string>& grad_ins_fwd_slotname_map,
    const std::map<std::string, std::string>& grad_ins_grad_slotname_map,
    const std::map<std::string, std::string>& grad_outs_slotname_map,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        grad_ins,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        grad_outs,
    const std::string& op_type, const std::vector<proto::OpProto::Var>& in_vars,
    const std::vector<proto::OpProto::Var>& out_vars) {
  /*
    // Forward Function Example:
  std::tuple<vector<Tensor>, Tensor, vector<Tensor>>
  kernel_function(vector<Tensor>& X, Tensor& Y, const paddle::AttributeMap&
  attr_map, size_t
  Out0Num, size_t Out1Num) {

        // Forward Function Body
        // According to fwd_inputs_name_pos_map
        std::map<std::string, std::vector<std::shared_ptr<egr::EagerTensor>>>
  ins =
                { {"X" , SyncToVars(X)}, { "Y" , SyncToVars(Y)} };

        std::map<std::string, std::vector<std::shared_ptr<egr::EagerTensor>>>
  outs =
  {
          {"Out0" , ConstructDuplicableOutput(Out0Num)}, {"Out1"
  ,ConstructDuplicableOutput(Out1Num)} };

        // According to op_proto->attrs()

        egr::legacy::RunOp("op_type", ins, outs, attr_map,
  Controller.Instance().GetExpectedPlace(), {});

        // According to fwd_outputs_names
        std::vector<egr::EagerTensor> Out0 = GGetOutputetOutputs(outs["Out0"]);
        egr::EagerTensor Out1 = GetOutputs(outs["Out1"][0]);
        std::vector<egr::EagerTensor> Out2 = GetOutputs(outs["Out2"]);

        // Grad Node Generation Codes
        ...

        return std::make_tuple(Out0, Out1, Out2);
    }
  */
  VLOG(6) << "Generating Dygraph Forward Function";

  std::string generated_function_body = "";
  std::string dygraph_function_args_str = "";

  /* ------ Dygraph forward function generation ------ */
  generated_function_body += "  // Dygraph Forward Pass\n";
  generated_function_body += "\n";

  // [Generation] Get Ins Map
  std::string ins_contents_str = "";
  std::vector<std::string> input_args_str_list(in_vars.size());
  for (const proto::OpProto::Var& input : in_vars) {
    const std::string& input_name = input.name();
    size_t input_position = fwd_inputs_name_pos_map.at(input_name);
    if (input.duplicable()) {
      const char* FWD_INS_ARG_TEMPLATE =
          "const std::vector<egr::EagerTensor>& %s";
      input_args_str_list[input_position] =
          paddle::string::Sprintf(FWD_INS_ARG_TEMPLATE, input_name);
    } else {
      const char* FWD_INS_ARG_TEMPLATE = "const egr::EagerTensor& %s";
      input_args_str_list[input_position] =
          paddle::string::Sprintf(FWD_INS_ARG_TEMPLATE, input_name);
    }
    const char* FWD_INS_CONTENT_TEMPLATE =
        "{ \"%s\", egr::EagerUtils::SyncToVars(%s) },";
    ins_contents_str += paddle::string::Sprintf(FWD_INS_CONTENT_TEMPLATE,
                                                input_name, input_name);
  }
  if (ins_contents_str.size() > 0)
    ins_contents_str.pop_back();  // // Remove trailing ","

  for (const std::string& arg : input_args_str_list) {
    dygraph_function_args_str += arg;
    dygraph_function_args_str += ",";
  }
  if (dygraph_function_args_str.size() > 0)
    dygraph_function_args_str.pop_back();

  const char* FWD_INS_MAP_TEMPLATE =
      "  std::map<std::string, "
      "std::vector<std::shared_ptr<egr::EagerTensor>>> ins = { "
      "%s };\n";
  std::string ins_map_str =
      paddle::string::Sprintf(FWD_INS_MAP_TEMPLATE, ins_contents_str);
  generated_function_body += ins_map_str;
  generated_function_body += "\n";

  VLOG(6) << "Generated Ins Map";

  // [Generation] Get Outs Map
  std::string outs_contents_str = "";
  for (const proto::OpProto::Var& output : out_vars) {
    const std::string& output_name = output.name();
    std::string outnum = "1";
    if (output.duplicable()) {
      outnum = output_name + "Num";

      const char* FWD_NUM_ARG_TEMPLATE = ", size_t %s";
      std::string arg_str =
          paddle::string::Sprintf(FWD_NUM_ARG_TEMPLATE, outnum);
      dygraph_function_args_str += arg_str;
      const char* FWD_OUTS_CONTENT_TEMPLATE =
          "{ \"%s\", egr::EagerUtils::ConstructDuplicableOutput(%s) },";
      outs_contents_str += paddle::string::Sprintf(FWD_OUTS_CONTENT_TEMPLATE,
                                                   output_name, outnum);
    } else {
      const char* FWD_OUTS_CONTENT_TEMPLATE =
          "{ \"%s\", "
          "{std::make_shared<egr::EagerTensor>(egr::Controller::Instance()."
          "GenerateUniqueName())}},";
      outs_contents_str +=
          paddle::string::Sprintf(FWD_OUTS_CONTENT_TEMPLATE, output_name);
    }
  }
  if (outs_contents_str.size() > 0)
    outs_contents_str.pop_back();  // Remove trailing ","

  const char* FWD_OUTS_MAP_TEMPLATE =
      "  std::map<std::string, "
      "std::vector<std::shared_ptr<egr::EagerTensor>>> outs = { "
      "%s };\n";
  std::string outs_map_str =
      paddle::string::Sprintf(FWD_OUTS_MAP_TEMPLATE, outs_contents_str);
  generated_function_body += outs_map_str;
  generated_function_body += "\n";

  VLOG(6) << "Generated Outs Map";

  // [Generation] Get Attrs
  dygraph_function_args_str +=
      ", const paddle::framework::AttributeMap& attr_map";

  // [Generation] Get TraceOp
  const char* FWD_TRACE_OP_TEMPLATE =
      "  paddle::framework::AttributeMap attrs = attr_map;\n"
      "  paddle::framework::AttributeMap default_attrs;\n"
      "  egr::legacy::RunOp(\"%s\", ins, outs, attrs, \n"
      "     egr::Controller::Instance().GetExpectedPlace(),\n"
      "     &default_attrs, true, {});\n";
  std::string trace_op_str =
      paddle::string::Sprintf(FWD_TRACE_OP_TEMPLATE, op_type);
  generated_function_body += trace_op_str;
  generated_function_body += "\n";

  VLOG(6) << "Generated AttrMap & TraceOp";

  // [Generation] Convert output VarBase to Vector/Tensor
  size_t output_size = out_vars.size();
  std::vector<std::string> return_contents(output_size);
  std::vector<std::string> return_types(output_size);
  for (const proto::OpProto::Var& output : out_vars) {
    const std::string& output_name = output.name();
    std::string out_tensor_str;
    size_t return_position = fwd_outputs_name_pos_map.at(output_name);
    std::string output_varname = LegalizeVariableName(output_name);

    if (output.duplicable()) {
      const char* FWD_OUT_TENSORS_TEMPLATE =
          "  std::vector<egr::EagerTensor> %s = "
          "egr::EagerUtils::GetOutputs(outs[\"%s\"]);\n";
      out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSORS_TEMPLATE,
                                               output_varname, output_name);
      return_types[return_position] = "std::vector<egr::EagerTensor>";
    } else {
      const char* FWD_OUT_TENSOR_TEMPLATE =
          "  egr::EagerTensor %s = "
          "egr::EagerUtils::GetOutput(outs[\"%s\"][0]);\n";
      out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSOR_TEMPLATE,
                                               output_varname, output_name);
      return_types[return_position] = "egr::EagerTensor";
    }

    return_contents[return_position] = output_varname;
    generated_function_body += out_tensor_str;
  }
  generated_function_body += "\n";
  VLOG(6) << "Converted Output VarBase to EagerTensor(s)";

  // [Generation] ComputeRequireGrad -> GradNodeCreation
  if (!generate_forward_only) {
    std::string grad_node_creation_body_str = GenerateGradNodeCreationContent(
        fwd_inputs_name_pos_map, fwd_outputs_name_pos_map,
        grad_ins_fwd_slotname_map, op_type, in_vars, out_vars);
    generated_function_body += grad_node_creation_body_str;
    generated_function_body += "\n";
    VLOG(6) << "Generated GradNode Creation codes";
  }

  // [Generation] Handle return: Tuple/Vector/Tensor
  generated_function_body += "\n";
  std::string return_str = "";
  std::string return_type_str = "";
  std::string function_proto_return_type_str = "";
  if (return_contents.size() > 1) {
    // Return tuple
    std::string return_content_str = "";
    for (const std::string& s : return_contents) {
      return_content_str += s + ",";
    }
    return_content_str.pop_back();  // Remove trailing ","

    for (const std::string& s : return_types) {
      return_type_str += s + ",";
    }
    return_type_str.pop_back();  // Remove trailing ","

    const char* FWD_TUPLE_RETURN_TEMPLATE = "  return std::make_tuple(%s);";
    return_str =
        paddle::string::Sprintf(FWD_TUPLE_RETURN_TEMPLATE, return_content_str);

    const char* FWD_FUNCTION_PROTO_RETURN_TEMPLATE = "std::tuple<%s>";
    function_proto_return_type_str = paddle::string::Sprintf(
        FWD_FUNCTION_PROTO_RETURN_TEMPLATE, return_type_str);

  } else if (return_contents.size() == 1) {
    // Return vector<Tensor> or Tensor
    return_type_str = return_types[0];
    const char* FWD_TENSOR_RETURN_TEMPLATE = "  return %s;";
    return_str =
        paddle::string::Sprintf(FWD_TENSOR_RETURN_TEMPLATE, return_contents[0]);
    function_proto_return_type_str = return_type_str;

  } else {
    return_str = "return nullptr;";
    function_proto_return_type_str = "void*";
  }

  generated_function_body += return_str;
  generated_function_body += "\n";
  VLOG(6) << "Generated return codes";

  // [Generation] Get Full Function
  std::string function_name = op_type + "_dygraph_function";

  if (dygraph_function_args_str.size() > 0) {
    auto iter = dygraph_function_args_str.begin();
    if ((*iter) == ',') dygraph_function_args_str.erase(iter);
  }

  const char* FWD_FUNCTION_TEMPLATE = "%s %s(%s) {\n\n%s\n}\n\n";
  std::string fwd_function_str = paddle::string::Sprintf(
      FWD_FUNCTION_TEMPLATE, function_proto_return_type_str, function_name,
      dygraph_function_args_str, generated_function_body);

  // [Generation] Generate forward functions header
  const char* FWD_HEADER_TEMPLATE = "%s %s(%s);\n";
  std::string dygraph_function_declaration_str = paddle::string::Sprintf(
      FWD_HEADER_TEMPLATE, function_proto_return_type_str, function_name,
      dygraph_function_args_str);

  return {fwd_function_str, dygraph_function_declaration_str};
}

/* ---------------------------------------------- */
/* --------- CodeGen: GradNode::operator() ------ */
/* ---------------------------------------------- */
static std::string GenerateGradNodeCCContents(
    const std::vector<std::string>& grad_op_types,
    const std::unordered_map<std::string, size_t>& fwd_inputs_name_pos_map,
    const std::unordered_map<std::string, size_t>& fwd_outputs_name_pos_map,
    const std::map<std::string, std::string>& grad_ins_fwd_slotname_map,
    const std::map<std::string, std::string>& grad_ins_grad_slotname_map,
    const std::map<std::string, std::string>& grad_outs_slotname_map,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        grad_ins,
    const std::map<
        std::string,
        std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>&
        grad_outs,
    const std::string& op_type, const std::vector<proto::OpProto::Var>& in_vars,
    const std::vector<proto::OpProto::Var>& out_vars) {
  VLOG(6) << "Generating Grad Node CC";

  /* [Outline]

  vector<vector<Tensor>> GradNodeXXX::operator()(vector<vector<Tensor>>& grads)
  {

    const std::shared_ptr<Tracer>& tracer = imperative::GetCurrentTracer();

    // Comes from "grad_ins"
    std::map<std::string, std::vector<std::shared_ptr<VarBase>>> ins =
            {
            "X" : this->"X", "Y" : this->"Y",
            "Out0@Grad":
  SyncToVars(grads["fwd_outputs_name_pos_map[grad_ins_grad_slotname_map["Out0@Grad"]]"]),
            "Out1@Grad":
  TensorsToVarBases(grads["fwd_outputs_name_pos_map[grad_ins_grad_slotname_map["Out1@Grad"]]"])
             };

    // Comes from "grad_outs"
    std::map<std::string, std::vector<std::shared_ptr<VarBase>>> outs =
            {
            "X@Grad" :
  ConstructDuplicableOutput(this->OutputMeta()["fwd_inputs_name_pos_map[grad_outs_slotname_map["X@Grad"]]"].Size()),
            "Y@Grad" :
  ConstructDuplicableOutput(this->OutputMeta()["fwd_inputs_name_pos_map[grad_outs_slotname_map["Y@Grad"]]"].Size())
             };

    // Visit each OpBase
    for(auto iter = "grad_node->begin()"; iter < "grad_node->end()"; iter++) {
        // Simply pass entire attribute map to kernels
        egr::legacy::RunOp("iter->Type()", ins, outs, this->attr_map_,
            egr::Controller::Instance().ExpectedPlace(), false, {});
    }

    vector<vector<egr::EagerTensor>> outputs(outs.size());
    for(auto& kv : outs) {
        outputs["fwd_inputs_name_pos_map[grad_outs_slotname_map[kv.first]]"] =
  GetOutputs(outs["kv.first"]);
    }

    return outputs;
  }
  */

  std::string generated_grad_function_body = "";

  // [Generation] Get Tracer
  generated_grad_function_body += "\n";
  generated_grad_function_body += "\n";

  // [Generation] Get Ins Map
  std::string ins_contents_str = "";
  for (auto iter : grad_ins) {
    const std::string& grad_input_name = iter.first;

    if (grad_ins_fwd_slotname_map.count(grad_input_name)) {
      // Fwd Tensor
      std::string struct_fwd_input_name =
          grad_ins_fwd_slotname_map.at(grad_input_name) + "_";
      const char* GRAD_INS_FWD_CONTENT_TEMPLATE =
          "{ \"%s\", "
          "egr::EagerUtils::SyncToVars(egr::EagerUtils::RecoverTensorWrapper(&"
          "this->%s, "
          "nullptr)) },";
      ins_contents_str +=
          paddle::string::Sprintf(GRAD_INS_FWD_CONTENT_TEMPLATE,
                                  grad_input_name, struct_fwd_input_name);

    } else if (grad_ins_grad_slotname_map.count(grad_input_name)) {
      // Fwd Tensor's Grad
      size_t fwd_output_position = fwd_outputs_name_pos_map.at(
          grad_ins_grad_slotname_map.at(grad_input_name));
      const char* GRAD_INS_GRAD_CONTENT_TEMPLATE =
          "{ \"%s\", egr::EagerUtils::SyncToVars(grads[%d]) },";
      ins_contents_str += paddle::string::Sprintf(
          GRAD_INS_GRAD_CONTENT_TEMPLATE, grad_input_name, fwd_output_position);

    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "Detected mismatched slot names."
          "Unable to find forward slot name that matches %s",
          grad_input_name));
    }
  }
  if (ins_contents_str.size() > 0)
    ins_contents_str.pop_back();  // // Remove trailing ","

  const char* BWD_INS_MAP_TEMPLATE =
      "  std::map<std::string, "
      "std::vector<std::shared_ptr<egr::EagerTensor>>> ins = { "
      "%s };\n";
  std::string ins_map_str =
      paddle::string::Sprintf(BWD_INS_MAP_TEMPLATE, ins_contents_str);
  generated_grad_function_body += ins_map_str;

  VLOG(6) << "Generated Ins Map";

  // [Generation] Get Outs Map
  std::unordered_set<std::string> duplicable_input_name_set;
  for (const auto& in : in_vars) {
    if (in.duplicable()) duplicable_input_name_set.insert(in.name());
  }

  std::string outs_contents_str = "";
  for (auto iter : grad_outs) {
    const std::string& grad_output_name = iter.first;

    if (grad_outs_slotname_map.count(grad_output_name)) {
      // Fwd Tensor
      const std::string& fwd_name = grad_outs_slotname_map.at(grad_output_name);

      /* Handle Special Case: "PullSparseOp", etc

          Forward:

             Ids  W
              |   |
           PullSparseOp
                |
               Out

          Backward:

             Ids  GradOut  W
              |      |     |
             PullSparseGradOp
                     |
                  GradOut

          Its grad output "GradOut" corresponds to forward output "Out",
          where there is a hiden inplace involved. So we find "GradOut"'s index
         in
          grads, and perform the inplace operation by constructing outs =
         {{"Out", grads[i]}}

          GradOut -> Out -> fwd_output_pos -> grads position -> grads[i]
          outs = {{"Out", grads[i]}}

          For returns, append "GradOut" to the very end of return list.
      */
      if (!fwd_inputs_name_pos_map.count(fwd_name)) {
        PADDLE_ENFORCE(fwd_outputs_name_pos_map.count(fwd_name),
                       paddle::platform::errors::Fatal(
                           "fwd_name not found in fwd_inputs_name_pos_map nor "
                           "fwd_outputs_name_pos_map"));

        size_t grads_position = fwd_outputs_name_pos_map.at(fwd_name);
        std::string grad_ptr_name = fwd_name + "_ptrs";
        const char* GET_GRADS_PTR_TEMPLATE =
            "  std::vector<std::shared_ptr<egr::EagerTensor>> %s;\n"
            "  for(const auto& t : grads[%d]) {\n    "
            "%s.emplace_back(std::move(std::make_shared<egr::EagerTensor>(t)));"
            "\n  }\n";
        std::string grads_ptr_str =
            paddle::string::Sprintf(GET_GRADS_PTR_TEMPLATE, grad_ptr_name,
                                    grads_position, grad_ptr_name);
        generated_grad_function_body += grads_ptr_str;
        generated_grad_function_body += "\n";

        const char* GRAD_OUTS_CONTENT_TEMPLATE = "{ \"%s\", %s },";
        outs_contents_str += paddle::string::Sprintf(
            GRAD_OUTS_CONTENT_TEMPLATE, grad_output_name, grad_ptr_name);

      } else {
        size_t fwd_input_position = fwd_inputs_name_pos_map.at(fwd_name);
        if (duplicable_input_name_set.count(fwd_name)) {
          const char* GRAD_OUTS_CONTENT_TEMPLATE =
              "{ \"%s\", egr::EagerUtils::ConstructDuplicableOutput( "
              "this->OutputMeta()[%d].Size() ) },";
          outs_contents_str += paddle::string::Sprintf(
              GRAD_OUTS_CONTENT_TEMPLATE, grad_output_name, fwd_input_position);
        } else {
          const char* GRAD_OUTS_CONTENT_TEMPLATE =
              "{ \"%s\", "
              "{std::make_shared<egr::EagerTensor>(egr::Controller::Instance()."
              "GenerateUniqueName())}},";
          outs_contents_str += paddle::string::Sprintf(
              GRAD_OUTS_CONTENT_TEMPLATE, grad_output_name);
        }
      }
    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "Detected mismatched slot names."
          "Unable to find forward slot name that matches %s",
          grad_output_name));
    }
  }
  if (outs_contents_str.size() > 0)
    outs_contents_str.pop_back();  // // Remove trailing ","

  const char* BWD_OUTS_MAP_TEMPLATE =
      "  std::map<std::string, "
      "std::vector<std::shared_ptr<egr::EagerTensor>>> outs = { "
      "%s };\n";
  std::string outs_map_str =
      paddle::string::Sprintf(BWD_OUTS_MAP_TEMPLATE, outs_contents_str);
  generated_grad_function_body += outs_map_str;
  generated_grad_function_body += "\n";

  VLOG(6) << "Generated Outs Map";

  // [Generation] Get Attrs Map
  std::string trace_opbase_str = "";
  for (size_t i = 0; i < grad_op_types.size(); i++) {
    const std::string& op_base_type = grad_op_types[i];

    const char* TRACE_OP_TEMPLATE =
        "  // Pass the entire attribute map to TraceOp\n"
        "  // The underlying kernel will pickup whatever attribute they need "
        "at runtime\n"
        "  egr::legacy::RunOp(\"%s\", ins, outs, this->attr_map_,\n"
        "      egr::Controller::Instance().GetExpectedPlace(),\n"
        "      &this->default_attr_map_, false, {});\n";
    trace_opbase_str = paddle::string::Sprintf(TRACE_OP_TEMPLATE, op_base_type);
  }

  generated_grad_function_body += trace_opbase_str;

  VLOG(6) << "Generated Attrs Map";

  // [Generation] Get Return
  std::string outputs_str = "";
  size_t num_appended_outputs = 0;
  for (auto iter : grad_outs) {
    const std::string& grad_out_name = iter.first;
    const std::string& fwd_name = grad_outs_slotname_map.at(grad_out_name);

    if (fwd_inputs_name_pos_map.count(fwd_name)) {
      size_t fwd_input_position = fwd_inputs_name_pos_map.at(fwd_name);
      const char* BWD_OUTPUT_TEMPLATE =
          "  outputs[%d] = egr::EagerUtils::GetOutputs(outs[\"%s\"]);\n";
      outputs_str += paddle::string::Sprintf(BWD_OUTPUT_TEMPLATE,
                                             fwd_input_position, grad_out_name);
      num_appended_outputs++;
    } else {
      PADDLE_ENFORCE(fwd_outputs_name_pos_map.count(fwd_name),
                     paddle::platform::errors::Fatal(
                         "fwd_name not found in fwd_inputs_name_pos_map nor "
                         "fwd_outputs_name_pos_map"));
    }
  }

  /* Handle Special Case: "PullSparseOp", etc
     For returns, append "GradOut" to the very end of return list. */
  for (auto iter : grad_outs) {
    const std::string& grad_out_name = iter.first;
    const std::string& fwd_name = grad_outs_slotname_map.at(grad_out_name);

    if (fwd_outputs_name_pos_map.count(fwd_name)) {
      const char* BWD_OUTPUT_TEMPLATE =
          "  outputs[%d] = egr::EagerUtils::GetOutputs(outs[\"%s\"]);\n";
      outputs_str += paddle::string::Sprintf(
          BWD_OUTPUT_TEMPLATE, num_appended_outputs, grad_out_name);
      num_appended_outputs++;
    }
  }

  const char* BWD_RETURN_TEMPLATE =
      "  std::vector<std::vector<egr::EagerTensor>> "
      "outputs(outs.size());\n%s\n  "
      "return outputs;";
  std::string return_str =
      paddle::string::Sprintf(BWD_RETURN_TEMPLATE, outputs_str);

  generated_grad_function_body += "\n";
  generated_grad_function_body += return_str;

  // [Generation] Get Full Grad Function
  const char* GRAD_FUNCTION_TEMPLATE =
      "std::vector<std::vector<egr::EagerTensor>> "
      "GradNode%s::operator()(const "
      "std::vector<std::vector<egr::EagerTensor>>& grads) {\n%s\n}";
  std::string grad_function_str = paddle::string::Sprintf(
      GRAD_FUNCTION_TEMPLATE, op_type, generated_grad_function_body);

  VLOG(6) << "Generated returns";

  return grad_function_str;
}

/* ----------------------------------------- */
/* --------- CodeGen: GradNode Header ------ */
/* ----------------------------------------- */
static std::string GenerateGradNodeHeaderContents(
    const std::map<std::string, std::string>& grad_ins_fwd_slotname_map,
    const std::string& op_type, const std::vector<proto::OpProto::Var>& in_vars,
    const std::vector<proto::OpProto::Var>& out_vars) {
  VLOG(6) << "Generating Grad Node Header";

  const char* GRAD_NODE_TEMPLATE =
      "class GradNode%s : public egr::GradNodeBase {\n"
      " public:\n"
      "  GradNode%s() : egr::GradNodeBase() {}\n"
      "  GradNode%s(size_t bwd_in_slot_num, size_t bwd_out_slot_num) : "
      "egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}\n"
      "  ~GradNode%s() override = default;\n"
      "\n"
      "  virtual std::vector<std::vector<egr::EagerTensor>> "
      "operator()(const "
      "std::vector<std::vector<egr::EagerTensor>>& grads) "
      "override;\n"
      "\n"
      "  // SetX, SetY, ...\n"
      "%s\n"
      "  // SetAttrMap\n"
      "%s\n"
      "\n"
      " private:\n"
      "   // TensorWrappers\n"
      "%s\n"
      "   // Attribute Map\n"
      "%s\n"
      "};";

  // [Generation] Handle Attributes
  std::string set_attr_map_str =
      "   void SetAttrMap(paddle::framework::AttributeMap&& attr_map) {\n     "
      "attr_map_ = std::move(attr_map);\n   }\n";
  set_attr_map_str +=
      "   void SetDefaultAttrMap(paddle::framework::AttributeMap&& "
      "default_attr_map) {\n     default_attr_map_ = "
      "std::move(default_attr_map);\n   }\n";
  std::string attr_members_str =
      "   paddle::framework::AttributeMap attr_map_;\n";
  attr_members_str += "   paddle::framework::AttributeMap default_attr_map_;";

  VLOG(6) << "Generated SetAttr";

  // [Generation] Handle TensorWrappers
  std::unordered_set<std::string> duplicable_tensors;
  for (const proto::OpProto::Var& input : in_vars) {
    if (input.duplicable()) {
      duplicable_tensors.insert(input.name());
    }
  }
  for (const proto::OpProto::Var& output : out_vars) {
    if (output.duplicable()) {
      duplicable_tensors.insert(output.name());
    }
  }

  std::string set_tensor_wrappers_str = "";
  std::string tensor_wrapper_members_str = "";
  for (const auto& kv : grad_ins_fwd_slotname_map) {
    const std::string& tensor_wrapper_name = kv.second;
    const std::string& struct_tensor_wrapper_name = kv.second + "_";

    std::string tensor_wrapper_arg_str;
    std::string tensor_wrapper_body_str;
    if (duplicable_tensors.count(tensor_wrapper_name)) {
      const char* ATTR_TENSOR_WRAPPER_ARG_TEMPLATE =
          "const std::vector<egr::EagerTensor>& %s";
      tensor_wrapper_arg_str = paddle::string::Sprintf(
          ATTR_TENSOR_WRAPPER_ARG_TEMPLATE, tensor_wrapper_name);

      const char* TENSOR_WRAPPER_MEMBER_TEMPLATE =
          "   std::vector<egr::TensorWrapper> %s;\n";
      tensor_wrapper_members_str += paddle::string::Sprintf(
          TENSOR_WRAPPER_MEMBER_TEMPLATE, struct_tensor_wrapper_name);

      const char* SET_TENSOR_WRAPPER_BODY_TEMPLATE =
          "for(const auto& eager_tensor : %s) {\n"
          "          %s.emplace_back( egr::TensorWrapper(eager_tensor, true "
          "/*full_reserved*/) );\n"
          "      }\n";
      tensor_wrapper_body_str = paddle::string::Sprintf(
          SET_TENSOR_WRAPPER_BODY_TEMPLATE, tensor_wrapper_name,
          struct_tensor_wrapper_name);

    } else {
      const char* ATTR_TENSOR_WRAPPER_ARG_TEMPLATE =
          "const egr::EagerTensor& %s";
      tensor_wrapper_arg_str = paddle::string::Sprintf(
          ATTR_TENSOR_WRAPPER_ARG_TEMPLATE, tensor_wrapper_name);

      const char* TENSOR_WRAPPER_MEMBER_TEMPLATE =
          "   egr::TensorWrapper %s;\n";
      tensor_wrapper_members_str += paddle::string::Sprintf(
          TENSOR_WRAPPER_MEMBER_TEMPLATE, struct_tensor_wrapper_name);

      const char* SET_TENSOR_WRAPPER_BODY_TEMPLATE =
          "%s = egr::TensorWrapper(%s, true /*full_reserved*/);";
      tensor_wrapper_body_str = paddle::string::Sprintf(
          SET_TENSOR_WRAPPER_BODY_TEMPLATE, struct_tensor_wrapper_name,
          tensor_wrapper_name);
    }

    const char* SET_TENSOR_WRAPPER_TEMPLATE =
        "   void SetTensorWrapper%s(%s) {\n     %s\n   }\n";
    set_tensor_wrappers_str += paddle::string::Sprintf(
        SET_TENSOR_WRAPPER_TEMPLATE, tensor_wrapper_name,
        tensor_wrapper_arg_str, tensor_wrapper_body_str);
  }
  VLOG(6) << "Generated TensorWrapper";

  std::string grad_node_str = paddle::string::Sprintf(
      GRAD_NODE_TEMPLATE, op_type, op_type, op_type, op_type,
      set_tensor_wrappers_str, set_attr_map_str, tensor_wrapper_members_str,
      attr_members_str);

  return grad_node_str;
}

/* --------------------------------- */
/* --------- FileGeneration --------- */
/* ---------------------------------- */
static void GenerateForwardHFile(const std::string& output_dir,
                                 const std::string& dygraph_forward_api_str) {
  std::string dygraph_forward_api_path = output_dir + "/dygraph_forward_api.h";
  std::ofstream forward_header_stream(dygraph_forward_api_path, std::ios::out);
  forward_header_stream << dygraph_forward_api_str;
  forward_header_stream.close();
}

static void GenerateForwardDygraphFile(const std::string& output_dir,
                                       const std::string& fwd_function_str) {
  std::string forwards_dir = output_dir + "/forwards/";
  std::string forward_cc_filename = "dygraph_forward_functions.cc";
  std::string forward_cc_path = forwards_dir + forward_cc_filename;
  const char* FORWARD_INCLUDE_TEMPLATE =
      "#include "
      "\"paddle/fluid/eager/api/generated/fluid_generated/"
      "dygraph_forward_api.h\"\n"
      "#include "
      "\"paddle/fluid/eager/api/generated/fluid_generated/nodes/nodes.h\"\n\n"
      "#include \"paddle/fluid/eager/api/utils/global_utils.h\"\n"
      "#include \"paddle/fluid/eager/legacy/op_runner.h\"\n";
  std::string forward_cc_include_str =
      paddle::string::Sprintf(FORWARD_INCLUDE_TEMPLATE);
  std::ofstream forward_cc_stream(forward_cc_path, std::ios::out);
  forward_cc_stream << forward_cc_include_str;
  forward_cc_stream << fwd_function_str;
  forward_cc_stream.close();
}

static void GenerateNodeHFile(const std::string& output_dir,
                              const std::string& grad_node_str) {
  std::string nodes_dir = output_dir + "/nodes/";
  std::string node_h_filename = "nodes.h";
  std::string node_h_path = nodes_dir + node_h_filename;
  std::string node_h_include_str =
      "#pragma once\n"
      "#include \"paddle/fluid/eager/tensor_wrapper.h\"\n"
      "#include \"paddle/fluid/eager/legacy/op_runner.h\"\n"
      "#include \"paddle/fluid/eager/grad_node_info.h\"\n\n";
  std::ofstream node_h_stream(node_h_path, std::ios::out);
  node_h_stream << node_h_include_str;
  node_h_stream << grad_node_str;
  node_h_stream.close();
}

static void GenerateNodeCCFile(const std::string& output_dir,
                               const std::string& grad_function_str) {
  std::string nodes_dir = output_dir + "/nodes/";
  std::string node_cc_filename = "nodes.cc";
  std::string node_cc_path = nodes_dir + node_cc_filename;
  const char* NODE_CC_INCLUDE_TEMPLATE =
      "#include \"glog/logging.h\"\n"
      "#include \"paddle/pten/api/all.h\"\n"
      "#include \"paddle/fluid/imperative/tracer.h\"\n"
      "#include \"paddle/fluid/framework/op_registry.h\"\n"
      "#include \"paddle/fluid/eager/utils.h\"\n"
      "#include \"paddle/fluid/eager/api/utils/global_utils.h\"\n"
      "#include "
      "\"paddle/fluid/eager/api/generated/fluid_generated/nodes/nodes.h\"\n\n";
  std::string node_cc_include_str =
      paddle::string::Sprintf(NODE_CC_INCLUDE_TEMPLATE);
  std::ofstream node_cc_stream(node_cc_path, std::ios::out);
  node_cc_stream << node_cc_include_str;
  node_cc_stream << grad_function_str;
  node_cc_stream.close();
}

static std::string GenerateDygraphHFileIncludes() {
  std::string dygraph_forward_api_includes_str =
      "#pragma once\n"
      "#include \"glog/logging.h\"\n"
      "#include \"paddle/fluid/eager/autograd_meta.h\"\n"
      "#include \"paddle/pten/api/all.h\"\n"
      "#include \"paddle/fluid/eager/utils.h\"\n"
      "#include \"paddle/fluid/framework/op_registry.h\"\n\n";

  return dygraph_forward_api_includes_str;
}

static void DygraphCodeGeneration(const std::string& output_dir) {
  std::string dygraph_forward_api_str = GenerateDygraphHFileIncludes();
  std::string fwd_function_str = "";
  std::string grad_node_h_str = "";
  std::string grad_node_cc_str = "";

  auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();

  for (auto& pair : op_info_map) {
    const OpInfo& op_info = pair.second;
    proto::OpProto* op_proto = op_info.proto_;

    if (!CheckOpProto(op_proto)) continue;
    const std::string& op_type = op_proto->type();

    /* ----------------------------- */
    /* ---- Collect Information ---- */
    /* ----------------------------- */
    std::vector<std::string> grad_op_types;
    std::vector<proto::OpProto::Var> in_vars;
    std::vector<proto::OpProto::Var> out_vars;
    std::map<std::string, std::string> grad_outs_slotname_map;
    std::map<std::string, std::string> grad_ins_fwd_slotname_map;
    std::map<std::string, std::string> grad_ins_grad_slotname_map;
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
        grad_ins;
    std::map<std::string,
             std::vector<std::shared_ptr<paddle::imperative::VariableWrapper>>>
        grad_outs;

    VLOG(6) << "-------- CollectInformationFromOpInfo -------";

    CollectForwardInformationFromOpInfo(op_info, &in_vars, &out_vars);

    bool generate_forward_only = false;
    bool is_available = CollectGradInformationFromOpInfo(
        op_info, &generate_forward_only, &grad_op_types,
        &grad_outs_slotname_map, &grad_ins_fwd_slotname_map,
        &grad_ins_grad_slotname_map, &grad_ins, &grad_outs);

    if (!is_available && !generate_forward_only) {
      VLOG(6) << "Skipped operator: " << op_type;
      continue;
    }

    VLOG(6) << "-------- PurifyOpProto -------";
    std::unordered_map<std::string, size_t> fwd_inputs_name_pos_map;
    std::unordered_map<std::string, size_t> fwd_outputs_name_pos_map;
    PurifyForwardOpProto(*op_proto, &fwd_inputs_name_pos_map,
                         &fwd_outputs_name_pos_map, &in_vars, &out_vars);

    if (!generate_forward_only) {
      PurifyGradOpProto(*op_proto, &grad_outs_slotname_map,
                        &grad_ins_fwd_slotname_map, &grad_ins_grad_slotname_map,
                        &grad_ins, &grad_outs);
    }

    /* --------------------------- */
    /* --------- CodeGen --------- */
    /* --------------------------- */
    /* ---- forward_dygraph_functions.cc ---- */
    VLOG(6) << "-------- GenerateForwardFunctionContents -------";
    std::pair<std::string, std::string> body_and_declaration =
        GenerateForwardFunctionContents(
            generate_forward_only, fwd_inputs_name_pos_map,
            fwd_outputs_name_pos_map, grad_ins_fwd_slotname_map,
            grad_ins_grad_slotname_map, grad_outs_slotname_map, grad_ins,
            grad_outs, op_type, in_vars, out_vars);

    fwd_function_str += body_and_declaration.first + "\n";

    /* ---- dygraph_forward_api.h ---- */
    std::string fwd_function_declare_str = body_and_declaration.second;
    dygraph_forward_api_str += fwd_function_declare_str;

    if (generate_forward_only) continue;

    /* ---- nodes.h ---- */
    VLOG(6) << "-------- GenerateGradNodeHeaderContents -------";
    grad_node_h_str +=
        GenerateGradNodeHeaderContents(grad_ins_fwd_slotname_map, op_type,
                                       in_vars, out_vars) +
        "\n";

    /* ---- nodes.cc ---- */
    VLOG(6) << "-------- GenerateGradNodeCCContents -------";
    grad_node_cc_str += GenerateGradNodeCCContents(
                            grad_op_types, fwd_inputs_name_pos_map,
                            fwd_outputs_name_pos_map, grad_ins_fwd_slotname_map,
                            grad_ins_grad_slotname_map, grad_outs_slotname_map,
                            grad_ins, grad_outs, op_type, in_vars, out_vars) +
                        "\n";

    VLOG(6) << op_type << ": Finished Generating Op: " << op_type;
  }
  /* ---- dygraph_forward_function.cc ---- */
  VLOG(6) << "-------- GenerateDygraphForwardCCFile -------";
  GenerateForwardDygraphFile(output_dir, fwd_function_str);

  /* ---- dygraph_forward_api.h ---- */
  VLOG(6) << "-------- GenerateForwardHFile -------";
  GenerateForwardHFile(output_dir, dygraph_forward_api_str);

  /* ---- nodes.h ---- */
  VLOG(6) << "-------- GenerateNodeHFile -------";
  GenerateNodeHFile(output_dir, grad_node_h_str);

  /* ---- nodes.cc ---- */
  VLOG(6) << "-------- GenerateNodeCCFile -------";
  GenerateNodeCCFile(output_dir, grad_node_cc_str);
}

static void PrepareAttrMapForOps() {
  // Handle "fused_elemwise_add_activation"
  std::vector<std::string> functor_list = {"a", "b"};
  operators_with_attrs["fused_elemwise_add_activation"] = {};
  operators_with_attrs["fused_elemwise_add_activation"]["functor_list"] =
      functor_list;

  // Handle "fused_elemwise_activation"
  operators_with_attrs["fused_elemwise_activation"] = {};
  operators_with_attrs["fused_elemwise_activation"]["functor_list"] =
      functor_list;

  // Handle "reverse"
  std::vector<int> axis = {0};
  operators_with_attrs["reverse"] = {};
  operators_with_attrs["reverse"]["axis"] = axis;

  // Handle "flip"
  operators_with_attrs["flip"] = {};
  operators_with_attrs["flip"]["axis"] = axis;

  // Handle "cast"
  operators_with_attrs["cast"] = {};
  operators_with_attrs["cast"]["out_dtype"] = 5;
  operators_with_attrs["cast"]["in_dtype"] = 5;

  // Handle "transfer_dtype"
  operators_with_attrs["transfer_dtype"] = {};
  operators_with_attrs["transfer_dtype"]["out_dtype"] = 5;
  operators_with_attrs["transfer_dtype"]["in_dtype"] = 5;

  // Handle "c_split"
  operators_with_attrs["c_split"] = {};
  operators_with_attrs["c_split"]["nranks"] = 1;
}

static void CollectOperatorsToCodeGen(const std::string& op_list_path) {
  std::string line;
  std::ifstream op_list_file(op_list_path);
  if (op_list_file.is_open()) {
    while (getline(op_list_file, line)) {
      operators_to_codegen.insert(line);
    }
    op_list_file.close();
  } else {
    PADDLE_THROW(
        paddle::platform::errors::Fatal("Unable to open op_list.txt file"));
  }
}

}  // namespace framework
}  // namespace paddle

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cerr << "argc must be 3" << std::endl;
    return -1;
  }

  std::string eager_root = argv[1];
  std::string op_list_path = argv[2];

  paddle::framework::CollectOperatorsToCodeGen(op_list_path);
  paddle::framework::PrepareAttrMapForOps();

  paddle::framework::DygraphCodeGeneration(eager_root);

  return 0;
}