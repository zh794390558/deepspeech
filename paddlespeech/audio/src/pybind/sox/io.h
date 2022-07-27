// Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
// All rights reserved.

#pragma once

#include "paddlespeech/audio/src/pybind/sox/utils.h"

namespace py = pybind11;

namespace paddleaudio {
namespace sox_io {

auto get_info_file(const std::string &path, const std::string &format)
    -> std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>;

auto get_info_fileobj(py::object fileobj, const std::string &format)
    -> std::tuple<int64_t, int64_t, int64_t, int64_t, std::string>;

auto load_audio_fileobj(
    py::object fileobj,
    tl::optional<int64_t> frame_offset,
    tl::optional<int64_t> num_frames,
    tl::optional<bool> normalize,
    tl::optional<bool> channels_first,
    tl::optional<std::string> format)
    -> tl::optional<std::tuple<py::array, int64_t>>;

void save_audio_fileobj(
    py::object fileobj,
    py::array tensor,
    int64_t sample_rate,
    bool channels_first,
    tl::optional<double> compression,
    tl::optional<std::string> format,
    tl::optional<std::string> encoding,
    tl::optional<int64_t> bits_per_sample);

}  // namespace paddleaudio
}  // namespace sox_io
