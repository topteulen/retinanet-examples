/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once 

#include <NvInfer.h>

#include <cassert>
#include <vector>

#include "../cuda/decode_rotate.h"

using namespace nvinfer1;

#define RETINANET_PLUGIN_NAME "RetinaNetDecodeRotate"
#define RETINANET_PLUGIN_VERSION "1"
#define RETINANET_PLUGIN_NAMESPACE ""

namespace retinanet {

class DecodeRotatePlugin : public IPluginV2Ext {
  float _score_thresh;
  int _top_n;
  std::vector<float> _anchors;
  float _scale;

  size_t _height;
  size_t _width;
  size_t _num_anchors;
  size_t _num_classes;

  mutable int size = -1;

protected:
  void deserialize(void const* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    read(d, _score_thresh);
    read(d, _top_n);
    size_t anchors_size;
    read(d, anchors_size);
    while( anchors_size-- ) {
      float val;
      read(d, val);
      _anchors.push_back(val);
    }
    read(d, _scale);
    read(d, _height);
    read(d, _width);
    read(d, _num_anchors);
    read(d, _num_classes);
  }

  size_t getSerializationSize() const override {
    return sizeof(_score_thresh) + sizeof(_top_n)
      + sizeof(size_t) + sizeof(float) * _anchors.size() + sizeof(_scale)
      + sizeof(_height) + sizeof(_width) + sizeof(_num_anchors) + sizeof(_num_classes);
  }

  void serialize(void *buffer) const override {
    char* d = static_cast<char*>(buffer);
    write(d, _score_thresh);
    write(d, _top_n);
    write(d, _anchors.size());
    for( auto &val : _anchors ) {
      write(d, val);
    }
    write(d, _scale);
    write(d, _height);
    write(d, _width);
    write(d, _num_anchors);
    write(d, _num_classes);
  }

public:
  DecodeRotatePlugin(float score_thresh, int top_n, std::vector<float> const& anchors, int scale)
    : _score_thresh(score_thresh), _top_n(top_n), _anchors(anchors), _scale(scale) {}

  DecodeRotatePlugin(float score_thresh, int top_n, std::vector<float> const& anchors, int scale,
    size_t height, size_t width, size_t num_anchors, size_t num_classes)
    : _score_thresh(score_thresh), _top_n(top_n), _anchors(anchors), _scale(scale),
      _height(height), _width(width), _num_anchors(num_anchors), _num_classes(num_classes) {}

  DecodeRotatePlugin(void const* data, size_t length) {
      this->deserialize(data, length);
  }

  const char *getPluginType() const override {
    return RETINANET_PLUGIN_NAME;
  }

  const char *getPluginVersion() const override {
    return RETINANET_PLUGIN_VERSION;
  }

  int getNbOutputs() const override {
    return 3;
  }

  Dims getOutputDimensions(int index,
                                     const Dims *inputs, int nbInputDims) override {
    assert(nbInputDims == 2);
    assert(index < this->getNbOutputs());
    return Dims3(_top_n * (index == 1 ? 6 : 1), 1, 1);
  }

  bool supportsFormat(DataType type, PluginFormat format) const override {
    return type == DataType::kFLOAT && format == PluginFormat::kLINEAR;
  }


  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override {
    if (size < 0) {
      size = cuda::decode_rotate(maxBatchSize, nullptr, nullptr, _height, _width, _scale,
        _num_anchors, _num_classes, _anchors, _score_thresh, _top_n,
        nullptr, 0, nullptr);
    }
    return size;
  }

  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override {
    return cuda::decode_rotate(batchSize, inputs, outputs, _height, _width, _scale,
      _num_anchors, _num_classes, _anchors, _score_thresh, _top_n,
      workspace, getWorkspaceSize(batchSize), stream);
  }

  void destroy() override {
    delete this;
  };

  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }

  void setPluginNamespace(const char *N) override {

  }

  // IPluginV2Ext Methods
  DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const
  {
    assert(index < 3);
    return DataType::kFLOAT;
  }

  bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted,
    int nbInputs) const { return false; }

  bool canBroadcastInputAcrossBatch(int inputIndex) const { return false; }

  void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
  {
    assert(*inputTypes == nvinfer1::DataType::kFLOAT &&
      floatFormat == nvinfer1::PluginFormat::kLINEAR);
    assert(nbInputs == 2);
    assert(nbOutputs == 3);
    auto const& scores_dims = inputDims[0];
    auto const& boxes_dims = inputDims[1];
    assert(scores_dims.d[1] == boxes_dims.d[1]);
    assert(scores_dims.d[2] == boxes_dims.d[2]);
    _height = scores_dims.d[1];
    _width = scores_dims.d[2];
    _num_anchors = boxes_dims.d[0] / 6;
    _num_classes = scores_dims.d[0] / _num_anchors;
  }

  IPluginV2Ext *clone() const override {
    return new DecodeRotatePlugin(_score_thresh, _top_n, _anchors, _scale, _height, _width,
      _num_anchors, _num_classes);
  }

private:
  template<typename T> void write(char*& buffer, const T& val) const {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
  }

  template<typename T> void read(const char*& buffer, T& val) {
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
  }
};

class DecodeRotatePluginCreator : public IPluginCreator {
public:
  DecodeRotatePluginCreator() {}

  const char *getPluginName () const override {
    return RETINANET_PLUGIN_NAME;
  }

  const char *getPluginVersion () const override {
    return RETINANET_PLUGIN_VERSION;
  }
 
  const char *getPluginNamespace() const override {
    return RETINANET_PLUGIN_NAMESPACE;
  }

  IPluginV2 *deserializePlugin (const char *name, const void *serialData, size_t serialLength) override {
    return new DecodeRotatePlugin(serialData, serialLength);
  }

  void setPluginNamespace(const char *N) override {}
  const PluginFieldCollection *getFieldNames() override { return nullptr; }
  IPluginV2 *createPlugin (const char *name, const PluginFieldCollection *fc) override { return nullptr; }
};

REGISTER_TENSORRT_PLUGIN(DecodeRotatePluginCreator);

}

#undef RETINANET_PLUGIN_NAME
#undef RETINANET_PLUGIN_VERSION
#undef RETINANET_PLUGIN_NAMESPACE
