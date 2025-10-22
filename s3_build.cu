#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <bits/stdc++.h>

namespace {

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << msg << "\n";
    }
  }
};

} // anonymous namespace

int main(int argc, char **argv) {
  Logger logger;
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  uint32_t flag = 1U << static_cast<uint32_t>(
                      nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));
  std::string file_path = "mnist.onnx";
  parser->parseFromFile(file_path.c_str(),
                        static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);

  auto engine = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));

  std::ofstream engine_file("mnist.engine", std::ios::binary);
  assert(engine_file.is_open() && "Failed to open engine file");
  engine_file.write((char *)engine->data(), engine->size());
  engine_file.close();

  std::cout << "Engine build success!\n";
}
