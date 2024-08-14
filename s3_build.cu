#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <bits/stdc++.h>

using namespace std;
using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      cout << msg << endl;
  }
};

int main(int argc, char **argv) {
  Logger logger;
  auto builder = unique_ptr<IBuilder>(createInferBuilder(logger));
  uint32_t flag = 1U << static_cast<uint32_t>(
                      NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = unique_ptr<INetworkDefinition>(builder->createNetworkV2(flag));

  auto parser = unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));
  std::string file_path = "mnist.onnx";
  parser->parseFromFile(file_path.c_str(),
                        static_cast<int32_t>(ILogger::Severity::kWARNING));

  auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 20);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);

  auto engine = unique_ptr<IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));

  ofstream engine_file("mnist.engine", ios::binary);
  assert(engine_file.is_open() && "Failed to open engine file");
  engine_file.write((char *)engine->data(), engine->size());
  engine_file.close();

  cout << "Engine build success!" << endl;
}
