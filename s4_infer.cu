#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace nvinfer1;

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING)
      cout << msg << endl;
  }
};

std::vector<unsigned char> load_engine_file(const std::string &file_name) {
  std::vector<unsigned char> engine_data;
  std::ifstream engine_file(file_name, std::ios::binary | std::ios::ate);
  assert(engine_file.is_open() && "Unable to load engine file.");
  int length = engine_file.tellg();
  engine_data.resize(length);
  engine_file.seekg(0, engine_file.beg);
  engine_file.read(reinterpret_cast<char *>(engine_data.data()), length);
  return engine_data;
}

int softmax(const float (&rst)[10]) {
  float cache = 0;
  int idx = 0;
  for (int i = 0; i < 10; i += 1) {
    if (rst[i] > cache) {
      cache = rst[i];
      idx = i;
    };
  };
  return idx;
}

int main(int argc, char **argv) {
  Logger logger;

  auto runtime = unique_ptr<IRuntime>(createInferRuntime(logger));
  auto plan = load_engine_file("mnist.engine");
  auto engine = shared_ptr<ICudaEngine>(
      runtime->deserializeCudaEngine(plan.data(), plan.size()));
  auto context =
      unique_ptr<IExecutionContext>(engine->createExecutionContext());

  auto idims = engine->getTensorShape("input.1");
  auto odims = engine->getTensorShape("23");
  Dims4 inputDims = {1, idims.d[1], idims.d[2], idims.d[3]};
  Dims2 outputDims = {1, 10};
  context->setInputShape("input.1", inputDims);

  void *buffers[2];
  const int inputIndex = 0;
  const int outputIndex = 1;

  cudaMalloc(&buffers[inputIndex], 1 * 28 * 28 * sizeof(float));
  cudaMalloc(&buffers[outputIndex], 10 * sizeof(float));

  context->setTensorAddress("input.1", buffers[inputIndex]);
  context->setTensorAddress("23", buffers[outputIndex]);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  vector<string> file_names = {"MNIST/images_test/0/mnist_test_3.png",
                               "MNIST/images_test/1/mnist_test_2.png",
                               "MNIST/images_test/2/mnist_test_1.png",
                               "MNIST/images_test/3/mnist_test_18.png",
                               "MNIST/images_test/4/mnist_test_4.png",
                               "MNIST/images_test/5/mnist_test_8.png",
                               "MNIST/images_test/6/mnist_test_11.png",
                               "MNIST/images_test/7/mnist_test_0.png",
                               "MNIST/images_test/8/mnist_test_61.png",
                               "MNIST/images_test/9/mnist_test_7.png"};

  for (auto file_name : file_names) {
    cv::Mat img0;
    img0 = cv::imread(file_name, 0);
    if (img0.empty()) {
      std::cout << "Could not open or find the image" << std::endl;
      return -1;
    }
    cv::Mat img;
    img0.convertTo(img, CV_32F);

    cudaMemcpyAsync(buffers[inputIndex], img.data, 1 * 28 * 28 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    float rst[10];
    cudaMemcpyAsync(&rst, buffers[outputIndex], 1 * 10 * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cout << file_name << " result: " << softmax(rst) << endl;
  }

  cudaStreamDestroy(stream);
  cudaFree(buffers[inputIndex]);
  cudaFree(buffers[outputIndex]);
}
