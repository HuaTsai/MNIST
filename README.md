# MNIST

Reference: https://github.com/cyberyang123/Learning-TensorRT

1. Train the model `python s1_train.py`
2. Export onnx file `python s2_export.py`
    - Optional: simplify model `python -m onnxsim mnist.onnx mnist_sim.onnx` (no effect in this example)
3. Generate engine
    - By compile c++ code
        - Build: `nvcc s3_build.cu -lnvinfer -lnvonnxparser`
            - Modify environment variable `CPATH` and `LIBRARY_PATH` for `nvcc` to see the TensorRT library
            - Or specify include path by `-I` and library path by `-L`
        - Another option: write `CMakeLists.txt`
    - By `trtexec`
        - Run: `trtexec --onnx=mnist.onnx --workspace=1 --fp16 --saveEngine=mnist.engine`
4. Inference: `nvcc s4_infer.cu -I/usr/include/opencv4 -lnvinfer -lopencv_core -lopencv_imgcodecs`

