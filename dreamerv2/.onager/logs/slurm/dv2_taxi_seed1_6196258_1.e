2022-09-22 02:37:31.315254: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-09-22 02:37:33.085328: E tensorflow/stream_executor/cuda/cuda_blas.cc:2981] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2022-09-22 02:37:38.122531: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /gpfs/runtime/opt/python/3.9.0/lib:/gpfs/runtime/opt/gcc/10.2/lib64:/gpfs/runtime/opt/ffmpeg/3.2.4/lib:/gpfs/runtime/opt/cudnn/8.2.0/lib64:/gpfs/runtime/opt/cuda/11.3.1/cuda/lib64:/gpfs/runtime/opt/cuda/11.3.1/src/lib64:/gpfs/runtime/opt/intel/2017.0/lib/intel64/:/gpfs/runtime/opt/intel/2017.0/mkl/lib/intel64:/gpfs/runtime/opt/java/8u111/jre/lib/amd64
2022-09-22 02:37:38.122722: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /gpfs/runtime/opt/python/3.9.0/lib:/gpfs/runtime/opt/gcc/10.2/lib64:/gpfs/runtime/opt/ffmpeg/3.2.4/lib:/gpfs/runtime/opt/cudnn/8.2.0/lib64:/gpfs/runtime/opt/cuda/11.3.1/cuda/lib64:/gpfs/runtime/opt/cuda/11.3.1/src/lib64:/gpfs/runtime/opt/intel/2017.0/lib/intel64/:/gpfs/runtime/opt/intel/2017.0/mkl/lib/intel64:/gpfs/runtime/opt/java/8u111/jre/lib/amd64
2022-09-22 02:37:38.122736: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
2022-09-22 02:37:48.459903: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-09-22 02:37:50.676146: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1616] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 22291 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3090, pci bus id: 0000:e1:00.0, compute capability: 8.6
/gpfs/data/tserre/ssunda11/IRL/factored_rep/factored-reps/venv/lib/python3.9/site-packages/tensorflow/python/data/ops/structured_function.py:264: UserWarning: Even though the `tf.config.experimental_run_functions_eagerly` option is set, this option does not apply to tf.data functions. To force eager execution of tf.data functions, please use `tf.data.experimental.enable_debug_mode()`.
  warnings.warn(
2022-09-22 02:37:52.167766: I tensorflow/stream_executor/cuda/cuda_dnn.cc:384] Loaded cuDNN version 8200
2022-09-22 02:37:54.415197: I tensorflow/stream_executor/cuda/cuda_blas.cc:1614] TensorFloat-32 will be used for the matrix multiplication. This will only be logged once.
