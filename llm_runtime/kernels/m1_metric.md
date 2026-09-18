[rmsnorm] triton.testing.Benchmark latency
m1-rmsnorm-latency:
shape  PyTorch (latency (ms))  Triton (latency (ms))
0  base                0.102689               0.024417
1  long                0.173677               0.032144

[rope] triton.testing.Benchmark latency
m1-rope-latency:
shape  PyTorch (latency (ms))  Triton (latency (ms))
0  base                0.212241               0.042188
1  long                0.408530               0.070223

[swiglu] triton.testing.Benchmark latency
m1-swiglu-latency:
shape  PyTorch (latency (ms))  Triton (latency (ms))
0  base                0.036139               0.027119
1  long                0.061603               0.048318

op provider shape latency_ms max_abs_error peak_memory_mib peak_memory_pct
rmsnorm torch base 0.0955 0 57.227 0.4660
rmsnorm triton base 0.0171 0.00195312 48.217 0.3926
rmsnorm torch long 0.1616 0 69.230 0.5637
rmsnorm triton long 0.0304 0.00195312 51.217 0.4170
rope torch base 0.2088 0 75.717 0.6165
rope triton base 0.0448 0.000976562 51.217 0.4170
rope torch long 0.4760 0 106.217 0.8649
rope triton long 0.0735 0.00195312 57.217 0.4659
swiglu torch base 0.0366 0 51.217 0.4170
swiglu triton base 0.0295 0.0078125 48.217 0.3926
swiglu torch long 0.0644 0 57.217 0.4659
swiglu triton long 0.0481 0.0078125 51.217 0.4170
