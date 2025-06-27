## Contrastive CTC

### Quick Start!

**1. Compile the operator**

```sh
python cmake_presets.py
cmake --preset conda-debug && cmake --build --preset conda-debug
./conda-debug/main
```

If you can not find your `cmake`, replace `cmake` with `$(which cmake)`, it will use the `cmake` in your conda environment.


**2. Use the extension to get the decoding map**

```python
import ctc_decode_module

# Invoke
# The parameters are same with torch.nn.functional.ctc_loss
loss, decode_map = ctc_decode_module.ctc_decode_fn(log_probs, targets, input_lengths, target_lengths)
```


**3. Check our minimum runable demo**

We provide a minimum demo to show the usage, check the script `demo.py` and run it, you may get the result:

```sh
Loss: 156.6901397705078
Decode Map shape: torch.Size([4, 50, 35])
```