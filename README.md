# ContrastiveCTC
Contrastive loss weight with CTC decoding path, implemented by CUDA

# How to Use

To install the CUDA extension, the paths in `setup.py` should be modified:

```python
...
"sources": [...], # Change the relative paths here
```

Then just run the commands

```bash
python setup.py install
```

If everything goes successfully, the function can be called in Python scripts:

```python
# Import the lib
import ctc_mask_cuda

# Invoke the provided interface
# The inputs are same with `torch.nn.functional.ctc_loss`
# The outputs are (losses (no gradient), (alpahs, betas), (masks, alpha_masks, beta_masks)) for debug purposes
res = ctc_mask_cuda.forward(
  log_probs,
  targets,
  input_lengths,
  target_lengths,
)
```

NOTE: the token `blank` is zero by default.

# TODO

The weighted loss are still in progress but the basic function is provided (such as get the decoding paths by invoking `forward`), so there may be some bugs. And the lib will be developped in the future, where the whole loss weighted strategies, the performance and the training loop will be released.
