# Optimizers

Optimizers update model parameters based on accumulated gradients.

Typically, the training flow looks like:

1. Forward pass
2. Compute loss
3. Backward pass (`loss.backward()`)
4. Optimizer step (`opt.step()`)
5. Clear gradients (`opt.zero_grad()`)

Exact method names and behavior are documented in each optimizer’s docstring.

---

::: keydnn.Adam
    options:
      show_root_heading: true

::: keydnn.SGD
    options:
      show_root_heading: true

---

## Notes

- Optimizers usually take an iterable of parameters (often from a model).
- If KeyDNN supports weight decay, momentum, or learning-rate schedules, document those options
  in the optimizer docstrings.
- If gradients can be accumulated across steps, clarify whether `zero_grad()` is required.
