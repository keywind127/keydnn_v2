# Losses

KeyDNN exposes common loss functions as **functions** in the public API.

Loss functions typically accept prediction tensors and target tensors and return a scalar `Tensor`
(or a reduced tensor, depending on configuration).

---

## Categorical Cross Entropy Loss (`cce_loss`)

::: keydnn.cce_loss
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Mean Squared Error Loss (`mse_loss`)

::: keydnn.mse_loss
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Sum of Squared Errors (`sse_loss`)

::: keydnn.sse_loss
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Binary Cross Entropy Loss (`bce_loss`)

::: keydnn.bce_loss
    options:
      show_root_heading: false
      show_root_toc_entry: false

---

## Notes

- Make sure your targets match the expected format (e.g., class indices vs one-hot vectors).
- If a loss supports logits vs probabilities, document it clearly in the docstring.
- If reductions are supported (e.g., `mean`/`sum`), prefer documenting the default explicitly.
