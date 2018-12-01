# In class competition for MSBD 5001

# Runner Up Solution

### Feature Engineering
- Baisc Processing
    + One hot encode `penalty`
    + Drop `id`, `random_state`
    + For `n_jobs`, replace  `-1` to `16`
    + Taylor Series for `n_jobs`

### Model Selection
Models like Linear Regression and Neural Networks suffer from the feature `n_jobs` with multiplying power. Instead of selecting a better model to formula, I change the task from predicting computation time to evaluating performance between machines which means predicting difference or ratio between TA's computing resource and my computing resource.

### Esemble Learning
For final submission, I stack two LightGBM model each predicting  `Time`/`LocalTime`.