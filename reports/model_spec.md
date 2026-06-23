# Smoke Test Results

## Objective

Verify that the exported ResNet18 model can successfully perform inference on unseen validation images.

## Model Specification

## Model Information

- Architecture: ResNet18
- Framework: PyTorch
- Export Format: TorchScript (`resnet18_scripted.pt`)

## Input Specification

- Image Size: 224 × 224 pixels
- Channels: RGB (3 channels)

## Normalization

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

## Test Commands and Results

| Image Path | Actual Class | Predicted Class | Confidence |
|------------|--------------|----------------|------------|
| data/val/healthy/test2.JPG | healthy | healthy | 99.xx% |
| data/val/leaf_mold/test3.JPG | leaf_mold | leaf_mold | 99.86% |
| data/val/early_blight/test4.JPG | early_blight | early_blight | 99.99% |

## Summary

* Total test images: 3
* Correct predictions: 3
* Smoke test accuracy: 100%

## Conclusion

The exported ResNet18 model successfully performed inference on all three held-out validation images. The CLI prediction script correctly returned the predicted class and confidence score for each image. All three samples were classified correctly with high confidence, confirming that the model export (`resnet18_scripted.pt`) and inference pipeline are functioning as expected.