## Real and Fake Image Detection in the Terminal using a Pretrained Model

To test a trained model, execute the following script:

```bash
python3 fake_face_detection/demo/demo_without_kamera/live_demo.py
```

### Note:

Please make the following adjustments in the `live_demo.py` script:

- Enter the correct model path in `MODEL_PATH`.

- Enter the path to either real or fake images in the following format:
  ```python
  image_path = f"data/fake_faces/{i + 1}_fake_faces.jpg"
  ```