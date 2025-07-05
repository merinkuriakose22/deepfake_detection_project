🎥 Deepfake Detection Using Frame Extraction and CNN
This project is a simple and effective deepfake detection pipeline using:

Frame extraction from videos

Image comparison using MSE and SSIM

CNN-based image classification

✅ Dataset: Small-scale Deepfake Forgery Video Dataset (SDFVD)

✅ Tech Stack: Python, TensorFlow, OpenCV, scikit-image

deepfake_detection_project/
│
├── SDFVD Small-scale Deepfake Forgery Video Dataset/
│   └── SDFVD/
│       ├── real/         # Real videos
│       └── fake/         # Fake videos
│
├── frames/
│   ├── real_frames/      # Extracted real video frames
│   └── fake_frames/      # Extracted fake video frames
│
├── cnn_trainer.py        # CNN model training script
├── frame_extractor.py    # Frame extraction module
├── image_comparator.py   # MSE and SSIM comparison module
├── main.py               # Frame extraction and comparison driver
├── plot_metrics.py       # Plotting MSE and SSIM trends
├── predict.py            # Predict real/fake on new frames
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

