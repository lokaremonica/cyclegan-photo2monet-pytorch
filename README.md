# 🎨 Unpaired Image-to-Image Translation with CycleGAN (Photo ↔ Monet)

This project implements a **CycleGAN** using PyTorch for unpaired image-to-image translation between real-world photographs and Monet-style paintings. The model is trained on separate image folders without aligned pairs and successfully learns two-way transformation: Photo → Monet and Monet → Photo.

---

## 📂 Dataset Access (Google Drive)

This project uses an unpaired dataset of **Photos and Monet-style paintings** for CycleGAN training and testing.

📥 **Download Dataset**  
The full dataset (Photos & Monets) is hosted on Google Drive:

🔗 [Google Drive – Photos and Monets Dataset](https://drive.google.com/drive/folders/1KkSYKqdmO1GE6gSP38UCsOrYZ7QUGj-x?usp=sharing)

📁 Folder Structure:

```bash
Photo2MonetDataset/
├── trainA/       # Real Photos
├── trainB/       # Monet Paintings
├── testA/        # Real Photos (test)
└── testB/        # Monet Paintings (test)
```
---

## 🧾 Dataset Details

| Domain A    | Domain B        |
| ----------- | --------------- |
| Real Photos | Monet Paintings |

* Images are **unpaired**, loaded via a custom `Photo2MonetDataset`
* Each image is **resized to 256x256**
* Applied transforms: Resize, RandomHorizontalFlip, Normalize

---

## 🧠 Model Architecture

### 🔹 Generator: `GeneratorResNet`

* Built with reflection padding + 4 residual blocks
* Downsampling (stride-2 convolutions)
* Upsampling via `nn.Upsample` + `Conv2d`
* Output layer with `Tanh`

### 🔹 Discriminator: PatchGAN-style

* 5 Conv2d layers with increasing filters
* Final output: 1x30x30 map (real/fake prediction)

---

## ⚙️ Training Details

| Component       | Value                              |
| --------------- | ---------------------------------- |
| Epochs          | 10                                 |
| Batch Size      | 4 (train), 5 (test)                |
| Optimizer       | Adam (lr=0.0002, β1=0.5, β2=0.999) |
| Losses          | GAN + Cycle + Identity             |
| Mixed Precision | ✅ `GradScaler`, `autocast`         |

```python
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()
```

---

## 📉 Training Losses

| Epoch | Generator Loss | D\_A Loss | D\_B Loss |
| ----- | -------------- | --------- | --------- |
| 1     | 8.1260         | 0.1875    | 0.1925    |
| 5     | 5.5873         | 0.2345    | 0.1727    |
| 10    | 5.6611         | 0.0634    | 0.1691    |

> Loss curves show stable convergence and balance between generator and discriminator training.

---

## 🖼️ Visual Results

### 📷 Photo → Monet

Generated artistic images that retain structural content and adopt Monet’s painting style.

### 🖼️ Monet → Photo

Restored near-photorealistic outputs while maintaining the core texture and layout.

---

## 🧠 Key Learnings

* CycleGAN can learn complex mappings between unpaired visual domains
* Residual blocks improve detail preservation and stylization
* Mixed precision training accelerates convergence on GPU
* Stable loss curves indicate well-balanced adversarial training

---

## 🏁 Conclusion

This project shows the power of CycleGAN for **style transfer without paired datasets**, enabling conversion of real photos into Monet-style paintings and vice versa.

---
