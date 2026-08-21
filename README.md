# Plant Disease Detection for Ghanaian Smallholder Farmers — README

**Group 9 — Introduction to Artificial Intelligence Final Project**

This README walks through every code section of `Plant_Disease_Detection.ipynb` in the order it's meant to be run, explains what each piece does and why, and interprets the results the notebook produced.

---

## Project Summary

The notebook trains a Convolutional Neural Network (CNN) to classify leaf images of four crops important to Ghanaian agriculture — **Tomato, Corn/Maize, Potato, and Pepper** — into 19 healthy/disease classes, using the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset). The goal is a proof-of-concept tool that could eventually let a farmer photograph a leaf and get an instant disease diagnosis, reducing reliance on scarce agricultural extension officers.

---

## 1. Setup & Data Download

**What the code does:**

```python
!pip install -q kaggle
```
Installs Kaggle's command-line tool inside Colab so the notebook can talk to Kaggle's API directly.

```python
from google.colab import files
files.upload()  # upload your kaggle.json here
```
Opens a file-picker so you can upload your personal `kaggle.json` API credentials (downloaded from your Kaggle account settings). This file authenticates the download request below — without it, Kaggle's API refuses all requests.

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```
Moves the credentials to the folder Kaggle's CLI expects and locks the file down to owner-only read/write (`600`). Kaggle's CLI explicitly rejects credential files with looser permissions than this — it's a security check on their end, not a stylistic choice.

```python
!kaggle datasets download -d abdallahalidev/plantvillage-dataset
!unzip -q plantvillage-dataset.zip -d plantvillage_data
```
Downloads the full PlantVillage dataset (tens of thousands of leaf images across many crops) and silently unzips it into `plantvillage_data/`.

**Why it matters:** this is a one-time, per-session setup — every teammate running the notebook fresh needs their own `kaggle.json`, since Kaggle credentials aren't something that should be hardcoded or shared in the notebook itself.

---

## 2. Filter to Study Crops & Split Data

**What the code does:**

```python
source_dir = 'plantvillage_data/plantvillage dataset/color'
Study_crops = ['Tomato', 'Corn_(maize)', 'Potato', 'Pepper']

for folder_name in os.listdir(source_dir):
    if any(crop in folder_name for crop in Study_crops):
        ...
        shutil.copytree(src_path, dst_path)
```
PlantVillage's class folders are named things like `Tomato___Late_blight` or `Corn_(maize)___healthy` — crop and disease encoded together in the folder name. This loop checks every folder name for a substring match against the four target crops and copies only the matching folders into a new `Crop_data/` directory. Everything else (grape, apple, strawberry, etc.) is left out entirely.

**Why it matters:** PlantVillage covers ~14 crops. Restricting to four keeps training time manageable on Colab's free-tier GPU and keeps the project scoped to crops that are actually economically significant in Ghana, per the problem statement.

```python
!pip install -q split-folders
import splitfolders
splitfolders.ratio("Crop_data", output="data_split", seed=42, ratio=(0.8, 0.1, 0.1))
```
Uses the `split-folders` library to automatically split each class folder into `train` (80%), `val` (10%), and `test` (10%) subfolders, preserving the class structure. `seed=42` makes the split reproducible — every teammate who runs this cell gets the *identical* split, which matters for fair comparison of results.

**Why an 80/10/10 split:** training needs the bulk of the data to learn robust features; validation needs enough data to give a meaningful signal on generalization during training; test needs to be large enough for a statistically meaningful final accuracy number without being so large it starves training.

---

## 3. Data Generators (with Augmentation)

**What the code does:**

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, zoom_range=0.15, horizontal_flip=True,
    brightness_range=[0.7, 1.3], fill_mode='nearest'
)
val_test_datagen = ImageDataGenerator(rescale=1./255)
```
`rescale=1./255` converts pixel values from the raw 0–255 range into 0–1, which is standard practice for feeding images into a neural network (keeps gradients well-scaled during training).

The rest of the arguments on `train_datagen` are **data augmentation** — applied only to training images:
- `rotation_range`, `width_shift_range`, `height_shift_range`, `shear_range`, `zoom_range` — randomly rotate/shift/skew/zoom each image slightly on every pass.
- `horizontal_flip` — randomly mirrors images left-right.
- `brightness_range=[0.7, 1.3]` — randomly darkens or brightens the image by up to 30%, which is specifically meant to help the model tolerate the inconsistent lighting of real smartphone photos, since PlantVillage's own images are studio-consistent.

`val_test_datagen` deliberately has **no augmentation** beyond rescaling — validation and test accuracy need to reflect performance on realistic, unaltered images, or the numbers stop being trustworthy.

```python
train_generator = train_datagen.flow_from_directory('data_split/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
```
`flow_from_directory` streams images directly off disk in batches (rather than loading the entire dataset into memory at once), resizing each to `128×128` and grouping them into batches of 32. `class_mode='categorical'` one-hot encodes the labels for multi-class classification. The test generator additionally sets `shuffle=False` so that predictions stay index-aligned with true labels — this is required for building an accurate confusion matrix later; shuffling would scramble that alignment.

---

## 4. Model Definition & Training

**What the code does (Cell 22, currently commented out — training already ran once and the result was saved):**

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```
A minimal two-block CNN:
- **Conv2D(32) + MaxPooling2D** — the first block learns simple, low-level visual features: edges, color blobs, texture gradients. Pooling shrinks the spatial size while keeping the strongest signals, which both reduces compute and adds a bit of translation-invariance.
- **Conv2D(64) + MaxPooling2D** — the second block combines those low-level features into more complex patterns — the kind of thing that starts to resemble "leaf spot" or "blight lesion" shapes.
- **Flatten + Dense(128)** — collapses the 2D feature maps into a single vector and lets the network combine spatial features into higher-level, non-spatial representations.
- **Dense(num_classes, softmax)** — the output layer, producing a probability distribution over all 19 crop/disease classes.

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
Adam is a standard, well-behaved optimizer for this kind of task. `categorical_crossentropy` is the correct loss function for multi-class, single-label classification with one-hot labels.

```python
history = model.fit(train_generator, epochs=20, validation_data=val_generator)
```
Trains for 20 epochs, checking validation performance after every epoch so the team could watch for overfitting (training accuracy climbing while validation accuracy stalls or drops) as training progressed.

**Why this cell is commented out now:** training is the slowest, most resource-intensive step. Once a model has been trained once and saved (next section), there's no need to burn GPU time retraining from scratch every time the notebook is reopened — the cell is left in place (commented) as documentation of exactly how the saved model was produced.

---

## Saving & Reloading the Model

```python
from google.colab import drive
drive.mount('/content/drive')
model.save('/content/drive/MyDrive/plant_disease_model.keras')
```
Colab's local disk is wiped when the runtime disconnects, so the trained model is saved to Google Drive in Keras's native `.keras` format for persistence across sessions.

```python
model = load_model('/content/drive/MyDrive/plant_disease_model.keras')
```
Reloads that saved model in any future session — this is the cell that actually needs to run before evaluation/demo cells will work, since the training cell above is commented out.

---

## 5. Evaluation

**What the code does:**

```python
test_loss, test_acc = model.evaluate(test_generator)
```
Runs the model over the entire (unaugmented, unshuffled) test set and reports overall loss and accuracy — a single honest number for how the model performs on data it never saw during training.

```python
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
```
`model.predict` returns a softmax probability vector per image; `argmax` collapses that to a single "most likely class" prediction. `test_generator.classes` gives the true labels in the same (unshuffled) order, so `y_pred` and `y_true` line up index-for-index.

```python
print(classification_report(y_true, y_pred, target_names=class_labels))
```
Prints **precision, recall, and F1-score per class** — this is far more informative than overall accuracy alone, because accuracy can hide the fact that a model is very good on some classes and quite bad on others (especially with imbalanced class sizes, which is the case here).

```python
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ...)
```
Builds a 19×19 grid where rows are true classes and columns are predicted classes. The diagonal is correct predictions; anything off the diagonal shows exactly which classes get mixed up with which — this is the tool that let us pinpoint *specific* confusion patterns rather than just knowing "accuracy is 92%."

---

## Interpreting the Results

### Headline numbers
- **Test accuracy: 92.53%** (loss 0.2306) on 2,679 held-out test images across 19 classes.
- **Macro-avg F1: 0.91**, **weighted-avg F1: 0.93** — the gap between these two tells you performance is a bit weaker on rarer classes than the 92.53% headline suggests, since the weighted average is pulled up by large classes like *Tomato_Yellow_Leaf_Curl_Virus* (537 test images).

### What the model gets right
Classes with very distinctive visual symptoms are essentially solved: **Corn Common Rust** (precision/recall 1.00/1.00), **Corn healthy** (1.00/0.99), **Tomato Bacterial Spot** (0.99/0.97), **Tomato Yellow Leaf Curl Virus** (0.99/0.98). These all have strong, consistent visual signatures (pustules, mottled yellowing) that a CNN can learn cleanly from images.

### Where it struggles — and why that matters more than the accuracy number
Two patterns stand out in the confusion matrix:

1. **Corn Cercospora leaf spot ↔ Corn Northern Leaf Blight** are confused in both directions (16 Northern Leaf Blight images predicted as Cercospora; 5 Cercospora images predicted as Northern Leaf Blight). Both diseases produce similar elongated gray/brown lesions on maize leaves — a visually plausible mix-up, not a random error.

2. **The model over-predicts "Tomato healthy."** *Tomato healthy* has perfect recall (1.00 — every actual healthy image was caught) but only 0.74 precision, meaning many diseased images get misclassified as healthy. The clearest case is **Tomato Target Spot**, whose recall is just 0.69 — 31 of its 141 test images were wrongly called "healthy," and another 9 were called Septoria leaf spot.

This second finding is the most important one in the whole evaluation: **a false "healthy" prediction is a worse failure than mixing up two diseases**, because it's the exact scenario where a farmer using this tool would be told (incorrectly) that no treatment is needed. Overall accuracy hides this bias completely — it's only visible by reading recall per class and the confusion matrix directly.

### The honest caveat
Every number above was measured on PlantVillage's studio-style images — uniform backgrounds, consistent lighting, no real-world clutter. The notebook's own Live Demo section explicitly flags that accuracy and confidence will likely drop on real field photos, but that drop was never actually measured against a real-world image set. **92.53% is the model's best-case performance, not a guarantee of field performance.**

---

## 6. Demo Samples

```python
for class_name in os.listdir(test_dir):
    ...
    chosen = random.choice(images)
    shutil.copy(src, dst)  # renamed ClassName__filename.jpg
```
Grabs one random image per class from the test set and copies it into `demo_samples/`, with the true class name baked into the filename. This is purely a convenience step so the team can quickly grab labeled example images for a presentation without manually digging through 19 test folders.

```python
def demo_from_test_set(model, class_labels, ...):
    ...
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    plt.imshow(img); plt.title(f"True: ... Predicted: ... Confidence: ...")
```
A repeatable sanity-check helper: picks one random test image, runs it through the model, and displays it with true label, predicted label, and confidence side by side. Useful for spot-checking individual predictions beyond the aggregate metrics.

---

## 7. Live Demo

```python
def quick_demo(model, class_labels, ...):
    uploaded = files.upload()
    ...
```
Lets anyone upload a single image directly in Colab and see the model's prediction and confidence — the simplest possible way to test the model on a brand-new (non-dataset) photo.

```python
def launch_gradio_demo(model, class_labels, ...):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr
    ...
    demo = gr.Interface(fn=predict_disease, inputs=gr.Image(type="pil"), outputs=gr.Label(num_top_classes=3), ...)
    demo.launch(share=True)
```
Installs and launches [Gradio](https://gradio.app/), a library that spins up an interactive web UI with drag-and-drop image upload. `outputs=gr.Label(num_top_classes=3)` shows the **top-3** predicted classes as a confidence bar chart rather than just the single top guess — more informative for a live audience, and it also surfaces cases where the model is genuinely torn between two plausible diseases. `share=True` generates a temporary public URL so the demo can be shown from any device, which is well-suited to a final presentation.

---

## Quick Reference: Files Produced by the Notebook

| Path | What it is |
|---|---|
| `plantvillage_data/` | Raw, unfiltered PlantVillage dataset (all crops) |
| `Crop_data/` | Filtered to Tomato/Corn/Potato/Pepper only |
| `data_split/{train,val,test}/` | 80/10/10 split of `Crop_data/`, used directly by the generators |
| `/content/drive/MyDrive/plant_disease_model.keras` | The trained model, saved to Google Drive |
| `demo_samples/` | One labeled example image per class, for quick demoing |
