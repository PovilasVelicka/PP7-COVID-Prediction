# COVID-19 Classifier from Chest X-ray Images 🫁

## ⚠️ Python Version Requirement

Make sure you're using **Python 3.10 or lower**.  
TensorFlow may not work correctly with Python 3.11+ without recompilation.

## 🔗 Dataset Source

Paper: [https://pmc.ncbi.nlm.nih.gov/articles/PMC7372265/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7372265/)  
Dataset: [https://github.com/PovilasVelicka/Covid.git](https://github.com/PovilasVelicka/Covid.git)

1. Downloaded image archive and `metadata.csv`, placed them in `dataset/`
2. Analyzed structure: 584 `Pneumonia/Viral/COVID-19` cases, only 22 `No Finding` cases
3. Decided to treat all COVID cases as `covid`, and others (excluding `todo`, `Unknown`, `CT`) as `non_covid`
4. Filtered only `X-ray` modality and projections `PA`, `AP`, added `AP Supine` when lacking
5. Created final dataset with two classes: `covid` and `non_covid`
6. Applied stratified split into `train`, `val`, `test`
7. Copied images into: `dataset/split/{train,val,test}/{covid,non_covid}`

## 🧠 Initial Model Training

- Initially used `ResNet50` pretrained on ImageNet
- Model head: GlobalAveragePooling + Dense layers + `sigmoid` activation
- Trained with `ImageDataGenerator` and saved in `.h5` format

### 🛠 Issue Faced:

The model initially predicted only `non_covid` because:
- Class label for `covid` was `0`, and `non_covid` was `1`
- During classification report analysis, the labels were interpreted in reverse

**Solution:** manually specified class order:  
```python
classes = ["non_covid", "covid"]
```

## 🔁 Model Update: Lightweight Architecture

To improve training efficiency and deployment readiness,  
`ResNet50` was replaced with **MobileNetV2**, a lightweight and fast architecture.

- `MobileNetV2` is also pretrained on ImageNet
- Lower 2/3 of layers are frozen
- Classification head: GAP → Dense → Dropout → Output

## ⚙️ Training Details

- Augmentations include: rotation, shift, zoom, flip, brightness
- `class_weight` used to balance the classes
- Callbacks: `ModelCheckpoint`, `EarlyStopping`, `ReduceLROnPlateau`
- Training stops early if validation loss stagnates

## 📉 Performance Before Fine-Tuning (15 Epochs)

| Class       | Precision | Recall | F1-score |
|-------------|-----------|--------|----------|
| `covid`     | 0.51      | 0.95   | 0.67     |
| `non_covid` | 0.95      | 0.51   | 0.67     |

**Interpretation:**
- The model was **highly sensitive to COVID** (high recall)
- But it often predicted `covid` even for `non_covid` (low precision)

## 🔁 Fine-tuning (continued training)

Fine-tuning was applied using `MobileNetV2`:

- Reduced learning rate to `1e-5`
- Used class weights and callbacks
- Training stopped on early epochs based on validation loss

### 📊 Final Results After Fine-tuning:

```
🔍 Classification Report:
              precision    recall  f1-score   support
   non_covid       0.50      0.89      0.64        19
       covid       0.90      0.51      0.65        35

✅ Sensitivity (Recall COVID): 0.895  
✅ Specificity (True Negative Rate non-COVID): 0.514
```

## 🧪 How to Test the Model

1. Make sure the directory contains:
   - `app.py` — inference script
   - `covid_classifier_mobilenetv2.keras` — trained model
   - `class_indices.json` — class mapping
   - `covid.png` — example of a COVID image
   - `non_covid.png` — example of a non-COVID image

2. Run predictions:

```bash
python app.py covid.png
python app.py non_covid.png
```

3. Sample Output:

```
🔎 Result:
  ➤ Class      : covid
  ➤ Confidence : 92.3%
```

## 📦 Summary

- Complete pipeline from dataset parsing to image prediction
- Switched to efficient architecture (`MobileNetV2`) for speed and deployability
- Model achieves high recall on COVID class, balanced with class weighting
- Can be deployed or further fine-tuned on new X-ray data
