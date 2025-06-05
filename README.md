# Deep Learning-Based Speaker Verification System  
*MFCC + CNN Embeddings*

Note: Currently, I am working on the second task and will update the Jupyter notebook on the same repository as soon as possible, if not by this evening. I am putting in my best efforts to complete it promptly.
This README reflects my understanding of the First task so far. Thanks :)

update- I've attempted the second task as well. Project report corresponding to `Task 2` is "TASK2_Report_License_Plate_Recognition" 
Dataset used in `Task 2: ` `https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset` 


---

## 1  Objective
Build a lightweight **speaker-verification** pipeline that decides whether two audio clips are spoken by the **same person**.  The task is framed as binary verification (“match / no-match”) rather than multi-class identification, making the system applicable to authentication scenarios (voice unlock, call-center security, etc.).

---

## 2  Dataset
We use a **3-speaker subset** of Google’s *Speech Commands v0.02* (clean 1 s, 16 kHz WAVs).  
For each speaker we select several utterances of the keyword **“zero”**:

- **jackson:** 500 clips (~500 seconds)  
- **nicolas:** 500 clips (~500 seconds)  
- **theo:** 500 clips (~500 seconds)

LINK: <https://github.com/Jakobovski/free-spoken-digit-dataset> 

> Although originally intended for keyword spotting, the dataset is ideal for a controlled proof-of-concept: clips are noise-free, uniformly sampled, and tagged with speaker IDs.

---

## 3  Feature Extraction — MFCC
Each clip is transformed into **Mel-Frequency Cepstral Coefficients**:

* `40` coefficients  
* 25 ms window, 10 ms hop  
* Output shape per clip → **(40 × 11) ≈ 0.11 s of frames**

We zero-pad / truncate so every sample fits a fixed tensor size **(40, 11, 1)**, well suited to 2-D CNN kernels.

---

## 4  Model Architecture
We use a lightweight Convolutional Neural Network (CNN) to encode MFCCs into 128-dimensional speaker embeddings. The model architecture is as follows:

Input: MFCC (40 × 11 × 1)
**Model Architecture Overview:**

- **Input:** MFCC spectrogram of shape **(40 × 11 × 1)**
- **Conv2D Layer 1:** 32 filters, 3×3 kernel, ReLU activation
- **MaxPooling Layer 1:** 2×2 pool size
- **Conv2D Layer 2:** 64 filters, 3×3 kernel, ReLU activation
- **MaxPooling Layer 2:** 2×2 pool size
- **Flatten Layer**
- **Dense Layer:** 128 units, ReLU activation → **Speaker Embedding (128-D)**


Input Layer: Takes in 40 Mel-frequency cepstral coefficients over 11 time frames.

Convolutional Layers: Two stacked 2D convolutional blocks extract local temporal-frequency features.

Flatten Layer: Converts the 2D feature map into a 1D vector.

Dense Layer: Outputs a compact 128-D embedding that represents the speaker's voiceprint.

This embedding is later used for cosine similarity-based speaker verification.

During training, a softmax classification head (3 units) is appended to the embedding layer and supervised with speaker IDs. Once trained, we discard the softmax head and retain the embedding model for inference.


---

## 5  Verification Pipeline
1. **Embedding extraction** – run each waveform through the CNN to get a 128-D vector.  
2. **Cosine similarity** – measure similarity between two embeddings.  
3. **Thresholding** – classify “same speaker” if similarity ≥ τ, else “different”.  

The optimal τ is chosen via the **Equal Error Rate (EER)** point on the ROC curve.

---

## 6  Evaluation
Balanced pair list: 750 “same-speaker” + 750 “different-speaker” pairs (cached embeddings, GPU inference).


- **Accuracy:** 0.917
- **Precision:** 0.917
- **Recall:** 0.916
- **F1-score:** 0.917
- **AUC(ROC):** 0.977
- **Flatten Layer**
- **EER:** 0.083 at **τ = 0.781**
 

### Visuals
* **Confusion matrix** @ EER threshold  
* **ROC curve** with EER point highlighted  
* **3-D t-SNE** scatter of embeddings → clear clusters by speaker, validating separability.

---

## 7  Strengths
* **Low compute footprint** – < 0.5 MB model, real-time capable on CPU or mobile hardware.  
* **High separability** demonstrated by AUC ≈ 0.98.  
* **Interpretability** – t-SNE plots and cosine threshold give intuitive insight.

---

## 8  Limitations
* **Limited speaker diversity** – only 3 speakers; generalisation to unseen voices is untested.  
* **Clean-speech bias** – no background noise or channel variability, so robustness in real environments is unknown.  
* **Closed-set assumption** – verification pairs are drawn only from known speakers.

---

## 9  Potential Improvements
* **Expand dataset** – scale to dozens of speakers (e.g., VoxCeleb mini split).  
* **Data augmentation** – add noise, reverberation, tempo & pitch shifts.  
* **Siamese / Triplet loss** – optimise embeddings directly for verification instead of indirect soft-max supervision.  
* **Pre-trained backbones** – swap custom CNN for `ECAPA-TDNN`, `Wav2Vec2`, or `HuBERT` to reach sub-5 % EER.  
* **Voice Activity Detection (VAD)** – strip silence to improve embedding quality and latency.

---

## 10  Conclusion
This project demonstrates that a **simple MFCC + CNN pipeline** can achieve **> 91 % accuracy** and **EER ≈ 8 %** on a lightweight dataset.  The approach balances clarity, speed, and effectiveness, making it an excellent foundation for more advanced, large-scale speaker verification systems.
