---
layout: default
title: "Lights Under Attack: Stress-Testing Noise-Coded Illumination"
---

# **Lights Under Attack: Stress-Testing Noise-Coded Illumination**

*A concise, descriptive title for your project.*

![Project Banner](./assets/img/banner-placeholder.png)  
<sub>*(Optional: Replace with a conceptual figure or meaningful image.)*</sub>

---

## 👥 **Team**

- Maxwell Jung ([email](maxwelljung@ucla.edu), [GitHub](https://github.com/MaxwellJung))  
- Wentao Chen ([email](wentac4@ucla.edu ), [GitHub](https://github.com/wentac4))  
- Steve Zang ([email](zangbruin007@ucla.edu), [GitHub](https://github.com/SteveZ-Cal))  

---

## 📝 **Abstract**

Noise-Coded Illumination (NCI) is recognized for its potential to provide robust forensic authentication for video footage. Existing research has demonstrated its capability to embed and recover temporal watermarks, creating a powerful asymmetry against manipulators. However, there remains limited exploration into the resilience of NCI against informed adversarial attacks, where an attacker with knowledge of the system attempts to bypass detection. This project aims to assess the security of NCI by evaluating several attack strategies under realistic conditions. Our work will implement a baseline NCI pipeline and systematically test adversarial bypass methods, analyzing their success and proposing potential countermeasures to guide the secure deployment of this promising technology.

---

## 📑 **Slides**

- [Midterm Checkpoint Slides](https://docs.google.com/presentation/d/1JbTUeoli6I7b-AFx3gMX-jX8nmnHsw5_VxXokDmLpW4/edit?usp=sharing)  
- [Final Presentation Slides](https://docs.google.com/presentation/d/1eZMDApG3otnrJUzfvpvx2tNXskiX9aYgf0qJhQyEvnM/edit?usp=sharing)
- [Google Drive](https://drive.google.com/drive/folders/17nz-i6D9IX33ADJJrDn0S3bhEgEBukR5?usp=drive_link)

---

## 🎛️ **Media**

- [38_edited_sampling_mult.mp4](https://drive.google.com/file/d/1c9GZ5Hwqy3Y4azfgi7JGC7BIhy3PX_Uh/view?usp=drive_link)

- [38_edited_sampling_mult_r_estimate.mp4](https://drive.google.com/file/d/12FuLq4AwDjDIo_oq-uR6NVYgyKB8aj5g/view?usp=drive_link)

- [71_edited_sampling_mult.mp4](https://drive.google.com/file/d/1hTYSCr0sIotjzVKb_edzKM9ds5cwCqVf/view?usp=drive_link)

- [71_edited_sampling_mult_r_estimate.mp4](https://drive.google.com/file/d/10VK6CmIZM6dbS-KHPcc2woX5X1kfQxee/view?usp=drive_link)

- [Other media](https://drive.google.com/drive/folders/17nz-i6D9IX33ADJJrDn0S3bhEgEBukR5?usp=drive_link)

---

# **1. Introduction**  
Producing realistic looking fake videos have gotten easier due to advancements in video editing tools and generative AI. Malicious actors can use these videos to spread disinformation with catastrophic consequences; as a result, verifying the authenticity of videos is a vital and challenging task. Noise-Coded Illumination (NCI) proposed by Michael, Hao, Belongie, and Davis is one method of watermarking a video using a special light source illuminating the scene. Modification of a video captured under NCI can be easily detected by correlating the signal from the light source with the signal from the video. The authors claim that NCI can effectively detect pixel manipulation, video cuts, video acceleration/deceleration, and other common forms of video editing. In this report, we attempt to test the limits of NCI as informed adversaries. We demonstrate video editing strategies that can reliably bypass NCI’s watermarking.

### **1.1 Motivation & Objective**  
NCI is a promising standard for watermarking videos as it requires minimal modification to video production setup. Any method that can bypass this watermark is fatal. This project highlights NCI’s weaknesses to guide future iterations.

### **1.2 State of the Art & Its Limitations**  
Authors briefly explore possible attacks against NCI, but they do not cover them in detail. They identified manipulations that change the reflectance in a scene without changing its geometry or remapping time could potentially bypass NCI.

### **1.3 Novelty & Rationale**  
We demonstrate three techniques that modify a video while preserving the NCI watermark.

1. Estimate c from y alone and evaluate how close it is to the real c
2. Multiply the region of y by some constant alpha and see if the modification can be detected in the alignment matrix or code image.
3. Replace a region of y using another different region of y and see if the modification can be detected in the alignment matrix or code image.

### **1.4 Potential Impact**  
If successful, the project will reveal the fundamental weakness in the design of NCI, and improve upon the current limitations of NCI, which prompt the development of more robust watermarking schemes, before it gets real-life deployment. 

### **1.5 Challenges**  
Technical Challenges: 

1. staging light setup
2. reproduce NCI pipeline

Methodological Challenges:

1. come up with attack methods
2. verify those attack works

### **1.6 Metrics of Success**  
1. Percent of the edited videos that bypass NCI detection: measured as the proportion of successfully attacked videos that remain undetected by the NCI verification algorithm.
2. Stealthiness of the attack: perceptual quality of attacked videos compared to originals, and human perceptual studies.
3. Time and resources required to execute the attack: computational cost (in seconds) and hardware/software resources needed to generate adversarial edits.

---

# **2. Related Work**  
Our work seeks to identify potential vulnerabilities in the recent light-watermarking technique introduced by Michael et al. [1]. As new techniques emerge for watermarking recorded video content along efforts to prevent or at the very least allow for easy detection of attempts to manipulate recorded videos, before such techniques can be confidently and reliably deployed in high-stakes scenarios, their robustness or potential vulnerability to various attacks must be comprehensively verified.

One such family of techniques are singular value decomposition (SVD)-based watermarking schemes, such as the method proposed by Sathya and Ramakrishnan [2]. In this method, keyframes are selected based on the Fibonacci sequence, where the initial seeds of the Fibonacci sequence serve as the authentication key, and secret images, scrambled using the Fibonacci-Lucas transform, are embedded into the LH sub-band of selected frames, with singular values (SVs) of the scrambled watermark added to the SVs of selected frames [2]. Prasetyo et al. [3] attempt malicious attacks on this scheme and find it robust to attacks such as noise injection, cropping, scaling, etc. [4], but identify a weakness in this approach in terms of its susceptibility to the False-Positive-Problem (FPP), wherein counterfeit watermark images can easily be reconstructed by a malicious attacker. They find that by using singular vectors associated with arbitrary “counterfeit” images in the extraction process, even when using the correct key, the recovered watermark images appear nearly identical to the counterfeit images, making this method unsuitable for critical applications such as providing proof of copyright or ownership of a video [3]. Prasetyo et al. [3] then propose a fix for this vulnerability, by embedding the principal components of the watermark image (including left singular vectors and singular values), such that using counterfeit singular vectors no longer reconstructs a discernable watermark image.

Frame-by-frame video watermarking techniques such as spread-spectrum (SS)-based techniques [5], where noise-like signals generated from a key are embedded into the video, have also shown promise [6], but have also been found to be susceptible to attacks [7]. For example, SS-based watermarking schemes where each frame gets a different, pseudorandom watermark are susceptible to Temporal Frame Averaging (TFA) attacks, where adjacent frames are averaged in order to remove watermarking [7]. In fact, Doerr and Dugelay [7] show that even when this scheme is enhanced to prevent TFA attacks, such that the scheme now randomly chooses a watermark for each from from a finite set of orthonormal watermarks and the watermark detector checks for the presence of all watermarks in the set, a new, more sophisticated Watermark Estimation Clusters Remodulation (WECR) attack can still successfully remove the watermark.

Especially as Michael et al. [1] themselves assert that the NCI watermarking technique is closely related to direct sequence spread spectrum techniques to spread signal transmission over broad frequency bands through modulation with pseudorandom noise, the introduction of this novel watermarking technique brings along with it a gap in understanding of its robustness and vulnerability to potential adversarial attacks, which, should it be able to verify originality and prevent tampering of recorded videos in critical, high-stakes scenarios such those presented by the authors (such as political campaigning), must be comprehensively evaluated.

---

# **3. Technical Approach**  
This project consists of two parts. The first part is recreating the NCI setup as outlined in Michael’s paper. The second part is testing and attacking the recreated NCI setup. 

### **3.1 Recreating NCI Setup**  
The complete NCI setup as outlined by Michael consists of a) recording a video illuminated by a noise coded light source (figure n) and b) analyzing a video for temporal or spatial manipulation (figure n).

##### **3.1.1 Noise Coded Light Source**  
Noise coded light source is created by modulating the brightness of a light according to some “code signal” (referred to as c). According to Michael’s paper, the code signal must be a) random, b) zero-mean, c) uncorrelated with other code signals, and d) bandlimited to half of video frame rate. The simplest way to achieve this is by generating the code signal in frequency-domain and transforming it to time-domain using Fourier Transform. Figure n shows generating 1024 samples of the code signal. We first generate a random discrete spectrum bandlimited to 9 Hz using 1024 frequency bins. 9 Hz is chosen because 9 Hz is less than half of 30 Hz, the target video frame rate. 1024 bins are chosen because it is the power of 2 closest to 1000. To ensure the final time-domain signal is real-valued, the first half of the spectrum must be a mirrored and complex conjugate version of the second half. 1024 point Inverse-FFT is used to convert the signal in frequency-domain to time-domain. Then, the time-domain code signal is scaled such that the maximum amplitude is 1. To play the code signal on a monitor, each value of the code signal is mapped to a greyscale value ranging from 0 (black) to 255 (white) and outputted at a rate of 30 Hz (monitor brightness changes every 1/30 seconds). After outputting all 1024 samples, the process is repeated with a new random 1024 frequency bins.

##### **3.1.2 Video Generation**  
A scene is illuminated with a noise coded light source. In our case, we used an LCD monitor as our noise coded light source. Video is recorded using iPhone 13/15/17 at 30fps in 1080p resolution. The video is exported as a .mov file.

##### **3.1.3 Tamper Detection Algorithm**  
The goal of the tamper detection algorithm (src/analyze.py) is to generate an Alignment Matrix and Reflectance Estimate from the video and code signal. The Alignment Matrix detects temporal manipulation in the video while Reflectance Estimate detects spatial manipulation in the video.

The first step in the tamper detection algorithm is to preprocess the .mov or .mp4 file. Figure below shows the video preprocessing pipeline. Preprocessing is required to reduce computation time and resource usage. It also serves to amplify the signals we want for generating the Alignment Matrix. 

The video file is loaded into memory as a 4-dimensional uint8 pixel array where each dimension represents frame number (n), pixel height location (y), pixel width location (x), and color channel (rgb). The pixel array is downscaled by 4 by decimating in x and y direction. Linear gamma correction is applied by normalizing each pixel and raising the value to a power of 2.2. Temporal Bilateral Filter is applied to all pixels using a temporal window size of 5 with temporal and range sigma values 0.5 and 0.03 respectively. Temporal window size of 5 is chosen to match the same parameter from Michael’s paper while temporal and range sigma values were chosen arbitrarily.

##### **3.1.3.1 Alignment Matrix Calculation**
Definitions:
y = video signal (1-d vector)
c = coded light signal (1-d vector)
y’ = window of y
c’ = window of c

To generate the Alignment Matrix, each frame of the video needs to be somehow compressed into a single value. This transforms the 4-d pixel array into a 1-d array referred to as the “Global Vector y”.

We first separate the color components of the video to get a red, green, and blue version of the video. For each color version of video, the mean pixel value is computed for each frame, to obtain 3 different 1-d vectors. Each 1-d vector is normalized independently then averaged to create a single 1-d Global Vector y. This process ensures each channel contributes equally to the final signal, regardless of its inherent brightness or variation.

Imagine your camera has unequal red, green, and blue sensitivity (very common in real cameras). The red channel might naturally be much brighter than blue. Without per-channel normalization, the red channel would dominate the alignment process, and any variations in blue would be drowned out.

By normalizing each channel independently, it’s equivalent as "Treat variations in red, green, and blue equally, regardless of their absolute brightness levels." This makes the alignment more robust to color imbalances in your camera.
 
[Alignment Matrix math]
The alignment matrix tells us which part of y corresponds to which part of c. To quantify correlation between sample y[n] and c[m], a window centered around the two samples are generated i.e. y’ = [y[n-w/2], y[n-w/2+1], … y[n-1], y[n], y[n+1], … y[n+w/2-1], y[n+w/2]] and c’ = [c[m-w/2], c[m-w/2+1], … c[m-1], c[m], c[m+1], … c[m+w/2-1], c[m+w/2]], where w is the alignment matrix window size. We chose w = 511 for the alignment matrix. The dot product between y’ and c’ indicates the strength of correlation. Vectorizing this process across all combinations of n and m yields the alignment matrix.

##### **3.1.3.2 Reflectance Estimate Calculation**  
[Reflectance Estimate math]
p = pixel array (4-d vector)

To recover the reflectance value of the pixel located at coordinate (height=y, width=x) in the n-th frame, i.e. p[n, y, x], we first find c[m] that maximally correlates with y[n] using the alignment matrix. Then we take another window of c centered around c[m] i.e. c’ = c[m-w/2], c[m-w/2+1], … c[m-1], c[m], c[m+1], … c[m+w/2-1], c[m+w/2] and a time window of the pixel i.e. p’ = p[n-w/2, y, x], p[n-w/2+1, y, x], …, p[n-1, y, x], p[n, y, x], p[n+1, y, x], … p[n+w/2-1, y, x], p[n+w/2, y, x], where w is the reflectance estimate window size. We chose w = 127 for the reflectance estimate. The reflectance of that pixel is calculated as the dot product between c’ and p’ divided by magnitude squared of c’. Vectorizing this process across all combinations of n, y, and x yields the reflectance estimate video.

### **3.2 Testing & Attacking NCI Setup**  


##### **3.2.1 Code Signal Extraction Attack**  
If the adversary can obtain the code signal, they can record a new video under the coded light to produce a new authentic video. For example, the adversary can recreate the scene from the original video but with some modifications. Assuming the code signal is private to the authors of the original video, the adversary will have to extract it from the video alone.

##### **3.2.2 Spatial Manipulation Attacks**  


##### **3.2.2.1 Basic Overlay**  
[Diagram of Basic Overlay]
The basic overlay attack serves as our control. This is the simplest form of pixel manipulation where the original pixel value is simply overwritten with a new pixel value. In this attack, any pixel that has been edited stays at a constant value.

##### **3.2.2.2 Pixel Multiplication**  
[Diagram of Attack #2 Pixel Multiplication Attack]
This attack performs pixel modification by multiplying the pixel value at a desired location by some constant alpha such that the final value matches the desired pixel value. alpha > 1 will make the pixel brighter while alpha < 1 will make the pixel dimmer. Each color channel can be multiplied independently to produce a wide range of colors. The premise behind this attack is that multiplying a pixel value by alpha is mathematically equivalent to a pixel under the same coded signal but with the reflectance multiplied by alpha and the noise variance multiplied by alpha^2.

##### **3.2.2.3 Pixel Sampling**  
[Diagram of Attack #3 Pixel Sampling Attack]
This attack performs pixel modification by replacing a pixel at a desired location with another pixel from the same video. The premise behind this attack is that NCI only watermarks a video across time; the analysis algorithm is agnostic to the order of the pixels. For example, in alignment matrix generation, the 1 dimensional global vector y is computed from an average of all the pixels in a frame. Since averaging is commutative, the order of the pixels in the original video has no effect on the alignment matrix. Reflectance estimate is also unaffected because the pixel simply inherits the reflectance of the sampled pixel.

##### **3.2.2.3 Pixel Sampling + Multiplication**  
[Diagram of Attack #3+2 Pixel Sampling + Mult. Attack]


---

# **4. Evaluation & Results**  
An LCD monitor playing our code signal illuminates an indoor scene. Windows were covered with blinds to minimize any uncontrollable light source from outside. The illuminated scene is captured using a smartphone camera (iPhone 17) at 1080p 30fps. The scene was kept static by mounting the camera on a table.
Figure n shows four random frames from one of the videos we recorded. We recommend checking out the videos on our website to get a full sense of the brightness fluctuations created by NCI.

### **4.1 Code Signal Extraction Attack Result**  

### **4.1 Basic Attack (Control) Result**  

### **4.2 Pixel Multiplication Attack Result**  

### **4.3 Pixel Sampling Attack Result**  

### **4.4 Pixel Multiplication + Sampling Attack Result**  

### **4.5 Attack Comparison**  


---

# **5. Discussion & Conclusions**

### **5.1 Potential Defense Strategy**  

### **5.2 Multiple Noise Coded Light Sources**  

---

# **6. References**  
[1] Michael, Peter, Zekun Hao, Serge Belongie, and Abe Davis. "Noise-Coded Illumination for Forensic and Photometric Video Analysis." ACM Transactions on Graphics 44, no. 5 (2025): 1-16. https://dl.acm.org/doi/pdf/10.1145/3742892

[2] Ponni alias Sathya, S., and S. Ramakrishnan. "Fibonacci based key frame selection and scrambling for video watermarking in DWT–SVD domain." Wireless Personal Communications 102.2 (2018): 2011-2031.

[3] Prasetyo, Heri, Chih-Hsien Hsia, and Chin-Hua Liu. "Vulnerability attacks of SVD-based video watermarking scheme in an IoT environment." IEEE Access 8 (2020): 69919-69936.

[4] Aberna, P., and Loganathan Agilandeeswari. "Digital image and video watermarking: methodologies, attacks, applications, and future directions." Multimedia Tools and Applications 83.2 (2024): 5531-5591.

[5] Hartung, Frank, and Bernd Girod. "Watermarking of uncompressed and compressed video." Signal processing 66.3 (1998): 283-301.

[6] Doerr, Gwenael, and Jean-Luc Dugelay. "A guide tour of video watermarking." Signal processing: Image communication 18.4 (2003): 263-282.

[7] Doërr, Gwenaël, and J-L. Dugelay. "Security pitfalls of frame-by-frame approaches to video watermarking." IEEE Transactions on Signal Processing 52.10 (2004): 2955-2964.

---

# **7. Supplementary Material**

## **7.a. Datasets**

Describe each dataset:
* Source and URL
* Data format
* Preprocessing steps
* Labeling/annotation efforts

Include your internal dataset if you collected one.
## **7.b. Software**

List:
* External libraries or models
* Internal modules you wrote
* Links to repos or documentation

---

> [!NOTE] 
> Read and then delete the material from this line onwards.

# 🧭 **Guidelines for a Strong Project Website**

- Include multiple clear, labeled figures in every major section.  
- Keep the writing accessible; explain acronyms and algorithms.  
- Use structured subsections for clarity.  
- Link to code or datasets whenever possible.  
- Ensure reproducibility by describing parameters, versions, and preprocessing.  
- Maintain visual consistency across the site.

---

# 📊 **Minimum vs. Excellent Rubric**

| **Component**        | **Minimum (B/C-level)**                                         | **Excellent (A-level)**                                                                 |
|----------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Introduction**     | Vague motivation; little structure                             | Clear motivation; structured subsections; strong narrative                                |
| **Related Work**     | 1–2 citations; shallow summary                                 | 5–12 citations; synthesized comparison; clear gap identification                          |
| **Technical Approach** | Text-only; unclear pipeline                                  | Architecture diagram, visuals, pseudocode, design rationale                               |
| **Evaluation**       | Small or unclear results; few figures                          | Multiple well-labeled plots, baselines, ablations, and analysis                           |
| **Discussion**       | Repeats results; little insight                                | Insightful synthesis; limitations; future directions                                      |
| **Figures**          | Few or low-quality visuals                                     | High-quality diagrams, plots, qualitative examples, consistent style                      |
| **Website Presentation** | Minimal formatting; rough writing                           | Clean layout, good formatting, polished writing, hyperlinks, readable organization        |
| **Reproducibility**  | Missing dataset/software details                               | Clear dataset description, preprocessing, parameters, software environment, instructions   |