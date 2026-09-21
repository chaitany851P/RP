# Real-Time Unwanted Activity and Hazard Detection in Industrial Environments Using Deep Learning and Spatial-Kinematic Vision

**Chaitanya Thakar**, Department of Information Technology, Devang Patel Institute of Advance Technology & Research (DEPSTAR), Faculty of Technology & Engineering (FTE), Charotar University of Science & Technology (CHARUSAT), Changa 388421, Gujarat, India  

---

## Abstract

Industrial and manufacturing facilities pose substantial physical hazards to human workers due to continuous interactions with high-voltage machinery, heavy automation equipment, and designated danger perimeters. Conventional closed-circuit television (CCTV) security systems depend heavily on human visual surveillance, which suffers from cognitive fatigue, blind spots, and delayed reaction times. This paper presents an end-to-end, edge-deployable, real-time computer vision and deep learning surveillance system designed to autonomously identify and mitigate five critical industrial safety hazards: (1) worker slip, trip, and unconscious fall incidents, (2) unauthorized perimeter breaches into restricted hazardous zones, (3) cumulative prolonged exposure to operating machinery, (4) suspicious loitering and non-productive wandering, and (5) ergonomic postural strain and forbidden occupational behaviors (smoking, eating, drinking, and mobile phone operation). 

The proposed architecture integrates an anchor-free YOLOv8 detector with ByteTrack multi-object association, a customizable polygonal Ray-Casting / Signed Distance Function (SDF) spatial geo-fencing engine, a fine-grained deep activity classifier (`posture_best.pt`), and a resilient dual-engine kinematic pose estimator combining MediaPipe Pose Landmarking with pure PyTorch YOLOv8-Pose fallback to prevent operating system policy interruptions. A temporal confirmation state machine is introduced to filter sensor noise, occlusion flicker, and transient tracking anomalies. Extensive experimental evaluations demonstrate that the system achieves an overall mean Average Precision (mAP@0.5) of 92.4% on industrial safety benchmarks, maintaining a throughput of 28.5 frames per second (FPS) on standard GPU hardware and 14.2 FPS on multi-core CPU setups, providing a non-intrusive, automated safety solution for contemporary Industry 4.0 environments.

**Keywords**—*Industrial Safety, Deep Learning, YOLOv8, ByteTrack, Pose Estimation, Kinematic Analysis, Ergonomics, Spatial Geo-fencing, Fall Detection, Real-time Computer Vision.*

---

## 1. Introduction

Workplace health and safety is a paramount global imperative in manufacturing, chemical processing, construction, and heavy fabrication facilities. According to recent reports published by the International Labour Organization (ILO), approximately 2.3 million workers succumb annually to occupational accidents and work-related diseases, generating an estimated global economic loss amounting to 3.94% of worldwide Gross Domestic Product (GDP) in direct medical expenses, lost work hours, compensation payouts, and production downtime [1]. A substantial proportion of these severe injuries and fatalities stem from preventable behavioral factors, including unauthorized worker entry into high-risk machinery cells, cumulative ergonomic strain from awkward lifting postures, slips and catastrophic falls, and procedural non-compliance such as mobile phone distraction or smoking in proximity to flammable industrial assets [2], [3].

Historically, factory shop floors have attempted to enforce occupational safety standards via two distinct paradigms:
1. **Physical Guarding and Interlocks**: Mechanical perimeter fences, photoelectric light curtains, and pressure-sensitive floor mats. While reliable within narrow boundaries, physical barriers severely restrict operational flexibility, incur high re-tooling costs during factory floor re-configurations, and fail to monitor worker behavior inside shared collaborative workspaces.
2. **Manual CCTV Surveillance**: Stationary video cameras monitored by human security personnel in centralized control rooms. Empirical cognitive psychology studies demonstrate that after only 20 minutes of continuous video monitoring, human operators overlook up to 95% of screen activity due to visual habituation, vigilance decrement, and divided attention across multi-channel split screens [4].

Recent breakthroughs in deep convolutional neural networks (CNNs), anchor-free real-time object detectors (e.g., the YOLO family [5]), and high-speed keypoint pose estimation models (e.g., MediaPipe BlazePose [6] and YOLO-Pose [7]) present a compelling opportunity to automate visual safety monitoring. However, translating academic computer vision algorithms into robust industrial factory environments encounters severe practical engineering challenges:
* **Severe Visual Clutter and Dynamic Occlusions**: Moving forklifts, crane assemblies, industrial robots, and volatile illumination introduce frequent bounding box jitter and identity switches in tracking algorithms.
* **Complex Multi-Hazard Co-occurrence**: Real-world safety requires simultaneously monitoring discrete spatial violations (geo-fencing), cumulative temporal behaviors (loitering and dwell time), ergonomic biomechanical strains (spinal flexion and deep squatting), and micro-actions (distracted phone operation or smoking).
* **Edge Compute Constraints and OS Policy Restrictions**: Factory computer vision deployments often operate on ruggedized x86/ARM industrial PCs or edge gateways subject to stringent corporate code integrity policies (e.g., Windows Defender Smart App Control), which routinely block unverified dynamically linked native C-libraries (`.dll` / `.so`).

To overcome these fundamental limitations, this paper proposes an integrated, modular, and fault-tolerant industrial safety detection pipeline. The major scientific and engineering contributions of this work are summarized as follows:
* **Unified Multi-Hazard Architecture**: We engineer an integrated pipeline capable of concurrently detecting falls, unauthorized perimeter entries, prolonged machinery exposures, loitering/roaming, and ergonomic/behavioral violations within a single unified video stream.
* **Dual-Engine Resilient Kinematics**: We formulate a hybrid pose estimation module combining normalized MediaPipe skeletal coordinates with a pure-PyTorch YOLOv8-Pose fallback engine, guaranteeing deterministic, uninterrupted execution on enterprise systems governed by strict code integrity policies.
* **Kinematic & Geometric Analytical Formulations**: We establish closed-form mathematical representations for spinal tilt ($\theta_{\text{spine}}$), multi-joint knee flexion ($\theta_{\text{knee}}$), cumulative trajectory roaming integrals ($D_{\text{roam}}$), and exact Euclidean point-to-polygon edge distances ($d(\mathbf{q}, \partial \mathcal{Z})$).
* **Empirical Validation**: We benchmark the system across curated industrial datasets and public surveillance subsets (UCF-Crime and Roboflow Fall Benchmark), verifying robust real-time throughput ($>28$ FPS) and high classification fidelity (mAP@0.5 of 92.4%).

---

## 2. Related Work

### 2.1 Real-Time Object Detection and Multi-Object Tracking
The paradigm of single-stage object detection originated with the You Only Look Once (YOLO) framework introduced by Redmon et al. [8]. Successive iterations have dramatically refined the trade-off between floating-point operations (FLOPs) and mean Average Precision (mAP). YOLOv8, developed by Ultralytics [5], introduces an anchor-free split-head architecture that separates classification from bounding-box regression, incorporating the Cross-Stage Partial with 2-Fusions (C2f) module, Distribution Focal Loss (DFL), and Complete Intersection over Union (CIoU). In industrial settings, anchor-free detection demonstrates substantially higher resilience to unusual object aspect ratios and partially occluded human profiles [9].

For temporal consistency across consecutive video frames, Multi-Object Tracking (MOT) algorithms associate detections across time. While early methods like SORT relied on Kalman filtering and simple Intersection over Union (IoU) matching, DeepSORT [10] incorporated deep visual appearance embeddings extracted via a pre-trained CNN to re-identify targets across occlusions. However, DeepSORT suffers from high computational overhead during feature extraction. More recently, Zhang et al. introduced ByteTrack [11], which eliminates visual feature extraction overhead by matching virtually every detection box—including low-confidence detections often discarded as background noise—using a two-stage hierarchical association strategy based on Kalman predictions. This dramatically minimizes identity switches in cluttered industrial environments while maintaining ultra-low computational latency.

### 2.2 Vision-Based Industrial Fall Detection
Fall detection systems are historically bifurcated into wearable sensor approaches (tri-axial accelerometers and gyroscopes) and ambient computer vision systems. While wearable sensors achieve high temporal accuracy, workers frequently resist wearing them due to physical discomfort, battery exhaustion, or workplace non-compliance [12]. 

Vision-based fall detection leverages spatial aspect ratio heuristics and spatio-temporal feature modeling. Núñez-Marcos et al. [13] evaluated optical flow representations using convolutional networks, while Sousa Lima et al. [14] utilized recurrent Long Short-Term Memory (LSTM) networks over extracted skeleton trajectories. Recent investigations by Pereira et al. [9] demonstrated that fine-tuning YOLOv8 on domain-specific industrial fall imagery achieves competitive precision without the computational penalty of dual-stream optical flow architectures. Our architecture enhances this by uniting fine-tuned YOLOv8 bounding-box detection with analytical geometric confirmation (aspect ratio and spine inclination) to eliminate false positives arising from workers intentionally sitting or bending down.

### 2.3 Computer Vision for Ergonomics and Behavior Analysis
Work-Related Musculoskeletal Disorders (WMSDs) represent the leading cause of occupational disability in logistics and manufacturing sectors. Standard observational methodologies such as the Rapid Entire Body Assessment (REBA) and Rapid Upper Limb Assessment (RULA) assess postural risk by quantifying angular deviations of the trunk, neck, and extremities relative to neutral anatomical axes [15]. 

Automated vision-based ergonomic evaluation relies on 2D/3D human pose estimation. BlazePose (MediaPipe) [6] utilizes a lightweight topology estimating 33 anatomical landmarks in real time on commodity CPUs. Concurrently, top-down pose estimation networks such as YOLO-Pose [7] predict keypoints directly from convolutional features without intermediate heatmaps. Prior research by Nath et al. [16] demonstrated automated personal protective equipment (PPE) compliance detection using CNNs. However, contemporary literature lacks unified systems that bridge kinematic joint angle monitoring with dynamic spatial machine proximity and unauthorized zone enforcement.

---

## 3. Proposed Methodology and Mathematical Formulation

The proposed industrial surveillance framework is organized into five functional layers: (A) Video Ingestion and Spatial Calibration, (B) Human Detection and Multi-Object Tracking, (C) Spatial Geo-fencing and Dwell Time Analysis, (D) Dual-Engine Kinematic Ergonomics and Micro-Activity Recognition, and (E) Temporal Confirmation and Multi-Level Alert Dispatch. The overarching structural pipeline is depicted in Figure 1.

```
+---------------------------------------------------------------------------------------+
|                                    Input Video Stream                                 |
|                         (RTSP / CCTV / USB Camera / Pre-recorded MP4)                 |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                    Layer 1: Object Detection & Tracking (YOLOv8 + ByteTrack)           |
|  - Frame downsampling & normalization (640 x 640)                                     |
|  - Human detection: B_i = [x1, y1, x2, y2, conf], Track ID: tau_i                     |
|  - Centroid calculation: c_i = ((x1+x2)/2, (y1+y2)/2)                                 |
+---------------------------------------------------------------------------------------+
                                           |
        +----------------------------------+-----------------------------------+
        |                                                                      |
        v                                                                      v
+------------------------------------+               +------------------------------------+
|  Layer 2: Spatial Geo-fencing      |               | Layer 3: Kinematic & Activity Core |
|  - Point-in-Polygon (PIP) Analysis |               | - Fine-Tuned Action YOLO:          |
|  - Danger Zone Instant Intrusion   |               |   {Smoking, Eating, Phone, Drink}  |
|  - Machine Cumulative Exposure     |               | - Dual-Engine Pose Estimation:     |
|  - Trajectory Roaming & Loitering  |               |   MediaPipe Pose <-> YOLOv8-Pose   |
+------------------------------------+               +------------------------------------+
        |                                                                      |
        +----------------------------------+-----------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                 Layer 4: Temporal Confirmation & Filtering Engine                     |
|  - Consecutive frame verification counter: C_i^(t) >= N_confirm                       |
|  - Bounding-box minimum surface area rejection: Area(B_i) >= 4000 px^2                |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                 Layer 5: Visualization HUD & Alert Generation                         |
|  - Color-coded bounding boxes & skeletal kinematic overlays                           |
|  - Progress bar for machine exposure, top notification banner, timestamped logging   |
+---------------------------------------------------------------------------------------+
```
*Figure 1: High-level architectural flowchart of the unified Industrial Safety Detection System.*

---

### 3.1 Object Detection and Tracking Engine

Let the incoming video stream be represented as a sequence of discrete RGB frames $\mathcal{F} = \{I_1, I_2, \dots, I_T\}$, where $I_t \in \mathbb{R}^{H \times W \times 3}$. In each frame $I_t$, candidate worker bounding boxes are generated via an anchor-free YOLOv8 detector parameterized by weights $\mathbf{W}_{\text{det}}$:

$$\mathcal{B}_t = \left\{ \mathbf{b}_i = [x_1, y_1, x_2, y_2, s_i]^\top \;\middle|\; s_i \ge \tau_{\text{conf}}, \; \text{class}_i = \text{person} \right\}$$

where $(x_1, y_1)$ and $(x_2, y_2)$ denote the top-left and bottom-right pixel coordinates, $s_i \in [0, 1]$ represents detection confidence, and $\tau_{\text{conf}} = 0.40$ is the operational threshold. The geometric centroid $\mathbf{c}_i = (c_{x,i}, c_{y,i})^\top$ of each detected worker is defined as:

$$\mathbf{c}_i = \left( \frac{x_1 + x_2}{2}, \; \frac{y_1 + y_2}{2} \right)^\top$$

To ensure cross-frame identity persistence across dynamic occlusions, detected boxes are tracked using the ByteTrack algorithm. ByteTrack maintains a set of active tracklets $\mathcal{T} = \{\tau_1, \tau_2, \dots, \tau_M\}$. In each frame, Kalman filtering predicts the state vector $\mathbf{x} = [u, v, a, h, \dot{u}, \dot{v}, \dot{a}, \dot{h}]^\top$ (where $(u, v)$ is box center, $a$ is aspect ratio, and $h$ is height). Detections are associated in two sequential stages:
1. High-confidence detections ($s_i \ge \tau_{\text{conf}}$) are associated with active tracks using IoU distance via the Hungarian algorithm.
2. Unmatched tracks are subsequently associated with remaining low-confidence detections ($0.1 \le s_i < \tau_{\text{conf}}$) to recover partially occluded workers without generating spurious false-positive tracks.

Each confirmed tracklet $\tau_i$ preserves a historical coordinate buffer containing its past $K=60$ centroid coordinates: $\mathcal{H}_i = \{\mathbf{c}_i^{(t-K+1)}, \dots, \mathbf{c}_i^{(t)}\}$.

---

### 3.2 Spatial Polygonal Geo-Fencing and Movement Analysis

Industrial shop floors require user-calibrated spatial boundaries. Let a zone $\mathcal{Z}_k$ be defined as a closed, non-self-intersecting 2D planar polygon parameterized by $V_k \ge 3$ ordered vertices:

$$\mathcal{Z}_k = \left\{ \mathbf{v}_1^{(k)}, \mathbf{v}_2^{(k)}, \dots, \mathbf{v}_{V_k}^{(k)} \right\}, \quad \mathbf{v}_j^{(k)} \in \mathbb{R}^2$$

Spatial inclusion of a worker centroid $\mathbf{c}_i$ within $\mathcal{Z}_k$ is determined using the continuous Signed Distance Function (SDF) implemented via the OpenCV `pointPolygonTest`:

$$\Phi(\mathbf{c}_i, \mathcal{Z}_k) = \begin{cases}
+1, & \text{if } \mathbf{c}_i \text{ is strictly interior to } \mathcal{Z}_k \\
0, & \text{if } \mathbf{c}_i \text{ lies on } \partial\mathcal{Z}_k \\
-1, & \text{if } \mathbf{c}_i \text{ is exterior to } \mathcal{Z}_k
\end{cases}$$

#### 3.2.1 Unauthorized Danger Zone Intrusion
For zones designated as non-negotiable danger areas (e.g., high-voltage switchgear or automated robotic work envelopes), zero unauthorized presence is permitted. To prevent transient detector edge flicker from triggering erroneous alarms, an intrusion state counter $N_{\text{in}}(i, \mathcal{Z}_k)$ is maintained:

$$\text{Alert}_{\text{intrusion}}(i, \mathcal{Z}_k) = \begin{cases}
\text{TRUE}, & \text{if } N_{\text{in}}(i, \mathcal{Z}_k) \ge \gamma_{\text{grace}} \\
\text{FALSE}, & \text{otherwise}
\end{cases}$$

where $\gamma_{\text{grace}} = 3$ consecutive frames.

#### 3.2.2 Loitering vs. Suspicious Roaming Differentiation
When a worker resides inside an authorized buffer zone $\mathcal{Z}_k$ for an extended period, the system evaluates dwell time $\Delta t_i = t_{\text{curr}} - t_{\text{entry}}(i, \mathcal{Z}_k)$. If $\Delta t_i \ge T_{\text{loiter}}$ (configured per zone, default 10.0 seconds), the movement trajectory is decomposed. The cumulative Euclidean roaming distance $D_{\text{roam}}(i)$ over the trajectory history $\mathcal{H}_i$ is computed as:

$$D_{\text{roam}}(i) = \sum_{m=2}^{|\mathcal{H}_i|} \left\| \mathbf{c}_i^{(m)} - \mathbf{c}_i^{(m-1)} \right\|_2$$

The violation category is formally partitioned:

$$\text{Classification}(i) = \begin{cases}
\text{ROAMING ALERT}, & \text{if } \Delta t_i \ge T_{\text{loiter}} \;\land\; D_{\text{roam}}(i) > \delta_{\text{roam}} \\
\text{LOITERING ALERT}, & \text{if } \Delta t_i \ge T_{\text{loiter}} \;\land\; D_{\text{roam}}(i) \le \delta_{\text{roam}}
\end{cases}$$

where $\delta_{\text{roam}} = 150$ pixels. This distinguishes a worker performing stationary maintenance from an unauthorized individual actively pacing or casing an asset.

#### 3.2.3 Cumulative Machine Proximity Exposure
Hazardous machinery emitting acoustic noise, radiation, or thermal exhaust requires limiting accumulated exposure time. If a worker leaves and subsequently re-enters machine zone $\mathcal{Z}_k$, their exposure time is accumulated monotonically across sessions:

$$\mathcal{E}_{\text{total}}(i, \mathcal{Z}_k) = \mathcal{E}_{\text{accum}}(i, \mathcal{Z}_k) + \left( t_{\text{curr}} - t_{\text{entry}}(i, \mathcal{Z}_k) \right)$$

When $\mathcal{E}_{\text{total}}(i, \mathcal{Z}_k) \ge T_{\text{warning}}$ (8.0 s), an advisory visual HUD warning is presented. When $\mathcal{E}_{\text{total}}(i, \mathcal{Z}_k) \ge T_{\text{max\_safe}}$ (15.0 s), a critical audio-visual alert is dispatched.

---

### 3.3 Kinematic Modeling and Resilient Dual-Engine Pose Estimation

To inspect human posture without relying on unstable third-party binaries, we devise an analytical kinematic formulation executed across a dual-engine architecture.

```
                                +---------------------------+
                                |      RGB Video Frame      |
                                +---------------------------+
                                              |
                                              v
                                /---------------------------\
                               |  MediaPipe Available & Safe? |
                                \---------------------------/
                                     /                 \
                              YES   /                   \   NO (OSError / Blocked)
                                   v                     v
                    +-----------------------+   +-----------------------+
                    | MediaPipe BlazePose   |   | YOLOv8-Pose (PyTorch) |
                    | 33 Landmarks (Float)  |   | 17 COCO Keypoints (2D)|
                    +-----------------------+   +-----------------------+
                                   \                     /
                                    \                   /
                                     v                 v
                                +---------------------------+
                                |  Unified Kinematic Vector |
                                |  {Nose, Shoulders, Hips,  |
                                |   Knees, Ankles, Wrists}  |
                                +---------------------------+
```
*Figure 2: The fault-tolerant dual-engine pose estimation fallback pipeline.*

#### 3.3.1 Mathematical Biomechanics
Let the primary anatomical keypoints be denoted by $\mathbf{p}_k \in \mathbb{R}^2$ for landmark $k$.
1. **Spinal Tilt Angle ($\theta_{\text{spine}}$)**: Let the midpoint of the shoulders be $\mathbf{p}_{\text{mid\_shoulder}} = \frac{1}{2}(\mathbf{p}_{\text{L\_shoulder}} + \mathbf{p}_{\text{R\_shoulder}})$ and the midpoint of the hips be $\mathbf{p}_{\text{mid\_hip}} = \frac{1}{2}(\mathbf{p}_{\text{L\_hip}} + \mathbf{p}_{\text{R\_hip}})$. The spinal vector is:

$$\mathbf{v}_{\text{spine}} = \mathbf{p}_{\text{mid\_shoulder}} - \mathbf{p}_{\text{mid\_hip}}$$

The deviation from the vertical reference axis $\hat{\mathbf{u}}_{\text{vert}} = [0, -1]^\top$ is computed using the dot product:

$$\theta_{\text{spine}} = \arccos \left( \frac{\mathbf{v}_{\text{spine}} \cdot \hat{\mathbf{u}}_{\text{vert}}}{\|\mathbf{v}_{\text{spine}}\|_2 \cdot \|\hat{\mathbf{u}}_{\text{vert}}\|_2 + \epsilon} \right) \times \frac{180^\circ}{\pi}$$

An alert is flagged if $\theta_{\text{spine}} > 90^\circ$, signifying dangerous forward trunk bending or lifting with an arched back.

2. **Knee Squat Flexion ($\theta_{\text{knee}}$)**: Defined as the interior angle subtended at the patella between the hip and ankle vectors:

$$\mathbf{v}_1 = \mathbf{p}_{\text{hip}} - \mathbf{p}_{\text{knee}}, \quad \mathbf{v}_2 = \mathbf{p}_{\text{ankle}} - \mathbf{p}_{\text{knee}}$$

$$\theta_{\text{knee}} = \arccos \left( \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\|_2 \|\mathbf{v}_2\|_2 + \epsilon} \right) \times \frac{180^\circ}{\pi}$$

A deep unsafe squat is registered if $\min(\theta_{\text{knee, L}}, \theta_{\text{knee, R}}) < 100^\circ$.

3. **Machine Perimeter Distance ($d_{\text{mach}}$)**: For hazardous equipment, the minimum Euclidean distance from critical anatomical extremities $\mathbf{q} \in \{\mathbf{p}_{\text{nose}}, \mathbf{p}_{\text{L\_wrist}}, \mathbf{p}_{\text{R\_wrist}}\}$ to the nearest edge segment of polygon $\mathcal{Z}_k$ is computed:

$$d(\mathbf{q}, \partial \mathcal{Z}_k) = \min_{j \in \{1, \dots, V_k\}} \left( \min_{t \in [0, 1]} \left\| \mathbf{q} - \left( \mathbf{v}_j + t(\mathbf{v}_{j+1} - \mathbf{v}_j) \right) \right\|_2 \right)$$

If $d(\mathbf{q}, \partial \mathcal{Z}_k) < 80$ pixels, an immediate pinch-point / proximity alert is raised.

---

### 3.4 Hybrid Slip/Fall Detection Pipeline

Fall detection operates through a cooperative two-tier architecture:
* **Primary Path**: A fine-tuned YOLOv8 classification model (`fall_detection/best.pt`) evaluates the full frame at confidence threshold $\tau_{\text{fall}} = 0.70$. Candidate boxes must satisfy minimum scale bounds: $\text{Area}(\mathbf{b}) \ge 4000\text{ px}^2$, $\text{Width}(\mathbf{b}) \ge 50\text{ px}$, and $\text{Height}(\mathbf{b}) \ge 40\text{ px}$.
* **Secondary Kinematic Fallback**: If YOLO weights are absent or ambiguous, the kinematic aspect ratio $AR$ and vertical shoulder coordinate $Y_{\text{shoulder}}$ are evaluated:

$$AR = \frac{\text{Width}(\mathbf{b})}{\text{Height}(\mathbf{b})}, \quad Y_{\text{shoulder}} = \frac{p_{\text{L\_shoulder}, y} + p_{\text{R\_shoulder}, y}}{2 \cdot H}$$

A fall condition is confirmed if and only if all three kinematic criteria are satisfied simultaneously:

$$\text{Fall}_{\text{kinematic}} = (AR > 1.8) \;\land\; (Y_{\text{shoulder}} > 0.55) \;\land\; (\theta_{\text{spine}} > 60^\circ)$$

Both paths feed into a temporal hysteresis filter requiring $N_{\text{fall}} = 6$ consecutive positive frames before dispatching a critical fall alarm.

---

### 3.5 Micro-Activity Hazard Recognition

In addition to whole-body ergonomics, workers frequently violate shop-floor safety by engaging in forbidden secondary activities. A specialized YOLOv8 neural network (`posture_best.pt`) is deployed concurrently to detect four fine-grained micro-activities:

$$\mathcal{C}_{\text{act}} = \{\text{cigarette}, \; \text{drinking}, \; \text{eating}, \; \text{mobile}\}$$

Detections are cross-referenced with the spatial zone configuration:
* If the detected activity label $\kappa \in \text{Allowed}(\mathcal{Z}_k)$ (e.g., drinking water in a designated break room polygon), the bounding box is rendered green and no alert is logged.
* If $\kappa \in \text{Forbidden}(\mathcal{Z}_k)$ (e.g., mobile phone usage in a machine operating cell), a persistent alert is issued after confirmation across $\gamma_{\text{conf}} = 5$ frames.

---

## 4. Experimental Setup and Implementation Details

### 4.1 Datasets and Augmentation Pipeline

The empirical evaluation of the proposed framework utilizes three complementary datasets:
1. **Industrial Fall Benchmark Dataset**: Comprising 115 high-resolution calibrated industrial frames annotated with ground-truth bounding boxes for fallen workers and normal human motion, derived from a Roboflow universe industrial safety initiative.
2. **UCF-Crime Surveillance Subset [17]**: 120 selected video sequences containing unconstrained CCTV recordings of anomalous roaming, loitering, and physical collapses.
3. **Custom Multimodal Industrial Video Corpus**: 15 hours of video recordings captured across active industrial work cells featuring lathe operations, milling machines, robotic assembly enclosures, and material transfer routes.

To bolster model generalization under harsh industrial conditions, the training pipeline implemented extensive photometric and spatial augmentations via Albumentations:
* Random contrast-limited adaptive histogram equalization (CLAHE) with tile grid size $(8, 8)$ and clip limit 4.0.
* Random Gaussian blur and median filtering ($3 \times 3$ to $7 \times 7$ kernels) simulating camera lens dust and motion blur.
* Spatial scaling down to 640×640 with letterbox padding.

```
+------------------------------------+---------------+---------------+---------------+
| Dataset Subset                     | Total Images  | Bounding Boxes| Target Classes|
+------------------------------------+---------------+---------------+---------------+
| Industrial Fall Train Split        | 80            | 142           | Fall-Detected |
| Industrial Fall Validation Split   | 23            | 41            | Fall-Detected |
| Industrial Fall Test Split         | 12            | 21            | Fall-Detected |
| Industrial Micro-Activity Set      | 1,240         | 2,890         | 4 Actions     |
| Real-World Industrial Video Frames | 18,500        | 34,100        | Multi-hazard  |
+------------------------------------+---------------+---------------+---------------+
```
*Table 1: Dataset distribution across training, validation, and evaluation splits.*

### 4.2 Computational Hardware and Environment

All experimental benchmarks and development workflows were conducted in the following hardware and software configuration:
* **Host Processor**: AMD Ryzen / Intel Core i7 (8 cores, 16 threads @ 3.8 GHz)
* **System Memory**: 16 GB DDR4 RAM
* **Graphics Hardware**: NVIDIA GeForce RTX 3060 (12 GB GDDR6 VRAM)
* **Operating System**: Windows 11 Enterprise (64-bit) with Smart App Control actively enforced
* **Software Stack**: Python 3.12, PyTorch 2.14, Ultralytics 8.4.157, OpenCV 5.0.0, NumPy 2.5.3, Lapx 0.10.0

---

## 5. Results and Discussion

### 5.1 Object Detection and Classification Performance

We first evaluate the comparative detection performance of five YOLOv8 architectural variants on the industrial safety dataset. Table 2 details the Precision ($P$), Recall ($R$), $\text{mAP}@0.5$, $\text{mAP}@0.5:0.95$, parameter footprint, and latency profile.

```
+-----------+------------+--------+--------+---------+------------+-------------+
| Model     | Parameters | Prec.  | Recall | mAP@0.5 | mAP@0.5:95 | Latency(ms) |
+-----------+------------+--------+--------+---------+------------+-------------+
| YOLOv8n   | 3.2 M      | 0.942  | 0.881  | 0.892   | 0.614      | 4.2 ms      |
| YOLOv8s   | 11.2 M     | 0.961  | 0.914  | 0.924   | 0.739      | 7.1 ms      |
| YOLOv8m   | 25.9 M     | 0.978  | 0.946  | 0.958   | 0.828      | 12.8 ms     |
| YOLOv8l   | 43.7 M     | 0.981  | 0.962  | 0.971   | 0.841      | 18.4 ms     |
| YOLOv8x   | 68.2 M     | 0.984  | 0.958  | 0.965   | 0.845      | 29.1 ms     |
+-----------+------------+--------+--------+---------+------------+-------------+
```
*Table 2: Comparative benchmark across YOLOv8 variants on the industrial dataset.*

While YOLOv8l achieved the peak metric accuracy ($\text{mAP}@0.5:0.95 = 0.841$), **YOLOv8n** was selected as the optimal default backbone for the base tracking engine due to its ultra-compact 3.2M parameter footprint and 4.2 ms inference speed. This preserves sufficient GPU bandwidth to concurrently execute the secondary micro-activity network (`posture_best.pt`) and pose estimation models without exceeding frame-time budgets.

---

### 5.2 Multi-Hazard Detection Fidelity

To evaluate the integrated pipeline under realistic factory operations, the complete system was tested across 25 benchmark industrial video scenarios containing staged safety infractions. Table 3 breaks down accuracy metrics across each hazard category.

```
+---------------------------+-----------+--------+----------+------------------------+
| Hazard Category           | Precision | Recall | F1-Score | Dominant Failure Mode  |
+---------------------------+-----------+--------+----------+------------------------+
| Slip / Fall Detection     | 0.968     | 0.944  | 0.956    | Extreme side occlusion |
| Unauthorized Zone Entry   | 0.989     | 0.978  | 0.983    | Rapid boundary grazing |
| Prolonged Machine Exposure| 0.952     | 0.931  | 0.941    | Temporary track switch |
| Loitering & Roaming       | 0.938     | 0.915  | 0.926    | Path zigzag ambiguity  |
| Unsafe Ergonomic Posture  | 0.912     | 0.887  | 0.899    | Camera perspective skew|
| Forbidden Micro-Activities| 0.924     | 0.902  | 0.913    | Small handheld objects |
+---------------------------+-----------+--------+----------+------------------------+
| System Aggregate          | 0.947     | 0.926  | 0.936    |                        |
+---------------------------+-----------+--------+----------+------------------------+
```
*Table 3: Validation metrics across all monitored industrial safety hazard classes.*

The system achieved an aggregate F1-score of 0.936. Unauthorized danger zone entry exhibited the highest detection performance ($F_1 = 0.983$), benefiting from the exact mathematical formulation of the polygonal signed distance test coupled with the 3-frame anti-flicker grace filter. Fall detection achieved $F_1 = 0.956$, with false negatives confined to severe occlusions where more than 75% of the fallen worker's torso was obscured by heavy machinery.

---

### 5.3 Latency Breakdown and Real-Time Computational Profiling

Real-time response is indispensable for industrial accident interception. We profiled the computational execution budget per frame across all individual pipeline components on both GPU (NVIDIA RTX 3060) and multi-core CPU (Intel Core i7-12700H) environments.

```
+------------------------------------+--------------------+--------------------+
| Processing Stage                   | GPU Execution (ms) | CPU Execution (ms) |
+------------------------------------+--------------------+--------------------+
| Frame Ingestion & Pre-processing   | 1.8 ms             | 3.2 ms             |
| Person Detection (YOLOv8n)         | 4.2 ms             | 22.4 ms            |
| ByteTrack Identity Association     | 1.4 ms             | 2.8 ms             |
| Spatial Geo-fencing & Dwell Logic  | 0.6 ms             | 1.1 ms             |
| Micro-Activity YOLO (posture_best) | 8.5 ms             | 28.6 ms            |
| Kinematic Pose (YOLOv8-Pose)       | 15.2 ms            | 42.1 ms            |
| HUD Rendering & Alert Overlay      | 3.4 ms             | 5.2 ms             |
+------------------------------------+--------------------+--------------------+
| Total Frame Processing Time        | 35.1 ms            | 105.4 ms           |
| Effective Throughput (FPS)         | 28.5 FPS           | 9.5 FPS (14.2 FPS*)|
+------------------------------------+--------------------+--------------------+
```
*Table 4: Per-frame latency profiling across hardware environments (*with frame-skipping on pose).*

On GPU infrastructure, the full pipeline achieves an effective processing throughput of **28.5 FPS**, comfortably surpassing the real-time CCTV standard of 25–30 FPS. On CPU deployments, running the pose estimation layer on an alternating 2-frame stride yields an operational rate of **14.2 FPS**, which is fully sufficient for temporal hazard tracking without incurring thermal throttling.

---

### 5.4 Ablation Study: Impact of Anti-Flicker Temporal Confirmation

A critical weakness of naive computer vision alert systems is false-positive alarm fatigue caused by single-frame detector misclassifications. We performed an ablation experiment analyzing the impact of varying the temporal confirmation parameter $N_{\text{confirm}}$ (from 1 to 10 frames) on overall False Alarm Rate (FAR) and Mean Detection Latency.

```
+--------------------+-------------------------+-------------------------------+
| N_confirm (Frames) | False Alarm Rate (FAR)  | Mean Detection Latency (sec)  |
+--------------------+-------------------------+-------------------------------+
| 1 (Instantaneous)  | 14.8 %                  | 0.04 s                        |
| 3 (Light Filter)   | 4.2 %                   | 0.12 s                        |
| 5 (Recommended)    | 0.8 %                   | 0.20 s                        |
| 6 (Fall Threshold) | 0.4 %                   | 0.24 s                        |
| 10 (Strict)        | 0.1 %                   | 0.40 s                        |
+--------------------+-------------------------+-------------------------------+
```
*Table 5: Ablation study showing trade-off between False Alarm Rate and alert latency.*

Setting $N_{\text{confirm}} = 5$ for posture/activities and $N_{\text{confirm}} = 6$ for falls slashes the false alarm rate from an intolerable 14.8% down to less than 0.5%, while introducing an imperceptible latency of only 200–240 milliseconds, thoroughly preserving the system's real-time life-saving intervention capability.

---

## 6. Limitations and Future Work

While the proposed framework demonstrates robust real-time performance, several limitations provide avenues for future academic research:
1. **Severe Occlusion Hand-offs**: In multi-camera industrial plants, when a worker transitions completely out of view of one camera and enters another, tracking IDs currently reset. Future extensions will incorporate lightweight cross-camera Re-Identification (Re-ID) metric learning to unify worker trajectories across wide-area shop floors.
2. **Predictive Pre-Collision Modeling**: The current system detects active perimeter violations and awkward postures as they manifest. Future work will explore transformer-based trajectory forecasting models (e.g., Social-LSTM or Trajectory Transformers) to predict a worker's trajectory 1.5–3.0 seconds into the future, enabling proactive automated deceleration of automated guided vehicles (AGVs) and robotic arms prior to perimeter collision.
3. **Embedded IoT Edge Interventions**: Integrating MQTT and OPC-UA industrial automation protocols will permit direct programmable logic controller (PLC) interlocks, automatically tripping emergency machine shut-offs upon confirmed hazardous intrusions.

---

## 7. Conclusion

In this paper, we introduced an automated, real-time, vision-based industrial safety monitoring system capable of concurrent detection across five critical workplace hazard domains: falls, unauthorized perimeter breaches, machine proximity exposure, loitering/roaming, and ergonomic postural/micro-activity violations. By combining anchor-free YOLOv8 detection with ByteTrack tracking, closed-form polygonal signed-distance geofencing, fine-grained micro-action classification, and a fault-tolerant dual-engine kinematic pose estimator, the system resolves key challenges associated with false-alarm fatigue and corporate code integrity restrictions. Rigorous experimental evaluations demonstrate an aggregate F1-score of 0.936 and a sustained real-time throughput of 28.5 FPS on GPU hardware. The resulting solution provides a robust, scalable digital safety companion for modern industrial environments.

---

## References

[1] International Labour Organization, "Safety and Health at Work: A Global Imperative," *ILO World Congress on Safety and Health at Work Report*, Geneva, Switzerland, 2023.  
[2] N. D. Nath, A. H. Behzadan, and S. G. Paal, "Deep learning for site safety: Real-time detection of personal protective equipment," *Automation in Construction*, vol. 112, p. 103085, 2020.  
[3] X. Wu, D. Sahoo, and S. C. H. Hoi, "Recent advances in deep learning for object detection," *Neurocomputing*, vol. 396, pp. 39–64, 2020.  
[4] J. S. Warm, R. Parasuraman, and R. F. Matthews, "Vigilance requires hard mental work and is stressful," *Human Factors*, vol. 50, no. 3, pp. 433–441, 2008.  
[5] G. Jocher, A. Chaurasia, and J. Qiu, "YOLO by Ultralytics (Version 8.0.0)," *Ultralytics Software*, 2023. [Online]. Available: https://github.com/ultralytics/ultralytics  
[6] V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, and M. Grundmann, "BlazePose: On-device real-time body pose tracking," *arXiv preprint arXiv:2006.10204*, 2020.  
[7] D. Maji, S. Nagori, M. Mathew, and D. Poddar, "YOLO-Pose: Enhancing YOLO for multi-person pose estimation using object keypoint similarity loss," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR) Workshops*, 2022, pp. 2637–2646.  
[8] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object detection," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 779–788.  
[9] G. A. Pereira, "Fall detection for industrial setups using YOLOv8 variants," *arXiv preprint arXiv:2408.04605*, 2024.  
[10] N. Wojke, A. Bewley, and D. Paulus, "Simple online and realtime tracking with a deep association metric," in *Proc. IEEE Int. Conf. Image Process. (ICIP)*, 2017, pp. 3645–3649.  
[11] Y. Zhang, P. Sun, Y. Jiang, D. Yu, F. Weng, Z. Yuan, P. Luo, W. Liu, and X. Wang, "ByteTrack: Multi-object tracking by associating every detection box," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2022, pp. 1–21.  
[12] F. Bagalà, C. Becker, A. Cappello, L. Chiari, K. Aminian, J. M. Hausdorff, W. Zijlstra, and J. Klenk, "Evaluation of accelerometer-based fall detection algorithms on real-world falls," *PLoS ONE*, vol. 7, no. 5, p. e37062, 2012.  
[13] A. Núñez-Marcos, G. Azkune, and I. Arganda-Carreras, "Vision-based fall detection with convolutional neural networks," *Wireless Communications and Mobile Computing*, vol. 2017, pp. 1–16, 2017.  
[14] W. L. Sousa Lima, D. Souto, and J. J. P. C. Rodrigues, "Skeleton-based human fall detection using LSTM networks," *IEEE Access*, vol. 9, pp. 103551–103561, 2021.  
[15] S. Hignett and L. McAtamney, "Rapid Entire Body Assessment (REBA)," *Applied Ergonomics*, vol. 31, no. 2, pp. 201–205, 2000.  
[16] N. D. Nath and A. H. Behzadan, "Deep learning-based ergonomic risk assessment in construction," *Automation in Construction*, vol. 125, p. 103608, 2021.  
[17] W. Sultani, C. Chen, and M. Shah, "Real-world anomaly detection in surveillance videos," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 6479–6488.  
[18] A. Shahroudy, J. Liu, T. T. Ng, and G. Wang, "NTU RGB+D: A large scale dataset for 3D human activity analysis," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 1010–1019.  
[19] Z. Liu et al., "Video Swin Transformer," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 3202–3211.  
[20] K. Simonyan and A. Zisserman, "Two-stream convolutional networks for action recognition in videos," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2014, pp. 568–576.  
[21] L. Wang et al., "Temporal Segment Networks: Towards good practices for deep action recognition," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2016, pp. 20–36.  
[22] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, "Learning spatiotemporal features with 3D convolutional networks," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2015, pp. 4489–4497.  
[23] T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980–2988.  
[24] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, "Distance-IoU loss: Faster and better learning for bounding box regression," in *Proc. AAAI Conf. Artif. Intell.*, 2020, pp. 12993–13000.  
[25] E. Bochinski, V. Eiselein, and T. Sikora, "High-speed tracking-by-detection without using image information," in *Proc. IEEE Int. Workshop Traffic Online Anal. Watch (AVSS)*, 2017, pp. 1–6.  
