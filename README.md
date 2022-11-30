<h1 align="center">:fire: FLAME Universe :fire:</h1>


<p align="center"> 
<img src="gifs/model_variations.gif">
</p>

FLAME is a lightweight and expressive generic head model learned from over 33,000 of accurately aligned 3D scans. FLAME combines a linear identity shape space (trained from head scans of 3800 subjects) with an articulated neck, jaw, and eyeballs, pose-dependent corrective blendshapes, and additional global expression blendshapes. For details please see the [scientific publication](https://ps.is.tuebingen.mpg.de/uploads_file/attachment/attachment/400/paper.pdf).

## Code

List of public repositories that use FLAME.

- [BFM_to_FLAME](https://github.com/TimoBolkart/BFM_to_FLAME): Conversion from Basel Face Model (BFM) to FLAME.
- [DECA](https://github.com/YadiraF/DECA):  Reconstruction of 3D faces with animatable facial expression detail from a single image.
- [EMOCA](https://github.com/radekd91/emoca): Reconstruction of emotional 3D faces from a single image.
- [FaceFormer](https://github.com/EvelynFan/FaceFormer): Speech-driven facial animation of meshes in FLAME mesh topology.
- [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch): FLAME PyTorch layer. 
- [flame-fitting](https://github.com/Rubikplayer/flame-fitting): Fitting of FLAME to scans. 
- [FLAME-Blender-Add-on](https://github.com/TimoBolkart/FLAME-Blender-Add-on): FLAME Blender Add-on.
- [GIF](https://github.com/ParthaEth/GIF): Generating face images with FLAME parameter control. 
- [learning2listen](https://github.com/evonneng/learning2listen): Modeling interactional communication in dyadic conversations. 
- [MICA](https://github.com/Zielon/MICA): Reconstruction of metrically accurated 3D faces from a single image. 
- [neural-head-avatars](https://github.com/philgras/neural-head-avatars): Building a neural head avatar from video sequences. 
- [photometric_optimization](https://github.com/HavenFeng/photometric_optimization): Fitting of FLAME to images using differentiable rendering. 
- [RingNet](https://github.com/soubhiksanyal/RingNet): Reconstruction of 3D faces from a single image. 
- [SAFA](https://github.com/Qiulin-W/SAFA): Animation of face images.  
- [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME): Fit FLAME to 2D/3D landmarks, FLAME meshes, or sample textured meshes. 
- [video-head-tracker](https://github.com/philgras/video-head-tracker): Track 3D heads in video sequences. 
- [VOCA](https://github.com/TimoBolkart/voca): Speech-driven facial animation of meshes in FLAME mesh topology.

## Datasets

List of datasets with meshes in FLAME topology. 

- [VOCASET](https://github.com/TimoBolkart/voca): 12 subjects, 40 speech sequences each with synchronized audio
- [CoMA dataset](https://coma.is.tue.mpg.de/download.php): 12 subjects, 12 extreme dynamic expressions each.
- [D3DFACS](https://flame.is.tue.mpg.de/download.php): 10 subjects, 519 dynamic expressions in total
- [LYHM](https://www-users.cs.york.ac.uk/~nep/research/Headspace/): 1216 subjects, one neutral expression mesh each. 
- [Stirling](https://github.com/Zielon/MICA/tree/master/datasets): 133 subjects, one neutral expression mesh each. 
- [Florence 2D/3D](https://github.com/Zielon/MICA/tree/master/datasets): 53 subjects, one neutral expression mesh each. 
- [FaceWarehouse](http://kunzhou.net/zjugaps/facewarehouse/): 150 subjects, one neutral expression mesh each. 
- [FRGC](https://github.com/Zielon/MICA/tree/master/datasets): 531 subjects, one neutral expression mesh each. 
- [BP4D+](https://github.com/Zielon/MICA/tree/master/datasets): 127 subjects, one neutral expression mesh each. 

## Publications

List of FLAME-based scientific publications.

#### 2022

- [Next3D: Generative Neural Texture Rasterization for 3D-Aware Head Avatars](https://arxiv.org/pdf/2211.11208.pdf)
- [Visual Speech-Aware Perceptual 3D Facial Expression Reconstruction from Videos](https://arxiv.org/pdf/2207.11094.pdf)
- [Realistic One-shot Mesh-based Head Avatars (ECCV 2022)](https://arxiv.org/pdf/2206.08343.pdf)
- [Towards Metrical Reconstruction of Human Faces (ECCV 2022)](https://arxiv.org/pdf/2204.06607.pdf)
- [Towards Racially Unbiased Skin Tone Estimation via Scene Disambiguation (ECCV 2022)](https://arxiv.org/pdf/2205.03962.pdf)
- [Neural Emotion Director: Speech-preserving semantic control of facial expressions in “in-the-wild” videos (CVPR 2022)](https://arxiv.org/pdf/2112.00585.pdf)
- [RigNeRF: Fully Controllable Neural 3D Portraits (CVPR 2022)](https://arxiv.org/pdf/2206.06481.pdf)
- [I M Avatar: Implicit Morphable Head Avatars from Videos (CVPR 2022)](https://arxiv.org/pdf/2112.07471.pdf)
- [Neural head avatars from monocular RGB videos (CVPR 2022)](https://arxiv.org/pdf/2112.01554.pdf)
- [Learning to Listen: Modeling Non-Deterministic Dyadic Facial Motion (CVPR 2022)](https://arxiv.org/pdf/2204.08451.pdf)
- [Simulated Adversarial Testing of Face Recognition Models (CVPR 2022)](https://arxiv.org/pdf/2106.04569.pdf)
- [EMOCA: Emotion Driven Monocular Face Capture and Animation (CVPR 2022)](https://arxiv.org/pdf/2204.11312.pdf)
- [Generating Diverse 3D Reconstructions from a Single Occluded Face Image (CVPR 2022)](https://arxiv.org/pdf/2112.00879.pdf)
- [Accurate 3D Hand Pose Estimation for Whole-Body 3D Human Mesh Estimation (CVPR-W 2022)](https://arxiv.org/pdf/2011.11534.pdf)
- [MOST-GAN: 3D Morphable StyleGAN for Disentangled Face Image Manipulation (AAAI 2022)](https://arxiv.org/pdf/2111.01048.pdf)

#### 2021

- [MorphGAN: One-Shot Face Synthesis GAN for Detecting Recognition Bias (BMVC 2021)](https://arxiv.org/pdf/2012.05225.pdf)
- [SIDER : Single-Image Neural Optimization for Facial Geometric Detail Recovery (3DV 2021)](https://arxiv.org/pdf/2108.05465.pdf)
- [SAFA: Structure Aware Face Animation (3DV 2021)](https://arxiv.org/pdf/2111.04928.pdf)
- [Learning an Animatable Detailed 3D Face Model from In-The-Wild Images (SIGGRAPH 2021)](https://arxiv.org/pdf/2012.04012.pdf)
- [Monocular Expressive Body Regression through Body-Driven Attention (ECCV 2020)](https://arxiv.org/pdf/2008.09062.pdf)

#### 2020

- [GIF: Generative Interpretable Faces (3DV 2020)](https://arxiv.org/pdf/2009.00149.pdf)

#### 2019

- [Learning to Regress 3D Face Shape and Expression from an Image without 3D Supervision (CVPR 2019)](https://arxiv.org/pdf/1905.06817.pdf)





