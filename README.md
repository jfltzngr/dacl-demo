# dacl-demo

*dacl-demo* is a tutorial repo to demonstrate how to use baselines from [*dacl.ai*](https://dacl.ai) or rather [*bikit*](https://github.com/phiyodr/building-inspection-toolkit) for inference. As soon as you have installed the requirements you can unleash the whole dacl power inside the jupyter notebook `demo.ipynb`.

Feel free to load different images into the assets directory and evaluate the *dacl* models on your own data!

|<p align="mid"><img src="assets/11_001990.jpg" alt="drawing" width="300"/></p>|<p align="mid"><img src="assets/11_008121.jpg" alt="drawing" width="300"/></p>|
|:--:|:--:| 
|<p align="mid"><img src="assets/11_010057.jpg" alt="drawing" width="300"/></p>|<p align="mid"><img src="assets/11_010332.jpg" alt="drawing" width="300"/></p>|


***Examples of images representing detectable damage with available dacl-models.** Crack (Top left); Spalling, Effloresence, BarsExposed, Rust (Top right); Crack, Efflorescence (Bottom left); Spalling, Effloresence, BarsExposed, Rust (Bottom right)*

## Available Models

| Modelname             | Dataset           | EMR   | F1   | Tag          | Checkpoint                |CorrespNameOnBikit*                   |
|-----------------------|-------------------|-------|------|--------------|---------------------------|--------------------------------------|
| Code_res_dacl         | codebrim_balanced | 73.73 | 0.85 | ResNet       | Code_res_dacl.pth         |CODEBRIMbalanced_ResNet50_hta         |
| Code_mobilev2_dacl    | codebrim_balanced |70.41  | 0.84 | MobileNetV2  | Code_mobilev2_dacl.pth    |CODEBRIMbalanced_MobileNetV2          |
| Code_mobile_dacl      | codebrim_balanced | 69.46 | 0.83 | MobileNet    | Code_mobile_dacl.pth      |CODEBRIMbalanced_MobileNetV3Large_hta |
| Code_eff_dacl         | codebrim_balanced | 68.67 | 0.84 | EfficientNet | Code_eff_dacl.pth         |CODEBRIMbalanced_EfficientNetV1B0_dhb |
| McdsBikit_mobile_dacl | mcds_bikit        | 54.44 | 0.66 | MobileNet    | McdsBikit_mobile_dacl.pth |MCDSbikit_MobileNetV3Large_hta        |
| McdsBikit_eff_dacl    | mcds_bikit        | 51.85 | 0.65 | EfficientNet | McdsBikit_eff_dacl.pth    |MCDSbikit_EfficientNetV1B0_dhb        |
| McdsBikit_res_dacl    | mcds_bikit        | 48.15 | 0.62 | ResNet       | McdsBikit_res_dacl.pth    |MCDSbikit_ResNet50_dhb                |

**CorrespNameOnBikit* displays the name which you can utilize to download the model via *bikit*. For further information about how to get the baselines from *bikit* check out the ***Models*** section in the README of [*bikit*](https://github.com/phiyodr/building-inspection-toolkit). 

## Structure

```
├── assets
│   └── *.jpg # example images
├── cat_to_name.json # Contains labels for each dataset
├── demo.ipynb # Main code
├── LICENSE
├── models
│   └── *.pth # checkpoints
├── README.md
└── requirements.txt
```