![Static Badge](https://img.shields.io/badge/course-NTUST-blue)
![Static Badge](https://img.shields.io/badge/AI-Attack_and_Defense-orange)
# Face De Identification Attacks and Defenses

## Files
```bash
📂Face_De-Identification_Attacks_and_Defenses/
    ├── 📃attack.ipynb # CNN Re-identification attack 
    ├── 📃Demo.ipynb # Demo all different image obfuscation
    ├── 📄HW_Multimedia Security_Face De-Identification.pdf # project step by step description
    ├── 📃main_program.py
    ├── 📂src/
    │   ├── 📃data_utils.py # essential data utility functions
    │   ├── 📃draw_plot.py # draw comparison plots
    │   ├── 📃image_obfuscator.py # all methods implement here
    │   └── 📃__init__.py
    └── 📂results/ # all different comparison figures
        ├── 📊epsilon_comparison_mse.png 
        ├── 📊epsilon_comparison_ssim.png
        ├── 📊methods_comparison_eps0.5.png
        ├── 📊step 3_gaussian_blur.png
        ├── 📊step 3_pixelized.png
        ├── 📊step1_gaussian_blur.png
        └── 📊step1_pixelized.png
```


## Dataset
* [AT&T Face Dataset](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/)

## Environment
1. git clone this repo
2. run docker container
```bash
docker run -it --name face-de-
identification --gpus all --ipc=host -v ./:/workspace --rm -p 8888:8888 allenlin316/face-de-identification
```
3. run jupyter notebook at `/workspace` 
```bash
jupyter notebook
```