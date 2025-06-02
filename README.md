<p align="center">
  <h3 align="center"><strong>ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding</strong></h3>

<p align="center">
    <a href="https://jamesyjl.github.io/">Junliang Ye</a><sup>1,2*</sup>,
    <a href="https://thuwzy.github.io/">Zhengyi Wang</a><sup>1,2*</sup>,
    <a href="https://zhaorw02.github.io/">Ruowen Zhao</a><sup>1*</sup>,
    <a href="">Shenghao Xie</a><sup>3</sup>,
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml">Jun Zhu</a><sup>1,2†</sup>
    <br>
    <sup>*</sup>Equal Contribution.
    <br>
    <sup>†</sup>Corresponding authors.
    <br>
    <sup>1</sup>Tsinghua University,
    <sup>2</sup>ShengShu,
    <sup>3</sup>Peking University,
</p>

<div align="center">

<a href='https://arxiv.org/abs/2503.15265'><img src='https://img.shields.io/badge/arXiv-2503.15265-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://zhaorw02.github.io/DeepMesh/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/yejunliang23/ShapeLLM-7B-omni"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://huggingface.co/yejunliang23/3DVQVAE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-HF-orange"></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://huggingface.co/datasets/yejunliang23/3D-Alpaca'><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HF-orange">

</div>

<p align="center">
    <img src="assets/head.jpg">
</p>

## Release
- [6/03] 🔥🔥We released the pretrained weights for both **ShapeLLM-omni** (7B) and **3DVQVAE**.
- [6/03] 🔥🔥We released 50k high-quality 3D edited data pairs.

## Inference
We suggest using Gradio UI for visualizing inference.
```
python app.py
```

## Important Notes
- Please refer to our [project_page](https://zhaorw02.github.io/DeepMesh/) for more examples.
## Todo
- [ ] Release of training code.
- [ ] Release of model weights featuring multi-turn dialogue and 3D editing capabilities.
- [ ] Release of the entire 3D-Alpaca dataset.

## Acknowledgement
Our code is based on these wonderful repos:
* **[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)**
* **[TRELLIS](https://github.com/microsoft/TRELLIS)**
* **[PointLLM](https://github.com/OpenRobotLab/PointLLM)**
* **[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)**
* [LLaMA-Mesh](https://github.com/nv-tlabs/LLaMA-Mesh)


