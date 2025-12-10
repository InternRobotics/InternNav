<div align="center">

[![demo](assets/InternNav.gif "demo")](https://www.youtube.com/watch?v=fD0F1jIax5Y)

[![HomePage](https://img.shields.io/badge/HomePage-144B9E?logo=ReactOS&logoColor=white)](https://internrobotics.github.io/internvla-n1.github.io/)
[![Technique Report](https://img.shields.io/badge/Paper-B31B1B?logo=arXiv&logoColor=white)](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf)
[![doc](https://img.shields.io/badge/Document-FFA500?logo=readthedocs&logoColor=white)](https://internrobotics.github.io/user_guide/internnav/index.html)
[![GitHub star chart](https://img.shields.io/github/stars/InternRobotics/InternNav?style=square)](https://github.com/InternRobotics/InternNav)
[![GitHub Issues](https://img.shields.io/github/issues/InternRobotics/InternNav)](https://github.com/InternRobotics/InternNav/issues)
<a href="https://cdn.vansin.top/taoyuan.jpg"><img src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white" height="20" style="display:inline"></a>
[![Discord](https://img.shields.io/discord/1373946774439591996?logo=discord)](https://discord.gg/5jeaQHUj4B)

</div>

## 🏠 Introduction

InternNav is an All-in-one open-source toolbox for embodied navigation based on PyTorch, Habitat and Isaac Sim.

### Highlights
- Modular Support of the Entire Navigation System

We support modular customization and study of the entire navigation system, including vision-language navigation with discrete action space (VLN-CE), visual navigation (VN) given point/image/trajectory goals, and the whole VLN system with continuous trajectory outputs.

- Compatibility with Mainstream Simulation Platforms

The toolbox is compatible with different training and evaluation requirements, supporting different environments for the usage of mainstream simulation platforms such as Habitat and Isaac Sim.

- Comprehensive Datasets, Models and Benchmarks

The toolbox supports the most comprehensive 6 datasets \& benchmarks and 10+ popular baselines, including both mainstream and our established brand new ones.

- State of the Art

The toolbox supports the most advanced high-quality navigation dataset, InternData-N1, which includes 3k+ scenes and 830k VLN data covering diverse embodiments and scenes, and the first dual-system navigation foundation model with leading performance on all the benchmarks and zero-shot generalization capability in the real world, InternVLA-N1.

## 🔥 News

| Time   | Update |
|---------|--------|
| 2025/11 | InternNav v0.2.0 released — added distributed evaluation support for VLN-PE.|
| 2025/10 | Add a [inference-only demo](scripts/notebooks/inference_only_demo.ipynb) of InternVLA-N1. |
| 2025/10 | InternVLA-N1 [technical report](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf) is released. Please check our [homepage](https://internrobotics.github.io/internvla-n1.github.io/). |
| 2025/09 | Real-world deployment code of InternVLA-N1 released. Upload 3D printing [files](assets/3d_printing_files/go2_stand.STEP) for Unitree Go2. |
| 2025/07 | Hosting the 🏆 IROS 2025 Grand Challenge (see updates at [official website](https://internrobotics.shlab.org.cn/challenge/2025/)) |
| 2025/07 | InternNav v0.1.1 released |

## 📋 Table of Contents
- [🏠 Introduction](#-introduction)
- [🔥 News](#-news)
- [📚 Getting Started](#-getting-started)
- [📦 Overview of Benchmark \& Model Zoo](#-overview-of-benchmark-and-model-zoo)
- [🔧 Customization](#-customization)
- [👥 Contribute](#-contribute)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 📚 Getting Started

Please refer to the [documentation](https://internrobotics.github.io/user_guide/internnav/quick_start/index.html) for quick start with InternNav, from installation to training or evaluating supported models.

## 📦 Overview of Benchmark and Model Zoo

### Datasets \& Benchmarks

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
         <b>System2 (VLN-CE)</b>
      </td>
      <td>
         <b>System1 (VN)</b>
      </td>
      <td>
         <b>Whole-system (VLN)</b>
      </td>
   </tr>
   <tr align="center" valign="top">
      <td>
         <ul>
            <li align="left"><a href="">VLN-CE R2R</a></li>
            <li align="left"><a href="">VLN-CE RxR</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">Cluttered Envs</a></li>
            <li align="left"><a href="">GRScenes-100</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">VLN-CE</a></li>
            <li align="left"><a href="">VLN-PE</a></li>
         </ul>
      </td>
   </tbody>
</table>

### Models

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
         <b>System2 (VLN-CE)</b>
      </td>
      <td>
         <b>System1 (VN)</b>
      </td>
      <td>
         <b>Whole-system (VLN)</b>
      </td>
   </tr>
   <tr align="center" valign="top">
      <td>
         <ul>
            <li align="left"><a href="">StreamVLN</a></li>
            <li align="left"><a href="">InternVLA-N1-Preview (S2)</a></li>
            <li align="left"><a href="">InternVLA-N1 (S2)</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">DD-PPO</a></li>
            <li align="left"><a href="">iPlanner</a></li>
            <li align="left"><a href="">ViPlanner</a></li>
            <li align="left"><a href="">GNM</a></li>
            <li align="left"><a href="">ViNT</a></li>
            <li align="left"><a href="">NoMad</a></li>
            <li align="left"><a href="">NavDP</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">Seq2Seq</a></li>
            <li align="left"><a href="">CMA</a></li>
            <li align="left"><a href="">RDP</a></li>
            <li align="left"><a href="">InternVLA-N1-Preview</a></li>
            <li align="left"><a href="">InternVLA-N1</a></li>
         </ul>
      </td>
   </tbody>
</table>

### Benchmark Results

#### VLN-CE Task
| Model  | Dataset/Benchmark | NE | OS | SR | SPL | Download |
| ------ | ----------------- | -- | -- | --------- |  -- | --------- |
| `InternVLA-N1 (S2)` | R2R | 4.89 | 60.6 | 55.4 | 52.1| [Model](https://huggingface.co/InternRobotics/InternVLA-N1-S2) |
| `InternVLA-N1` | R2R | **4.83** | **63.3** | **58.2** | **54.0** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1) |
| `InternVLA-N1 (S2)` | RxR | 6.67 | 56.5 | 48.6 | 42.6 | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-S2) |
| `InternVLA-N1` | RxR | **5.91** | **60.8** | **53.5** | **46.1** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1) |
| `InternVLA-N1-Preview (S2)` | R2R | 5.09 | 60.9 | 53.7 | 49.7 | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview-S2) |
| `InternVLA-N1-Preview` | R2R | **4.76** | **63.4** | **56.7** | **52.6** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview) |
| `InternVLA-N1-Preview (S2)` | RxR | 6.39 | 60.1 | 50.5 | 43.3 | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview-S2) |
| `InternVLA-N1-Preview` | RxR | **5.65** | **63.2** | **53.5** | **45.7** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview) |

#### VLN-PE Task
| Model  | Dataset/Benchmark | NE | OS | SR | SPL | Download |
| ------ | ----------------- | -- | -- | -- | --- | --- |
| `Seq2Seq` | Flash | 8.27 | 43.0 | 15.7 | 9.7 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `CMA` | Flash | 7.52 | 45.0 | 24.4 | 18.2 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `RDP` | Flash | 6.98 | 42.5 | 24.9 | 17.5 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `InternVLA-N1-Preview` | Flash | **4.21** | **68.0** | **59.8** | **54.0** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview) |
| `InternVLA-N1` | Flash | **4.13** | **67.6** | **60.4** | **54.9** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1) |
| `Seq2Seq` | Physical | 7.88 | 28.1 | 15.1 | 10.7 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `CMA` | Physical | 7.26 | 31.4 | 22.1 | 18.6 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `RDP` | Physical | 6.72 | 36.9 | 25.2 | 17.7 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `InternVLA-N1-Preview` | Physical | **5.31** | **49.0** | **42.6** | **35.8** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview) |
| `InternVLA-N1` | Physical | **4.73** | **56.7** | **50.6** | **43.3** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1) |

#### Visual Navigation Task - PointGoal Navigation
| Model  | Dataset/Benchmark | SR | SPL | Download |
| ------ | ----------------- | -- | -- | --------- |
| `iPlanner` | ClutteredEnv | 84.8 | 83.6 | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `ViPlanner` | ClutteredEnv | 72.4 | 72.3 | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `InternVLA-N1 (S1)` | ClutteredEnv | **89.8** | **87.7** | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `iPlanner` | InternScenes | 48.8 | 46.7 | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `ViPlanner` | InternScenes | 54.3 | 52.5 | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `InternVLA-N1 (S1)` | InternScenes | **65.7** | **60.7** | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |



**NOTE:**
- VLN-CE RxR benchmark and StreamVLN will be supported soon.

## 🔧 Customization

Please refer to the [tutorial](https://internrobotics.github.io/user_guide/internnav/tutorials/index.html) for advanced usage of InternNav, including customization of datasets, models and experimental settings.

## 👥 Contribute

If you would like to contribute to InternNav, please check out our [contribution guide]().
For example, raising issues, fixing bugs in the framework, and adapting or adding new policies and data to the framework.

**Note:** We welcome the feedback of the model's zero-shot performance when deploying in your own environment. Please show us your results and offer us your future demands regarding the model's capability. We will select the most valuable ones and collaborate with users together to solve them in the next few months :)

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@misc{internnav2025,
    title = {{InternNav: InternRobotics'} open platform for building generalized navigation foundation models},
    author = {InternNav Contributors},
    howpublished={\url{https://github.com/InternRobotics/InternNav}},
    year = {2025}
}
```

If you use the specific pretrained models and benchmarks, please kindly cite the original papers involved in our work. Related BibTex entries of our papers are provided below.

<details><summary>Related Work BibTex</summary>

```BibTex
@misc{internvla-n1,
    title = {{InternVLA-N1: An} Open Dual-System Navigation Foundation Model with Learned Latent Plans},
    author = {InternNav Team},
    year = {2025},
    booktitle={arXiv},
}
@inproceedings{vlnpe,
  title={Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities},
  author={Wang, Liuyi and Xia, Xinyuan and Zhao, Hui and Wang, Hanqing and Wang, Tai and Chen, Yilun and Liu, Chengju and Chen, Qijun and Pang, Jiangmiao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
@misc{streamvln,
    title = {StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling},
    author = {Wei, Meng and Wan, Chenyang and Yu, Xiqian and Wang, Tai and Yang, Yuqiang and Mao, Xiaohan and Zhu, Chenming and Cai, Wenzhe and Wang, Hanqing and Chen, Yilun and Liu, Xihui and Pang, Jiangmiao},
    booktitle={arXiv},
    year = {2025}
}
@misc{navdp,
    title = {NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance},
    author = {Wenzhe Cai, Jiaqi Peng, Yuqiang Yang, Yujian Zhang, Meng Wei, Hanqing Wang, Yilun Chen, Tai Wang and Jiangmiao Pang},
    year = {2025},
    booktitle={arXiv},
}
```

</details>


## 📄 License

InternNav's codes are [MIT licensed](LICENSE).
The open-sourced InternData-N1 data are under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>.
Other datasets like VLN-CE inherit their own distribution licenses.

## 👏 Acknowledgement

- [InternUtopia](https://github.com/InternRobotics/InternUtopia) (Previously `GRUtopia`): The closed-loop evaluation and GRScenes-100 data in this framework relies on the InternUtopia framework.
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): Diffusion policy implementation.
- [LongCLIP](https://github.com/beichenzbc/Long-CLIP): Long-text CLIP model.
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE): Vision-and-Language Navigation in Continuous Environments based on Habitat.
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL): The pretrained vision-language foundation model.
- [LeRobot](https://github.com/huggingface/lerobot): The data format used in this project largely follows the conventions of LeRobot.
