<div align="center">

[![demo](assets/InternNav.gif "demo")](https://www.youtube.com/watch?v=fD0F1jIax5Y)

[![HomePage](https://img.shields.io/badge/HomePage-144B9E?logo=ReactOS&logoColor=white)](https://internrobotics.github.io/internvla-n1.github.io/)
[![Technical Report — InternVLA-N1](https://img.shields.io/badge/Technical_Report-InternVLA--N1-BB2649?logo=adobeacrobatreader&logoColor=white)](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf)
[![DualVLN Paper — arXiv](https://img.shields.io/badge/arXiv-DualVLN-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.08186)
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
| 2026/01 | InternNav v0.3.0 released. |
| 2025/12 | We introduce Interactive Instance Goal Navigation (IIGN) and release VL-LN Bench to enable InternVLA-N1 to solve this task, with large-scale dialog-trajectory collection plus training and evaluation support. See [our website](https://0309hws.github.io/VL-LN.github.io/) for details.|
| 2025/12 | Training code for InternVLA-N1 and the corresponding [usage doc](https://internrobotics.github.io/user_guide/internnav/quick_start/training.html) is now available. This release provides two model configurations: InternVLA-N1 (Dual System)<span style="color: #28a745; font-size: 0.9em"> with NavDP*</span> and InternVLA-N1 (Dual System)<span style="color: #28a745; font-size: 0.9em"> DualVLN </span>. For model architecture and training details, please refer to the [DualVLN paper](https://arxiv.org/abs/2512.08186).|
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
- [📦 Overview of Benchmark \& Model Zoo](#-overview)
- [🔧 Customization](#-customization)
- [👥 Contribute](#-contribute)
- [🚀 Community Deployment & Best Practices](#-community-deployment--best-practices)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [👏 Acknowledgements](#-acknowledgements)

## 📚 Getting Started

Please refer to the [documentation](https://internrobotics.github.io/user_guide/internnav/quick_start/index.html) for quick start with InternNav, from installation to training or evaluating supported models.

## 📦 Overview

### 🧪 Supported Benchmarks

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
         <b>VLN Benchmarks</b>
      </td>
      <td>
         <b>VN Benchmarks</b>
      </td>
   </tr>
   <tr align="center" valign="top">
      <td>
         <ul>
            <li align="left"><a href="https://arxiv.org/abs/2004.02857">VLN-CE</a></li>
            <li align="left"><a href="https://arxiv.org/abs/2507.13019">VLN-PE</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="https://arxiv.org/abs/2505.08712">Cluttered Environments</a></li>
            <li align="left"><a href="https://arxiv.org/abs/2505.08712">GRScenes-100</a></li>
         </ul>
      </td>
   </tbody>
</table>

### 🤗 Model Zoo & Downloads

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
         <b>🧠 VLN Single-System</b>
      </td>
      <td>
         <b>🎯 VN System (System1)</b>
      </td>
      <td>
         <b>🤝 VLN Multi-System</b>
      </td>
   </tr>
   <tr align="center" valign="top">
      <td>
         <ul>
            <li align="left"><a href="https://huggingface.co/InternRobotics/VLN-PE">Seq2Seq</a></li>
            <li align="left"><a href="https://huggingface.co/InternRobotics/VLN-PE">CMA</a></li>
            <li align="left"><a href="https://huggingface.co/InternRobotics/VLN-PE">RDP</a></li>
            <li align="left"><a href="https://github.com/InternRobotics/StreamVLN">StreamVLN</a> <em>(coming soon)</em></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library">DD-PPO</a></li>
            <li align="left"><a href="https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library">iPlanner</a></li>
            <li align="left"><a href="https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library">ViPlanner</a></li>
            <li align="left"><a href="https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library">GNM</a></li>
            <li align="left"><a href="https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library">ViNT</a></li>
            <li align="left"><a href="https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library">NoMad</a></li>
            <li align="left"><a href="https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library">NavDP <small>InternVLA-N1 (System 1)</small></a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="https://huggingface.co/InternRobotics/InternVLA-N1-System2">InternVLA-N1 (System 2)</a> + <a href="https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library" style="color: #1e90ff;">Decoupled System1</a></li>
            <li align="left"><a href="https://huggingface.co/InternRobotics/InternVLA-N1-w-NavDP">InternVLA-N1 (Dual System) <small>w/ NavDP*</small> </a>  <small> (NavDP*</small> indicates joint tuning with System 2)</li>
            <li align="left"><a href="https://huggingface.co/InternRobotics/InternVLA-N1-DualVLN">InternVLA-N1 (Dual System) <small>DualVLN</small></a></li>
         </ul>
      </td>
   </tbody>
</table>

<!-- **📝 Note:**
- VLN-CE RxR benchmark and StreamVLN model will be supported soon.
- **NE**: Navigation Error (lower is better) • **OS**: Oracle Success (higher is better) • **SR**: Success Rate (higher is better) • **SPL**: Success weighted by Path Length (higher is better) -->


### 📊 Benchmark Results


#### <u>VLN-CE Benchmarks</u>

**📍 R2R Dataset**
| Model | Observation | NE ↓ | OS ↑ | SR ↑ | SPL ↑ |
|-------|-------------|------|------|------|-------|
| InternVLA-N1-wo-dagger (S2) + [ShortestPathFollower](https://aihabitat.org/docs/habitat-lab/habitat.tasks.nav.shortest_path_follower.ShortestPathFollower.html) | - | 4.89 | 60.6 | 55.4 | 52.1 |
| InternVLA-N1-wo-dagger (Dual System) <span style="color: #28a745; font-size: 0.9em"> with NavDP*</span>  | RGB-D | 4.83 | 63.3 | 58.2 | 54.0 |
| InternVLA-N1 (S2) + [ShortestPathFollower](https://aihabitat.org/docs/habitat-lab/habitat.tasks.nav.shortest_path_follower.ShortestPathFollower.html) | - | 4.25 | 68.3 | 60.9 | 55.2 |
| InternVLA-N1 (Dual System)<span style="color: #28a745; font-size: 0.9em"> with NavDP*</span> | RGB-D | 4.22 | 70.4 | 64.1 | 58.1 |
| InternVLA-N1 (Dual System)<span style="color: #28a745; font-size: 0.9em"> DualVLN </span> | RGB | **4.05** | **70.7** | **64.3** | **58.5** |

**📍 RxR Dataset**
| Model | Observation | NE ↓ |  SR ↑ | SPL ↑ | nDTW ↑ |
|-------|-------------|------|------|------|-------|
| InternVLA-N1 (S2) + [ShortestPathFollower](https://aihabitat.org/docs/habitat-lab/habitat.tasks.nav.shortest_path_follower.ShortestPathFollower.html) | - | 5.71 | 63.5 | 55.0 | 46.8 |
| InternVLA-N1 (Dual System)<span style="color: #28a745; font-size: 0.9em"> with NavDP*</span> | RGB-D | 4.70 | 59.7 | 50.6 | 69.7 |
| InternVLA-N1 (Dual System)<span style="color: #28a745; font-size: 0.9em"> DualVLN </span> | RGB | **4.58** | **61.4** | **51.8** | **70.0** |

#### <u>VLN-PE Benchmarks</u>

**📍 Flash Controller on R2R Unseen**
| Model | NE ↓ | OS ↑ | SR ↑ | SPL ↑ |
|-------|------|------|------|-------|
| Seq2Seq | 8.27 | 43.0 | 15.7 | 9.7 |
| CMA | 7.52 | 45.0 | 24.4 | 18.2 |
| RDP | 6.98 | 42.5 | 24.9 | 17.5 |
| InternVLA-N1 (System 2) + iPlanner | 4.91 | 55.53 | 47.07 | 41.09 |
| InternVLA-N1 (System 2) + NavDP | 4.22 | 67.33 | 58.72 | 50.98 |
| InternVLA-N1 (Dual System)<span style="color: #28a745; font-size: 0.9em"> DualVLN </span> | **3.90** | **69.93** | **63.62** | **56.49** |

**📍 Physical Controller on R2R Unseen**
| Model | NE ↓ | OS ↑ | SR ↑ | SPL ↑ |
|-------|------|------|------|-------|
| Seq2Seq | 7.88 | 28.1 | 15.1 | 10.7 |
| CMA | 7.26 | 31.4 | 22.1 | 18.6 |
| RDP | 6.72 | 36.9 | 25.2 | 17.7 |
| InternVLA-N1 (Dual System)<span style="color: #28a745; font-size: 0.9em"> DualVLN </span> | **4.66** | **55.9** | **51.6** | **42.49** |


#### <u>Visual Navigation Benchmarks</u>

**📍 ClutteredEnv Dataset**
| Model | SR ↑ | SPL ↑ |
|-------|------|-------|
| iPlanner | 84.8 | 83.6 |
| ViPlanner | 72.4 | 72.3 |
| NavDP <InternVLA-N1 (System 1)> | **89.8** | **87.7** |

**📍 InternScenes Dataset**
| Model | SR ↑ | SPL ↑ |
|-------|------|-------|
| iPlanner | 48.8 | 46.7 |
| ViPlanner | 54.3 | 52.5 |
| NavDP <InternVLA-N1 (System 1)> | **65.7** | **60.7** |

## 🔧 Customization

Please refer to the [tutorial](https://internrobotics.github.io/user_guide/internnav/tutorials/index.html) for advanced usage of InternNav, including customization of datasets, models and experimental settings.

## 👥 Contribute

If you would like to contribute to InternNav, please check out our [contribution guide]().
For example, raising issues, fixing bugs in the framework, and adapting or adding new policies and data to the framework.

**Note:** We welcome the feedback of the model's zero-shot performance when deploying in your own environment. Please show us your results and offer us your future demands regarding the model's capability. We will select the most valuable ones and collaborate with users together to solve them in the next few months :)

## 🚀 Community Deployment & Best Practices

We are excited to see InternNav being deployed and extended by the community across different robots and real-world scenarios.
Below are selected community-driven deployment guides and solution write-ups, which may serve as practical references for advanced users.

- **IROS Challenge Nav Track: Champion Solution (2025)**
  A complete system-level solution and design analysis for Vision-and-Language Navigation in Physical Environments.
  🔗 https://zhuanlan.zhihu.com/p/1969046543286907790

- **Go2 Series Deployment Tutorial (ShanghaiTech University)**
  Step-by-step edge deployment guide for InternNav-based perception and navigation.
  🔗 https://github.com/cmjang/InternNav-deploy

- **G1 Series Deployment Tutorial (Wuhan University)**
  Detailed educational materials on vision-language navigation deployment.
  🔗 [*Chapter 5: Vision-Language Navigation (Part II)*](https://mp.weixin.qq.com/s/p3cJzbRvecMajiTh9mXoAw)

## 🔗 Citation

If you find our InternVLA-N1 (Dual System) model helpful, please cite our ICLR paper and previous technical report:
```bibtex
@misc{wei2025groundslowfastdualsystem,
      title={Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Language Navigation},
      author={Meng Wei and Chenyang Wan and Jiaqi Peng and Xiqian Yu and Yuqiang Yang and Delin Feng and Wenzhe Cai and Chenming Zhu and Tai Wang and Jiangmiao Pang and Xihui Liu},
      year={2025},
      eprint={2512.08186},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.08186},
}
@misc{internvla-n1,
    title = {{InternVLA-N1: An} Open Dual-System Navigation Foundation Model with Learned Latent Plans},
    author = {InternNav Team},
    year = {2025},
    booktitle={arXiv},
}
```

If you use this InternNav codebase to develop your method, please cite our codebase:
```bibtex
@misc{internnav2025,
    title = {{InternNav: InternRobotics'} open platform for building generalized navigation foundation models},
    author = {InternNav Contributors},
    howpublished={\url{https://github.com/InternRobotics/InternNav}},
    year = {2025}
}
```


<details><summary>If you use the specific pretrained models and benchmarks, please kindly cite the original papers below.</summary>

```BibTex

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
