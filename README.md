
## <div align="center"> Chasing Faster ConvNet for Efficient Super-Resolution </div>

**Overview:** The repository records a path of chasing faster ConvNet.

The repo is still under construction!

---

☁️ EFDN for NTIRE 2022 ESR
---
> ***Edge-enhanced Feature Distillation Network for Efficient Super-Resolution*** \
> [Yan Wang](https://scholar.google.com/citations?user=SXIehvoAAAAJ&hl=en) \
> Nankai University

<a href="https://arxiv.org/abs/2204.08759" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2204.08759-b31b1b.svg?style=flat" /></a>
<a href="https://data.vision.ee.ethz.ch/cvl/ntire22/posters/Wang_Edge_074-poster-Edge-enhanced%20Feature%20Distillation%20Network%20for%20Efficient%20Super-Resolution.pdf" alt="Poster">
    <img src="https://img.shields.io/badge/poster-NTIRE 2022-brightgreen" /></a> 
<a href="https://github.com/icandle/EFDN" alt="Poster">
    <img src="https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge%3Fref%3Dmaster&style=flat" /></a>
</p>


**Summary**: **5th** solution of **Model Complexity** in the [NTIRE 2022](https://cvlai.net/ntire/2022/) Challenge on Efficient Super-Resolution. Involoving the modification of convolution and network architecture.
- 🌟 *Convolution*: edge-ehanced reparameter block (EDBB) with a corresponding edge loss .
- 📦 *Attention*: original ESA.
- 📦 *Backbone*: backbone searched by network-level NAS.



🌥️ PFDN for NTIRE 2023 ESR
---
> ***Partial Feature Distillation Network for Efficient Super-Resolution*** \
> [Yan Wang](https://scholar.google.com/citations?user=SXIehvoAAAAJ&hl=en), [Erlin Pan](https://scholar.google.com/citations?user=Z6RyGacAAAAJ&hl=en&oi=ao)<sup>†</sup>, [Qixuan Cai](https://scholar.google.com/citations?user=tPbL7HMAAAAJ&hl=en&oi=ao), Xinan Dai \
> Nankai University, University of Electronic Science and Technology of China, Tianjin University

<a href="https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Li_NTIRE_2023_Challenge_on_Efficient_Super-Resolution_Methods_and_Results_CVPRW_2023_paper" alt="Report">
    <img src="https://img.shields.io/badge/report-NTIRE 2023-367DBD" /></a>
<a href="https://github.com/icandle/PlainUSR/blob/main/2023_PFDN_NTIRE/factsheet/08-PFDN-Factsheet.pdf">
    <img src="https://img.shields.io/badge/docs-factsheet-8A2BE2" /></a>
<a href="https://github.com/icandle/PlainUSR/blob/main/models/team08_PFDN.py" alt="Report">
    <img src="https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge%3Fref%3Dmaster&style=flat" /></a>
</p>

**Summary**: **Winner** of **Overall Evaluation** and **4th** of **Runtime** in the [NTIRE 2023](https://cvlai.net/ntire/2023/) Challenge on Efficient Super-Resolution. Involoving the modification of convolution and network architecture.
- ⭐️ *Convolution*: integrating partial convolution and RRRB.
- 📦 *Attention*: efficient ESA.
- 📦 *Backbone*: ResNet-style backbone.


| <sub> Model </sub> | <sub> Runtime[ms] </sub> | <sub> Params[M] </sub> | <sub> Flops[G] </sub> |  <sub> Acts[M] </sub> | <sub> GPU Mem[M] </sub> |
|  :----:  | :----:  |  :----:  | :----:  |  :----:  | :----:  |
|  RFDN  | 35.54  |  0.433  | 27.10  |  112.03  | 788.13  |
|  PFDN  | 20.49  |  0.272  | 16.76  |  65.10  | 296.45  |

⛅️ PFDNLite for NTIRE 2024 ESR
---
> ***Lightening Partial Feature Distillation Network for Efficient Super-Resolution*** \
> [Yan Wang](https://scholar.google.com/citations?user=SXIehvoAAAAJ&hl=en), Yi Liu, Qing Wang, Gang Zhang, Liou Zhang, Shijie Zhao<sup>†</sup> \
> Nankai University, ByteDance

<a href="https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Ren_The_Ninth_NTIRE_2024_Efficient_Super-Resolution_Challenge_Report_CVPRW_2024_paper.pdf" alt="Report">
    <img src="https://img.shields.io/badge/report-NTIRE 2024-367DBD" /></a>
<a href="https://github.com/icandle/BSR/blob/main/factsheet/NTIRE_2024_ESR.pdf">
    <img src="https://img.shields.io/badge/docs-factsheet-8A2BE2" /></a>
<a href="https://github.com/icandle/BSR" alt="Report">
    <img src="https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge%3Fref%3Dmaster&style=flat" /></a>
</p> 

**Summary**: **3rd** of **Overall Evaluation** and **3rd** of **Runtime** in the [NTIRE 2024](https://cvlai.net/ntire/2024/) Challenge on Efficient Super-Resolution. Involoving the modification of convolution, attention and network pruning.
- 📦 *Convolution*: RepMBConv in PlainUSR.
- 📦 *Attention*: LIA in PlainUSR.
- ⭐️ *Backbone*: ABPN-style backbone and block pruning.

🌤️ PlainUSR for ACCV 2024
---
> ***PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution*** \
> [Yan Wang](https://scholar.google.com/citations?user=SXIehvoAAAAJ&hl=en), [Yusen Li](https://scholar.google.com/citations?user=4EJ9aekAAAAJ&hl=en&oi=ao)<sup>†</sup>, Gang Wang, Xiaoguang Liu \
> Nankai University 

<a href="https://arxiv.org/abs/2409.13435" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2409.13435-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/icandle/PlainUSR/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-MIT--License-%23B7A800" /></a>
</p>

**Summary**:  we present PlainUSR incorporating three pertinent modifications (convolution, attention, and backbone) to expedite ConvNet for efficient SR.
- 🌟 *Convolution*: Reparameterized MobileNetV3 Convolution (RepMBConv).
- ⭐️ *Attention*: Local Importance-based Attention (LIA).
- 🌟 *Backbone*: Plain U-Net.

🌤️ ESPAN for NTIRE 2025 ESR
---
> ***Expanded SPAN for Efficient Super-Resolution*** \
> [Qing Wang](https://scholar.google.com/citations?user=FT9ZYSwAAAAJ&hl=en&oi=sra), [Yan Wang](https://scholar.google.com/citations?user=SXIehvoAAAAJ&hl=en), [Hongyu An](https://scholar.google.com/citations?user=pPsK7L4AAAAJ&hl=en&oi=sra), Yi Liu, Liou Zhang, Shijie Zhao<sup>†</sup> \
> ByteDance

<a href="https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Wang_Expanded_SPAN_for_Efficient_Super-Resolution_CVPRW_2025_paper.pdf" alt="Report">
    <img src="https://img.shields.io/badge/CVF-NTIRE 2025-367DBD" /></a>
<a href="">
    <img src="https://img.shields.io/badge/docs-factsheet-8A2BE2" /></a>
<a href="" alt="Report">
    <img src="https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge%3Fref%3Dmaster&style=flat" /></a>
</p> 

**Summary**: **5th** of **Overall Evaluation** and **3rd** of **Runtime** in the [NTIRE 2024](https://cvlai.net/ntire/2024/) Challenge on Efficient Super-Resolution.


☀️ PlainUSRv2 
---

To be updated.

💖 Acknowledgments
---
We would thank [BasicSR](https://github.com/XPixelGroup/BasicSR), [ECBSR](https://github.com/xindongzhang/ECBSR), [DBB](https://github.com/DingXiaoH/DiverseBranchBlock), [ETDS](https://github.com/ECNUSR/ETDS), [FasterNet](https://github.com/JierunChen/FasterNet), etc, for their enlightening work!

🎓 Citation
---
```
@inproceedings{wang2022edge,
  title={Edge-enhanced Feature Distillation Network for Efficient Super-Resolution},
  author={Wang, Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={777--785},
  year={2022}
}

@inproceedings{wang2024plainusr,
  title={PlainUSR: Chasing Faster ConvNet for Efficient Super-Resolution},
  author={Wang, Yan and Li, Yusen and Wang, Gang and Liu, Xiaoguang},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={4262--4279},
  year={2024}
}

@inproceedings{wang2025expanded,
  title={Expanded SPAN for Efficient Super-Resolution},
  author={Wang, Qing and Wang, Yang and An, Hongyu and Liu, Yi and Zhang, Liou and Zhao, Shijie},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={967--976},
  year={2025}
}
```
