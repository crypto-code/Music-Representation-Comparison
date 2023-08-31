# Music-Representation-Comparison
This is the repo with the code to conduct a comparative analysis of different audio representation models.


## Reproducability

This repo using the [MagnaTagATune](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset) dataset to evaluate the performance of different music representation model in the downstream task of music tagging.

### Dataset

The audio files for MagnaTagATune dataset can be downloaded [here](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset). Extract the audio files to audio directory in MTT folder. The directory structure will be as shown below:

```
.               
├── MTT
│   ├── audios
│   │   │── 0
│   │   │── 1
│   │   │── ...
│   ├── magnatagatune.json
├── evaluate_clap.py
├── evaluate_mert.py
└── ...
```
We use the same split as [Jukebox](https://github.com/p-lambda/jukemir).

### Model Evaluation

We evaluate the following music representation models in this paper:
- [MERT](evaluate_mert.py) ([https://arxiv.org/abs/2306.00107](https://arxiv.org/abs/2306.00107))
- [CLAP](evaluate_clap.py) ([https://arxiv.org/abs/2211.06687](https://arxiv.org/abs/2211.06687))
- [Imagebind](evaluate_imagebind.py) ([https://arxiv.org/abs/2305.05665](https://arxiv.org/abs/2305.05665))
- [Wav2CLIP](evaluate_wav2clip.py) ([https://arxiv.org/abs/2110.11499](https://arxiv.org/abs/2110.11499))

### Model Performance

The comparison of the models are shown below:
| Model                  | **MTT<sub>AUC</sub>** | **MTT<sub>AP</sub>** |
|------------------------|----------------------|-----------------------|
| ImageBind              | 88.55\%              | 40.19\%               |
| JukeBox                | 91.50\%              | 41.40\%               |
| OpenL3                 | 89.35\%              | 42.88\%               |
| CLAP                   | 70.04\%              | 27.95\%               |
| Wav2CLIP               | 90.15\%              | 49.12\%               |
| **_MERT_**             | **93.91\%**          | **59.57\%**           |

