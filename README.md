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
