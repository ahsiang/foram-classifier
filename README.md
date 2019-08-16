# foram-classifier
Automated planktonic foraminifera classifier

This repository contains the code and weights from the best-performing model as described in:

>    Hsiang AY, Brombacher A, Rilo MC, Mleneck-Vautravers MJ, Conn S, Lordsmith S, Jentzen A, Henehan MJ, Metcalfe B, Fenton I, Wade B, Fox L, Meilland J, Davis CV, Baranowski U, Groeneveld J, Edgar KM, Movellan A, Aze T, Dowsett H, Miller G, Rios N & Hull PM. Endless Forams: >34,000 modern planktonic foraminiferal images for taxonomic training and automated species recognition using convolutional neural networks. Paleoceanography and Paleoclimatology. 34(7):1157-1177. ([https://doi.org/10.1029/2019PA003612](https://doi.org/10.1029/2019PA003612))

The classifier can identify 35 species of planktonic foraminifera (+1 "Not a planktonic foraminifer" category) with a top-1 accuracy of 87.4% and a top-3 accuracy of 97.7%. The training set used consisted of 27,737 images from the [Yale Peabody Museum](http://peabody.yale.edu/) and [Natural History Museum, London](https://www.nhm.ac.uk/) collections.

Training images and a taxonomic training module can be found at [Endless Forams](http://www.endlessforams.org).

## FAQ
The weights file is versioned using [Git LFS](https://git-lfs.github.com/) (Large File Storage). The usual `git clone` command may fail on checkout as a result. In this case, using `git lfs clone` should fix the problem.
