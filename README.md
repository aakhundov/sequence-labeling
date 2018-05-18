Sequence Labeling: A Practical Approach
===============

Accompanying repository of the paper "Sequence Labeling: A Practical Approach". Works well with TensorFlow 1.3 and 1.4 (for some unknown reason, the performance drops substantially under the version 1.5 and above). The resources within the repo are structured as follows:

* **convert** - the folder with several scripts for converting the standard sequence labeling datasets (including those mentioned in the paper) to the standardized format recognizable by the code. The instructions on converting each particular standard dataset are in the heading comments of the respective conversion script. The recognizable format is: 
    * two files with the data - **train.txt** and **val.txt** - each containing the respective part of the dataset (training and validation/development set). Each line of these files should contain the space-separated list of tokens of one sentence, separated by a tab from the space-separated list of the corresponding labels.
    * one file with the set of all available labels - **labels.txt** - containing one label in each of its lines. The labels in the file should come in the alphabetical order. The exception is the null-label (e.g. "O") in multi-token labeling tasks (e.g. IOB or IOBES tagging scheme used for NER or Chunking), which should be in the last line.

* The files with the word embeddings should be copied into the sub-folders of the **data/embeddins** folder. Each sub-folder corresponds to one of the supported embedding types - [GloVe](https://nlp.stanford.edu/projects/glove/), [Polyglot](https://sites.google.com/site/rmyeid/projects/polyglot), or [Senna](https://ronan.collobert.com/senna/) - and contains the specific instructions.

* **train.py** trains and saves a sequence labeling model, given the path to the folder with the input data in the recognizable format (see above). The specification of the command-line arguments can be obtained by running the script with "-h" key. The training results are written to a sub-folder of the "results" folder named after the input folder plus a timestamp. These results may be used further for evaluation on and/or annotation of new data.

* **evaluate.py** evaluates the trained model, given the path to the training results and the name of the data file (within the original data folder) to evaluate on (e.g. "test.txt" containing the testing set). The specification of the command-line arguments can be obtained by running the script with "-h" key.

* **annotate.py** annotates new data, given the path to the training results (containing the trained model to be used for the annotation) and the path to the data to be annotated in the recognizable format (see above; labels are not required).

* **logs** folder contains the detailed results of the experiments on the eight standard datasets mentioned in the paper. Each dataset folder contains six sub-folders corresponding to six different scenarios used in the ablation studies (with or without byte embeddings, word embeddings, and CRF layer). Each scenario folder, in turn, contains the following three files: **training_log.txt** (detailed log of the model training with the validation results after each epoch), **evaluation_log.txt** (the results of the evaluation performed using the official [CoNLL evaluation script](https://www.clips.uantwerpen.be/conll2000/chunking/conlleval)), and **labeled_test_set.zip** (the test set with the predicted and ground truth labels, ready to be evaluated by the above-mentioned CoNLL evaluation script; for privacy reasons, all tokens in the file are replaced with a "W" token).
