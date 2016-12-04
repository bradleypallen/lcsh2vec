# lcsh2vec
A Jupyter notebook that implements an approach to generating embeddings for Library of Congress Subject Headings from their labels.

## Requirements

* Python 3.5.2
* jupyter 4.2.0

## Package dependencies

* numpy 1.11.1
* pandas 0.18.1
* matplotlib 1.5.3
* Keras 1.1.2
* scikit-learn 0.17.1
* h5py 2.6.0
* hdf5 1.8.17

## Implementation notes

The Keras model architecture is shown below:

![[Keras architecture for LCSH embedding generation]](lcsh2vec_arch.png)

The model uses categorical cross-entropy as a loss function and uses Adam for optimization. We run training for 40 epochs and then save the weights from the emoji embedding layer of the model checkpoint with the maximum categorical accuracy. Training takes approximately n minutes per epoch, using Tensorflow as a backend for Keras on an Amazon Web Services EC2 p2-xlarge GPU compute instance.

## Results

## Usage

Simply run the notebook using the standard Jupyter command:

    $ jupyter notebook

Apart from running the notebook, one can view a t-SNE visualization of the computed embeddings by running the following command:

    $ python visualize.py

## License

MIT. See the LICENSE file for the copyright notice.

## References

[[1]](https://arxiv.org/abs/1609.08359) Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, and Sebastian Riedel. “emoji2vec: Learning Emoji Representations from their Description,” in Proceedings of the 4th International Workshop on Natural Language Processing for Social Media at EMNLP 2016 (SocialNLP at EMNLP 2016), November 2016.

[[2]](http://nlp.stanford.edu/pubs/glove.pdf) Jeffrey Pennington, Richard Socher, and Christopher D. Manning. "GloVe: Global Vectors for Word Representation," in Proceedings of the 2014 Conference on Empirical Methods In Natural Language Processing (EMNLP 2014), October 2014.

[[3]](#eisner-personal-communication) Ben Eisner. Private communication, 22 November 2016.

[[4]](https://arxiv.org/abs/1607.04853v2) Anirban Laha and Vikas Raykar. "An Empirical Evaluation of various Deep Learning Architectures for Bi-Sequence Classification Tasks," in Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, p. 2762–2773, Osaka, Japan, 11-17 December 2016.

[[5]](https://explosion.ai/blog/deep-learning-formula-nlp) Matthew Honnibal. "Embed, encode, attend, predict: The new deep learning formula for state-of-the-art NLP models", 10 November 2016. Retrieved at https://explosion.ai/blog/deep-learning-formula-nlp on 1 December 2016.
