<h2> An AI-Generated Music Detector </h2>

<em> This is the repository of our final project at Le Wagon Data Science & AI Bootcamp. </em>

<h3> Goals & context </h3>
As AI-generated music plays a growing role in music production, it becomes crucial to distinguish between human-composed and AI-generated music. The idea of our project takes place in this context.
Our goal is simple: perform a classification task and predict if a music is AI generated or not.

Because of its actuality, many scientific publications have been released in this field. The following ones were particularly helpful to our project:

  - Luca Comanducci, Paolo Bestagini, Stefano Tubaro "FakeMusicCaps: a Dataset for Detection and Attribution of Synthetic Music Generated via Text-to-Music Models" (25th Sep. 2024) — details are available on <a href=https://github.com/polimi-ispl/FakeMusicCaps target="_blank" rel="nofollow">Github</a> and in this <a href=https://arxiv.org/abs/2409.10684 target="_blank" rel="nofollow">paper</a>;

  - Md Awsafur Rahman, Zaber Ibn Abdul Hakim, Najibul Haque Sarker, Bishmoy Paul, Shaikh Anowarul Fattah, "SONICS: Synthetic Or Not - Identifying Counterfeit Songs" (first published the 26th Aug. 2024, last revised the 25th February 2025) — the dataset and the models are described on <a href=https://github.com/awsaf49/sonics target="_blank" rel="nofollow">Github</a> and in this <a href=https://arxiv.org/abs/2408.14080 target="_blank" rel="nofollow">paper</a>.

In addition, we studied the model constructed by the Deezer research team:
  - Darius Afchar, Gabriel Meseguer-Brocal, Romain Hennequin, AI-Generated Music Detection and its Challenges (17th Jan 2025) — the code is available on <a href=https://github.com/deezer/deepfake-detector target="_blank" rel="nofollow">Github</a> and the results are presented in this <a href=https://arxiv.org/abs/2501.10111 target="_blank" rel="nofollow">paper</a>.

<h3> Datasets </h3>
We used open datasets:

  - The Free Music Archive in its small split (available on <a href=https://github.com/mdeff/fma target="_blank" rel="nofollow">Github</a>; described in this <a href=https://arxiv.org/abs/1612.01840 target="_blank" rel="nofollow">paper</a>) : 8,000 tracks (30 seconds each), balanced across genres, which we split into 10-second extracts, resulting in 24,000 tracks;
  - MusicCaps: 5374 tracks of 10 seconds each (available on <a href=https://huggingface.co/datasets/google/MusicCaps target="_blank" rel="nofollow">HuggingFace</a> and described in this <a href=https://arxiv.org/abs/2301.11325 target="_blank" rel="nofollow">paper</a>);
  - FakeMusicCaps dataset: 27,674 audio files of 10 seconds each, generated using six Text-to-Music models based on MusicCaps captions.

Our dataset was split into 29,374 "real" songs and 27,674 AI-generated songs, providing a well-balanced set.

<h3> Preprocessing </h3>
We generated the spectrograms of the audio files using Fourier transforms and then converted them to mel-spectrograms. We used the librosa package for audio processing (<a href=https://librosa.org/doc/main/index.html target="_blank" rel="nofollow">Librosa Documentation</a>).

We flattened the resulting arrays and stored them with their dimensions in CSV files. When loading the CSV for training, we were able to reshape the arrays accordingly.


<h3> Models </h3>
The dataset was split into training, validation, and test sets.

We trained several models to achieve our classification task.
We began with relatively simple models: <strong>two SVC models</strong>, one with a <strong>kernel linear</strong> and the other one with a <strong>kernel polynomial</strong>.

We also built <strong>two CNN models</strong>, one simple and the other one more complex.

<h3> Results </h3>
We achieved high accuracy scores across all our models. Specifically, we obtained an accuracy of 0.91 with the SVC Linear model and nearly 0.95 with our simple CNN.

We were pleasantly surprised by these results, especially with the simpler machine learning models. We initially expected CNNs to perform better, as they are commonly the go-to models for image-related tasks. However, it seems that the patterns within our mel-spectrograms were well captured by the classic ML models as well.

<h3> Discussion & going further </h3>
We encountered some challenges when dealing with the mel-spectrograms. Storing them as CSV files turned out to be difficult due to the size of the arrays. In hindsight, we should have saved them in a more efficient format, such as .npy files, and loaded them in batches. We also considered using the h5py package. But we ran out of time.

We trained our models only on 10 seconds extracts, whereas we prepared datasets of 2 seconds and 1 second. A good next step would be to challenge our results using these shorter audio segments.

Furthermore, our dataset of AI-generated music consists of tracks generated with six Text-to-Music models. To improve the robustness of our model, it would be beneficial, if not necessary, to expand the variety of models considered, particularly by including Music-to-Music models.
