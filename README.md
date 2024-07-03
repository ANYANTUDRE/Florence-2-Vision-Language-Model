<div align="center">
  <h1>Florence-2 - Microsoft's Cutting-edge Vision Language Models</h1>
  <p align="center">
    🕸 <a href="https://www.linkedin.com/in/anyantudre">LinkedIn</a> • 
    📙 <a href="https://www.kaggle.com/waalbannyantudre">Kaggle</a> • 
    💻 <a href="https://anyantudre.medium.com/">Medium Blog</a> • 
    🤗 <a href="https://huggingface.co/anyantudre">Hugging Face</a> • 
  </p>
</div>
<br/>

<a href=""> <img src="" alt="Open In "></a>

# Description
Florence-2, released by Microsoft in June 2024, is a lightweight foundation vision-language model open-sourced under the MIT license. This model is very attractive because of its small size (0.2B and 0.7B) and strong performance on a variety of computer vision and vision-language tasks.
Despite its small size, it achieves results on par with models many times larger, like Kosmos-2. The model's strength lies not in a complex architecture but in the large-scale FLD-5B dataset, consisting of 126 million images and 5.4 billion comprehensive visual annotations.

Florence supports many tasks out of the box:
-  **captioning**, 
-  **object detection**,
-  **OCR**,
-  **grounding**,
-  **segmentation**,
-  

ou can try out the model via HF Space or Google Colab.

# Unified Representation
Vision tasks are diverse and vary in terms of spatial hierarchy and semantic granularity. Instance segmentation provides detailed information about object locations within an image but lacks semantic information. On the other hand, image captioning allows for a deeper understanding of the relationships between objects, but without reference to their actual locations.

<a href=""> <img src="" alt="Open In "></a>
Figure 1. Illustration showing the level of spatial hierarchy and semantic granularity expressed by each task. 
Source: Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.

The authors of Florence-2 decided that instead of training a series of separate models capable of executing individual tasks, they would unify their representation and train a single model capable of executing over 10 tasks. However, this requires a new dataset.

# Building Comprehensive Dataset
Unfortunately, there are currently no large, unified datasets available. Existing large-scale datasets cover limited tasks for single images. SA-1B, the dataset used to train Segment Anything (SAM), only contains masks. COCO, while supporting a wider range of tasks, is relatively small.

Manual labeling is expensive, so to build a unified dataset, the authors decided to automate the process using existing specialized models. This led to the creation of FLD-5B, a dataset containing 126 million images and 5 billion annotations, including boxes, masks, and a variety of captions at different levels of granularity. Notably, the dataset doesn't contain any new images; all images originally belong to other computer vision datasets.


<a href=""> <img src="" alt="Open In "></a>
An illustrative example of an image and its corresponding annotations in the FLD-5B dataset. Source: Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.

FLD-5B is not yet publicly available, but the authors announced its upcoming release during CVPR 2024.

<a href=""> <img src="" alt="Open In "></a>
Summary of size, spatial hierarchy, and semantic granularity of top datasets. Source: Florence-2 CVPR 2024 poster.


# Model Architecture
The model takes images and task prompts as input, generating the desired results in text format. It uses a DaViT vision encoder to convert images into visual token embeddings. These are then concatenated with BERT-generated text embeddings and processed by a transformer-based multi-modal encoder-decoder to generate the response.

<a href=""> <img src="" alt="Open In "></a>
Overview of Florence-2 architecture. Source: Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.

For region-specific tasks, location tokens representing quantized coordinates are added to the tokenizer's vocabulary.
- Box Representation (x0, y0, x1, y1): Location tokens correspond to the box coordinates, specifically the top-left and bottom-right corners.
- Polygon Representation (x0, y0, ..., xn, yn): Location tokens represent the polygon's vertices in clockwise order.

# Capabilities
Florence-2 is smaller and more accurate than its predecessors. The Florence-2 series consists of two models: Florence-2-base and Florence-2-large, with 0.23 billion and 0.77 billion parameters, respectively. This size allows for deployment on even mobile devices.

Despite its small size, Florence-2 achieves better zero-shot results than Kosmos-2 across all benchmarks, even though Kosmos-2 has 1.6 billion parameters.


# Citations
- Piotr Skalski. (Jun 20, 2024). Florence-2: Open Source Vision Foundation Model by Microsoft. [Roboflow Blog](https://blog.roboflow.com/florence-2/)


# Structure du repository

Les répertoires de ce repo sont organisés comme suit :  

## 1. Mini-Projets & Applications Pratiques

| Titre | Mot Clés | Description | Notebook | Status |
| --- | --- | --- | --- | --- |
| **1. Sentiment Analysis** | **Keras, RNN, LSTMs** | Analyser les sentiments de commentaires en ligne | [Open or Download](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/blob/main/01.%20Notebooks%20-%20Mini%20Projets%20et%20Applications%20Pratique/01.%201er%20Mini%20Projet%20-%20Sentiment%20Analysis%20with%20Keras.ipynb) | Done ✅ |
| **2. L'API pipeline de 🤗** | **transformers, pipeline** | Utilisation de l'API pipeline pour effectuer des taches de one shot classification, traduction, résumé, NER etc... | [Open or Download](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/blob/main/01.%20Notebooks%20-%20Mini%20Projets%20et%20Applications%20Pratique/02.%20Transformers%20-%20La%20fonction%20pipeline%20.ipynb) | Done ✅ |
| **3. Initiation à Gradio** | **Gradio, transformers** | Developper des interfaces UI pour un LLM et deploiement sur 🤗 | [Open or Download](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/blob/main/01.%20Notebooks%20-%20Mini%20Projets%20et%20Applications%20Pratique/03.%20Intro%20%C3%A0%20Gradio%20-%20Developper%20des%20interfaces%20UI%20pour%20un%20LLM.ipynb) | Done ✅ |
| **4. Utilisation des transformers** | **transformers, AutoTokenizer, TFAutoModel, TFBertModel...** | Familiarization avec la bibliothèque transformers | [Open or Download](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/blob/main/01.%20Notebooks%20-%20Mini%20Projets%20et%20Applications%20Pratique/04-%20Utilisation%20des%20transformers.ipynb) | Done ✅ |
| **5. Prompt Engineering** | **In context learning: zero, one and few shot inference avec le modèle prétrainé FLAN-T5** | Application des techniques de prompt engineering pour  améliorer les complétions des LLMs. | [Open or Download](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/blob/main/01.%20Notebooks%20-%20Mini%20Projets%20et%20Applications%20Pratique/05.%20Prompt%20Engineering%20avec%20FLAN-T5.ipynb) | Not Yet 🔜 |
| **6. Déploiement d'un modèle multilingue sur 🤗** | ** ** | ** ** | [Open or Download]() | Not Yet 🔜 |
| **7. Full instruction Fine-tuning 🤗** | ** ** | ** ** | [Open or Download]() | Not Yet 🔜 |
| **8. Fine-tuning with LoRA and QLoRA 🤗** | ** ** | ** ** | [Open or Download]() | Not Yet 🔜 |



## 2. Séances de Formations & Supports de Cours

| Titre | Date | Objectifs | Supports | 
|---------|---------|-------------------|--------------------------------|
| **1. Intro au Deep Learning & NLP** | 17/04/2024, 19h GMT | Introduire les concepts fondamentaux de **Réseaux de Neurones**, de **Backpropagation** ainsi que le **pipeline classique du NLP** | [Supports](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/tree/main/02.%20Supports%20de%20Cours%20-%20Formations/01.%20S%C3%A9ance%201)|
| **2. 1er Projet et Intro to Generative AI & Transformers** | 27/04/2024, 19h GMT | Pratiquer les notions de **OneHotEncoding, Embedding, RNNs, LSTMs** & Vue d'ensemble sur l'**IA Générative**, le **Prompt Engineering** et l'**architecture des Transformateurs** (high level) | [Supports](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/tree/main/02.%20Supports%20de%20Cours%20-%20Formations/02.%20S%C3%A9ance%202) |
| **3. Architecture détaillée des Transformers & 2ème Projet** |  | Détails sur les **Encodeurs, les Décodeurs et les Encodeurs-Décodeurs**  | [Supports](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/tree/main/02.%20Supports%20de%20Cours%20-%20Formations/03.%20S%C3%A9ance%203) |
| **4. Utilisation des Transformers avec Hugging Face** | 19/05/2024, 19h GMT  | Introduction à la bibliothèque 🤗 **transformers**: Utilisation des classes de **Tokenizers**, de **Modèles prétrainés (BERT base & DistilBERT)**, techniques de **padding, truncation, batching** etc...  | [Supports](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/tree/main/02.%20Supports%20de%20Cours%20-%20Formations/04.%20S%C3%A9ance%204%20-%20Utilisation%20des%20Transformers%20de%20Hugging%20Face) |
| **5. Rappels & Quantization** | 25/05/2024, 20h GMT  | - Retour sur la bibliothèque 🤗 **transformers**    - Technique de **Quantization** | [Supports](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/tree/main/02.%20Supports%20de%20Cours%20-%20Formations/05.%20S%C3%A9ance%205%20-%20Quantization%20%26%20Rappels) |
| **6. Multi-task instruction Fine-tuning & 3ème & 4ème projets** | 26/05/2024, 20h GMT  | **- Projet Prompt Engineering  - Déploiement d'un modèle multilingue sur Hugging Face avec Gradio - Notion de Fine-tuning d'un LLM: Processus de Fine-tuning, Catastrophic Forgetting & Multi-task instruction Fine-tuning**| [Supports](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/tree/main/02.%20Supports%20de%20Cours%20-%20Formations/06.%20S%C3%A9ance%206%20-%20%20Fine-tuning%20process%20%26%20Instruction%20Fine-tuning) |
| **7. Evaluation metrics & PEFT (LoRA et QLoRA) & 5ème projet** | Coming soon...  | **- Evaluation des LLMs: scores ROUGE et BLEU  - PEFT et Techniques de Reparamétrisation: LoRA et QLoRA**| [Supports](https://github.com/ANYANTUDRE/Stage-IA-Selever-GO-AI-Corp/tree/main/02.%20Supports%20de%20Cours%20-%20Formations/07.%20S%C3%A9ance%207%20-%20M%C3%A9triques%20%26%20%20PEFTs%20-%20LoRA%20%26%20QLoRA) |





## 3. Les Ressources à consulter absolument

| Titre | Connaissances à acquérir | Brève Description  | Liens |
|---------|--------------------|-------------------------------|----------------------------------------------------------|
| **1. NLP Course on Hugging Face** |  **Transformers** | Cours sur le NLP et la bibliothèque transformers de Hugging Face| [Lien](https://github.com/ANYANTUDRE/NLP-Course-Hugging-Face) |
| **2. Formation FIDLE** | **Maths & Backpropagation, CNNs, RNNs, Transformers, GNNs etc...**  | Formation d'Introduction au Deep Learning sur Youtube | [Lien](https://www.youtube.com/playlist?list=PLlI0-qAzf2SZQ_ZRAh4u4zbb-MQHOW7IZ) |
| **3. The Illustrated Transformer** | **L'architecture des Transformers et la Self-Attention détaillée et illustrée**  | Article - Blog post de Jay Alammar | [Lien](https://jalammar.github.io/illustrated-transformer/) |



## 4.  Les Ressources supplémentaires pour approfondir vos connaissances

| Titre | Connaissances à acquérir | Brève Description  | Liens |
|---------|--------------------|-------------------------------|------------------------|
| **1. Formation FIDLE** | **Maths & Backpropagation, CNNs, RNNs, Transformers, GNNs etc...**  | Formation d'Introduction au Deep Learning sur Youtube | [Lien](https://www.youtube.com/playlist?list=PLlI0-qAzf2SZQ_ZRAh4u4zbb-MQHOW7IZ) |
| **2. Documentation officielle des Transformers** | **Transformes en général**  | Doc officielle sur Hugging Face | [Lien](https://huggingface.co/docs/transformers/index) |

