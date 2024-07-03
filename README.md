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

<a href="" style="align-items:center"> <img src="https://github.com/ANYANTUDRE/Florence-2-Vision-Language-Model/blob/main/img/card.png" alt="Open In" class="center"></a>

# 🔗 Short Links
- [Florence-2 technical report](https://arxiv.org/abs/2311.06242)
- [HuggingFace's transformers implementation of Florence-2 model](https://huggingface.co/microsoft/Florence-2-large)

# 📃 Model Description
Florence-2, released by Microsoft in June 2024, is an advanced, lightweight foundation **vision-language model open-sourced** under the MIT license. This model is very attractive because of its small size (0.2B and 0.7B) and strong performance on a variety of computer vision and vision-language tasks.
Despite its small size, it achieves results on par with models many times larger, like Kosmos-2. The model's strength lies not in a complex architecture but in the large-scale FLD-5B dataset, consisting of 126 million images and 5.4 billion comprehensive visual annotations.

#### **Florence-2 model series:**

| Model   | Model size | Model Description | 
| ------- | ------------- |   ------------- |  
| Florence-2-base[[HF]](https://huggingface.co/microsoft/Florence-2-base) | 0.23B | Pretrained model with FLD-5B  
| Florence-2-large[[HF]](https://huggingface.co/microsoft/Florence-2-large) | 0.77B  | Pretrained model with FLD-5B  
| Florence-2-base-ft[[HF]](https://huggingface.co/microsoft/Florence-2-base-ft) | 0.23B  | Finetuned model on a colletion of downstream tasks
| Florence-2-large-ft[[HF]](https://huggingface.co/microsoft/Florence-2-large-ft) | 0.77B | Finetuned model on a colletion of downstream tasks

#### **Tasks**
Florence 2 supports many tasks out of the box:
-  **Caption**,
-  **Detailed Caption**,
-  **More Detailed Caption**,
-  **Dense Region Caption**,
-  **Object Detection**,
-  **OCR**,
-  **Caption to Phrase Grounding**,
-  **segmentation**,
-  **Region proposal**,
-  **OCR**,
-  **OCR with Region**.

You can try out the model via HF Space or Google Colab.


# 🕸 Unified Representation
Vision tasks are diverse and vary in terms of spatial hierarchy and semantic granularity. Instance segmentation provides detailed information about object locations within an image but lacks semantic information. On the other hand, image captioning allows for a deeper understanding of the relationships between objects, but without reference to their actual locations.

<a href=""> <img src="https://github.com/ANYANTUDRE/Florence-2-Vision-Language-Model/blob/main/img/representation.jpeg" alt="Open In "></a>
Figure 1. Illustration showing the level of spatial hierarchy and semantic granularity expressed by each task. 
Source: Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.

The authors of Florence-2 decided that instead of training a series of separate models capable of executing individual tasks, they would unify their representation and train a single model capable of executing over 10 tasks. However, this requires a new dataset.


# 💎 Dataset
Unfortunately, there are currently no large, unified datasets available. Existing large-scale datasets cover limited tasks for single images. SA-1B, the dataset used to train Segment Anything (SAM), only contains masks. COCO, while supporting a wider range of tasks, is relatively small.

Manual labeling is expensive, so to build a unified dataset, the authors decided to automate the process using existing specialized models. This led to the creation of FLD-5B, a dataset containing 126 million images and 5 billion annotations, including boxes, masks, and a variety of captions at different levels of granularity. Notably, the dataset doesn't contain any new images; all images originally belong to other computer vision datasets.

<a href=""> <img src="https://github.com/ANYANTUDRE/Florence-2-Vision-Language-Model/blob/main/img/annotation.jpeg" alt="Open In "></a>
An illustrative example of an image and its corresponding annotations in the FLD-5B dataset. Source: Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.

FLD-5B is not yet publicly available, but the authors announced its upcoming release during CVPR 2024.

<a href=""> <img src="https://github.com/ANYANTUDRE/Florence-2-Vision-Language-Model/blob/main/img/dataset.jpeg" alt="Open In "></a>
Summary of size, spatial hierarchy, and semantic granularity of top datasets. Source: Florence-2 CVPR 2024 poster.


# 🧩 Architecture and Pre-training details 
Regardless of the computer vision task being performed, Florence-2 formulates the problem as a sequence-to-sequence task. Florence-2 takes an image and text as inputs, and generates text as output. The model has a simple structure. It uses a DaViT vision encoder to convert images into visual embeddings, and BERT to convert text prompts into text and location embeddings. The resulting embeddings are then processed by a standard encoder-decoder transformer architecture, generating text and location tokens. Florence-2's strength doesn't stem from its architecture, but from the massive dataset it was pre-trained on. The authors noted that leading computer vision datasets typically contain limited information - WIT only includes image/caption pairs, SA-1B only contains images and associated segmentation masks. Therefore, they decided to build a new FLD-5B dataset containing a wide range of information about each image - boxes, masks, captions, and grounding. The dataset creation process was largely automated. The authors used off-the-shelf task-specific models and a set of heuristics and quality checks to clean the obtained results. The result was a new dataset containing over 5 billion annotations for 126 million images, which was used to pre-train the Florence-2 model.

The model takes images and task prompts as input, generating the desired results in text format. It uses a DaViT vision encoder to convert images into visual token embeddings. These are then concatenated with BERT-generated text embeddings and processed by a transformer-based multi-modal encoder-decoder to generate the response.

<a href=""> <img src="https://github.com/ANYANTUDRE/Florence-2-Vision-Language-Model/blob/main/img/architecture.png" alt="Open In "></a>
Overview of Florence-2 architecture. Source: Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks.

For region-specific tasks, location tokens representing quantized coordinates are added to the tokenizer's vocabulary.
- Box Representation (x0, y0, x1, y1): Location tokens correspond to the box coordinates, specifically the top-left and bottom-right corners.
- Polygon Representation (x0, y0, ..., xn, yn): Location tokens represent the polygon's vertices in clockwise order.

# 🦾 Capabilities
Florence-2 is smaller and more accurate than its predecessors. The Florence-2 series consists of two models: Florence-2-base and Florence-2-large, with 0.23 billion and 0.77 billion parameters, respectively. This size allows for deployment on even mobile devices.

Despite its small size, Florence-2 achieves better zero-shot results than Kosmos-2 across all benchmarks, even though Kosmos-2 has 1.6 billion parameters.

#### Examples


# 🏋🏾‍♂️ Finetuning
Even if Florence-2 supports many tasks, maybe your task or domain might not be supported, or you may want to better control the model's output for your task. That's when you will need to fine-tune.
This post shows an example on fine-tuning Florence on DocVQA.


# 🗂 Useful Resources

| Title | Articles | Brief Description  | Links |
|---------|--------------------|-------------------------------|----------------------------------------------------------|
| **Vision Language Models Explained** |  Blog articles | something here | [Link](https://huggingface.co/blog/vlms) |
| **Florence-2 Demo** |   | something here| [Link]() |
| **Florence-2 DocVQA Demo** |   | something here| [Link]() |
| **Florence-2 Inference Notebook** |  | something here | [Link]() |
| **Florence-2 Finetuning Notebook** |  | something here | [Link]() |


# 🔗 Citations and References
- @article{xiao2023florence,
  title={Florence-2: Advancing a unified representation for a variety of vision tasks},
  author={Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
  journal={arXiv preprint arXiv:2311.06242},
  year={2023}
}

- Piotr Skalski. (Jun 20, 2024). Florence-2: Open Source Vision Foundation Model by Microsoft. [Roboflow Blog](https://blog.roboflow.com/florence-2/)
- [Fine-tuning Florence-2 - Microsoft's Cutting-edge Vision Language Models](https://huggingface.co/blog/finetune-florence2)
