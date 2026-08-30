# **Full Report**

# **The State of the Art in Cursive Font OCR/HTR: Applications, Research, and Multilingual Perspectives**

## **1\. Introduction**

### **The Evolving Landscape of Cursive Script Recognition**

The digitization of historical and contemporary handwritten materials represents a significant challenge and an area of burgeoning technological advancement. Cursive scripts, with their flowing and interconnected characters, have historically posed substantial barriers to automated transcription. However, the demand for accessing the vast wealth of information locked within handwritten documents—ranging from historical archives and literary manuscripts to medical records, financial ledgers, and personal correspondence—has spurred considerable research and development in Optical Character Recognition (OCR) and, more specifically, Handwritten Text Recognition (HTR).1 While OCR traditionally focused on machine-printed text, HTR addresses the more complex task of deciphering handwritten content. The transition from OCR systems adept at handling uniform printed text to HTR systems capable of interpreting the nuances of cursive handwriting signifies a major technological leap. This evolution is not merely an incremental improvement but a qualitative shift, enabling the processing of data previously deemed too complex for automation and unlocking vast repositories of human knowledge.4 The ability to accurately convert handwritten cursive into machine-readable text has profound implications, facilitating large-scale data analysis, preservation, and accessibility across numerous fields.1

### **Inherent Challenges in Digitizing Handwritten Cursive Text**

The primary difficulty in HTR, particularly for cursive scripts, stems from the inherent variability and complexity of human handwriting. Unlike printed text, which adheres to standardized fonts and spacing, handwriting exhibits significant inter-writer and intra-writer variations in style, slant, size, and character formation.1 Cursive writing, by its nature, involves connected characters, ligatures (where two or more letters are joined as a single glyph), and often elaborate flourishes, making segmentation into individual characters a formidable task. Furthermore, the condition of many source documents, especially historical artifacts, introduces additional complexities. Degradation such as faded ink, stains, tears, and background noise, coupled with poor image quality from initial scanning or photography, can severely impede recognition accuracy.8 These factors necessitate robust HTR systems capable of handling diverse and imperfect inputs.

### **Report Objectives and Structure**

This report aims to provide a comprehensive analysis of the current state-of-the-art in cursive OCR/HTR technology. It will delve into both research advancements and practical applications, with a specific focus on the Arabic language, which presents unique challenges due to its scriptorial properties. Comparative examples from Spanish, English, and other non-Western languages, including Indic and East Asian scripts, will be included to offer a broader multilingual perspective. The analysis will cover four key areas:

1. The ability to recognize structured data (e.g., in forms, tables) versus unstructured data (e.g., free-flowing manuscripts).  
2. The availability and role of training data, including the challenges of data scarcity and the use of synthetic data and augmentation techniques.  
3. The evolution of methodologies, comparing traditional heuristic and statistical methods with modern Artificial Intelligence (AI), Machine Learning (ML), and Large Language Model (LLM) based approaches.  
4. Performance evaluation, examining metrics such as precision and recall, both generally and within specific domains like historical document analysis and medical prescription reading.

The success of HTR technologies is transforming fields reliant on handwritten records. By making large-scale analysis of these documents feasible, HTR is opening new avenues for research in digital humanities, enabling more efficient archival science, and improving data processing in legal and medical domains.

## **2\. Technological Evolution in Cursive OCR/HTR**

### **2.1. From Heuristics to Deep Learning: A Paradigm Shift**

The journey of automated cursive script recognition has been marked by a significant methodological evolution, moving from early systems reliant on explicit rules and handcrafted features to sophisticated deep learning models that learn representations directly from data.

Early Approaches: Rule-Based, Template Matching, and Statistical Models  
Initial forays into HTR employed rule-based systems and template matching.1 These methods compared input character images against a predefined set of templates or applied heuristic rules derived from script characteristics. While functional in controlled environments with limited handwriting variability, they proved inadequate for the diverse and often inconsistent nature of real-world cursive scripts. The inherent variability in character shapes, sizes, and connections quickly overwhelmed these rigid approaches.5  
A subsequent advancement came with the introduction of statistical methods, most notably Hidden Markov Models (HMMs).1 HMMs offered improved performance by learning statistical properties from labeled data, allowing them to model sequences of characters, typically at the word or line level. This was a step towards more data-driven recognition. However, HMM-based systems often struggled with the complexities of cursive scripts and typically required explicit segmentation of text into characters or sub-character units, a process that is itself a major challenge for connected scripts.1 While HMMs represented progress, their ability to capture the intricate spatial and long-range dependencies in handwriting remained limited.

The Deep Learning Revolution: CNNs, RNNs (LSTMs, BiLSTMs), CTC  
The advent of deep learning marked a true revolution in the field of HTR, leading to unprecedented improvements in accuracy and robustness.1 Two main types of neural networks became cornerstones of modern HTR systems:

* **Convolutional Neural Networks (CNNs):** CNNs excel at learning hierarchical visual features directly from raw pixel data.5 In HTR, CNN layers are typically used as powerful feature extractors, identifying relevant patterns like strokes, loops, and character components from the input image. Deep CNNs (DCNNs) can learn increasingly abstract representations from the initial pixel level to higher-level features crucial for differentiation.7  
* **Recurrent Neural Networks (RNNs):** Given the sequential nature of text, RNNs, particularly Long Short-Term Memory (LSTM) 1 and Bidirectional LSTM (BiLSTM) networks 17, became essential for modeling the contextual dependencies between characters in a line of handwritten text. BiLSTMs, by processing sequences in both forward and backward directions, can capture a richer contextual understanding.

A pivotal development that enabled truly end-to-end training of these hybrid CNN-RNN architectures was the **Connectionist Temporal Classification (CTC) loss function**.1 CTC obviates the need for pre-segmenting the input image into individual characters and aligning these segments with output labels. It allows the network to learn to transcribe a sequence of input frames (e.g., vertical slices of a text line image) into a sequence of characters, handling the variable lengths and alignments inherent in handwriting. This combination of CNNs for feature extraction, RNNs for sequence modeling, and CTC for training has been the dominant paradigm in HTR for many years.1

Transformer Architectures: TrOCR, Vision Transformers, and Beyond  
More recently, Transformer models, originally developed for Natural Language Processing (NLP), have made a significant impact on HTR.1 The self-attention mechanism in Transformers allows them to model long-range dependencies within sequences more effectively than traditional RNNs, which can be beneficial for understanding the global context of a handwritten line or even a document.  
A notable example is **TrOCR (Transformer OCR)**, which typically employs a vision transformer (ViT) as an image encoder and a text transformer as a decoder.4 TrOCR models are often pre-trained on large datasets of synthetic handwritten text and then fine-tuned on specific target domains or languages, achieving state-of-the-art results on various benchmarks.4 Other Transformer-based architectures, such as **LayoutLM** and its variants, integrate textual, visual, and layout information, making them suitable for document understanding tasks that go beyond simple text transcription, such as key-value extraction from forms.9

### **2.2. The Rise of Multimodal Large Language Models (MLLMs) in HTR**

The latest wave of innovation in AI has brought Multimodal Large Language Models (MLLMs) to the forefront. These models, such as OpenAI's GPT-4v/GPT-4o and Google's Gemini, are capable of processing and understanding information from multiple modalities, including images and text, simultaneously.4 Their application to HTR is a rapidly developing area.

MLLMs are being explored for HTR in several ways:

* **Direct Transcription:** Some MLLMs can perform direct transcription of handwritten text from images, often in a zero-shot (no specific training on the task) or few-shot (with a few examples provided in the prompt) setting. This is particularly appealing for scenarios where large, annotated HTR datasets are unavailable.  
* **OCR Post-Correction:** MLLMs can be used to refine and correct the output of existing OCR/HTR systems.32 By leveraging their strong language understanding capabilities and, in some cases, the original image, they can identify and fix errors made by the primary HTR engine.  
* **Enhanced Document Understanding:** Beyond transcription, MLLMs offer the potential for a more holistic understanding of documents, including interpreting complex layouts, understanding the relationship between text and visual elements, and extracting structured information from handwritten forms.42

Despite their promise, MLLMs also present challenges, including high computational costs, slower processing speeds compared to specialized HTR models, a tendency to "hallucinate" (generate plausible but factually incorrect information), and concerns regarding data privacy when using cloud-based APIs.6

The progression from rule-based systems, which were interpretable but limited, to statistical models, then to deep learning architectures (CNNs, RNNs, Transformers) that automatically learn features, and now towards MLLMs that integrate broad world knowledge and multimodal understanding, illustrates a clear trajectory. This path shows a continuous abstraction away from manual feature engineering and an increasing reliance on the model's capacity to learn from vast quantities of data.1 This evolution suggests a future where HTR systems become more adaptable, requiring less domain-specific tuning, and capable of handling increasingly diverse and noisy inputs. However, this increasing model complexity brings challenges. As models, particularly large Transformers and MLLMs, become more powerful, their internal decision-making processes become less transparent, leading to the "black box" problem.1 While accuracy may improve significantly 4, understanding the root cause of errors becomes more difficult compared to debugging a rule in a simpler heuristic system. This lack of interpretability can be a concern for critical applications where accountability and error analysis are paramount, such as in the processing of medical prescriptions or legal documents.9 Consequently, future research must balance the pursuit of higher accuracy with the development of techniques for explainable AI (XAI) to enhance the trustworthiness and reliability of these advanced HTR systems. The demand for large, diverse datasets also intensifies with more powerful models, yet the zero-shot or few-shot capabilities of MLLMs could, if current limitations are overcome, democratize HTR for low-resource languages or niche applications.

## **3\. Cursive OCR for Arabic Script: A Deep Dive**

The Arabic script, with its inherent cursiveness and unique characteristics, presents a distinct set of challenges for OCR and HTR technologies. Significant research efforts have been directed towards developing robust systems capable of accurately digitizing Arabic handwritten text.

### **3.1. Unique Linguistic and Scriptorial Challenges**

The accurate recognition of handwritten Arabic is complicated by several intrinsic features of the script:

* **Inherent Cursiveness and Connectivity:** Unlike many Latin-based scripts where characters can be written in a disconnected print style, Arabic is fundamentally cursive in both its printed and handwritten forms. Letters within a single word or sub-word are typically connected, making the explicit segmentation of words into individual characters an extremely difficult and error-prone task.1  
* **Contextual Letter Forms and Ligatures:** The visual form of an Arabic letter changes significantly depending on its position within a word: initial, medial, final, or isolated.1 This means a single character can have up to four different allographs. Furthermore, Arabic script makes extensive use of ligatures, where specific sequences of two or more characters are combined into a single, often complex, glyph (e.g., the Lam-Alif ligature). Recognizing these ligatures and correctly decomposing them into their constituent characters (or recognizing them holistically) is a major hurdle.14  
* **Diacritics, Dots, and Vocalization:** Many Arabic letters share the same basic shape (rasm) and are distinguished only by the number and placement of dots (i'jam) above or below the main character body.1 These dots are small and can be easily missed, misplaced, or merge with the character body or noise, especially in degraded or low-resolution images. Additionally, diacritical marks (Harakat) are used to indicate short vowels and other phonetic features. While often omitted in everyday handwriting and modern printed texts, their presence (especially in older manuscripts or pedagogical texts) is crucial for correct pronunciation and disambiguation of meaning. These marks add another layer of complexity for OCR systems.14  
* **Right-to-Left (RTL) Processing:** Arabic is written and read from right to left. HTR systems must be designed to process text flow in this direction, which impacts sequence modeling and layout analysis.28  
* **Variability:** As with all handwriting, Arabic script exhibits high variability in writing styles, stroke thickness, character size, and alignment, both between different writers and within the writing of a single individual.1 This is further compounded when considering historical manuscripts spanning different eras and geographical regions, each with distinct calligraphic traditions.

### **3.2. State-of-the-Art Models and Techniques for Arabic HTR**

Addressing the aforementioned challenges requires specialized approaches and model architectures.

Segmentation-based vs. Segmentation-free Approaches:  
Historically, many attempts at Arabic OCR involved explicit segmentation of words into characters before recognition. However, due to the cursive nature and ligatures, this approach is fraught with difficulty and often leads to cascading errors.1 Consequently, segmentation-free (or holistic) approaches have gained prominence. These methods treat entire words or text lines as single units for recognition, typically employing sequence-to-sequence models that learn to map an image of a word/line directly to its corresponding character sequence.18 Some systems use implicit segmentation, where the model internally learns to identify character-level features without explicit segmentation boundaries being provided during training.21  
Key Architectures:  
The current state-of-the-art in Arabic HTR is dominated by deep learning models:

* **CNN-Transformer Architectures:** Systems like the "Invizo" model demonstrate the power of combining CNNs for robust visual feature extraction with Transformer-based sequence modeling.1 Transformers, with their self-attention mechanisms, are adept at capturing long-range contextual dependencies within the Arabic text, accommodating variations in handwriting styles, stroke thicknesses, alignments, and noise conditions. Such models have shown strong performance on both printed and handwritten Arabic texts.  
* **CNN-BiLSTM-CTC with Attention:** This hybrid architecture remains a popular and effective choice.1 CNN layers extract spatial features from the input image. These features are then fed into BiLSTM layers, which model the sequential nature of the text and capture bidirectional context (crucial for Arabic due to contextual shaping and ligatures). The CTC loss function enables end-to-end training without requiring precise pre-segmentation. The inclusion of attention mechanisms further enhances these models by allowing them to dynamically focus on the most relevant parts of the input image features when predicting each character, which is particularly beneficial for handling the intricate details of Arabic characters, diacritics, and ligatures.18

Specialized Preprocessing and Segmentation for Text Detection:  
While recognition models may be segmentation-free at the character level, accurate detection and segmentation of text lines or regions from a document page remain critical precursor steps. Advanced techniques are employed for this:

* **Differentiable Binarization (DBNet, DBNet++):** These methods are used for accurate text line detection and segmentation, particularly in complex layouts.1 DBNet++ incorporates an Adaptive Scale Fusion (ASF) module, which improves the model's robustness to text of varying scales by adaptively fusing features from different layers of the backbone network (e.g., ResNet50).1 The ASF module uses stage-wise and spatial attention to learn weights for feature maps of different scales and across spatial dimensions, respectively.55 The general OCR pipeline for Arabic, much like for other scripts, typically involves several stages: image acquisition and input, pre-processing (including binarization, noise reduction, normalization, and skew correction), text detection and segmentation (to isolate lines or words), feature extraction by the neural network, character/sequence recognition, and finally, post-processing or enhancement (which might involve language models or predefined word lists to correct errors).16

The efficacy of Arabic HTR systems is profoundly influenced by the synergy between these pipeline stages. Robust segmentation and tailored pre-processing are not merely preparatory steps but integral components that directly impact the final recognition accuracy. For instance, the unique characteristics of Arabic script, such as the intricate diacritics and the interconnectedness of characters, demand specialized handling in these early phases. An advanced recognition model may still falter if the input it receives from a poor segmentation module is fragmented or distorted. This interdependence suggests that future breakthroughs in Arabic HTR will likely emerge from holistic pipeline optimizations and the co-development of segmentation and recognition modules that operate in a more integrated fashion, possibly leading to more sophisticated end-to-end systems that implicitly manage these complexities.

### **3.3. Training Data for Arabic HTR**

The availability of large, diverse, and accurately annotated datasets is paramount for training effective deep learning-based HTR models.

Overview of Major Datasets:  
Several datasets have been instrumental in advancing Arabic HTR research:

* **Muharaf:** This is a particularly valuable resource for historical Arabic manuscripts. It comprises over 1,600 fully annotated and transcribed page images, featuring a wide diversity of handwriting styles (predominantly Ruq'ah script, dating from the early 19th to the early 21st century) and various document types, including personal letters, diaries, notes, poems, church records, and legal correspondences.1 Annotations are detailed, providing spatial polygonal coordinates for text lines and marking basic page elements like graphics, page numbers, and headings. Muharaf is classified as a category 2 dataset, meaning it consists of original historical manuscripts rather than text scribed under controlled conditions.58  
* **IFN/ENIT:** A widely recognized benchmark dataset in the Arabic HTR community, often used for evaluating systems on handwritten Arabic, particularly for tasks like recognizing Tunisian town and village names.1  
* **AMFDS (Arabic Multi-Fonts Dataset):** This dataset is used for training and testing systems on a variety of Arabic fonts, including those designed to mimic handwritten styles, thus helping models generalize across different printed representations of Arabic text.17  
* **Historical Arabic Handwritten Text Recognition Dataset (Mendeley Data):** This dataset offers 40 meticulously transcribed historical Arabic pages, each selected from distinct books and representing different geographical origins and centuries, contributing to deep learning OCR modeling for historical Arabic texts.1  
* **KHATT (Kalamu Waraqa Handwritten Arabic TexT):** Another important benchmark dataset for Arabic script recognition, providing a collection of handwritten Arabic text images for training and evaluation.1 Surveys by Lorigo and Govindaraju 1 and AlKendi et al. 1 also provide overviews of earlier datasets and HTR efforts for Arabic.

Data Augmentation and Synthetic Data Generation Strategies:  
Given that even these datasets can be limited for training very deep models or for covering all possible variations, data augmentation and synthetic data generation are crucial:

* **Geometric and Photometric Augmentations:** Common techniques include random rotations, scaling, shearing, elastic transformations (to mimic natural handwriting distortions), brightness and contrast adjustments, adding Gaussian noise, and random cropping.9  
* **Script-Specific Augmentations:** For Arabic, methods like Moving Least Squares (MLS) have been developed to deform images in a way that produces realistic variations of handwritten text, suitable for augmenting training data.21  
* **Adaptive Augmentation:** Some approaches propose adaptive data augmentation algorithms that focus on generating more samples for underrepresented characters or words in the dataset, aiming to create a more balanced training set.26  
* **Synthetic Data Generation:** While challenging to perfectly replicate the nuances of real handwriting, synthetic data generation is used to expand training sets. This can involve using existing fonts with various distortions or employing generative models like GANs or RNNs to create novel handwritten samples.21

The representativeness of training data is a critical factor. While datasets like IFN/ENIT serve as useful benchmarks for specific tasks (e.g., recognition of a limited vocabulary of names), the rich diversity captured in datasets like Muharaf—with its wide array of historical documents, varied calligraphic hands, and the common Ruq'ah script—is indispensable for developing HTR systems that can perform robustly on real-world, unconstrained Arabic historical texts.1 Models trained exclusively on cleaner, more homogeneous datasets may struggle when confronted with the "wild" nature of historical documents, which often feature degradation, idiosyncratic writing styles, and non-standard layouts. This underscores the continuous need for the creation and curation of larger, more diverse, and meticulously annotated datasets for Arabic HTR, particularly those that encompass a broader range of historical periods, calligraphic styles beyond Ruq'ah, and geographical variations. Such efforts are fundamental enablers for advancing the field.

### **3.4. Performance Benchmarks and Evaluation for Arabic**

The performance of Arabic HTR systems is typically evaluated using a set of standard metrics, allowing for comparison across different approaches and datasets.

**Common Metrics:**

* **Character Error Rate (CER):** Defined as the sum of character insertions, deletions, and substitutions required to transform the model's output into the ground truth, normalized by the total number of characters in the ground truth. A lower CER indicates higher accuracy. This is a primary metric for HTR.1  
* **Word Error Rate (WER):** Similar to CER, but calculated at the word level. It reflects the model's ability to correctly recognize whole words.1  
* **Precision, Recall, and F-measure (F1-score):** These metrics are often used for evaluating text detection performance (e.g., detecting text lines or regions) but can also apply to recognition tasks, especially in information extraction contexts.  
  * Precision: P=TP/(TP+FP)  
  * Recall: R=TP/(TP+FN)  
  * F-measure: F1=2⋅(P⋅R)/(P+R) (where TP \= True Positives, FP \= False Positives, FN \= False Negatives).1  
* **Character Recognition Rate (CRR) and Word Recognition Rate (WRR):** These are accuracy measures, often calculated as 1−CER and 1−WER respectively, or as the percentage of correctly recognized characters/words.1

**Table 1: Comparative Performance of Selected State-of-the-Art Arabic HTR Models**

| Model/System Name (Lead Author/Year) | Core Architecture | Key Datasets Used (Training/Test) | Reported CER (%) | Reported WER (%) | Other Key Metrics (Precision/Recall/F1/CRR/WRR) | Specific Focus |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Invizo (Waly et al., 2025\) 1 | CNN-Transformer | Custom Arabic (Handwritten/Printed) | 7.91 (HW) | 31.41 (HW) | Text Detection: P: 81.66%, R: 78.82%, F1: 79.07%. Printed: 0.59% CER, 1.72% WER | General purpose (Handwritten, Printed, Receipts, Numbers) |
| CNN-BiLSTM-CTC (Waly et al., 2024\) 1 | CNN-BiLSTM-CTC | AMFDS | \- | \- | CRR (single words 7-10 chars): 99.20%, WRR: 93.75%. CRR (sentences): 83.76% | Arabic Handwritten Text (Sentences and Words) |
| Attention-based CNN-RNN-CTC (Ben Aïcha Gader & Kacem Echi, 2022\) 1 | Attention-based CNN-BiLSTM-CTC | IFN/ENIT, IAM | 2.9 (IAM) | \- | Accuracy (IFN/ENIT): 97.1% | Arabic (IFN/ENIT) and English (IAM) Handwritten Text, Script Independence |
| Hybrid CNN-Transformer Ensemble (El et al., 2025\) 35 | CNNs \+ Transformers (Ensemble) | IFN/ENIT (customized) | \- | \- | Accuracy (Letter Classification): 96.38%, Accuracy (Positional Class.): 97.22% | Arabic Handwriting Recognition (Letter & Positional Classification) |
| Vision-Language Models (e.g., GPT-4, Gemini) on KITAB-Bench 65 | Vision-Language Models | KITAB-Bench | Avg. 60% lower than traditional OCR | \- | \- | Diverse Arabic Documents (Handwritten, Tables, Charts) |

*Note: HW \= Handwritten. Performance metrics can vary based on specific dataset splits and evaluation protocols. This table provides a snapshot based on reported results.*

The performance figures indicate substantial progress. For instance, the Invizo system reports a CER of 7.91% for handwritten Arabic, which is commendable given the script's complexity.27 Systems focusing on specific datasets like IFN/ENIT achieve very high accuracies (e.g., 97.1% by Ben Aïcha Gader & Kacem Echi 18). The emergence of benchmarks like KITAB-Bench, which evaluate modern vision-language models, suggests that these newer architectures can significantly outperform traditional OCR methods on diverse Arabic document types, including handwritten text and tables, by a large margin in CER.65 However, it is important to note that direct comparison across all systems is challenging due to variations in datasets, preprocessing pipelines, and evaluation methodologies.

## **4\. Cursive OCR Across Diverse Languages: Comparative Analysis**

While Arabic presents a unique set of challenges, cursive HTR research and application span a multitude of languages, each with its own scriptorial characteristics and levels of technological maturity. This section provides a comparative overview for Spanish, English (with a focus on historical documents), and selected non-Western Indic and East Asian scripts.

### **4.1. Spanish Cursive HTR**

Spanish, as a Latin-alphabet language, shares some foundational aspects with English HTR, but historical Spanish documents introduce specific paleographic complexities.

**Key Models and Datasets:**

* **Coloso Español (Transkribus):** This AI model, developed by the Transkribus platform, is designed for transcribing a broad range of Spanish texts, from medieval manuscripts to 20th-century documents, encompassing both handwritten and typewritten materials.66 Built using the PyLaia engine, it reports a Character Error Rate (CER) of 4.80% on its validation set, indicating strong performance on diverse historical Spanish scripts.  
* **Spanish TrOCR:** Leveraging the Transformer-based TrOCR architecture, researchers have fine-tuned models (small, base, and large variants) specifically for the Spanish language. These models are publicly available via HuggingFace (user: qantev).23 When evaluated on the XFUND Spanish dataset (primarily printed documents), the large TrOCR model achieved a CER of 0.0632 (6.32%) and a Word Error Rate (WER) of 0.1817 (18.17%).31  
* **Datasets:** Key datasets include the **XFUND Spanish dataset** (for forms and printed documents) 23 and the varied historical collections implicitly used to train models like Coloso Español, which cover centuries of Spanish handwriting.66

Performance Insights and Challenges:  
Current models demonstrate good capabilities for Spanish. TrOCR models show strong performance on structured or printed Spanish documents, while specialized models like Coloso Español are tailored for the complexities of historical handwritten Spanish. Challenges in historical Spanish HTR include the wide variability in calligraphic styles across different centuries (e.g., Visigothic, Caroline, Gothic, Humanistic, and later cursive hands), the extensive use of abbreviations and ligatures common in older manuscripts, and orthographic variations.

### **4.2. English Cursive HTR (with emphasis on historical documents)**

English HTR, particularly for historical documents, is a well-researched area, benefiting from relatively larger datasets and significant interest from the digital humanities community.

**Leading Models:**

* **TrOCR and other Transformer-based HTR models:** These serve as strong baselines and are often fine-tuned on specific historical English datasets.4  
* **Multimodal Large Language Models (MLLMs):** Models like GPT-4o, Gemini, and Claude Sonnet 3.5 are showing considerable promise for transcribing historical handwritten documents, including English.4 They are being used not only for end-to-end transcription but also for post-correction of outputs from conventional OCR/HTR systems, sometimes achieving CERs below 1% on challenging historical prints (e.g., German Fraktur, which shares some legibility challenges with older English scripts).41 Novel prompting strategies, such as the "+first page" method for multi-page documents (where the MLLM uses the first page image and OCR of the whole document to improve transcription), are being explored to leverage shared document features efficiently.42 Studies on historical tabular documents show LLMs can outperform traditional systems.32

**Relevant Datasets:**

* **IAM Handwriting Database:** A widely adopted benchmark for modern English handwriting.1  
* **RIMES Database:** Contains French and English handwritten documents, often used in competitions and research.1  
* **Historical Collections:** Datasets derived from specific archival collections, such as the Jeremy Bentham papers 4, and datasets from ICDAR (International Conference on Document Analysis and Recognition) competitions focusing on Early Modern English, German, and Latin scripts are crucial for training models on historical handwriting styles. The HathiTrust Digital Library also provides access to large datasets of historical English literature, though these are primarily printed texts, they can be useful for language modeling aspects.67

Addressing Degradation and Stylistic Variation:  
Successfully transcribing historical English cursive requires robust preprocessing to handle document degradation (stains, fading, foxing), data augmentation techniques tailored to the types of damage and script variations found in historical documents, and models capable of generalizing across diverse and often inconsistent handwriting styles.6 MLLMs are showing potential in zero-shot or few-shot settings for such challenging documents, reducing the need for extensive fine-tuning on every specific collection.42 Legibility tips for human writers, such as maintaining consistent letter forms and avoiding excessive slant, also highlight the features OCR systems find challenging.11

### **4.3. Non-Western Scripts: Indic Languages (e.g., Urdu, Devanagari, Bengali)**

Indic scripts, used for a multitude of languages across South Asia, present a rich and complex area for HTR development, characterized by unique scriptorial features and significant challenges related to resource availability.

**Specific Challenges:**

* **Conjunct Characters (Matras/Yuktakshars):** A hallmark of many Indic scripts is the formation of conjunct characters, where two or more consonants combine to form a single, often complex, graphical unit. These are distinct from simple ligatures and significantly increase the effective size of the character set.25  
* **Large Character Sets:** Most Indic scripts have a basic set of around 100 or more unique Unicode characters, including vowels, consonants, vowel modifiers (matras), and various other symbols. When combined with conjuncts, the number of distinct visual forms can be very large.69  
* **Script Diversity:** India alone has 22 official languages, many using distinct scripts (e.g., Devanagari for Hindi and Marathi, Bengali script, Tamil script, Telugu script, Gurmukhi for Punjabi, Gujarati script, Odia script, Kannada script, Malayalam script, and Perso-Arabic script for Urdu). Each script has its own orthographic rules, character shapes, and conjunct formation principles.25  
* **Data Scarcity:** A primary obstacle for Indic HTR is the limited availability of large, high-quality, annotated datasets, especially for handwritten text and for many of the less-resourced languages within the Indic family.3 This hampers the training of robust deep learning models.  
* **Other Challenges:** Include variations in handwriting styles, varying ink density, and complex document layouts with potential overlaps between words.69

**Promising Systems and Datasets:**

* **PLATTER (Page-Level hAndwriTTen TExt Recognition system):** This framework addresses page-level Indic HTR using a two-stage approach: word detection (often using a language-agnostic DBNet model) followed by word-level HTR using various models like CRNN, MASTER, or PARSEQ.34 On the CHIPS dataset, PLATTER with a CRNN-based HTR model achieved an end-to-end Character Recognition Rate (CRR) of approximately 84.44% across ten Indic languages.  
* **Datasets:**  
  * **CHIPS (Corpus of Handwritten Indic Scripts):** A novel page-level dataset created for the PLATTER project, containing labeled word bounding boxes and corresponding text in ten Indic languages (Bengali, Devanagari, Gujarati, Gurumukhi, Kannada, Malayalam, Odia, Tamil, Telugu, Urdu), designed as a benchmark for page-level Indic HTR.34  
  * **AIO-HB Dataset:** Contains handwritten text in Hindi and Bengali from 202 different writers, covering all letters of these scripts and suitable for various document analysis tasks.70  
  * **IIIT-Indic-HW-Words:** A word-level HTR benchmark dataset with over 1 million images across 10 Indic languages, used for training HTR models and as a basis for CHIPS.34  
  * **ICDAR 2023 Competition on Indic Handwriting Text Recognition Datasets:** The competition introduced new, manually annotated validation and test sets for ten Indic languages, each with 6000 word images from 100 writers per language.69

**Insights from Competitions:** The ICDAR 2023 IHTR competition underscored the dominance of deep learning architectures (CRNNs and Transformers like TrOCR) for Indic scripts.69 Key success factors included extensive data augmentation (using synthetic and real-world data), multi-stage training strategies, the use of language models for post-OCR error correction, and careful preprocessing (e.g., image binarization, skew correction, and script-specific handling like flipping images for right-to-left Urdu). The winning team, Upstage KR, achieved an average CRR of 95.94% and WRR of 88.31% across the ten languages, demonstrating significant progress.69

### **4.4. Non-Western Scripts: East Asian Languages (e.g., Chinese, Japanese Kuzushiji, Korean Cursive)**

East Asian scripts, with their logographic or syllabic nature and distinct cursive traditions, present another unique set of HTR challenges.

**Unique Script Complexities and Cursive Styles:**

* **Chinese:** Characterized by a very large set of characters (thousands in common use), each composed of multiple strokes. Recognition is challenged by the sheer number of classes, the visual similarity between many characters, and the evolution of script styles over millennia, from oracle bone script to various standard and cursive forms.72 Handwritten Chinese, especially in cursive or calligraphic styles, can be very difficult to decipher.12  
* **Japanese Kuzushiji:** This is a highly cursive script style that was used in Japan for over a thousand years for writing historical documents. It is largely unreadable by most contemporary Japanese people. Challenges for HTR include the extreme cursiveness leading to highly variable character shapes, long-range contextual dependencies, a large vocabulary of characters and their cursive variants, and often non-standardized or irregular layouts in historical documents.77  
* **Korean Cursive (흘림체 \- Heullimche, 초서 \- Choseo):** While research on Korean cursive HTR is less voluminous compared to Chinese or Japanese, it shares complexities with other East Asian cursive scripts, including connected strokes and stylistic variations. The recognition of Hanja (Chinese characters historically used in Korean writing) within cursive Korean text is also a relevant challenge.79

**Noteworthy Models and Datasets:**

* **Chinese:**  
  * **Models:** Deep learning methods, particularly CNN-based feature extractors combined with classifiers or sequence models, are state-of-the-art.76 Approaches include integrating traditional normalization-cooperated direction-decomposed feature maps (directMap) with CNNs 76, and prototype-based classifiers with CNN features.75 Writer adaptation techniques have also been shown to be effective.76  
  * **Datasets:** The **CASIA Online and Offline Chinese Handwriting Databases** are major resources, containing approximately 3.9 million isolated character samples and about 5,090 pages of handwritten texts from 1,020 writers, covering 7,356 character classes (including Chinese characters and symbols).73  
* **Japanese Kuzushiji:**  
  * **Models:** Deep learning is the primary approach.77 **KuroNet**, a model based on a residual U-Net architecture, was proposed for jointly recognizing an entire page of Kuzushiji text, aiming to handle long-range context and non-standard layouts without extensive pre-processing.78  
  * **Datasets:** Competitions and research initiatives have led to the creation of Kuzushiji datasets, often derived from digitized historical documents.  
* **Korean Cursive:**  
  * **Models:** CNNs have been applied to handwritten Korean character recognition.80 More recent work for classical Hanja cursive documents uses High-Resolution Networks (HRNet), a type of CNN, combined with data augmentation techniques like highly variable random cropping within document regions.79  
  * **Datasets:** Research often relies on smaller, specialized datasets. For instance, one study used 1,100 cursive Hanja documents, augmented significantly for training.79

General Challenges for East Asian HTR:  
Beyond script-specific issues, common challenges include document degradation in ancient texts (e.g., the Dunhuang manuscripts, which are often stained, torn, or have faded ink) 12, difficulties in the binarization stage of image processing (e.g., retaining faint strokes or handling colored ink appropriately) 12, and dealing with irregular layouts, character corrections, and margin notes commonly found in historical handwritten works.12  
The diverse nature of these scripts—ranging from alphabetic systems like Spanish and English, to abugidas like many Indic scripts, and logographic/syllabic systems like Chinese, Japanese, and Korean—necessitates tailored HTR solutions. Techniques that prove highly effective for Latin-based cursive may not directly transfer to the complexities of Arabic ligatures or Indic conjunct characters without substantial adaptation. This inherent scriptorial diversity means that the development of truly universal or easily adaptable multilingual HTR systems remains a significant research frontier. Furthermore, the "resource gap" is a recurring theme. While English and, to some extent, Spanish and Chinese HTR benefit from more established and larger datasets 5, many non-Western languages, particularly numerous Indic scripts and historical variants like Kuzushiji, face a significant scarcity of high-quality, annotated training data.3 This lack of data directly impedes the development and performance of data-hungry deep learning models. Consequently, initiatives focused on creating diverse, large-scale datasets for these under-resourced languages (such as the CHIPS dataset for Indic scripts 34 or the Muharaf dataset for historical Arabic 50) are as pivotal to progress as algorithmic innovations. For these languages, advanced data augmentation strategies and synthetic data generation techniques become even more critical tools.3

**Table 2: Overview of Cursive OCR/HTR for Selected Non-Arabic Languages**

| Language/Script Group | Key Model(s)/Architectures | Common Datasets | Typical CER/WER Range (Representative) | Key Challenges Specific to this Script |
| :---- | :---- | :---- | :---- | :---- |
| **Spanish (Historical & Modern)** | Coloso Español (PyLaia), Spanish TrOCR (Transformer) 31 | XFUND Spanish, Transkribus collections | Coloso: 4.80% CER. TrOCR (Large, XFUND): 6.32% CER, 18.17% WER 31 | Paleographic variations over centuries, abbreviations, ligatures in historical scripts. |
| **English (Historical)** | TrOCR, MLLMs (GPT-4o, Gemini, Claude) for transcription & post-correction 4 | IAM, RIMES, Jeremy Bentham papers, ICDAR historical collections, HathiTrust 5 | MLLM post-correction: \<1% CER (German Fraktur, similar complexity).41 LLMs outperform traditional on tabular HTR.32 | Extreme handwriting variability, document degradation (fading, stains), diverse historical cursive styles (e.g., secretary hand, italic hand). |
| **Indic Scripts (e.g., Devanagari, Bengali, Urdu)** | PLATTER (DBNet \+ CRNN/Transformer), CRNNs, TrOCR 34 | CHIPS, AIO-HB, IIIT-Indic-HW-Words, ICDAR 2023 IHTR datasets 34 | ICDAR Comp. Winner: \~95.9% CRR, \~88.3% WRR (avg. 10 langs). PLATTER (CRNN): \~84.4% CRR (avg. 10 langs).34 | Conjunct characters (matras), large character sets, multiple distinct scripts, diacritics, data scarcity. Urdu: RTL, Perso-Arabic script complexities. |
| **East Asian \- Chinese** | CNN+directMap, CNN+Prototype Classifier, Transformers 75 | CASIA Online/Offline DBs, ICDAR competition datasets 73 | Offline HCCR (ICDAR-2013): \>97% accuracy with advanced models.76 | Very large character set, stroke complexity, numerous similar characters, diverse calligraphic and cursive styles. |
| **East Asian \- Japanese (Kuzushiji)** | KuroNet (Res U-Net), Deep Learning general 77 | Kuzushiji-MNIST, Kuzushiji-49, specialized historical collections \[78 (implied)\] | Performance varies; a challenging HTR task. | Highly cursive and stylized, large vocabulary of cursive forms, non-standard layouts in historical documents, unreadable by most modern natives. |
| **East Asian \- Korean (Cursive/Hanja)** | CNNs, HRNet (for Hanja cursive) 79 | Smaller, specialized datasets; some use of augmented Hanja cursive documents 79 | Older CNN: \~96% accuracy (Korean characters).80 Hanja cursive is an active research area. | Cursive forms (Heullimche, Choseo), recognition of Hanja characters, limited large-scale public cursive datasets. |

## **5\. Core Technical Considerations in Modern Cursive OCR**

Beyond language-specific challenges, several core technical aspects underpin the development and application of modern cursive OCR/HTR systems. These include the ability to handle different data structures, the critical role of training data, the comparative strengths of AI versus traditional methods, and robust performance evaluation.

### **5.1. Recognizing Structured vs. Unstructured Data**

Cursive handwriting appears in diverse contexts, from free-flowing text in historical manuscripts (unstructured data) to entries in forms, tables, and receipts (structured data). OCR/HTR systems must adapt to these different data types.

Approaches for Structured Data (Forms, Tables, Receipts):  
Recognizing handwritten entries in structured documents often involves more than just transcribing the text; it requires understanding the document's layout to extract information from specific fields, a task known as key-value extraction.83

* **Layout-Aware Models:** Architectures like **LayoutLM** and its successors (e.g., LayoutLMv3) are pre-trained by incorporating textual information, visual image features, and positional layout data from scanned documents.9 LayoutLMv3, for instance, is a multimodal approach that processes text, layout, and image information together. While powerful for printed documents, its out-of-the-box performance on handwritten text might be limited without specific fine-tuning, and it may require tools like Tesseract OCR for initial text extraction from images.39  
* **Multimodal Large Language Models (MLLMs):** MLLMs are increasingly demonstrating capabilities in document intelligence, including the parsing of structured forms and tables, even those containing handwritten entries.42 Their ability to process visual layout and textual content holistically makes them promising for these tasks.  
* **Specialized Pipelines:** For specific structured data applications, such as extracting medicine names from handwritten prescriptions, hybrid approaches are common. For example, a Mask R-CNN model can be used for segmenting the relevant medicinal sections on a prescription form, followed by a TrOCR model to transcribe the handwritten text within those segmented regions.9 Similarly, the "Invizo" system for Arabic OCR is designed for general-purpose use, including the processing of documents or receipts, which are forms of structured data.27  
* **Traditional Methods:** Earlier systems for form processing often relied on predefined templates or rule-based zoning to locate fields, with OCR/HTR applied subsequently to the content of these zones.7 Modern systems tend to use machine learning for more flexible field detection.

Handling Unstructured Data (Free-Flowing Text in Manuscripts, Notes):  
This is the traditional domain of HTR, focusing on the transcription of continuous lines or paragraphs of handwritten text found in documents like historical manuscripts, personal notes, and letters.1

* **Line Segmentation:** For many HTR systems operating on full pages, accurate segmentation of the page into individual text lines is a critical preprocessing step.17  
* **Recognition Granularity:** Models typically process text line by line. However, recent advancements are pushing towards paragraph-level or even full document-level transcription, aiming to leverage broader contextual information.1

**The Blurring Lines:** The distinction between approaches for structured and unstructured data is becoming less rigid. Advanced AI models, particularly MLLMs and sophisticated Transformer architectures, are increasingly capable of understanding both the content and the layout of a document in a more holistic manner. This suggests a convergence towards unified frameworks that can handle diverse document types, whether they contain neatly formatted tables or free-flowing cursive script, potentially within a single model.42 Such models might learn to identify structured elements and transcribe their content while simultaneously processing surrounding unstructured text, leading to more versatile document intelligence solutions.

### **5.2. The Crucial Role of Training Data**

The performance of modern HTR systems, especially those based on deep learning, is inextricably linked to the quality, quantity, and diversity of the training data.

Data Scarcity in Low-Resource Languages and Specialized Domains:  
A persistent challenge in HTR is the scarcity of adequate training data, particularly for cursive scripts in low-resource languages and for specialized domains such as particular historical periods, unique calligraphic styles, or specific types of forms.3 Deep learning models are notoriously data-hungry, and insufficient or non-representative training data can lead to poor generalization and suboptimal performance on real-world tasks.3  
Advances in Synthetic Data Generation:  
To mitigate the problem of data scarcity, the generation of synthetic handwritten text images has become an important area of research.1

* Models like TrOCR are often pre-trained on massive datasets of synthetically generated handwriting before being fine-tuned on smaller, domain-specific real datasets.4  
* Generative Adversarial Networks (GANs) are increasingly employed to synthesize realistic handwritten text images that can mimic various styles and content.3  
* Other methods include using RNNs to generate handwriting stroke sequences which are then rendered as images 21, or employing existing fonts with various augmentations and distortions to simulate handwritten appearance.31 While synthetic data can significantly expand training sets, ensuring its diversity and realism to match real-world handwriting complexities remains an ongoing challenge.

Effective Data Augmentation Techniques for Cursive Scripts:  
Data augmentation involves applying transformations to existing labeled data to create new training samples, thereby increasing the size and diversity of the training set and improving model robustness.

* **Standard Image Manipulations:** These include geometric transformations like rotation, scaling, shearing, and elastic distortions (which are particularly effective for simulating handwriting variability), as well as photometric changes like adjustments to brightness and contrast, and the addition of various types of noise (e.g., Gaussian noise).3  
* **Script-Specific Augmentations:** Some techniques are tailored to the characteristics of specific scripts. For example, Moving Least Squares (MLS) deformation has been used to create realistic augmentations for Arabic handwritten text by applying localized, non-rigid transformations.21  
* **Adaptive Augmentation:** This strategy involves analyzing the training data to identify underrepresented characters or words and then selectively applying augmentation techniques to generate more samples for these less frequent classes, aiming for a more balanced dataset.26

Public Datasets:  
Publicly available, well-annotated datasets are indispensable for benchmarking HTR systems, fostering reproducible research, and driving progress in the field. Table 3 provides an overview of some prominent datasets.  
**Table 3: Prominent Public Datasets for Cursive HTR**

| Dataset Name | Primary Language(s) | Script Type(s) | Data Type | Size/Scope (Approx.) | Key Features/Annotations | Availability |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| IAM 5 | English | Latin Cursive & Printed | Modern Handwriting (forms) | \~1,500 forms, \~13,000 lines, \~115,000 words, 657 writers | Line, word, character segmentation; transcriptions | Public |
| RIMES 5 | French, English | Latin Cursive | Modern Handwriting (letters) | \~12,000 pages, \~1,300 writers | Line, word transcriptions | Public (via competition/request) |
| IFN/ENIT 18 | Arabic | Arabic Naskh/Ruq'ah like | Modern Handwriting (Tunisian town names) | \~26,000 word images, \~400 writers | Word transcriptions | Public |
| Muharaf 50 | Arabic | Arabic Ruq'ah (primarily) | Historical Manuscripts (19th-21st C.) | 1,644 images, \~36,000 lines | Polygonal line coordinates, transcriptions, page elements | Public (GitHub) |
| KHATT 18 | Arabic | Arabic various styles | Modern Handwriting | Paragraphs, lines, words | Transcriptions | Public |
| AMFDS 17 | Arabic | Various Arabic fonts (some mimicking handwriting) | Printed/Synthetic | \- | \- | Used in research, availability varies |
| Historical Arabic Dataset (Mendeley) 64 | Arabic | Historical Arabic | Historical Manuscripts | 40 pages | Textual ground truth | Public (Mendeley Data) |
| CHIPS 34 | 10 Indic Languages (Devanagari, Bengali, Urdu, etc.) | Various Indic scripts | Page-level Handwriting | Page images, word bounding boxes, transcriptions | Word bounding boxes, transcriptions | Public (associated with PLATTER) |
| AIO-HB 70 | Hindi, Bengali | Devanagari, Bengali | Modern Handwriting | 202 writers, multiple samples | Line/word segmentation, transcriptions | Described in research, availability varies |
| CASIA Online/Offline DBs 73 | Chinese | Chinese Characters | Modern Handwriting (isolated chars & texts) | \~3.9M isolated chars, \~5,090 text pages, 1,020 writers | Character-level annotations, online trajectory data | By request/application |
| XFUND 31 | Multilingual (incl. Spanish) | Various (primarily printed) | Forms (visually rich documents) | Few thousand examples per language | Key-value pairs, bounding boxes | Public |
| Coloso Español (training data) 66 | Spanish | Latin (medieval to modern) | Historical Manuscripts & Typewritten Docs | \- (underlying data for Transkribus model) | \- | Used by READ-COOP, model public |

The interplay between HTR model development and dataset creation forms a crucial feedback loop. As HTR models improve, they can be employed to semi-automatically transcribe larger volumes of historical or handwritten data. These initial, albeit imperfect, transcriptions can then be corrected by human annotators much more efficiently than transcribing from scratch.6 This accelerated creation of ground truth data, in turn, fuels the training of even more accurate and robust HTR models. This synergy is particularly vital for low-resource languages and specialized domains, where the initial hurdle of dataset creation is often the most significant barrier to progress.

### **5.3. Methodological Comparison: Advanced AI vs. Traditional OCR**

The trajectory of HTR technology clearly shows a shift from traditional OCR techniques to advanced AI-driven methods.

Limitations of Heuristic and Early Statistical Methods:  
Traditional OCR methods, including rule-based systems, template matching, and early statistical models like HMMs, faced significant limitations when applied to cursive handwriting.1

* **Handling Variability:** They struggled with the high degree of variability inherent in handwriting styles, character shapes, and connections. Their reliance on predefined rules or templates made them brittle and unable to generalize well to unseen writing.5  
* **Feature Engineering:** These methods often required extensive manual feature engineering, where domain experts would define specific visual characteristics to identify characters. This process was time-consuming, script-dependent, and often suboptimal.1  
* **Accuracy:** Consequently, their accuracy on challenging real-world cursive documents was generally much lower than what is achievable with modern AI approaches.5 Even with improvements like HMMs, the fundamental challenges of segmenting cursive script and modeling complex dependencies remained.1

Strengths and Adaptability of Deep Learning:  
Deep learning has fundamentally transformed HTR by addressing many of the limitations of earlier methods.1

* **Automatic Feature Learning:** Deep neural networks, particularly CNNs, can automatically learn relevant hierarchical features directly from the raw pixel data of images, eliminating the need for manual feature engineering.1 This allows models to discover complex patterns and representations that might be missed by human designers.  
* **Superior Performance on Variability and Noise:** Deep learning models have demonstrated remarkable success in handling the high variability, noise, and complex patterns characteristic of cursive handwriting.1 Architectures combining CNNs for visual processing, RNNs (especially LSTMs/BiLSTMs) for sequence modeling, and Transformers for capturing long-range dependencies provide powerful tools for image understanding and sequential data interpretation.1  
* **End-to-End Learning:** Techniques like CTC loss enable end-to-end training of HTR systems, simplifying the pipeline and allowing the model to learn an optimal mapping from input image to output text sequence directly.1

The Expanding Role of LLMs/MLLMs:  
Large Language Models and Multimodal Large Language Models are further pushing the boundaries of what is possible in HTR and document understanding.3

* **Few-Shot and Zero-Shot Capabilities:** MLLMs have shown the potential for performing HTR tasks with minimal or no task-specific training data (few-shot or zero-shot learning).3 This could be transformative for low-resource languages or highly specialized document types where collecting large annotated datasets is impractical.  
* **OCR Post-Correction:** LLMs, with their powerful language understanding, can be used to correct errors in the output of primary OCR/HTR engines, leveraging linguistic context to improve the accuracy and fluency of transcriptions.32  
* **Holistic Document Understanding:** MLLMs can process and integrate information from both the visual layout of a document and its textual content, enabling more comprehensive document intelligence beyond simple transcription.6 However, the application of MLLMs in HTR is still in its early stages. Challenges include their computational expense, processing speed, and the risk of generating "hallucinations" (plausible-sounding but incorrect information).6 Some studies have even noted that careless application of LLMs for post-correction can degrade transcription quality.32

### **5.4. Performance Evaluation: Precision, Recall, and Beyond**

Evaluating the performance of cursive OCR/HTR systems requires appropriate metrics that capture different aspects of accuracy and utility.

Standard Metrics:  
The most commonly used metrics for HTR are:

* **Character Error Rate (CER):** This metric quantifies the dissimilarity between the recognized text and the ground truth text at the character level. It is calculated as the minimum number of character insertions (I), deletions (D), and substitutions (S) required to transform the hypothesis into the reference, divided by the total number of characters (N) in the reference text: CER=(S+D+I)/N.1 CER is particularly important for tasks where small character-level errors can have significant consequences, such as in medical terms, numerical data, or highly inflected languages.91  
* **Word Error Rate (WER):** Analogous to CER, WER measures errors at the word level: WER=(Sw​+Dw​+Iw​)/Nw​, where Sw​,Dw​,Iw​ are word substitutions, deletions, and insertions, and Nw​ is the total number of words in the reference.1 WER often gives a better sense of overall readability and semantic correctness than CER.  
* **Accuracy (Character Recognition Rate \- CRR, Word Recognition Rate \- WRR):** These metrics represent the proportion of correctly recognized characters or words, often calculated as 1−CER or 1−WER, respectively, or directly as the count of correct items over the total items.1

Importance of Precision and Recall in Application-Specific Contexts:  
While CER and WER measure overall transcription fidelity, for many practical applications, especially those involving information extraction from structured or semi-structured documents, precision and recall for specific information fields become critical.

* **Precision:** Measures the proportion of recognized items (e.g., characters, words, extracted entities) that are correct. High precision is important when the cost of false positives is high.  
* **Recall (Sensitivity):** Measures the proportion of actual items in the ground truth that were correctly recognized by the system. High recall is crucial in domains where missing information is critical, such as failing to detect a medication name in prescription OCR 9 or missing a key term in legal document analysis.  
* **F-measure (F1-score):** The harmonic mean of precision and recall, providing a single, balanced measure of performance, especially useful when there's an uneven class distribution or when both precision and recall are equally important.1 These metrics are particularly relevant for evaluating the performance of text detection modules (e.g., correctly identifying text line boundaries 17) or the accuracy of key-value pair extraction from forms.

Benchmarking Practices and the Need for Diverse Test Sets:  
Standardized benchmark datasets like IAM, IFN/ENIT, and Muharaf are essential for the fair comparison of different HTR systems and for tracking progress in the field.1 However, many existing benchmarks have limitations; they might be focused on specific languages or writing styles and may not fully represent the diversity and complexity of real-world documents.34 There is an ongoing need for the development and adoption of more comprehensive test sets that cover a wider range of scripts, levels of document degradation, handwriting styles, and document types to ensure that HTR systems are robust and generalize well to unseen data.  
The evaluation of HTR systems, particularly those designed for integration into larger information extraction or document processing pipelines, must extend beyond generic transcription metrics like CER and WER. For applications such as automated form processing 7 or the critical task of reading medical prescriptions 9, the ultimate utility of the system often hinges on the correct extraction and interpretation of specific, predefined information fields (e.g., patient name, medication dosage, invoice total). A low overall CER might be achieved, but if critical data points within a form are misrecognized, the system fails its primary purpose. Therefore, task-specific evaluations, focusing on the precision and recall of extracting key information elements or named entities, are paramount. This implies that benchmark datasets should also evolve to include annotations and evaluation protocols that support these more application-oriented assessments, ensuring that HTR technology is not only accurate in transcription but also fit for its intended purpose in real-world workflows.

**Table 4: Key Performance Metrics in Cursive OCR/HTR**

| Metric | Definition/Formula | Relevance & Use Case in Cursive OCR/HTR |
| :---- | :---- | :---- |
| **Character Error Rate (CER)** 74 | CER=(S+D+I)/N \<br/\> (S: Substitutions, D: Deletions, I: Insertions, N: Total characters in reference) | Measures fine-grained accuracy at the character level. Crucial for highly inflected languages, connected scripts (e.g., Arabic), numerical data, codes, and domains where minor errors are critical (e.g., medical terms 9, legal names). Lower is better. |
| **Word Error Rate (WER)** 74 | WER=(Sw​+Dw​+Iw​)/Nw​ \<br/\> (Word-level S, D, I; Nw​: Total words in reference) | Measures accuracy at the word level, often better reflecting overall readability and semantic correctness. Lower is better. |
| **Precision** 17 | P=TP/(TP+FP) \<br/\> (TP: True Positives, FP: False Positives) | Proportion of recognized items (characters, words, entities, detected lines) that are correct. Important for minimizing false alarms or incorrect extractions, e.g., ensuring an extracted medication name is indeed correct. |
| **Recall (Sensitivity)** 17 | R=TP/(TP+FN) \<br/\> (FN: False Negatives) | Proportion of actual items in the ground truth that were correctly recognized. Critical when missing information is costly, e.g., failing to detect all medication entries on a prescription 9 or all relevant clauses in a historical document.6 |
| **F1-score (F-measure)** 17 | F1=2⋅(P⋅R)/(P+R) | Harmonic mean of Precision and Recall, providing a balanced measure, especially useful when classes are imbalanced or both P and R are important. Often used for text detection or entity extraction tasks. |
| **Character Recognition Rate (CRR)** / **Accuracy** 17 | CRR=(Number of correctly recognized characters)/(Total characters) \<br/\> Often 1−CER | Proportion of characters correctly identified. Higher is better. |
| **Word Recognition Rate (WRR)** / **Accuracy** 17 | WRR=(Number of correctly recognized words)/(Total words) \<br/\> Often 1−WER | Proportion of words correctly identified. Higher is better. |

## **6\. Current Challenges and Future Research Directions**

Despite significant advancements, the field of cursive OCR/HTR continues to face persistent challenges, while also benefiting from emerging research directions that promise to push the boundaries of capability and applicability.

### **6.1. Persistent Hurdles in Cursive OCR**

Several core difficulties continue to challenge researchers and developers:

* **Extreme Handwriting Variability and Ambiguity:** The sheer diversity in human handwriting—encompassing variations in style, slant, size, character formation, ligatures, and spacing, both between different writers and within the writing of a single individual over time—remains the most fundamental challenge.1 Ambiguous characters or those that are poorly formed can be difficult even for human readers to decipher, let alone automated systems.  
* **Robustness to Noise, Degradation, and Low-Quality Inputs:** Many documents of interest, particularly historical manuscripts or records stored in suboptimal conditions, suffer from degradation such as faded ink, stains, foxing, tears, and general noise.3 Low-resolution scans, poor lighting during capture, or complex backgrounds further complicate the recognition task by obscuring textual details or introducing artifacts.  
* **Writer Adaptation and Personalization:** While general HTR models aim to recognize a wide range of handwriting styles, their performance can often be significantly improved if they can adapt to the specific style of a new, unseen writer. Developing methods for rapid and effective writer adaptation, ideally with minimal writer-specific training data (e.g., few-shot or zero-shot adaptation), is an ongoing research area.23  
* **Handling Out-of-Vocabulary (OOV) Words and Rare Glyphs:** HTR systems, especially those incorporating language models, can struggle with words not present in their training vocabulary (OOV words). This is particularly problematic for historical texts containing archaic terms, proper nouns, or specialized jargon, as well as for scripts with a vast number of rare characters or unique calligraphic glyphs. While language models can help predict plausible words, the visual recognition of these rare forms remains difficult.  
* **Segmentation of Highly Cursive and Connected Scripts:** For scripts with extreme cursiveness and complex connectivity, such as Arabic or densely written English cursive, the accurate segmentation of text into lines, words, or characters (if required by the model architecture) remains a non-trivial problem.1 Errors in segmentation can propagate and severely degrade the performance of the subsequent recognition stage.

### **6.2. Advancing the Frontier**

Future research in cursive OCR/HTR is likely to focus on several key areas to overcome existing limitations and enhance capabilities:

* **Unsupervised and Self-Supervised Learning for HTR:** A major thrust is towards reducing the heavy reliance on large, meticulously labeled datasets. Unsupervised learning methods aim to discover patterns in data without any explicit labels, while self-supervised learning (SSL) techniques create pseudo-labels from the data itself (e.g., by masking parts of an image or text and training the model to predict the masked portions).3 Approaches like contrastive learning (e.g., SimCLR) and masked autoencoders (e.g., MAE) are showing promise in learning meaningful representations from unlabeled image and text data, which can then be fine-tuned for HTR tasks with smaller amounts of labeled data. This data-efficiency is crucial for tackling the "long tail" of languages, scripts, and historical document types for which extensive labeled data is unlikely to become available.  
* **Enhanced Multimodal Learning and Contextual Understanding with LLMs/MLLMs:** The integration of MLLMs is set to continue, with research focusing on leveraging their capabilities more effectively for HTR.4 This includes improving end-to-end transcription by better incorporating visual layout cues, understanding stylistic elements of handwriting, and utilizing broader linguistic and world knowledge. A key goal is to enhance few-shot and zero-shot HTR performance, making these powerful models more accessible for diverse applications without requiring extensive retraining.3  
* **Mitigating Hallucinations and Ensuring Reliability in LLM-based OCR:** A critical challenge for the practical deployment of MLLMs in HTR is their propensity to "hallucinate"—generating outputs that are plausible but inconsistent with the visual input or factual reality.6 Future work will need to develop robust mechanisms for detecting, quantifying, and mitigating such hallucinations. This may involve techniques like adversarial training, employing multiple LLM agents with voting mechanisms, integrating external knowledge retrieval for verification, improving model calibration for confidence scoring, and designing prompts that constrain outputs more effectively.48 Ensuring the factual correctness and faithfulness of MLLM-generated transcriptions is paramount for their adoption in sensitive applications.  
* **Development of More Efficient and Scalable Architectures:** The computational and memory demands of very large models, particularly Transformers and MLLMs, can be prohibitive for many applications.5 Research into model compression techniques (e.g., quantization, pruning), more efficient architectures (e.g., sparse neural networks, Mixture of Experts like Google's Switch Transformer), and dynamic computation methods will be essential for making advanced HTR solutions more scalable, cost-effective, and deployable on edge devices or with limited computational resources.  
* **Focus on Low-Resource Languages and Under-Represented Scripts:** Addressing the significant "resource gap" for many of the world's languages and historical scripts remains a priority.3 This involves not only continued efforts in dataset creation and annotation but also the development of cross-lingual transfer learning techniques, domain adaptation methods, and model architectures specifically designed to handle the unique complexities of these scripts with limited training data. The goal is to make HTR technology more equitable and accessible globally.

The pursuit of data-efficient learning strategies is a significant trend, driven by the practical limitations of acquiring large labeled datasets for every conceivable HTR task. Success in this area could dramatically broaden the applicability of HTR to a vast array of historical archives and languages that are currently underserved. Simultaneously, the increasing power of MLLMs, while promising, introduces the critical challenge of trustworthiness. Robust mechanisms for hallucination detection and confidence estimation will be indispensable for integrating these models reliably into HTR workflows, particularly where accuracy is non-negotiable. Human oversight and verification are likely to remain crucial components, especially in the initial phases of adopting these newer, more complex technologies.

## **7\. Conclusion and Recommendations**

The field of cursive Optical Character Recognition (OCR) and Handwritten Text Recognition (HTR) has undergone a remarkable transformation, evolving from rudimentary pattern matching techniques to highly sophisticated deep learning models and, more recently, to the integration of Multimodal Large Language Models (MLLMs). This technological progression has significantly advanced our ability to digitize and understand the vast corpus of handwritten information present in historical archives, personal records, and various forms of documentation.

Recapitulation of the State of the Art:  
The current state-of-the-art in cursive OCR/HTR is unequivocally dominated by deep learning methodologies. Architectures combining Convolutional Neural Networks (CNNs) for visual feature extraction, Recurrent Neural Networks (RNNs)—particularly LSTMs and BiLSTMs—for sequence modeling, and Transformer models with their powerful attention mechanisms, have demonstrated superior performance across a wide range of languages and script types. The Connectionist Temporal Classification (CTC) loss function has been pivotal in enabling end-to-end training for sequence transcription. The emergence of MLLMs like GPT-4v/GPT-4o and Gemini represents a new frontier, showing potential for zero-shot or few-shot transcription, OCR post-correction, and holistic document understanding that encompasses both layout and content. Despite these advancements, significant challenges persist, including handling extreme handwriting variability, robustness to document degradation, the scarcity of training data for many languages and specialized domains, and the computational demands and reliability concerns associated with the newest generation of AI models.  
Key Insights for Arabic and Other Language Groups:  
For Arabic cursive HTR, substantial progress has been made by tailoring deep learning pipelines to its unique scriptorial complexities, such as inherent cursiveness, contextual letter shaping, ligatures, and diacritics. Specialized segmentation techniques (e.g., DBNet++), CNN-Transformer, and attention-enhanced CNN-BiLSTM-CTC architectures have yielded promising results, particularly when trained on increasingly diverse datasets like Muharaf and IFN/ENIT. However, the interdependence of pipeline stages (segmentation, recognition) and the need for even larger, more representative datasets remain key considerations.  
For **Spanish** and **English** cursive, particularly in historical contexts, models like TrOCR and specialized systems (e.g., Coloso Español) achieve good accuracy. MLLMs are showing particular strength in handling degraded historical documents and for post-correction tasks in these Latin-based scripts. The availability of benchmark datasets like IAM and RIMES, alongside growing historical collections, supports ongoing research.

For other **non-Western languages**, the landscape is more varied. **Indic scripts** (e.g., Devanagari, Bengali, Urdu) present challenges due to conjunct characters, large character sets, and significant data scarcity. Systems like PLATTER and initiatives like the ICDAR 2023 IHTR competition are driving progress, emphasizing the need for page-level understanding and robust models for diverse scripts. **East Asian languages** (e.g., Chinese, Japanese Kuzushiji, Korean cursive) also pose unique difficulties stemming from large character sets (Chinese), extreme cursiveness in historical scripts (Kuzushiji), and document degradation. Large datasets like CASIA for Chinese exist, but specialized historical scripts often require bespoke solutions and dataset creation efforts. A common thread across these non-Western scripts is the amplified impact of the "resource gap"—the lack of extensive, high-quality annotated data significantly hinders development compared to English.

**Recommendations:**

1. **For Future Research:**  
   * **Data-Efficient Learning:** Prioritize research into unsupervised, self-supervised, and few-shot learning techniques to reduce dependence on large labeled datasets, especially for low-resource languages and specialized domains.  
   * **Interpretability and Reliability:** Develop and integrate explainable AI (XAI) methods into HTR systems, particularly for complex models like Transformers and MLLMs, to understand error modes and build trust, especially in critical applications. Address MLLM hallucination through robust detection and mitigation strategies.  
   * **Robustness and Generalization:** Focus on improving model robustness to extreme handwriting variability, severe document degradation, and out-of-vocabulary words. Enhance writer adaptation capabilities.  
   * **Specialized Architectures for Complex Scripts:** Continue developing and refining architectures tailored to the unique challenges of scripts like Arabic (ligatures, diacritics), Indic (conjuncts), and highly cursive East Asian styles.  
   * **Efficient Models:** Investigate model compression, sparse architectures, and other techniques to create more computationally efficient HTR models without sacrificing accuracy, facilitating wider deployment.  
2. **For Dataset Curation:**  
   * **Collaborative Creation of Diverse Datasets:** Encourage and fund collaborative, international efforts to create, annotate, and share large-scale, high-quality datasets for under-resourced languages, historical scripts, and diverse document types (including structured forms with cursive entries).  
   * **Standardization of Annotation:** Promote standardized annotation formats and guidelines (e.g., PAGE-XML) to ensure interoperability and comparability of datasets and models.  
   * **Leverage Improved HTR for Annotation:** Utilize state-of-the-art HTR models in human-in-the-loop annotation workflows to accelerate the creation of new ground truth data, particularly for historical and low-resource materials.  
   * **Synthetic Data Refinement:** Continue research into generating highly realistic and diverse synthetic handwritten data that can effectively augment real datasets and improve model generalization.  
3. **For Technology Adoption:**  
   * **Hybrid Approaches:** Consider hybrid systems that combine the strengths of different technologies (e.g., specialized HTR models for initial transcription, MLLMs for contextual refinement or post-correction, human-in-the-loop for verification in critical tasks).  
   * **Context-Specific Evaluation:** Evaluate HTR systems not only on generic metrics like CER/WER but also on task-specific metrics (e.g., precision/recall for key information extraction from forms) relevant to the intended application.  
   * **Awareness of Limitations:** Maintain an awareness of the current limitations of HTR technology, including the potential for errors (especially with MLLMs), the impact of data quality, and the computational costs, before deploying systems in production, particularly in high-stakes environments.  
   * **Ethical Considerations:** Address ethical implications related to data privacy (especially with personal documents or medical records), potential biases in training data and models, and the impact of automation on professions involved in transcription and archival work.

Concluding Remarks on the Trajectory of Cursive OCR/HTR Technology:  
The field of cursive OCR/HTR is exceptionally dynamic and continues its rapid evolution. AI-driven methods, from sophisticated deep neural networks to the emerging capabilities of MLLMs, are consistently pushing the boundaries of what is achievable. The overarching goal remains the accurate, efficient, and scalable conversion of the world's vast handwritten heritage and contemporary handwritten data into accessible, searchable, and usable digital formats. While challenges related to script complexity, data scarcity, and model reliability persist, the ongoing innovations in algorithms, the increasing availability of data, and the collaborative efforts within the research community suggest a future where even the most challenging cursive scripts will become increasingly amenable to automated understanding. This trajectory promises to unlock unprecedented access to human knowledge and streamline countless information-dependent workflows across diverse sectors.

#### **Works cited**

> 1. Handwritten Text Recognition: A Survey \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2502.08417v1](https://arxiv.org/html/2502.08417v1)  
> 2. Handwritten Text Recognition: A Survey \- arXiv, accessed May 15, 2025, [https://arxiv.org/pdf/2502.08417?](https://arxiv.org/pdf/2502.08417)  
> 3. Low-Resource Language OCR: New Possibilities with AI \- Sunway College Kathmandu, accessed May 15, 2025, [https://sunway.edu.np/low-resource-language-ocr-automation/](https://sunway.edu.np/low-resource-language-ocr-automation/)  
> 4. Handwriting Recognition in Historical Documents with Multimodal LLM \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2410.24034v1](https://arxiv.org/html/2410.24034v1)  
> 5. ADVANCING HANDWRITTEN TEXT RECOGNITION: METHODS, INNOVATIONS, AND CHALLENGES, accessed May 15, 2025, [https://www.irjmets.com/uploadedfiles/paper//issue\_1\_january\_2025/66642/final/fin\_irjmets1737738491.pdf](https://www.irjmets.com/uploadedfiles/paper//issue_1_january_2025/66642/final/fin_irjmets1737738491.pdf)  
> 6. \[Literature Review\] Handwriting Recognition in Historical Documents with Multimodal LLM, accessed May 15, 2025, [https://www.themoonlight.io/review/handwriting-recognition-in-historical-documents-with-multimodal-llm](https://www.themoonlight.io/review/handwriting-recognition-in-historical-documents-with-multimodal-llm)  
> 7. State-of-the-Art Review of Handwritten Digit Recognition \- ER Publications, accessed May 15, 2025, [https://www.erpublications.com/uploaded\_files/download/t-kavitha-karthikeyan-alamuthu-dr-t-sathis-kumar\_MhtSU.pdf](https://www.erpublications.com/uploaded_files/download/t-kavitha-karthikeyan-alamuthu-dr-t-sathis-kumar_MhtSU.pdf)  
> 8. Handwriting Recognition: Benefits, Challenges, and the Role of AI, accessed May 15, 2025, [https://instabase.com/blog/handwriting-recognition-ai/](https://instabase.com/blog/handwriting-recognition-ai/)  
> 9. Leveraging Deep Learning with Multi-Head Attention for Accurate Extraction of Medicine from Handwritten Prescriptions \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2412.18199v1](https://arxiv.org/html/2412.18199v1)  
> 10. Testing Handwriting OCR for Genealogy: Transcribing Death Records with AI \- Know Who Wears the Genes in Your Family, accessed May 15, 2025, [https://www.knowwhowearsthegenesinyourfamily.com/blog/testing-handwriting-ocr-for-genealogy-transcribing-death-records-with-ai](https://www.knowwhowearsthegenesinyourfamily.com/blog/testing-handwriting-ocr-for-genealogy-transcribing-death-records-with-ai)  
> 11. Handwriting OCR: Is cursive OCR compatible? | Adobe Acrobat, accessed May 15, 2025, [https://www.adobe.com/acrobat/hub/can-ocr-recognize-handwriting.html](https://www.adobe.com/acrobat/hub/can-ocr-recognize-handwriting.html)  
> 12. Handwritten Text Recognition of the Dunhuang manuscripts: the ..., accessed May 15, 2025, [https://blogs.bl.uk/digital-scholarship/2024/03/handwritten-text-recognition-of-the-dunhuang-manuscripts.html](https://blogs.bl.uk/digital-scholarship/2024/03/handwritten-text-recognition-of-the-dunhuang-manuscripts.html)  
> 13. Bridging the Gap: OCR Techniques for Noisy and Distorted Texts, accessed May 15, 2025, [https://ijsrcseit.com/index.php/home/article/view/CSEIT2511111](https://ijsrcseit.com/index.php/home/article/view/CSEIT2511111)  
> 14. ARABIC OCR SYSTEM ANALOGOUS TO HMM-BASED ASR SYSTEMS \- CiteSeerX, accessed May 15, 2025, [https://citeseerx.ist.psu.edu/document?repid=rep1\&type=pdf\&doi=d2e50d9beb9a895aa1a3ef8dbb994f56aa59924d](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=d2e50d9beb9a895aa1a3ef8dbb994f56aa59924d)  
> 15. (PDF) Data Augmentation for Offline Handwritten Text Recognition: A Systematic Literature Review \- ResearchGate, accessed May 15, 2025, [https://www.researchgate.net/publication/378124201\_Data\_Augmentation\_for\_Offline\_Handwritten\_Text\_Recognition\_A\_Systematic\_Literature\_Review](https://www.researchgate.net/publication/378124201_Data_Augmentation_for_Offline_Handwritten_Text_Recognition_A_Systematic_Literature_Review)  
> 16. Analysis of Recent Deep Learning Techniques for Arabic Handwritten-Text OCR and Post-OCR Correction \- ResearchGate, accessed May 15, 2025, [https://www.researchgate.net/publication/376555585\_Analysis\_of\_Recent\_Deep\_Learning\_Techniques\_for\_Arabic\_Handwritten-Text\_OCR\_and\_Post-OCR\_Correction](https://www.researchgate.net/publication/376555585_Analysis_of_Recent_Deep_Learning_Techniques_for_Arabic_Handwritten-Text_OCR_and_Post-OCR_Correction)  
> 17. Arabic Handwritten Document OCR Solution with Binarization and Adaptive Scale Fusion Detection \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2412.01601v1](https://arxiv.org/html/2412.01601v1)  
> 18. Attention-based deep learning model for Arabic handwritten text ..., accessed May 15, 2025, [https://mgv.sggw.edu.pl/article/view/2242](https://mgv.sggw.edu.pl/article/view/2242)  
> 19. Arabic Handwritten Document OCR Solution with Binarization and Adaptive Scale Fusion Detection \- ResearchGate, accessed May 15, 2025, [https://www.researchgate.net/publication/386375585\_Arabic\_Handwritten\_Document\_OCR\_Solution\_with\_Binarization\_and\_Adaptive\_Scale\_Fusion\_Detection](https://www.researchgate.net/publication/386375585_Arabic_Handwritten_Document_OCR_Solution_with_Binarization_and_Adaptive_Scale_Fusion_Detection)  
> 20. Liberated Arabic Handwritten Text Recognition using Convolutional Recurrent Neural Networks \- Iraqi Journal for Computer Science and Mathematics, accessed May 15, 2025, [https://ijcsm.researchcommons.org/cgi/viewcontent.cgi?article=1246\&context=ijcsm](https://ijcsm.researchcommons.org/cgi/viewcontent.cgi?article=1246&context=ijcsm)  
> 21. Data Augmentation for Offline Arabic Handwritten Text Recognition Using Moving Least Squares | IIETA, accessed May 15, 2025, [https://www.iieta.org/journals/ria/paper/10.18280/ria.380101](https://www.iieta.org/journals/ria/paper/10.18280/ria.380101)  
> 22. (PDF) A Survey on Arabic Handwritten Character Recognition, accessed May 15, 2025, [https://www.researchgate.net/publication/341195670\_A\_Survey\_on\_Arabic\_Handwritten\_Character\_Recognition](https://www.researchgate.net/publication/341195670_A_Survey_on_Arabic_Handwritten_Character_Recognition)  
> 23. accessed January 1, 1970, [https://arxiv.org/pdf/2502.08417](https://arxiv.org/pdf/2502.08417)  
> 24. accessed January 1, 1970, [https://arxiv.org/pdf/2412.01601](https://arxiv.org/pdf/2412.01601)  
> 25. www.psvpec.in, accessed May 15, 2025, [https://www.psvpec.in/jcres/2023\_2/17.pdf](https://www.psvpec.in/jcres/2023_2/17.pdf)  
> 26. Enhancing Arabic handwritten word recognition: a CNN-BiLSTM-CTC architecture with attention mechanism and adaptive augmentation | Scilit, accessed May 15, 2025, [https://www.scilit.com/publications/c1cf5dd9fd9f6c419dffeb08a4a6df03](https://www.scilit.com/publications/c1cf5dd9fd9f6c419dffeb08a4a6df03)  
> 27. Invizo: Arabic Handwritten Document Optical Character Recognition Solution \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2502.05277v1](https://arxiv.org/html/2502.05277v1)  
> 28. Challenges & Solutions in Developing Arabic OCR Technology, accessed May 15, 2025, [https://accurascan.com/blog/challenges-and-solutions-in-developing-accurate-arabic-ocr-technology/](https://accurascan.com/blog/challenges-and-solutions-in-developing-accurate-arabic-ocr-technology/)  
> 29. \[Papierüberprüfung\] Handwriting Recognition in Historical Documents with Multimodal LLM, accessed May 15, 2025, [https://www.themoonlight.io/de/review/handwriting-recognition-in-historical-documents-with-multimodal-llm](https://www.themoonlight.io/de/review/handwriting-recognition-in-historical-documents-with-multimodal-llm)  
> 30. Leveraging Deep Learning with Multi-Head Attention for Accurate Extraction of Medicine from Handwritten Prescriptions \- arXiv, accessed May 15, 2025, [https://arxiv.org/pdf/2412.18199](https://arxiv.org/pdf/2412.18199)  
> 31. Spanish TrOCR: Leveraging Transfer Learning for Language Adaptation \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2407.06950v1](https://arxiv.org/html/2407.06950v1)  
> 32. Early evidence of how LLMs outperform traditional systems on OCR/HTR tasks for historical records \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2501.11623v1](https://arxiv.org/html/2501.11623v1)  
> 33. Handwriting Recognition in Historical Documents with Multimodal LLM | PromptLayer, accessed May 15, 2025, [https://www.promptlayer.com/research-papers/unlocking-history-how-ai-is-deciphering-handwritten-texts](https://www.promptlayer.com/research-papers/unlocking-history-how-ai-is-deciphering-handwritten-texts)  
> 34. (PDF) PLATTER: A Page-Level Handwritten Text Recognition ..., accessed May 15, 2025, [https://www.researchgate.net/publication/388883429\_PLATTER\_A\_Page-Level\_Handwritten\_Text\_Recognition\_System\_for\_Indic\_Scripts](https://www.researchgate.net/publication/388883429_PLATTER_A_Page-Level_Handwritten_Text_Recognition_System_for_Indic_Scripts)  
> 35. Bridging the Gap: Fusing CNNs and Transformers to Decode the Elegance of Handwritten Arabic Script \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2503.15023v1](https://arxiv.org/html/2503.15023v1)  
> 36. \[2503.15023\] Bridging the Gap: Fusing CNNs and Transformers to Decode the Elegance of Handwritten Arabic Script \- arXiv, accessed May 15, 2025, [https://arxiv.org/abs/2503.15023](https://arxiv.org/abs/2503.15023)  
> 37. accessed January 1, 1970, [https://arxiv.org/pdf/2410.24034](https://arxiv.org/pdf/2410.24034)  
> 38. accessed January 1, 1970, [https://arxiv.org/pdf/2502.05277](https://arxiv.org/pdf/2502.05277)  
> 39. How Can LayoutLM Transform Text Extraction? \- Nitor Infotech, accessed May 15, 2025, [https://www.nitorinfotech.com/blog/how-can-layoutlm-transform-text-extraction/](https://www.nitorinfotech.com/blog/how-can-layoutlm-transform-text-extraction/)  
> 40. \[2410.24034\] Handwriting Recognition in Historical Documents with Multimodal LLM \- arXiv, accessed May 15, 2025, [https://arxiv.org/abs/2410.24034](https://arxiv.org/abs/2410.24034)  
> 41. Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2504.00414](https://arxiv.org/html/2504.00414)  
> 42. Judge a Book by Its Cover: Investigating Multi-Modal LLMs for Multi-Page Handwritten Document Transcription \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2502.20295v1](https://arxiv.org/html/2502.20295v1)  
> 43. What are the future trends in neural network research? \- Milvus, accessed May 15, 2025, [https://milvus.io/ai-quick-reference/what-are-the-future-trends-in-neural-network-research](https://milvus.io/ai-quick-reference/what-are-the-future-trends-in-neural-network-research)  
> 44. Daily Papers \- Hugging Face, accessed May 15, 2025, [https://huggingface.co/papers?q=General-Level](https://huggingface.co/papers?q=General-Level)  
> 45. The Future of OCR | Deep Analysis, accessed May 15, 2025, [https://www.deep-analysis.net/the-future-of-ocr/](https://www.deep-analysis.net/the-future-of-ocr/)  
> 46. Early evidence of how LLMs outperform traditional systems on OCR/HTR tasks for historical records | Papers With Code, accessed May 15, 2025, [https://paperswithcode.com/paper/early-evidence-of-how-llms-outperform](https://paperswithcode.com/paper/early-evidence-of-how-llms-outperform)  
> 47. Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2504.00414v1](https://arxiv.org/html/2504.00414v1)  
> 48. Minimizing Hallucinations and Communication Costs: Adversarial Debate and Voting Mechanisms in LLM-Based Multi-Agents \- MDPI, accessed May 15, 2025, [https://www.mdpi.com/2076-3417/15/7/3676](https://www.mdpi.com/2076-3417/15/7/3676)  
> 49. Hallucination of Multimodal Large Language Models: A Survey \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2404.18930v2](https://arxiv.org/html/2404.18930v2)  
> 50. Muharaf: Manuscripts of Handwritten Arabic Dataset for Cursive Text Recognition, accessed May 15, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/6b8cb6b291045e217c3ff3f854f2fd0f-Paper-Datasets\_and\_Benchmarks\_Track.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b8cb6b291045e217c3ff3f854f2fd0f-Paper-Datasets_and_Benchmarks_Track.pdf)  
> 51. Offline Arabic Handwriting Recognition: A Survey \- University at Buffalo, accessed May 15, 2025, [https://cubs.buffalo.edu/images/pdf/pub/offline-arabic-handwriting-recognition.pdf](https://cubs.buffalo.edu/images/pdf/pub/offline-arabic-handwriting-recognition.pdf)  
> 52. AN OCR FOR ARABIC CHARACTERS WITH ADVANCED BASELINE SEGMENTATION AND ENHANCED CONVOLUTIONAL NEURAL NETWORK FOR CLASSIFICATION, accessed May 15, 2025, [https://www.jatit.org/volumes/Vol101No17/25Vol101No17.pdf](https://www.jatit.org/volumes/Vol101No17/25Vol101No17.pdf)  
> 53. Invizo: Arabic Handwritten Document Optical Character Recognition Solution \- arXiv, accessed May 15, 2025, [https://arxiv.org/abs/2502.05277](https://arxiv.org/abs/2502.05277)  
> 54. Arabic Handwritten Document OCR Solution with Binarization and Adaptive Scale Fusion Detection | Request PDF \- ResearchGate, accessed May 15, 2025, [https://www.researchgate.net/publication/386058637\_Arabic\_Handwritten\_Document\_OCR\_Solution\_with\_Binarization\_and\_Adaptive\_Scale\_Fusion\_Detection](https://www.researchgate.net/publication/386058637_Arabic_Handwritten_Document_OCR_Solution_with_Binarization_and_Adaptive_Scale_Fusion_Detection)  
> 55. DBNet and DBNet++, accessed May 15, 2025, [https://www.mindspore.cn/resources/hub/details?MindSpore/2.5/dbnet\_mobilenetv3\_icdar2015](https://www.mindspore.cn/resources/hub/details?MindSpore/2.5/dbnet_mobilenetv3_icdar2015)  
> 56. mindocr/configs/det/dbnet/README.md at main \- GitHub, accessed May 15, 2025, [https://github.com/mindspore-lab/mindocr/blob/main/configs/det/dbnet/README.md](https://github.com/mindspore-lab/mindocr/blob/main/configs/det/dbnet/README.md)  
> 57. Arabic Handwritten Segmentation Model Trained on the Muharaf Corpus \- Zenodo, accessed May 15, 2025, [https://zenodo.org/records/14295555](https://zenodo.org/records/14295555)  
> 58. Muharaf: Manuscripts of Handwritten Arabic Dataset for Cursive Text Recognition \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2406.09630v2](https://arxiv.org/html/2406.09630v2)  
> 59. Daily Papers \- Hugging Face, accessed May 15, 2025, [https://huggingface.co/papers?q=Arabic%20dataset](https://huggingface.co/papers?q=Arabic+dataset)  
> 60. A JSON file format used in the Muharaf dataset, accessed May 15, 2025, [https://proceedings.neurips.cc/paper\_files/paper/2024/file/6b8cb6b291045e217c3ff3f854f2fd0f-Supplemental-Datasets\_and\_Benchmarks\_Track.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/6b8cb6b291045e217c3ff3f854f2fd0f-Supplemental-Datasets_and_Benchmarks_Track.pdf)  
> 61. Ruqʿah script \- Wikipedia, accessed May 15, 2025, [https://en.wikipedia.org/wiki/Ruq%CA%BFah\_script](https://en.wikipedia.org/wiki/Ruq%CA%BFah_script)  
> 62. Muharaf: Manuscripts of Handwritten Arabic Dataset for Cursive Text Recognition, accessed May 15, 2025, [https://huggingface.co/papers/2406.09630](https://huggingface.co/papers/2406.09630)  
> 63. accessed January 1, 1970, [https://arxiv.org/pdf/2406.09630](https://arxiv.org/pdf/2406.09630)  
> 64. Historical Arabic Handwritten Text Recognition Dataset \- Mendeley Data, accessed May 15, 2025, [https://data.mendeley.com/datasets/xz6f8bw3w8/1](https://data.mendeley.com/datasets/xz6f8bw3w8/1)  
> 65. KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding, accessed May 15, 2025, [https://mbzuai-oryx.github.io/KITAB-Bench/](https://mbzuai-oryx.github.io/KITAB-Bench/)  
> 66. Coloso Español \- Public Transkribus AI Model \- READ-COOP SCE, accessed May 15, 2025, [https://readcoop.eu/model/coloso-espanol/](https://readcoop.eu/model/coloso-espanol/)  
> 67. Requesting and Using Research Datasets – HathiTrust Digital Library, accessed May 15, 2025, [https://www.hathitrust.org/member-libraries/resources-for-librarians/data-resources/research-datasets/](https://www.hathitrust.org/member-libraries/resources-for-librarians/data-resources/research-datasets/)  
> 68. HTRC BookNLP Dataset for English-Language Fiction \- Documentation \- HathiTrust Research Center \- Atlassian, accessed May 15, 2025, [https://htrc.atlassian.net/wiki/spaces/COM/pages/43300179/HTRC+BookNLP+Dataset+for+English-Language+Fiction](https://htrc.atlassian.net/wiki/spaces/COM/pages/43300179/HTRC+BookNLP+Dataset+for+English-Language+Fiction)  
> 69. cvit.iiit.ac.in, accessed May 15, 2025, [https://cvit.iiit.ac.in/images/ConferencePapers/2023/icdar2033\_tr.pdf](https://cvit.iiit.ac.in/images/ConferencePapers/2023/icdar2033_tr.pdf)  
> 70. AIO-HB: A Handwritten Text Image Dataset of Hindi and Bengali ..., accessed May 15, 2025, [https://colab.ws/articles/10.1007%2F978-3-031-78495-8\_20](https://colab.ws/articles/10.1007%2F978-3-031-78495-8_20)  
> 71. PLATTER: A Page-Level Handwritten Text Recognition System for Indic Scripts \- arXiv, accessed May 15, 2025, [https://arxiv.org/html/2502.06172v1](https://arxiv.org/html/2502.06172v1)  
> 72. arXiv:2410.03787v1 \[cs.CL\] 3 Oct 2024, accessed May 15, 2025, [https://arxiv.org/pdf/2410.03787?](https://arxiv.org/pdf/2410.03787)  
> 73. CASIA Online and Offline Chinese Handwriting Databases, accessed May 15, 2025, [https://nlpr.ia.ac.cn/databases/download/ICDAR2011-CASIA%20databases.pdf](https://nlpr.ia.ac.cn/databases/download/ICDAR2011-CASIA%20databases.pdf)  
> 74. The Challenges of Recognizing Offline Handwritten Chinese: A Technical Review \- MDPI, accessed May 15, 2025, [https://www.mdpi.com/2076-3417/13/6/3500](https://www.mdpi.com/2076-3417/13/6/3500)  
> 75. Character-level Chinese Writer Identification using Path Signature Feature, DropStroke and Deep CNN \- Bohrium, accessed May 15, 2025, [https://bohrium.dp.tech/paper/arxiv/1505.04922](https://bohrium.dp.tech/paper/arxiv/1505.04922)  
> 76. \[1606.05763\] Online and Offline Handwritten Chinese Character Recognition: A Comprehensive Study and New Benchmark \- arXiv, accessed May 15, 2025, [https://arxiv.org/abs/1606.05763](https://arxiv.org/abs/1606.05763)  
> 77. \[2007.09637\] Survey on Deep Learning-based Kuzushiji Recognition \- arXiv, accessed May 15, 2025, [https://arxiv.org/abs/2007.09637](https://arxiv.org/abs/2007.09637)  
> 78. KuroNet: Pre-Modern Japanese Kuzushiji Character Recognition with Deep Learning \- arXiv, accessed May 15, 2025, [https://arxiv.org/abs/1910.09433](https://arxiv.org/abs/1910.09433)  
> 79. Enhancement of text recognition for hanja handwritten documents of Ancient Korea Abstract \- arXiv, accessed May 15, 2025, [https://arxiv.org/pdf/2412.10647](https://arxiv.org/pdf/2412.10647)  
> 80. Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks \- CS231n, accessed May 15, 2025, [https://cs231n.stanford.edu/reports/2016/pdfs/262\_Report.pdf](https://cs231n.stanford.edu/reports/2016/pdfs/262_Report.pdf)  
> 81. CASIA Online and Offline Chinese Handwriting Databases, accessed May 15, 2025, [https://nlpr.ia.ac.cn/databases/handwriting/home.html](https://nlpr.ia.ac.cn/databases/handwriting/home.html)  
> 82. 9 Biggest OCR Limitations And How To Overcome Them \- DocuClipper, accessed May 15, 2025, [https://www.docuclipper.com/blog/ocr-limitations/](https://www.docuclipper.com/blog/ocr-limitations/)  
> 83. Structured vs. Unstructured Data: A Complete Guide \- Talend, accessed May 15, 2025, [https://www.talend.com/resources/structured-vs-unstructured-data/](https://www.talend.com/resources/structured-vs-unstructured-data/)  
> 84. Structured vs unstructured data: What's the difference? \- SailPoint, accessed May 15, 2025, [https://www.sailpoint.com/identity-library/structured-unstructured-data](https://www.sailpoint.com/identity-library/structured-unstructured-data)  
> 85. Optical Character Recognition (OCR): What It Is & How It Works? \- Cflow, accessed May 15, 2025, [https://www.cflowapps.com/optical-character-recognition-ocr/](https://www.cflowapps.com/optical-character-recognition-ocr/)  
> 86. Handwriting OCR | Extract Text from Handwritten PDF From \- Unstract, accessed May 15, 2025, [https://unstract.com/blog/guide-to-extracting-data-from-handwritten-pdf-with-ocr/](https://unstract.com/blog/guide-to-extracting-data-from-handwritten-pdf-with-ocr/)  
> 87. A Deeper Dive on Deep Learning OCR \- Zebra Technologies, accessed May 15, 2025, [https://www.zebra.com/us/en/blog/posts/2024/deeper-dive-on-deep-learning-ocr.html](https://www.zebra.com/us/en/blog/posts/2024/deeper-dive-on-deep-learning-ocr.html)  
> 88. Open-Source OCR Libraries: A Comprehensive Study for Low Resource Language \- ACL Anthology, accessed May 15, 2025, [https://aclanthology.org/2024.icon-1.48.pdf](https://aclanthology.org/2024.icon-1.48.pdf)  
> 89. End-to-end subtitle detection and recognition for videos in East Asian languages via CNN ensemble \- Microsoft, accessed May 15, 2025, [https://www.microsoft.com/en-us/research/wp-content/uploads/2018/06/2018SCISPICEnd-to-end-subtitle-detection-and-recognition-for-videos-in-East-Asian-languages-via-CNN-ensemble.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2018/06/2018SCISPICEnd-to-end-subtitle-detection-and-recognition-for-videos-in-East-Asian-languages-via-CNN-ensemble.pdf)  
> 90. A survey of handwriting synthesis from 2019 to 2024: A comprehensive review, accessed May 15, 2025, [https://www.researchgate.net/publication/388254036\_A\_survey\_of\_handwriting\_synthesis\_from\_2019\_to\_2024\_A\_comprehensive\_review](https://www.researchgate.net/publication/388254036_A_survey_of_handwriting_synthesis_from_2019_to_2024_A_comprehensive_review)  
> 91. Evaluating AI Models: Understanding the Character Error Rate (CER) Metric \- Galileo AI, accessed May 15, 2025, [https://www.galileo.ai/blog/character-error-rate-cer-metric](https://www.galileo.ai/blog/character-error-rate-cer-metric)  
> 92. Assessing Transformers and Traditional Models for Spanish-English Code- Switched Hate Detection \- Preprints.org, accessed May 15, 2025, [https://www.preprints.org/manuscript/202504.0052/v1/download](https://www.preprints.org/manuscript/202504.0052/v1/download)  
> 93. TrOCR Transformer-based OCR for Handwritten Text using Python \- YouTube, accessed May 15, 2025, [https://www.youtube.com/watch?v=eoqH3WPMg0E](https://www.youtube.com/watch?v=eoqH3WPMg0E)

# **One Page Summary**

Okay, here is a one-page summary of the state-of-the-art in cursive OCR technology, focusing on Arabic with examples from Spanish, English, and other non-western languages, and considering the specified dimensions.

**State-of-the-Art Cursive OCR: A One-Page Summary (as of May 2025\)**

Cursive Optical Character Recognition (OCR), or Handwritten Text Recognition (HTR), has seen significant advancements, moving decisively from traditional heuristic methods to sophisticated Artificial Intelligence (AI) and Machine Learning (ML) approaches. Deep Learning (DL) models are now the standard, with Multimodal Large Language Models (MLLMs) representing the cutting edge.

**1\. Dominant Technologies & Methodologies (ML/AI/LLMs vs. Heuristic):** The field is dominated by DL architectures. Early DL successes involved Convolutional Neural Networks (CNNs) for feature extraction, combined with Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks and the Connectionist Temporal Classification (CTC) loss function, for sequence recognition. More recently, **Transformer-based models** (e.g., TrOCR, Vision Transformers) have shown superior performance due to their ability to capture long-range dependencies. The latest advancements involve **MLLMs** (e.g., GPT-4o, Gemini), which can perform direct HTR, offer powerful post-correction capabilities, and better understand context, significantly improving accuracy, especially for historical and complex documents. Traditional methods (template matching, feature engineering with SVMs/HMMs) are largely outdated for cursive script due to their inability to handle high variability.

**2\. Language-Specific State-of-the-Art:**

* **Arabic:**

  * **Challenges:** Highly cursive script, numerous ligatures, variable diacritics (Harakat) and dots (Ijam), context-dependent character shapes (initial, medial, final, isolated), and text written right-to-left.  
  * **SOTA & Research:** Hybrid models (CNN-Transformer, CNN-BiLSTM-CTC with attention mechanisms) show strong results. Systems like "Invizo" (CNN-Transformer) report promising CER/WER for handwritten (e.g., \~7.9% CER) and printed Arabic. Techniques like Differentiable Binarization (DBNet++) for segmentation are used. Research focuses on robust feature extraction for subtle diacritics and ligatures, often using attention to focus on relevant character parts.  
  * **Applications:** Digitization of historical manuscripts, processing handwritten forms and receipts, improving accessibility.  
* **Spanish:**

  * **SOTA & Research:** Models like "Coloso Español" (Transkribus, PyLaia-based) achieve good accuracy (e.g., 4.8% CER) on historical Spanish documents. Adaptation of Transformer models like TrOCR for Spanish, often fine-tuned on datasets like XFUND, shows strong performance (e.g., CER around 6% on XFUND).  
  * **Applications:** Transcription of medieval to modern manuscripts, archival research.  
* **English:**

  * **SOTA & Research:** MLLMs are making a significant impact, especially for historical documents, achieving very low CERs (\<1% with post-correction). TrOCR and similar models are effective, particularly when fine-tuned for specific domains like medical prescriptions (\~1.4% CER).  
  * **Applications:** Genealogical research, digitization of historical archives, processing handwritten notes and forms.  
* **Other Non-Western Languages:**

  * **Indic Scripts (Urdu, Devanagari, Bengali, etc.):** Significant challenges due to numerous characters, conjunct characters (ligatures), and diverse scripts. Systems like "PLATTER" (page-level HTR using DBNet and CRNN/Transformer) are emerging. The ICDAR 2023 Indic HTR competition saw CRNN and TrOCR-based models achieving high accuracy (winning CRR \~96%).  
  * **East Asian Scripts (Chinese, Japanese, Korean):**  
    * *Japanese Kuzushiji (cursive):* Models like "KuroNet" (U-Net based) and other DL approaches are used. Challenges include a vast character set and stroke variations.  
    * *Chinese Cursive (Cǎoshū, Xíngshū):* While large datasets like CASIA exist for general Chinese handwriting, specific SOTA for highly cursive styles is an active research area, often building on general HCCR advancements. Challenges include character complexity and artistic variability.  
    * *Korean Cursive:* Research is less extensive, with some work on Hanja (Chinese characters in Korean) cursive documents using models like HRNet.

**3\. Structured and Non-Structured Data Recognition:** Modern OCR systems, especially those leveraging layout analysis and MLLMs, are increasingly capable of handling both:

* **Unstructured Data:** Manuscripts, letters, notes (primary focus of much HTR).  
* **Structured Data:** Forms, tables, receipts. For Arabic, Invizo handles receipts. For English, MLLMs can parse entities from historical documents into structured formats, and specialized models like LayoutLM (though with limitations for handwriting) aim to understand document layouts. Spanish TrOCR has been tested on form datasets (XFUND). Extracting key-value pairs from cursive handwritten forms is an advancing area, often relying on joint text and layout understanding.

**4\. Availability of Training Data:** This remains a critical factor.

* **Arabic:** Datasets like Muharaf (historical), IFN/ENIT, KHATT, and AMFDS exist.  
* **Spanish:** XFUND (forms), and community-driven datasets (e.g., for Coloso Español).  
* **English:** IAM (general), HathiTrust (historical), and domain-specific custom datasets are common.  
* **Indic:** Growing efforts with datasets like CHIPS, IIIT-Indic-HW-Words, and AIO-HB, but data scarcity is still a major challenge.  
* **East Asian:** CASIA (large Chinese resource), Kuzushiji datasets for Japanese.  
* **Data Augmentation & Synthesis:** Crucial for all, especially low-resource languages. Techniques include traditional image manipulations (rotation, noise, elastic distortions), Moving Least Squares (MLS) for Arabic, and generative models like GANs (e.g., ScrabbleGAN for Indic) to create synthetic handwritten text. Few-shot learning and self-supervised learning are explored for low-resource scenarios.

**5\. Precision and Recall (Performance):** Metrics like Character Error Rate (CER), Word Error Rate (WER), and Accuracy are standard.

* **General Trend:** Significant improvements with DL models. Transformer-based models often outperform RNN-based ones. MLLMs show potential for SOTA, especially in English and for post-correction.  
* **Specifics:** Arabic handwritten CERs can range from \~7-10% for robust systems to lower for specific datasets/tasks. Spanish models show CERs around 5-6%. English historical documents can achieve \<1-5% CER with MLLMs/TrOCR. Indic competition winners achieve high CRR/WRR (\~96% CRR). Performance varies greatly based on script complexity, dataset quality, and model architecture. Precision and recall are not always reported directly, with CER/WER being more common for HTR tasks.

**Overall Challenges & Future Directions:** Key challenges persist: extreme handwriting variability, degradation of historical documents, data scarcity for many languages/scripts, and handling complex layouts and nuanced script features (e.g., overlapping characters, subtle diacritics). Future directions include more robust and generalizable MLLMs, advancements in unsupervised and self-supervised learning to reduce reliance on labeled data, improved synthetic data generation, better handling of document layout and structure within HTR, and specific architectural innovations for the unique complexities of non-Latin cursive scripts.

# **libraries and current solutions**

[https://huggingface.co/Nada2125/paligemma-3b-arabic-handwritten-ocr-1/tree/main](https://huggingface.co/Nada2125/paligemma-3b-arabic-handwritten-ocr-1/tree/main)  
[https://huggingface.co/youssef227/paligemma-3b-arabic-handwritten-ocr-1](https://huggingface.co/youssef227/paligemma-3b-arabic-handwritten-ocr-1)

[https://huggingface.co/cycloneboy/arabic\_PP-OCRv4\_rec\_infer](https://huggingface.co/cycloneboy/arabic_PP-OCRv4_rec_infer)

[https://huggingface.co/papers/2407.13559?utm\_source=chatgpt.com](https://huggingface.co/papers/2407.13559?utm_source=chatgpt.com)

[https://www.linkedin.com/in/asmaa-saeed-b5582b173/?originalSubdomain=eg](https://www.linkedin.com/in/asmaa-saeed-b5582b173/?originalSubdomain=eg)

[https://huggingface.co/AsmaaMahmoudSaeddd/Florence-2-Arabic-OCR](https://huggingface.co/AsmaaMahmoudSaeddd/Florence-2-Arabic-OCR)  
