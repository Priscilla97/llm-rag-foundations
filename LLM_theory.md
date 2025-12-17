# NLP, Transformer, LLM, RAG
## Intro
**NLP (Natural Language Processing)**
- is the broader field focused on enabling computers to understand, interpret, and generate human language.  
- it is not a model  
- **tasks:**  
    - text classification
    - question answering
    - sentiment analysis  
    - named entity recognition  
    - machine translation
    - summarization  


**LLMs (Large Language Models)**
- it is a *neural model* characterized by their massive size, extensive training data  
- ability to perform a wide range of language tasks with minimal task-specific training.  
- Models like the Llama, GPT, or Claude series are examples of LLMs
- it models the distribution of the language, not the true
- it knows linguistic patterns
- transformers -> decoder only
```math
P(t_{n+1}|t_{1},...,t_{n})
```

Limitations LLMs:
- **Hallucinations:** They can generate incorrect information confidently  
- **Lack of true understanding:** They lack true understanding of the world and operate purely on statistical patterns  
- **Bias:** They may reproduce biases present in their training data or inputs  
- **Context windows:** They have limited context windows (though this is improving)  
- **Computational resources:** They require significant computational resources


# HuggingFace
Hugging Face è una **piattaforma open-source per il machine learning**, nata soprattutto per lavorare con modelli di linguaggio (NLP) e LLM, ma oggi estesa a vision, audio e multimodale.

Hugging Face fornisce:
- Modelli pre-addestrati
- Librerie software standard
- Un hub centralizzato per condividere modelli, dataset e tokenizer

### Componenti principali:
1) **Repository online** con:
    - LLM (BERT, LLaMA, Mistral, T5, ecc.)
    - tokenizer
    - embedding model
    - modelli fine-tuned  
2) **Libreria transformers:**
    - permette di caricare qualsiasi modello compatibile senza conoscere l’implementazione interna.
    - Supporta training, inference, fine-tuning
3) **Tokenizer standardizzati:** implementazioni ufficiali di BPE, WordPiece, SentencePiece
4) **Datasets**


# Tokenizer

# Libreria "pipeline()"
In Hugging Face Transformers, la **pipeline()** è un’API di alto livello che permette di usare un modello pre-addestrato per un task specifico con pochissimo codice, senza gestire direttamente tokenizer e modello.

It connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer.

```
pipeline = task pronto all’uso (tokenizer + modello + post-processing)
```

When you write:

```
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
```
*pipeline:*
- selects the right tokenizer
- selects a particular pretrained model that has been fine-tuned for the task (sebntiment analysis)
- execute:
    1) The text is **preprocessed** into a format the model can understand.
    2) The preprocessed inputs are passed to the **model.**
    3) The predictions of the model are **post-processed**, so you can make sense of them.

```
testo → token → modello → output leggibile
```

## Different pipeline modalities
The pipeline() function supports multiple modalities, allowing you to work with text, images, audio, and even multimodal tasks.

**Text pipelines**  
- *text-generation:* Generate text from a prompt
- *text-classification:* Classify text into predefined categories
- *summarization:* Create a shorter version of a text while preserving key information
- *translation:* Translate text from one language to another
- *zero-shot-classification:* Classify text without prior training on specific labels
- *feature-extraction:* Extract vector representations of text

**Image pipelines**
- *image-to-text:* Generate text descriptions of images
- *image-classification:* Identify objects in an image
- *object-detection:* Locate and identify objects in images

**Audio pipelines**
- *automatic-speech-recognition:* Convert speech to text
- *audio-classification:* Classify audio into categories
- *text-to-speech:* Convert text to spoken audio

**Multimodal pipelines**
- *image-text-to-text:* Respond to an image based on a text prompt