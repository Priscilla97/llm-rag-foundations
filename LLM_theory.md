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
- carica il tokenizer corretto
- carica il modello adatto al task
- esegue:

```
testo → token → modello → output leggibile
```
