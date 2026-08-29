from faiss_store import search
from model import embedding_model

evaluation_data = [
  {
    "id": 1,
    "question": "What is Generative AI?",
    "expected_pages": [
      1
    ],
    "answer": "Generative AI refers to AI models that learn patterns and structure from existing data and generate new content such as text, images, music, code, or video.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 2,
    "question": "What major breakthrough in generative AI occurred in 2014?",
    "expected_pages": [
      2
    ],
    "answer": "Ian Goodfellow introduced Generative Adversarial Networks (GANs), which use a generator and discriminator in a competitive setup to create realistic synthetic data.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 3,
    "question": "What is the main purpose of generative modelling compared with discriminative modelling?",
    "expected_pages": [
      2
    ],
    "answer": "Generative modelling learns the data distribution and can generate new samples, whereas discriminative modelling focuses on predicting or classifying labels from inputs.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 4,
    "question": "What are the two competing networks in a GAN?",
    "expected_pages": [
      4
    ],
    "answer": "A GAN contains a Generator, which creates synthetic samples, and a Discriminator, which distinguishes real samples from generated ones.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 5,
    "question": "What is the role of the encoder in a VAE?",
    "expected_pages": [
      5,
      6
    ],
    "answer": "The encoder processes the input, extracts important features, and compresses the data into a lower-dimensional latent representation.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 6,
    "question": "What is an autoregressive generative model?",
    "expected_pages": [
      6,
      7
    ],
    "answer": "It generates a sequence step by step, predicting the next element using the elements generated so far.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 7,
    "question": "What is the primary objective of a language model?",
    "expected_pages": [
      15,
      16
    ],
    "answer": "A language model estimates the probability of word sequences and primarily predicts the next word or token from the preceding context.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 8,
    "question": "What is tokenization in a language model?",
    "expected_pages": [
      16,
      17
    ],
    "answer": "Tokenization breaks text into smaller units called tokens, which may be words, subwords, or characters, so the model can process the text numerically.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 9,
    "question": "What is GPT designed to do?",
    "expected_pages": [
      20
    ],
    "answer": "GPT is a large language model designed to understand and produce human-like text and perform tasks such as writing, summarization, question answering, and coding.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 10,
    "question": "What does hallucination mean in an LLM?",
    "expected_pages": [
      25
    ],
    "answer": "Hallucination is when an LLM produces factually incorrect, misleading, or fabricated information even though the response may appear fluent and confident.",
    "difficulty": "easy",
    "type": "direct"
  },
  {
    "id": 11,
    "question": "Why can a RAG system use newly added domain knowledge without retraining the entire LLM?",
    "expected_pages": [
      23,
      24
    ],
    "answer": "RAG retrieves information from an external knowledge source, so updating that knowledge base can provide new information without retraining the large language model.",
    "difficulty": "medium",
    "type": "semantic"
  },
  {
    "id": 12,
    "question": "If a user's query needs information that is not present in the model's original training data, which RAG capability addresses this problem?",
    "expected_pages": [
      23,
      24
    ],
    "answer": "RAG provides access to updated knowledge by retrieving relevant information from external sources before generating the answer.",
    "difficulty": "medium",
    "type": "semantic"
  },
  {
    "id": 13,
    "question": "Why are embeddings useful in a RAG pipeline?",
    "expected_pages": [
      24
    ],
    "answer": "Embeddings convert text into numerical vectors that capture semantic meaning, allowing the system to compare the user's query with stored document representations.",
    "difficulty": "medium",
    "type": "semantic"
  },
  {
    "id": 14,
    "question": "What is the difference between a vector database and a retriever in RAG?",
    "expected_pages": [
      24
    ],
    "answer": "The vector database stores embeddings and supports similarity search, while the retriever uses query similarity to find and return the most relevant chunks.",
    "difficulty": "medium",
    "type": "semantic"
  },
  {
    "id": 15,
    "question": "Why does masked attention matter when generating text?",
    "expected_pages": [
      18,
      19
    ],
    "answer": "Masked attention prevents the decoder from seeing future tokens, preserving left-to-right autoregressive generation.",
    "difficulty": "medium",
    "type": "semantic"
  },
  {
    "id": 16,
    "question": "Why does a Transformer need positional encoding even though it processes tokens in parallel?",
    "expected_pages": [
      17,
      18
    ],
    "answer": "Positional encoding supplies information about token order because the Transformer does not use recurrence and processes tokens in parallel.",
    "difficulty": "medium",
    "type": "semantic"
  },
  {
    "id": 17,
    "question": "What makes few-shot prompting different from giving the model only an instruction?",
    "expected_pages": [
      22
    ],
    "answer": "Few-shot prompting includes examples in the prompt so the model can infer the expected pattern or style, improving consistency and accuracy.",
    "difficulty": "medium",
    "type": "semantic"
  },
  {
    "id": 18,
    "question": "Why can RAG help reduce hallucinations but not guarantee that every answer is correct?",
    "expected_pages": [
      23,
      24,
      25
    ],
    "answer": "RAG supplies retrieved external context and is described as improving factual accuracy and reducing hallucinations, but the source does not state that it guarantees correctness.",
    "difficulty": "medium",
    "type": "semantic"
  },
  {
    "id": 19,
    "question": "A system receives a question, converts both the question and documents into vectors, finds similar document chunks, and then gives those chunks to an LLM. Which RAG components are involved, in order?",
    "expected_pages": [
      24
    ],
    "answer": "The query encoder creates a query vector, the vector database stores and searches embeddings, the retriever returns relevant chunks, and the prompt augmentation layer combines the retrieved chunks with the query before the LLM generates the response.",
    "difficulty": "hard",
    "type": "multi-hop"
  },
  {
    "id": 20,
    "question": "A company changes its internal policies every month. Why would the document-updating part of a RAG system be preferable to repeatedly retraining the LLM according to the material?",
    "expected_pages": [
      23,
      24
    ],
    "answer": "The material says RAG can keep knowledge current by updating the external knowledge base and optionally refreshing and re-embedding data, avoiding the time and resources required to retrain a massive LLM.",
    "difficulty": "hard",
    "type": "multi-hop"
  },
  {
    "id": 21,
    "question": "A user asks a question about a private document collection. Explain how RAG can provide a personalized answer without making the LLM itself store the entire collection in its parameters.",
    "expected_pages": [
      23,
      24
    ],
    "answer": "RAG can retrieve user-specific information from an external knowledge source, combine the retrieved chunks with the user's query, and have the LLM generate a response grounded in that retrieved context.",
    "difficulty": "hard",
    "type": "multi-hop"
  },
  {
    "id": 22,
    "question": "How do the encoder and decoder contribute different functions to a Transformer-based sequence-to-sequence system?",
    "expected_pages": [
      18,
      19
    ],
    "answer": "The encoder converts the input into contextual representations using self-attention and feed-forward layers, while the decoder generates the output sequence using masked self-attention and, in encoder-decoder systems, cross-attention to the encoder output.",
    "difficulty": "hard",
    "type": "multi-hop"
  },
  {
    "id": 23,
    "question": "A prompt is vague and the resulting answer is generic. Based on the document's prompt-engineering guidance, what changes could make the output more focused?",
    "expected_pages": [
      22
    ],
    "answer": "Make the prompt clear and specific, provide context, give explicit instructions, specify the desired output format, and add useful constraints such as scope, tone, or word limits.",
    "difficulty": "hard",
    "type": "multi-hop"
  },
  {
    "id": 24,
    "question": "How does GPT differ from BERT according to the document?",
    "expected_pages": [
      15,
      20,
      21
    ],
    "answer": "GPT is presented as a model for text generation, while BERT is presented primarily for language understanding.",
    "difficulty": "medium",
    "type": "comparison"
  },
  {
    "id": 25,
    "question": "How do GANs and VAEs differ in the way they generate data?",
    "expected_pages": [
      4,
      5,
      6
    ],
    "answer": "GANs use adversarial competition between a generator and discriminator, whereas VAEs learn a probabilistic latent space through an encoder-decoder structure and generate samples by sampling from that learned space.",
    "difficulty": "medium",
    "type": "comparison"
  },
  {
    "id": 26,
    "question": "How does RAG differ from fine-tuning as described in the material?",
    "expected_pages": [
      20,
      23,
      24
    ],
    "answer": "RAG supplies external knowledge at retrieval time, allowing the knowledge base to be updated without retraining the massive LLM, whereas fine-tuning adapts a pretrained model using task-specific data.",
    "difficulty": "hard",
    "type": "comparison"
  },
  {
    "id": 27,
    "question": "How are self-attention and cross-attention used differently in the Transformer description?",
    "expected_pages": [
      18,
      19
    ],
    "answer": "Self-attention lets tokens attend to other tokens within the same sequence, while cross-attention in the decoder lets generated output attend to the encoder's representations.",
    "difficulty": "hard",
    "type": "comparison"
  },
  {
    "id": 28,
    "question": "A GAN repeatedly produces only a small variety of outputs even though the training data contains many different examples. What problem described in the material does this indicate?",
    "expected_pages": [
      10
    ],
    "answer": "This indicates mode collapse, where the model generates only a few types of samples repeatedly and ignores the diversity of the training data.",
    "difficulty": "hard",
    "type": "reasoning"
  },
  {
    "id": 29,
    "question": "Why might an LLM give a confident answer containing a fabricated research paper?",
    "expected_pages": [
      25
    ],
    "answer": "The material identifies hallucination as a case where an LLM can generate fabricated information while appearing fluent and confident. Causes include noisy or incomplete training data, lack of real-time verification, over-generalization, and ambiguous prompts.",
    "difficulty": "hard",
    "type": "reasoning"
  },
  {
    "id": 30,
    "question": "In CLIP, what is optimized during training when an image is paired with its correct description?",
    "expected_pages": [
      31
    ],
    "answer": "CLIP uses separate image and text encoders, calculates cosine similarity between their embeddings, and trains to increase similarity for correct image-text pairs while decreasing similarity for incorrect pairs.",
    "difficulty": "hard",
    "type": "reasoning"
  }
]

def evaluate_question(index, chunks, question, expected_pages, k):
    question_embedding = embedding_model.encode(question)

    retrieved_chunks = search(
        index,
        question_embedding,
        chunks,
        k
    )

    for rank, result in enumerate(retrieved_chunks, start=1):
        page = result["chunk"]["page"]

        if page in expected_pages:
            return rank

    return None


def evaluate_all(index, chunks, evaluation_data, k):
    successful = 0

    for data in evaluation_data:
        question = data["question"]
        expected_pages = data["expected_pages"]

        rank = evaluate_question(
            index,
            chunks,
            question,
            expected_pages,
            k
        )

        if rank is not None:
            successful += 1

        print(f"Question: {question}")
        print(f"Expected pages: {expected_pages}")

        if rank is not None:
            print(f"First relevant result: Rank {rank}")
        else:
            print("First relevant result: NOT FOUND")

        print("-" * 60)

    return successful / len(evaluation_data) * 100