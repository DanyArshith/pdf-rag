from faiss_store import search
from model import embedding_model

evaluation_data = [
    {
        "question": "What is machine learning?",
        "expected_pages": [2]
    },
    {
        "question": "What are the three elements in the definition of learning according to Tom Mitchell?",
        "expected_pages": [2]
    },
    {
        "question": "What are the four basic components of the learning process?",
        "expected_pages": [2, 3]
    },
    {
        "question": "What is data storage in the learning process?",
        "expected_pages": [3]
    },
    {
        "question": "What is abstraction in machine learning?",
        "expected_pages": [3]
    },
    {
        "question": "What is generalization in machine learning?",
        "expected_pages": [3]
    },
    {
        "question": "What is evaluation in the learning process?",
        "expected_pages": [3]
    },
    {
        "question": "What are the applications of machine learning?",
        "expected_pages": [3, 4]
    },
    {
        "question": "What are the three categories of learning models?",
        "expected_pages": [4]
    },
    {
        "question": "What are logical models?",
        "expected_pages": [4, 5]
    },
    {
        "question": "What is concept learning?",
        "expected_pages": [4, 5]
    },
    {
        "question": "What are tree models and rule models?",
        "expected_pages": [4, 5]
    },
    {
        "question": "What are geometric models?",
        "expected_pages": [5, 6]
    },
    {
        "question": "What is the difference between linear models and distance-based models?",
        "expected_pages": [6]
    },
    {
        "question": "What are distance-based models?",
        "expected_pages": [6, 7]
    },
    {
        "question": "What is the difference between a centroid and a medoid?",
        "expected_pages": [7]
    },
    {
        "question": "What are probabilistic models?",
        "expected_pages": [7, 8]
    },
    {
        "question": "What is the difference between predictive and generative probabilistic models?",
        "expected_pages": [7, 8]
    },
    {
        "question": "Why is Naive Bayes considered a probabilistic classifier?",
        "expected_pages": [8]
    },
    {
        "question": "What is the difference between grouping and grading models?",
        "expected_pages": [8]
    },
    {
        "question": "What are the three elements T, P, and E in a learning system?",
        "expected_pages": [8, 9]
    },
    {
        "question": "What is a target function in machine learning?",
        "expected_pages": [9]
    },
    {
        "question": "What are the five design choices involved in designing a learning system?",
        "expected_pages": [9]
    },
    {
        "question": "What are the Task, Performance Measure, and Training Experience for the checkers learning problem?",
        "expected_pages": [9, 12]
    },
    {
        "question": "What is direct versus indirect training experience?",
        "expected_pages": [9, 10]
    },
    {
        "question": "What is the target function representation used in the checkers learning system?",
        "expected_pages": [12]
    },
    {
        "question": "What is temporal difference learning?",
        "expected_pages": [12, 13]
    },
    {
        "question": "What is the Least Mean Square training rule?",
        "expected_pages": [13]
    },
    {
        "question": "What are the four modules in the final design of the checkers learning system?",
        "expected_pages": [13]
    },
    {
        "question": "What is supervised learning?",
        "expected_pages": [14]
    },
    {
        "question": "What is the difference between classification and regression in supervised learning?",
        "expected_pages": [14]
    },
    {
        "question": "What is unsupervised learning?",
        "expected_pages": [14, 15]
    },
    {
        "question": "What is cluster analysis in unsupervised learning?",
        "expected_pages": [15]
    },
    {
        "question": "What is reinforcement learning?",
        "expected_pages": [15]
    },
    {
        "question": "How is reinforcement learning different from supervised learning?",
        "expected_pages": [15]
    },
    {
        "question": "What is the perspective of machine learning as a search problem?",
        "expected_pages": [15, 16]
    },
    {
        "question": "What are the major issues in machine learning?",
        "expected_pages": [16]
    },
    {
        "question": "What is a version space?",
        "expected_pages": [17]
    },
    {
        "question": "What does it mean for a hypothesis to be consistent with training examples?",
        "expected_pages": [17]
    },
    {
        "question": "What is the LIST-THEN-ELIMINATE algorithm?",
        "expected_pages": [17, 18]
    },
    {
        "question": "What are the general and specific boundaries of a version space?",
        "expected_pages": [18]
    },
    {
        "question": "What is the version space representation theorem?",
        "expected_pages": [18]
    },
    {
        "question": "What is the Candidate-Elimination algorithm?",
        "expected_pages": [18, 19]
    },
    {
        "question": "How does Candidate-Elimination update the general and specific boundaries?",
        "expected_pages": [19, 20, 21]
    },
    {
        "question": "What is PAC learning?",
        "expected_pages": [21, 22]
    },
    {
        "question": "What does probably approximately correct mean?",
        "expected_pages": [21]
    },
    {
        "question": "What are the instance space, concept class, hypothesis, and probability distribution in PAC learning?",
        "expected_pages": [22]
    },
    {
        "question": "What is PAC-learnability?",
        "expected_pages": [22]
    },
    {
        "question": "Why are axis-aligned rectangles used as an example of a PAC-learnable concept class?",
        "expected_pages": [22, 23]
    },
    {
        "question": "What is Vapnik-Chervonenkis dimension?",
        "expected_pages": [23]
    },
    {
        "question": "What is shattering in the context of VC dimension?",
        "expected_pages": [24]
    },
    {
        "question": "How many possible dichotomies exist for N examples?",
        "expected_pages": [24]
    },
    {
        "question": "What is the VC dimension of axis-aligned rectangles in two dimensions?",
        "expected_pages": [24, 25]
    },
    {
        "question": "How can four points be shattered by axis-aligned rectangles?",
        "expected_pages": [24, 25]
    }
]

def evaluate_question(index, chunks, question, expected_pages, k):
    question_embedding = embedding_model.encode(question)

    retrieved_chunks = search(index,question_embedding, chunks, k)
    retrieved_pages = {
        chunk["page"]
        for chunk in retrieved_chunks
    }
    hit = bool(retrieved_pages.intersection(set(expected_pages)))

    return hit, retrieved_pages

def evaluate_all(index, chunks, evaluation_data, k):
    successful = 0

    for data in evaluation_data:
        question = data["question"]
        expected_pages = data["expected_pages"]

        hit, retrieved_pages = evaluate_question(index, chunks, question, expected_pages, k)
        if hit:
            successful += 1

        print(f"Question: {question}")
        print(f"Expected pages: {expected_pages}")
        print(f"Retrieved pages: {sorted(retrieved_pages)}")
        print(f"Result: {'PASS' if hit else 'FAIL'}")
        print("-" * 60)

    accuracy = (successful / len(evaluation_data)) * 100

    print(f"Accuracy for k = {k} if {accuracy}")