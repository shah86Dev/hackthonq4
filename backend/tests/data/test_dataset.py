"""
Test dataset for evaluating the RAG chatbot responses
"""

# Sample test cases for book content
TEST_DATASET = [
    {
        "question": "What is the main theme of this book?",
        "expected_answer": "The main theme of this book is artificial intelligence and its applications in modern technology.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "Explain the concept of machine learning",
        "expected_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What are neural networks?",
        "expected_answer": "Neural networks are computing systems inspired by the human brain, consisting of interconnected nodes that process information.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What does this selected text mean?",
        "expected_answer": "The selected text explains that deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "book_id": "test-book-1",
        "selected_text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns in data."
    },
    {
        "question": "How does supervised learning work?",
        "expected_answer": "Supervised learning works by training a model on labeled data, where the input-output pairs are known, allowing the model to make predictions on new data.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What is the difference between supervised and unsupervised learning?",
        "expected_answer": "Supervised learning uses labeled data for training, while unsupervised learning finds patterns in unlabeled data without guidance.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "Explain the bias-variance tradeoff",
        "expected_answer": "The bias-variance tradeoff is a central problem in supervised learning where a model must balance between underfitting (high bias) and overfitting (high variance).",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What are hyperparameters in machine learning?",
        "expected_answer": "Hyperparameters are parameters set before the learning process begins that control the learning algorithm's behavior.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What is cross-validation?",
        "expected_answer": "Cross-validation is a technique for evaluating machine learning models by training them on different subsets of data and validating on the remaining parts.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "Describe the backpropagation algorithm",
        "expected_answer": "Backpropagation is an algorithm used in training neural networks that calculates the gradient of the loss function with respect to the weights.",
        "book_id": "test-book-1",
        "selected_text": None
    }
]

# Extended test dataset with more complex questions
EXTENDED_TEST_DATASET = [
    {
        "question": "Summarize the key concepts covered in the first chapter",
        "expected_answer": "The first chapter covers the fundamentals of artificial intelligence, including its history, key concepts, and major applications in various fields.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What are the ethical considerations in AI development?",
        "expected_answer": "Ethical considerations in AI development include fairness, transparency, accountability, privacy, and the potential impact on employment.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "Compare different types of neural networks",
        "expected_answer": "Different types of neural networks include feedforward networks, convolutional neural networks (CNNs) for image processing, and recurrent neural networks (RNNs) for sequence data.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What is the role of activation functions?",
        "expected_answer": "Activation functions determine whether a neuron should be activated or not, introducing non-linearity to the model which allows it to learn complex patterns.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "Explain ensemble methods in machine learning",
        "expected_answer": "Ensemble methods combine multiple models to improve overall performance, including techniques like bagging, boosting, and stacking.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What is the difference between classification and regression?",
        "expected_answer": "Classification predicts discrete categories while regression predicts continuous values.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "How do you handle overfitting in machine learning models?",
        "expected_answer": "Overfitting can be handled using techniques like cross-validation, regularization, early stopping, and increasing training data.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What is feature engineering?",
        "expected_answer": "Feature engineering is the process of selecting, transforming, and creating features from raw data to improve machine learning model performance.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "Describe the concept of transfer learning",
        "expected_answer": "Transfer learning is a technique where a pre-trained model is adapted to a new but related task, leveraging learned features to improve performance.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "What are the main challenges in deep learning?",
        "expected_answer": "Main challenges in deep learning include need for large amounts of data, computational requirements, interpretability, and risk of overfitting.",
        "book_id": "test-book-1",
        "selected_text": None
    }
]

# Test dataset for performance evaluation
PERFORMANCE_TEST_DATASET = [
    {
        "question": "What is AI?",
        "expected_answer": "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines.",
        "book_id": "test-book-1",
        "selected_text": None
    },
    {
        "question": "Define machine learning",
        "expected_answer": "Machine learning is a method of teaching computers to learn and adapt through experience.",
        "book_id": "test-book-1",
        "selected_text": None
    }
]

def get_test_dataset(dataset_name="basic"):
    """
    Get a specific test dataset by name
    """
    datasets = {
        "basic": TEST_DATASET,
        "extended": EXTENDED_TEST_DATASET,
        "performance": PERFORMANCE_TEST_DATASET
    }

    return datasets.get(dataset_name, TEST_DATASET)

def get_all_test_datasets():
    """
    Get all test datasets combined
    """
    all_tests = []
    for dataset in [TEST_DATASET, EXTENDED_TEST_DATASET, PERFORMANCE_TEST_DATASET]:
        all_tests.extend(dataset)
    return all_tests