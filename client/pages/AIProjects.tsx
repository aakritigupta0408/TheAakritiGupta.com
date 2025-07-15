import React, { useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Navigation from "../components/Navigation";

interface Resource {
  name: string;
  type:
    | "API"
    | "Dataset"
    | "Model"
    | "Framework"
    | "Company"
    | "Research Lab"
    | "Paper";
  description: string;
  link: string;
  pricing: string;
  category?: "Industry" | "Academic" | "Open Source" | "Commercial";
}

interface Project {
  id: number;
  title: string;
  category: string;
  description: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  timeToComplete: string;
  useCases: string[];
  trainingApproach: string;
  keySteps: string[];
  resources: Resource[];
  codeExample: string;
  icon: string;
  tags: string[];
  theoreticalConcepts: {
    fundamentals: string[];
    keyTheory: string;
    mathematicalFoundations: string;
    importantPapers: string[];
    businessImpact: string;
  };
}

const projects: Project[] = [
  {
    id: 1,
    title: "Image Classification",
    category: "Computer Vision",
    description:
      "Classify images into predefined categories using deep learning models.",
    difficulty: "Beginner",
    timeToComplete: "1-2 weeks",
    useCases: [
      "Medical image diagnosis",
      "Product quality control",
      "Content moderation",
      "Wildlife monitoring",
      "Food recognition apps",
    ],
    trainingApproach:
      "Transfer learning with pre-trained CNNs like ResNet, EfficientNet, or Vision Transformers. Fine-tune on your specific dataset.",
    keySteps: [
      "Collect and label training images",
      "Data preprocessing and augmentation",
      "Load pre-trained model (ImageNet)",
      "Replace classification head",
      "Fine-tune on your dataset",
      "Evaluate and optimize performance",
    ],
    resources: [
      {
        name: "Google Cloud Vision API",
        type: "API",
        description: "Ready-to-use image classification and object detection",
        link: "https://cloud.google.com/vision",
        pricing: "$1.50 per 1000 requests",
        category: "Commercial",
      },
      {
        name: "ImageNet Dataset",
        type: "Dataset",
        description: "1.2M labeled images across 1000 categories",
        link: "https://www.image-net.org/",
        pricing: "Free",
        category: "Academic",
      },
      {
        name: "PyTorch Vision Models",
        type: "Model",
        description: "Pre-trained models: ResNet, VGG, EfficientNet",
        link: "https://pytorch.org/vision/stable/models.html",
        pricing: "Free",
        category: "Open Source",
      },
      {
        name: "Google Research",
        type: "Company",
        description:
          "Leading AI research with EfficientNet, Vision Transformer innovations",
        link: "https://research.google/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Meta AI (FAIR)",
        type: "Research Lab",
        description: "ResNet, DETR, and computer vision breakthroughs",
        link: "https://ai.meta.com/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Stanford Vision Lab",
        type: "Research Lab",
        description: "ImageNet creators, computer vision research pioneers",
        link: "https://vision.stanford.edu/",
        pricing: "N/A",
        category: "Academic",
      },
      {
        name: "EfficientNet Paper",
        type: "Paper",
        description:
          "Rethinking Model Scaling for Convolutional Neural Networks",
        link: "https://arxiv.org/abs/1905.11946",
        pricing: "Free",
        category: "Academic",
      },
    ],
    codeExample: `import torch
import torchvision.models as models
from torchvision import transforms

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(2048, num_classes)

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(epochs):
    for batch in dataloader:
        outputs = model(batch['image'])
        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()`,
    icon: "📸",
    tags: ["CNN", "Transfer Learning", "PyTorch", "Classification"],
    theoreticalConcepts: {
      fundamentals: [
        "Convolutional Neural Networks (CNNs) - Local receptive fields detect spatial patterns",
        "Feature Maps - Each filter detects specific visual features like edges, textures",
        "Pooling Layers - Reduce spatial dimensions while retaining important information",
        "Transfer Learning - Leverage pre-trained features from large datasets like ImageNet",
        "Data Augmentation - Increase dataset diversity through transformations",
      ],
      keyTheory:
        "CNNs use translation-invariant filters that share weights across spatial locations, enabling efficient feature detection regardless of object position. The hierarchical architecture learns increasingly complex features: edges → shapes → objects → concepts.",
      mathematicalFoundations:
        "Convolution operation: (f * g)(x,y) = Σ Σ f(i,j) × g(x-i, y-j). Backpropagation through convolution layers requires computing gradients w.r.t filters and inputs using chain rule.",
      importantPapers: [
        "LeNet (1998) - First CNN for digit recognition",
        "AlexNet (2012) - Breakthrough in ImageNet competition",
        "ResNet (2015) - Skip connections for very deep networks",
        "EfficientNet (2019) - Scaling CNNs with compound coefficients",
      ],
      businessImpact:
        "Enables automated visual inspection, medical diagnosis, autonomous vehicles, content moderation, and quality control - reducing human error and processing time by 90%+",
    },
  },
  {
    id: 2,
    title: "Chatbot Development",
    category: "Natural Language Processing",
    description:
      "Build intelligent conversational agents using large language models.",
    difficulty: "Intermediate",
    timeToComplete: "2-4 weeks",
    useCases: [
      "Customer support automation",
      "Personal assistants",
      "Educational tutoring",
      "Mental health support",
      "E-commerce recommendations",
    ],
    trainingApproach:
      "Fine-tune pre-trained language models (GPT, BERT) on domain-specific conversation data or use retrieval-augmented generation (RAG).",
    keySteps: [
      "Define conversation scope and intents",
      "Collect/create training conversations",
      "Choose base model (GPT-3.5, GPT-4, Llama)",
      "Implement RAG or fine-tuning pipeline",
      "Add safety and content filtering",
      "Deploy and monitor performance",
    ],
    resources: [
      {
        name: "OpenAI GPT API",
        type: "API",
        description: "Access to GPT-3.5 and GPT-4 for chat applications",
        link: "https://platform.openai.com/docs/guides/gpt",
        pricing: "$0.002 per 1K tokens",
        category: "Commercial",
      },
      {
        name: "Hugging Face Transformers",
        type: "Framework",
        description: "Open-source library for transformer models",
        link: "https://huggingface.co/transformers/",
        pricing: "Free",
        category: "Open Source",
      },
      {
        name: "PersonaChat Dataset",
        type: "Dataset",
        description: "Conversations with personality traits",
        link: "https://huggingface.co/datasets/persona_chat",
        pricing: "Free",
        category: "Academic",
      },
      {
        name: "OpenAI",
        type: "Company",
        description: "Pioneer in large language models and conversational AI",
        link: "https://openai.com/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Anthropic",
        type: "Company",
        description: "Constitutional AI and Claude chatbot development",
        link: "https://anthropic.com/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Google DeepMind",
        type: "Research Lab",
        description: "Transformer architecture and Gemini conversational AI",
        link: "https://deepmind.google/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Attention Is All You Need",
        type: "Paper",
        description: "Original Transformer architecture paper",
        link: "https://arxiv.org/abs/1706.03762",
        pricing: "Free",
        category: "Academic",
      },
      {
        name: "InstructGPT Paper",
        type: "Paper",
        description: "Training language models to follow instructions",
        link: "https://arxiv.org/abs/2203.02155",
        pricing: "Free",
        category: "Academic",
      },
    ],
    codeExample: `from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load pre-trained model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(input_text, chat_history_ids=None):
    # Encode input
    new_user_input_ids = tokenizer.encode(
        input_text + tokenizer.eos_token, 
        return_tensors='pt'
    )
    
    # Append to chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    
    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        num_beams=5,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)`,
    icon: "💬",
    tags: ["NLP", "Transformers", "GPT", "Conversational AI"],
    theoreticalConcepts: {
      fundamentals: [
        "Transformer Architecture - Self-attention mechanism processes sequences in parallel",
        "Language Modeling - Predicting next word given context using probability distributions",
        "Tokenization - Converting text to numerical tokens that models can process",
        "Fine-tuning vs RAG - Adapting pre-trained models vs retrieving relevant context",
        "Context Windows - Maximum token limit that models can process at once",
      ],
      keyTheory:
        "Large Language Models learn statistical patterns in text through unsupervised learning on massive corpora. Attention mechanisms allow models to focus on relevant parts of input when generating responses, enabling coherent long-form conversations.",
      mathematicalFoundations:
        "Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V. Cross-entropy loss for next-token prediction: L = -Σ log P(w_t|w_1...w_{t-1})",
      importantPapers: [
        "Attention Is All You Need (2017) - Transformer architecture",
        "BERT (2018) - Bidirectional encoder representations",
        "GPT (2018) - Generative pre-training of language models",
        "ChatGPT/InstructGPT (2022) - Human feedback fine-tuning",
      ],
      businessImpact:
        "Automates customer service, reduces response time from hours to seconds, handles 80%+ of routine inquiries, and scales support without linear cost increases.",
    },
  },
  {
    id: 3,
    title: "Recommendation System",
    category: "Machine Learning",
    description:
      "Build personalized recommendation engines for products, content, or services.",
    difficulty: "Intermediate",
    timeToComplete: "3-5 weeks",
    useCases: [
      "E-commerce product recommendations",
      "Movie/music streaming suggestions",
      "News article recommendations",
      "Social media content feeds",
      "Job matching platforms",
    ],
    trainingApproach:
      "Collaborative filtering, content-based filtering, or hybrid approaches using matrix factorization, deep learning embeddings, or neural collaborative filtering.",
    keySteps: [
      "Collect user interaction data",
      "Data preprocessing and feature engineering",
      "Choose recommendation algorithm",
      "Train and validate model",
      "Implement real-time inference",
      "A/B test and optimize recommendations",
    ],
    resources: [
      {
        name: "Amazon Personalize",
        type: "API",
        description: "Fully managed ML service for recommendations",
        link: "https://aws.amazon.com/personalize/",
        pricing: "$0.05 per hour training",
        category: "Commercial",
      },
      {
        name: "MovieLens Dataset",
        type: "Dataset",
        description: "Movie ratings from 100K+ users",
        link: "https://grouplens.org/datasets/movielens/",
        pricing: "Free",
        category: "Academic",
      },
      {
        name: "Surprise Library",
        type: "Framework",
        description: "Python scikit for recommender systems",
        link: "https://surprise.readthedocs.io/",
        pricing: "Free",
        category: "Open Source",
      },
      {
        name: "Netflix",
        type: "Company",
        description:
          "Pioneered modern collaborative filtering and matrix factorization",
        link: "https://research.netflix.com/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Amazon",
        type: "Company",
        description:
          "Item-to-item collaborative filtering and personalization at scale",
        link: "https://www.amazon.science/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "YouTube AI",
        type: "Research Lab",
        description: "Deep neural networks for YouTube recommendations",
        link: "https://research.google/teams/brain/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Netflix Prize Papers",
        type: "Paper",
        description: "Matrix factorization techniques for recommender systems",
        link: "https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf",
        pricing: "Free",
        category: "Academic",
      },
      {
        name: "Neural Collaborative Filtering",
        type: "Paper",
        description: "Deep learning approach to collaborative filtering",
        link: "https://arxiv.org/abs/1708.05031",
        pricing: "Free",
        category: "Academic",
      },
    ],
    codeExample: `import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load data
df = pd.read_csv('ratings.csv')
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split data
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD(n_factors=50, lr_all=0.005, reg_all=0.02)
model.fit(trainset)

# Make predictions
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

# Get recommendations for user
def get_recommendations(user_id, n_recommendations=10):
    # Get list of all items
    all_items = df['item_id'].unique()
    
    # Get items user has already rated
    user_items = df[df['user_id'] == user_id]['item_id'].unique()
    
    # Predict ratings for unrated items
    predictions = []
    for item_id in all_items:
        if item_id not in user_items:
            pred = model.predict(user_id, item_id)
            predictions.append((item_id, pred.est))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n_recommendations]`,
    icon: "🎯",
    tags: ["Collaborative Filtering", "Matrix Factorization", "Embeddings"],
    theoreticalConcepts: {
      fundamentals: [
        "Collaborative Filtering - Users with similar preferences like similar items",
        "Content-Based Filtering - Recommend items similar to user's past preferences",
        "Matrix Factorization - Decompose user-item matrix into latent factors",
        "Cold Start Problem - Handling new users/items with no interaction history",
        "Embedding Spaces - Learn dense representations capturing user/item similarities",
      ],
      keyTheory:
        "Recommendation systems predict user preferences by learning latent factors that explain observed interactions. Matrix factorization discovers hidden patterns in user-item interactions to predict unobserved preferences.",
      mathematicalFoundations:
        "Matrix Factorization: R ≈ UV^T where U(users×factors), V(items×factors). Loss: ||R - UV^T||²_F + λ(||U||²_F + ||V||²_F)",
      importantPapers: [
        "Netflix Prize Papers (2009) - Matrix factorization techniques",
        "Deep Learning for Recommender Systems (2017)",
        "Neural Collaborative Filtering (2017)",
        "Wide & Deep Learning (2016) - Google's recommendation system",
      ],
      businessImpact:
        "Drives 35%+ of Amazon revenue, increases user engagement by 60%+, and improves conversion rates by personalizing content discovery and product recommendations.",
    },
  },
  {
    id: 4,
    title: "Object Detection",
    category: "Computer Vision",
    description:
      "Detect and localize multiple objects within images using bounding boxes.",
    difficulty: "Advanced",
    timeToComplete: "4-6 weeks",
    useCases: [
      "Autonomous vehicle perception",
      "Security surveillance systems",
      "Retail inventory management",
      "Sports analytics",
      "Medical imaging diagnostics",
    ],
    trainingApproach:
      "Use YOLO, R-CNN, or Transformer-based models (DETR). Transfer learning from COCO pre-trained models and fine-tune on custom datasets.",
    keySteps: [
      "Collect and annotate bounding box data",
      "Data augmentation for robustness",
      "Choose detection architecture (YOLO, R-CNN)",
      "Configure anchor boxes and loss functions",
      "Train with proper data loading pipeline",
      "Post-processing with NMS and evaluation",
    ],
    resources: [
      {
        name: "YOLOv8 by Ultralytics",
        type: "Model",
        description: "State-of-the-art object detection model",
        link: "https://github.com/ultralytics/ultralytics",
        pricing: "Free",
        category: "Open Source",
      },
      {
        name: "COCO Dataset",
        type: "Dataset",
        description: "330K images with 80 object categories",
        link: "https://cocodataset.org/",
        pricing: "Free",
        category: "Academic",
      },
      {
        name: "Detectron2",
        type: "Framework",
        description: "Facebook's detection and segmentation platform",
        link: "https://detectron2.readthedocs.io/",
        pricing: "Free",
        category: "Open Source",
      },
      {
        name: "Ultralytics",
        type: "Company",
        description: "YOLO series creators and computer vision solutions",
        link: "https://ultralytics.com/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Meta AI (FAIR)",
        type: "Research Lab",
        description: "Faster R-CNN, Mask R-CNN, and Detectron frameworks",
        link: "https://ai.meta.com/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Microsoft Research",
        type: "Research Lab",
        description: "R-CNN series and object detection innovations",
        link: "https://www.microsoft.com/en-us/research/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "YOLO Paper",
        type: "Paper",
        description: "You Only Look Once: Unified, Real-Time Object Detection",
        link: "https://arxiv.org/abs/1506.02640",
        pricing: "Free",
        category: "Academic",
      },
      {
        name: "Faster R-CNN Paper",
        type: "Paper",
        description:
          "Towards Real-Time Object Detection with Region Proposal Networks",
        link: "https://arxiv.org/abs/1506.01497",
        pricing: "Free",
        category: "Academic",
      },
    ],
    codeExample: `from ultralytics import YOLO
import cv2

# Load pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train on custom dataset
model.train(
    data='path/to/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# Inference on new image
results = model('path/to/image.jpg')

# Process results
for result in results:
    boxes = result.boxes
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        confidence = box.conf[0]
        class_id = box.cls[0]
        
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f'{model.names[int(class_id)]}: {confidence:.2f}', 
                   (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)`,
    icon: "🔍",
    tags: ["YOLO", "Object Detection", "Computer Vision", "Bounding Boxes"],
    theoreticalConcepts: {
      fundamentals: [
        "Two-Stage Detection - Region proposal then classification (R-CNN family)",
        "One-Stage Detection - Direct prediction of boxes and classes (YOLO, SSD)",
        "Anchor Boxes - Pre-defined box shapes at different scales and ratios",
        "Non-Maximum Suppression - Remove duplicate detections of same object",
        "IoU (Intersection over Union) - Metric for bounding box overlap",
      ],
      keyTheory:
        "Object detection combines classification (what) and localization (where). Modern detectors use feature pyramids to detect objects at multiple scales and aspect ratios efficiently in a single forward pass.",
      mathematicalFoundations:
        "IoU = |A ∩ B| / |A ∪ B|. Loss = Classification_loss + Localization_loss + Confidence_loss. Smooth L1 loss for bounding box regression.",
      importantPapers: [
        "R-CNN (2014) - First deep learning object detector",
        "YOLO (2016) - Real-time single-stage detection",
        "Faster R-CNN (2015) - End-to-end trainable detector",
        "EfficientDet (2020) - Efficient compound scaling",
      ],
      businessImpact:
        "Powers autonomous vehicles, security systems, medical imaging, and retail analytics - enabling real-time decision making with 95%+ accuracy in critical applications.",
    },
  },
  {
    id: 5,
    title: "Sentiment Analysis",
    category: "Natural Language Processing",
    description:
      "Analyze emotional tone and opinions in text data for business insights.",
    difficulty: "Beginner",
    timeToComplete: "1-2 weeks",
    useCases: [
      "Social media monitoring",
      "Customer review analysis",
      "Brand reputation management",
      "Market research insights",
      "Content recommendation",
    ],
    trainingApproach:
      "Fine-tune BERT-based models on sentiment datasets or use pre-trained sentiment analysis APIs for quick deployment.",
    keySteps: [
      "Collect labeled sentiment data",
      "Text preprocessing and tokenization",
      "Choose model architecture (BERT, RoBERTa)",
      "Fine-tune on sentiment dataset",
      "Evaluate on test data",
      "Deploy for real-time inference",
    ],
    resources: [
      {
        name: "Google Cloud Natural Language API",
        type: "API",
        description: "Ready-to-use sentiment analysis",
        link: "https://cloud.google.com/natural-language",
        pricing: "$1 per 1000 requests",
      },
      {
        name: "IMDB Movie Reviews",
        type: "Dataset",
        description: "50K movie reviews with sentiment labels",
        link: "https://ai.stanford.edu/~amaas/data/sentiment/",
        pricing: "Free",
      },
      {
        name: "DistilBERT",
        type: "Model",
        description: "Lightweight BERT for sentiment analysis",
        link: "https://huggingface.co/distilbert-base-uncased",
        pricing: "Free",
      },
    ],
    codeExample: `from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained sentiment model
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def analyze_sentiment(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Map to labels
    labels = ['negative', 'neutral', 'positive']
    scores = predictions[0].tolist()
    
    result = {
        'sentiment': labels[torch.argmax(predictions).item()],
        'scores': dict(zip(labels, scores))
    }
    
    return result

# Example usage
text = "I love this product! It's amazing."
sentiment = analyze_sentiment(text)
print(f"Sentiment: {sentiment['sentiment']}")
print(f"Confidence: {max(sentiment['scores'].values()):.3f}")`,
    icon: "😊",
    tags: ["BERT", "Text Classification", "NLP", "Sentiment"],
    theoreticalConcepts: {
      fundamentals: [
        "Word Embeddings - Dense vector representations capturing semantic similarity",
        "BERT Architecture - Bidirectional encoder for context-aware representations",
        "Attention Mechanism - Weights determine importance of each word for classification",
        "Transfer Learning - Fine-tune pre-trained language models on domain data",
        "Text Preprocessing - Tokenization, normalization, handling out-of-vocabulary words",
      ],
      keyTheory:
        "Sentiment analysis uses contextualized word representations to classify emotional polarity. BERT's bidirectional training captures nuanced semantic relationships that improve classification accuracy over traditional bag-of-words approaches.",
      mathematicalFoundations:
        "BERT uses multi-head self-attention: MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O. Classification head: P(class) = softmax(W·h_[CLS] + b)",
      importantPapers: [
        "BERT (2018) - Bidirectional encoder representations",
        "RoBERTa (2019) - Robustly optimized BERT approach",
        "DistilBERT (2019) - Lightweight BERT alternative",
        "Sentiment Treebank (2013) - Fine-grained sentiment dataset",
      ],
      businessImpact:
        "Monitors brand reputation in real-time, processes millions of social media posts daily, and provides actionable insights that drive marketing strategies and customer service improvements.",
    },
  },
  {
    id: 6,
    title: "Time Series Forecasting",
    category: "Machine Learning",
    description:
      "Predict future values based on historical time series data patterns.",
    difficulty: "Intermediate",
    timeToComplete: "2-4 weeks",
    useCases: [
      "Stock price prediction",
      "Sales forecasting",
      "Energy demand prediction",
      "Weather forecasting",
      "Traffic flow prediction",
    ],
    trainingApproach:
      "Use LSTM/GRU networks, Transformer models, or traditional methods like ARIMA. Consider seasonal patterns and external features.",
    keySteps: [
      "Data collection and cleaning",
      "Feature engineering (lags, rolling averages)",
      "Handle seasonality and trends",
      "Choose model (LSTM, Prophet, ARIMA)",
      "Train with proper validation strategy",
      "Evaluate forecast accuracy",
    ],
    resources: [
      {
        name: "Facebook Prophet",
        type: "Framework",
        description: "Automatic forecasting tool",
        link: "https://facebook.github.io/prophet/",
        pricing: "Free",
      },
      {
        name: "Yahoo Finance Data",
        type: "Dataset",
        description: "Historical stock and financial data",
        link: "https://finance.yahoo.com/",
        pricing: "Free",
      },
      {
        name: "TensorFlow Time Series",
        type: "Framework",
        description: "Deep learning for time series",
        link: "https://www.tensorflow.org/tutorials/structured_data/time_series",
        pricing: "Free",
      },
    ],
    codeExample: `import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Load time series data
df = pd.read_csv('timeseries_data.csv')
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['value']

# Initialize and fit Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

# Add custom seasonality if needed
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Fit model
model.fit(df[['ds', 'y']])

# Create future dataframe
future = model.make_future_dataframe(periods=365)  # 365 days forecast

# Make predictions
forecast = model.predict(future)

# Plot results
fig = model.plot(forecast)
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Get forecast components
fig2 = model.plot_components(forecast)
plt.show()

# Calculate accuracy metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

y_true = df['y'].values
y_pred = forecast.loc[:len(df)-1, 'yhat'].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")`,
    icon: "📈",
    tags: ["Time Series", "Prophet", "LSTM", "Forecasting"],
    theoreticalConcepts: {
      fundamentals: [
        "Temporal Dependencies - Past values influence future predictions",
        "Seasonality - Recurring patterns at regular intervals (daily, weekly, yearly)",
        "Trend Analysis - Long-term directional movement in data",
        "LSTM Memory Cells - Selectively remember/forget information over time",
        "Prophet Decomposition - Trend + Seasonality + Holidays + Error components",
      ],
      keyTheory:
        "Time series forecasting models temporal relationships to predict future values. LSTMs use gating mechanisms to learn long-term dependencies, while Prophet uses additive decomposition for interpretable forecasting.",
      mathematicalFoundations:
        "LSTM gates: f_t = σ(W_f·[h_{t-1}, x_t] + b_f). Prophet: y(t) = g(t) + s(t) + h(t) + ε_t where g=trend, s=seasonality, h=holidays",
      importantPapers: [
        "LSTM (1997) - Long Short-Term Memory networks",
        "Prophet (2017) - Facebook's forecasting tool",
        "DeepAR (2017) - Amazon's probabilistic forecasting",
        "Temporal Convolutional Networks (2018)",
      ],
      businessImpact:
        "Optimizes inventory management, predicts demand fluctuations, reduces stockouts by 30%+, and enables data-driven capacity planning across industries.",
    },
  },
  {
    id: 7,
    title: "Text Generation",
    category: "Natural Language Processing",
    description:
      "Generate human-like text for content creation, creative writing, and automation.",
    difficulty: "Advanced",
    timeToComplete: "3-5 weeks",
    useCases: [
      "Content marketing automation",
      "Creative writing assistance",
      "Code generation",
      "Email automation",
      "Language translation",
    ],
    trainingApproach:
      "Fine-tune GPT-based models on domain-specific text or use prompt engineering with large language models.",
    keySteps: [
      "Collect domain-specific text data",
      "Data preprocessing and tokenization",
      "Choose base model (GPT-2, GPT-3, T5)",
      "Fine-tune or prompt engineering",
      "Implement generation strategies",
      "Quality evaluation and filtering",
    ],
    resources: [
      {
        name: "OpenAI GPT-4 API",
        type: "API",
        description: "State-of-the-art text generation",
        link: "https://platform.openai.com/docs/models/gpt-4",
        pricing: "$0.03 per 1K tokens",
      },
      {
        name: "Common Crawl",
        type: "Dataset",
        description: "Web crawl data for training",
        link: "https://commoncrawl.org/",
        pricing: "Free",
      },
      {
        name: "GPT-2 by Hugging Face",
        type: "Model",
        description: "Open-source text generation model",
        link: "https://huggingface.co/gpt2",
        pricing: "Free",
      },
    ],
    codeExample: `from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add padding token
tokenizer.pad_token = tokenizer.eos_token

def generate_text(prompt, max_length=150, temperature=0.8, top_p=0.9):
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text[len(prompt):]

# Example usage
prompt = "The future of artificial intelligence is"
generated = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Generated: {generated}")

# Fine-tuning example
from transformers import Trainer, TrainingArguments

def fine_tune_gpt2(train_dataset, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=5e-5,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model()`,
    icon: "✍️",
    tags: ["GPT", "Text Generation", "Language Models", "Fine-tuning"],
    theoreticalConcepts: {
      fundamentals: [
        "Autoregressive Generation - Predict next token given previous context",
        "Temperature Sampling - Control randomness/creativity in generation",
        "Top-k/Top-p Sampling - Filter low-probability tokens for better quality",
        "Prompt Engineering - Craft inputs to guide model behavior",
        "Fine-tuning vs Few-shot - Adapt models through training vs examples",
      ],
      keyTheory:
        "Text generation models learn probability distributions over vocabulary given context. GPT uses decoder-only transformers with causal masking to generate coherent, contextually appropriate text through autoregressive prediction.",
      mathematicalFoundations:
        "Language modeling: P(w₁,...,w_n) = ∏P(w_i|w₁,...,w_{i-1}). Temperature scaling: P'(w_i) = exp(logits_i/T) / Σexp(logits_j/T)",
      importantPapers: [
        "GPT (2018) - Generative pre-training approach",
        "GPT-2 (2019) - Scaling language models",
        "GPT-3 (2020) - Few-shot learning capabilities",
        "InstructGPT (2022) - Training with human feedback",
      ],
      businessImpact:
        "Automates content creation, reduces writing time by 70%+, enables personalized communication at scale, and powers AI assistants generating billions of interactions daily.",
    },
  },
  {
    id: 8,
    title: "Anomaly Detection",
    category: "Machine Learning",
    description:
      "Identify unusual patterns or outliers in data for fraud detection and monitoring.",
    difficulty: "Intermediate",
    timeToComplete: "2-3 weeks",
    useCases: [
      "Credit card fraud detection",
      "Network intrusion detection",
      "Manufacturing quality control",
      "System monitoring and alerts",
      "Medical diagnosis assistance",
    ],
    trainingApproach:
      "Use isolation forests, autoencoders, or one-class SVM. For time series, use LSTM autoencoders or statistical methods.",
    keySteps: [
      "Collect normal and anomalous data",
      "Feature engineering and normalization",
      "Choose algorithm (Isolation Forest, Autoencoder)",
      "Train on normal data patterns",
      "Set anomaly threshold",
      "Validate and deploy monitoring",
    ],
    resources: [
      {
        name: "Scikit-learn Anomaly Detection",
        type: "Framework",
        description: "Built-in anomaly detection algorithms",
        link: "https://scikit-learn.org/stable/modules/outlier_detection.html",
        pricing: "Free",
      },
      {
        name: "KDD Cup 99 Dataset",
        type: "Dataset",
        description: "Network intrusion detection data",
        link: "http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html",
        pricing: "Free",
      },
      {
        name: "PyOD Library",
        type: "Framework",
        description: "Python outlier detection toolkit",
        link: "https://pyod.readthedocs.io/",
        pricing: "Free",
      },
    ],
    codeExample: `import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')
features = df.select_dtypes(include=[np.number])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Train Isolation Forest
iso_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of anomalies
    random_state=42,
    n_estimators=100
)

# Fit and predict
anomaly_labels = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.decision_function(X_scaled)

# Add results to dataframe
df['anomaly'] = anomaly_labels
df['anomaly_score'] = anomaly_scores

# Identify anomalies (-1 indicates anomaly)
anomalies = df[df['anomaly'] == -1]
normal_data = df[df['anomaly'] == 1]

print(f"Total samples: {len(df)}")
print(f"Anomalies detected: {len(anomalies)}")
print(f"Anomaly rate: {len(anomalies)/len(df)*100:.2f}%")

# Autoencoder approach for anomaly detection
import tensorflow as tf
from tensorflow.keras import layers

def create_autoencoder(input_dim, encoding_dim=32):
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim//2, activation='relu')(encoded)
    
    # Decoder
    decoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = tf.keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# Train autoencoder on normal data only
autoencoder = create_autoencoder(X_scaled.shape[1])
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=0)

# Calculate reconstruction error
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

# Set threshold (e.g., 95th percentile)
threshold = np.percentile(mse, 95)
anomalies_ae = mse > threshold

print(f"Autoencoder anomalies: {np.sum(anomalies_ae)}")`,
    icon: "🚨",
    tags: ["Anomaly Detection", "Isolation Forest", "Autoencoders", "Outliers"],
    theoreticalConcepts: {
      fundamentals: [
        "Outlier Detection - Identify data points significantly different from normal",
        "Isolation Forest - Anomalies are easier to isolate in random partitions",
        "Autoencoder Reconstruction - Normal patterns reconstruct well, anomalies don't",
        "One-Class SVM - Learn boundary around normal data distribution",
        "Statistical Methods - Z-score, IQR, and distribution-based detection",
      ],
      keyTheory:
        "Anomaly detection identifies rare patterns that deviate from expected behavior. Isolation forests exploit the fact that anomalies are few and different, requiring fewer splits to isolate in random feature space partitions.",
      mathematicalFoundations:
        "Isolation score: s(x,n) = 2^(-E(h(x))/c(n)) where E(h(x)) is average path length. Autoencoder loss: L = ||x - decoder(encoder(x))||²",
      importantPapers: [
        "Isolation Forest (2008) - Efficient anomaly detection",
        "One-Class SVM (2001) - Support vector approach",
        "Autoencoder-based Anomaly Detection (2015)",
        "Deep SVDD (2018) - Deep one-class classification",
      ],
      businessImpact:
        "Prevents fraud saving billions annually, detects cyber threats in real-time, identifies equipment failures before costly breakdowns, and ensures system reliability.",
    },
  },
  {
    id: 9,
    title: "Speech Recognition",
    category: "Natural Language Processing",
    description:
      "Convert spoken language to text using deep learning models for voice interfaces and transcription.",
    difficulty: "Intermediate",
    timeToComplete: "3-4 weeks",
    useCases: [
      "Voice assistants and smart speakers",
      "Automated transcription services",
      "Voice-controlled applications",
      "Accessibility tools for hearing impaired",
      "Call center automation",
    ],
    trainingApproach:
      "Use pre-trained models like Wav2Vec2, Whisper, or train custom models with CTC/attention mechanisms on speech datasets.",
    keySteps: [
      "Collect and preprocess audio data",
      "Feature extraction (MFCC, spectrograms)",
      "Choose architecture (Transformer, RNN-T)",
      "Train with CTC/attention loss",
      "Post-processing and language models",
      "Deploy real-time inference pipeline",
    ],
    resources: [
      {
        name: "OpenAI Whisper",
        type: "Model",
        description: "State-of-the-art multilingual speech recognition",
        link: "https://github.com/openai/whisper",
        pricing: "Free",
        category: "Open Source",
      },
      {
        name: "Google Speech-to-Text API",
        type: "API",
        description: "Cloud-based speech recognition service",
        link: "https://cloud.google.com/speech-to-text",
        pricing: "$0.006 per 15 seconds",
        category: "Commercial",
      },
      {
        name: "Common Voice Dataset",
        type: "Dataset",
        description: "Mozilla's open source voice dataset",
        link: "https://commonvoice.mozilla.org/",
        pricing: "Free",
        category: "Open Source",
      },
      {
        name: "OpenAI",
        type: "Company",
        description: "Whisper model creators, robust speech recognition",
        link: "https://openai.com/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Google Research",
        type: "Research Lab",
        description: "Listen, Attend and Spell architecture pioneers",
        link: "https://research.google/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Meta AI",
        type: "Research Lab",
        description: "Wav2Vec 2.0 self-supervised speech representation",
        link: "https://ai.meta.com/",
        pricing: "N/A",
        category: "Industry",
      },
      {
        name: "Whisper Paper",
        type: "Paper",
        description:
          "Robust Speech Recognition via Large-Scale Weak Supervision",
        link: "https://arxiv.org/abs/2212.04356",
        pricing: "Free",
        category: "Academic",
      },
      {
        name: "Wav2Vec 2.0 Paper",
        type: "Paper",
        description: "Self-supervised Learning of Speech Representations",
        link: "https://arxiv.org/abs/2006.11477",
        pricing: "Free",
        category: "Academic",
      },
    ],
    codeExample: `import whisper
import torch
import librosa
import numpy as np

# Load Whisper model
model = whisper.load_model("base")

def transcribe_audio(audio_path, language=None):
    # Load and preprocess audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Create mel spectrogram
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect language if not specified
    if language is None:
        _, probs = model.detect_language(mel)
        language = max(probs, key=probs.get)
        print(f"Detected language: {language}")

    # Decode audio
    options = whisper.DecodingOptions(language=language)
    result = whisper.decode(model, mel, options)

    return result.text

# Real-time speech recognition
import pyaudio
import threading

class RealTimeSpeechRecognizer:
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
        self.is_listening = False

    def start_listening(self, duration=5):
        self.is_listening = True

        # Audio recording parameters
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)

        print("Listening...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * duration)):
            if not self.is_listening:
                break
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0

        # Transcribe
        result = self.model.transcribe(audio_data)
        return result["text"]

    def stop_listening(self):
        self.is_listening = False

# Example usage
recognizer = RealTimeSpeechRecognizer()
transcription = recognizer.start_listening(duration=10)
print(f"Transcription: {transcription}")`,
    icon: "🎤",
    tags: ["Speech Recognition", "Whisper", "Audio Processing", "ASR"],
    theoreticalConcepts: {
      fundamentals: [
        "Acoustic Features - MFCC, mel spectrograms convert audio to learnable representations",
        "Connectionist Temporal Classification (CTC) - Aligns variable-length audio to text sequences",
        "Attention Mechanisms - Focus on relevant audio segments during decoding",
        "Language Models - Improve accuracy by incorporating linguistic context",
        "Wav2Vec2 - Self-supervised learning from raw audio waveforms",
      ],
      keyTheory:
        "Speech recognition transforms acoustic signals into text through feature extraction, sequence modeling, and language understanding. Modern approaches use transformer architectures with attention to align audio frames with text tokens efficiently.",
      mathematicalFoundations:
        "CTC Loss: L = -log(∑π∈B⁻¹(y) ∏ᵗ p(πₜ|x)). Attention: αᵢⱼ = exp(eᵢⱼ)/∑ₖ exp(eᵢₖ) where eᵢⱼ = score(sᵢ₋₁, hⱼ)",
      importantPapers: [
        "Deep Speech (2014) - End-to-end deep learning for speech",
        "Listen, Attend and Spell (2015) - Attention-based speech recognition",
        "Wav2Vec2.0 (2020) - Self-supervised speech representation learning",
        "Whisper (2022) - Robust speech recognition via large-scale weak supervision",
      ],
      businessImpact:
        "Powers voice assistants used by billions, enables hands-free computing, automates transcription reducing costs by 80%+, and provides accessibility solutions for disabilities.",
    },
  },
  {
    id: 10,
    title: "Generative Adversarial Networks (GANs)",
    category: "Computer Vision",
    description:
      "Generate realistic images, videos, and data using adversarial training between generator and discriminator networks.",
    difficulty: "Advanced",
    timeToComplete: "4-6 weeks",
    useCases: [
      "Synthetic data generation",
      "Image super-resolution",
      "Style transfer and art creation",
      "Face generation and deepfakes",
      "Data augmentation for training",
    ],
    trainingApproach:
      "Train generator and discriminator networks adversarially. Use techniques like progressive growing, spectral normalization, and self-attention for stable training.",
    keySteps: [
      "Design generator and discriminator architectures",
      "Implement adversarial loss functions",
      "Balance training between G and D networks",
      "Apply training stabilization techniques",
      "Evaluate with FID, IS, and visual quality",
      "Fine-tune and condition on specific domains",
    ],
    resources: [
      {
        name: "PyTorch GAN Zoo",
        type: "Framework",
        description: "Collection of GAN implementations",
        link: "https://github.com/facebookresearch/pytorch_GAN_zoo",
        pricing: "Free",
      },
      {
        name: "CelebA Dataset",
        type: "Dataset",
        description: "200K celebrity face images",
        link: "http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        pricing: "Free",
      },
      {
        name: "StyleGAN3",
        type: "Model",
        description: "NVIDIA's state-of-the-art image generation",
        link: "https://github.com/NVlabs/stylegan3",
        pricing: "Free",
      },
    ],
    codeExample: `import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, features_g=64):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x latent_dim x 1 x 1
            self._block(latent_dim, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features_g * 2, img_channels, 4, 2, 1),  # 64x64
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x img_channels x 64 x 64
            nn.Conv2d(img_channels, features_d, 4, 2, 1),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 8, 1, 4, 2, 0),  # 1x1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)

# Training function
def train_gan(generator, discriminator, dataloader, num_epochs=100, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loss and optimizers
    criterion = nn.BCELoss()
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            batch_size = real.shape[0]

            # Train Discriminator
            noise = torch.randn(batch_size, 100, 1, 1).to(device)
            fake = generator(noise)

            disc_real = discriminator(real).view(-1)
            disc_fake = discriminator(fake.detach()).view(-1)

            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            discriminator.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = discriminator(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

# Initialize models
latent_dim = 100
generator = Generator(latent_dim)
discriminator = Discriminator()`,
    icon: "🎨",
    tags: ["GANs", "Image Generation", "Adversarial Training", "Deep Learning"],
    theoreticalConcepts: {
      fundamentals: [
        "Adversarial Training - Generator creates fake data, discriminator learns to detect it",
        "Nash Equilibrium - Both networks reach optimal strategy where neither can improve",
        "Mode Collapse - Generator produces limited variety, common training issue",
        "Wasserstein Distance - Better loss function for stable GAN training",
        "Progressive Growing - Gradually increase resolution for high-quality generation",
      ],
      keyTheory:
        "GANs use game theory where generator and discriminator compete in a minimax game. The generator learns to map noise to realistic data by fooling an adversarially trained discriminator that distinguishes real from fake samples.",
      mathematicalFoundations:
        "GAN objective: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]. Wasserstein distance: W(P_r, P_g) = inf_γ E[(x,y)~γ][||x-y||]",
      importantPapers: [
        "Generative Adversarial Networks (2014) - Original GAN paper",
        "DCGAN (2015) - Deep convolutional GANs",
        "WGAN (2017) - Wasserstein GANs for stable training",
        "StyleGAN (2019) - High-quality image generation with style control",
      ],
      businessImpact:
        "Creates synthetic data worth billions for training AI, generates realistic media for entertainment, enables virtual try-ons in e-commerce, and produces art selling for millions in NFT markets.",
    },
  },
  {
    id: 11,
    title: "Reinforcement Learning Agent",
    category: "Machine Learning",
    description:
      "Train intelligent agents to make decisions in environments through trial and error learning.",
    difficulty: "Advanced",
    timeToComplete: "5-8 weeks",
    useCases: [
      "Game playing (chess, Go, video games)",
      "Autonomous vehicle control",
      "Robot navigation and manipulation",
      "Trading and portfolio optimization",
      "Resource allocation and scheduling",
    ],
    trainingApproach:
      "Use Q-learning, policy gradients, or actor-critic methods. Train agents in simulated environments with reward shaping and exploration strategies.",
    keySteps: [
      "Define environment and state/action spaces",
      "Design reward function and episode structure",
      "Choose RL algorithm (DQN, PPO, A3C)",
      "Implement experience replay and exploration",
      "Train with environment interaction",
      "Evaluate and fine-tune policy performance",
    ],
    resources: [
      {
        name: "OpenAI Gym",
        type: "Framework",
        description: "Standard RL environment interface",
        link: "https://gym.openai.com/",
        pricing: "Free",
      },
      {
        name: "Stable Baselines3",
        type: "Framework",
        description: "Reliable RL algorithm implementations",
        link: "https://stable-baselines3.readthedocs.io/",
        pricing: "Free",
      },
      {
        name: "Unity ML-Agents",
        type: "Framework",
        description: "Train agents in Unity environments",
        link: "https://unity.com/products/machine-learning-agents",
        pricing: "Free",
      },
    ],
    codeExample: `import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize target network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.choice(np.arange(self.action_size))

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training loop
def train_dqn_agent():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    scores = deque(maxlen=100)

    for episode in range(2000):
        state = env.reset()
        total_reward = 0

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

            if len(agent.memory) > 32:
                agent.replay(32)

        scores.append(total_reward)

        if episode % 100 == 0:
            agent.update_target_network()
            print(f"Episode {episode}, Average Score: {np.mean(scores):.2f}, Epsilon: {agent.epsilon:.2f}")

        if np.mean(scores) >= 195.0:
            print(f"Environment solved in {episode} episodes!")
            break

    return agent

# Run training
trained_agent = train_dqn_agent()`,
    icon: "🤖",
    tags: ["Reinforcement Learning", "DQN", "Policy Gradients", "Game AI"],
    theoreticalConcepts: {
      fundamentals: [
        "Markov Decision Process - Mathematical framework for sequential decision making",
        "Q-Learning - Learn action-value function to estimate long-term rewards",
        "Policy Gradients - Directly optimize policy parameters using gradient ascent",
        "Exploration vs Exploitation - Balance between trying new actions and using known good ones",
        "Temporal Difference Learning - Update value estimates using observed rewards",
      ],
      keyTheory:
        "Reinforcement learning optimizes sequential decision making through trial and error. Agents learn optimal policies by maximizing cumulative rewards while exploring the environment to discover better strategies.",
      mathematicalFoundations:
        "Bellman equation: Q(s,a) = E[r + γ max Q(s',a')]. Policy gradient: ∇θ J(θ) = E[∇θ log π(a|s)Q(s,a)]",
      importantPapers: [
        "Q-Learning (1989) - Model-free reinforcement learning",
        "DQN (2015) - Deep Q-Networks for Atari games",
        "AlphaGo (2016) - Combining tree search with deep RL",
        "PPO (2017) - Proximal policy optimization",
      ],
      businessImpact:
        "Powers game AI defeating world champions, optimizes data center cooling saving 40%+ energy, enables autonomous vehicles, and automates trading generating billions in profits.",
    },
  },
  {
    id: 12,
    title: "Neural Machine Translation",
    category: "Natural Language Processing",
    description:
      "Translate text between languages using sequence-to-sequence models with attention mechanisms.",
    difficulty: "Advanced",
    timeToComplete: "4-6 weeks",
    useCases: [
      "Real-time language translation",
      "Document translation services",
      "Multilingual customer support",
      "Content localization",
      "Cross-language information retrieval",
    ],
    trainingApproach:
      "Use transformer-based sequence-to-sequence models with attention. Train on parallel corpora with techniques like back-translation and multilingual training.",
    keySteps: [
      "Collect parallel translation datasets",
      "Tokenization and vocabulary creation",
      "Build encoder-decoder architecture",
      "Implement attention mechanisms",
      "Train with teacher forcing",
      "Evaluate with BLEU score and human evaluation",
    ],
    resources: [
      {
        name: "Google Translate API",
        type: "API",
        description: "Production-ready translation service",
        link: "https://cloud.google.com/translate",
        pricing: "$20 per 1M characters",
      },
      {
        name: "WMT Translation Datasets",
        type: "Dataset",
        description: "Annual workshop parallel corpora",
        link: "http://www.statmt.org/wmt21/",
        pricing: "Free",
      },
      {
        name: "MarianMT",
        type: "Model",
        description: "Fast neural machine translation models",
        link: "https://huggingface.co/Helsinki-NLP",
        pricing: "Free",
      },
    ],
    codeExample: `from transformers import MarianMTModel, MarianTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=128):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = str(self.src_texts[idx])
        tgt_text = str(self.tgt_texts[idx])

        # Tokenize source and target
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'src_input_ids': src_encoding['input_ids'].flatten(),
            'src_attention_mask': src_encoding['attention_mask'].flatten(),
            'tgt_input_ids': tgt_encoding['input_ids'].flatten(),
            'tgt_attention_mask': tgt_encoding['attention_mask'].flatten()
        }

class NeuralTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-de"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def translate(self, text, max_length=128):
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )

        # Decode output
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation

    def translate_batch(self, texts, batch_size=8):
        translations = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate translations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )

            # Decode outputs
            batch_translations = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]

            translations.extend(batch_translations)

        return translations

# Fine-tuning for domain adaptation
def fine_tune_translator(model, tokenizer, train_dataset, val_dataset, epochs=3):
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir='./translation_model',
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer.model

# Example usage
translator = NeuralTranslator("Helsinki-NLP/opus-mt-en-fr")

# Single translation
english_text = "Hello, how are you today?"
french_translation = translator.translate(english_text)
print(f"English: {english_text}")
print(f"French: {french_translation}")

# Batch translation
english_texts = [
    "Good morning!",
    "Thank you very much.",
    "Where is the nearest restaurant?",
    "I would like to book a hotel room."
]

french_translations = translator.translate_batch(english_texts)
for en, fr in zip(english_texts, french_translations):
    print(f"EN: {en} -> FR: {fr}")`,
    icon: "🌍",
    tags: ["Machine Translation", "Seq2Seq", "Transformers", "Multilingual"],
    theoreticalConcepts: {
      fundamentals: [
        "Sequence-to-Sequence Models - Encoder-decoder architecture for variable-length translation",
        "Attention Mechanism - Focus on relevant source words during translation",
        "Beam Search - Explore multiple translation hypotheses to find best output",
        "Subword Tokenization - Handle out-of-vocabulary words with BPE/SentencePiece",
        "Transfer Learning - Leverage multilingual pre-trained models for low-resource languages",
      ],
      keyTheory:
        "Neural machine translation uses encoder-decoder transformers to map source language sequences to target languages. Attention mechanisms enable the decoder to focus on relevant source positions, producing fluent and accurate translations.",
      mathematicalFoundations:
        "Seq2Seq probability: P(y|x) = ∏ᵢ P(yᵢ|y₁...yᵢ₋₁, x). Attention: context_i = Σⱼ αᵢⱼhⱼ where αᵢⱼ = exp(score(sᵢ,hⱼ))/Σₖ exp(score(sᵢ,hₖ))",
      importantPapers: [
        "Sequence to Sequence Learning (2014) - Original seq2seq with RNNs",
        "Neural Machine Translation by Jointly Learning to Align and Translate (2014) - Attention mechanism",
        "Attention Is All You Need (2017) - Transformer architecture",
        "Multilingual Neural Machine Translation (2019) - Massively multilingual models",
      ],
      businessImpact:
        "Breaks down language barriers for global commerce, enables real-time communication across cultures, automates content localization saving millions, and powers multilingual AI assistants.",
    },
  },
  {
    id: 13,
    title: "Medical Image Analysis",
    category: "Computer Vision",
    description:
      "Analyze medical images for diagnosis assistance using deep learning models trained on healthcare data.",
    difficulty: "Advanced",
    timeToComplete: "6-8 weeks",
    useCases: [
      "Radiology diagnosis assistance",
      "Cancer detection and staging",
      "Retinal disease screening",
      "Skin lesion classification",
      "Brain tumor segmentation",
    ],
    trainingApproach:
      "Use specialized CNN architectures with medical image preprocessing. Apply transfer learning from ImageNet and fine-tune on medical datasets with careful validation.",
    keySteps: [
      "Collect and curate medical image datasets",
      "Medical image preprocessing (DICOM, normalization)",
      "Design specialized CNN architecture",
      "Implement proper train/validation splits",
      "Apply medical-specific data augmentation",
      "Evaluate with medical metrics and expert validation",
    ],
    resources: [
      {
        name: "NIH Chest X-ray Dataset",
        type: "Dataset",
        description: "100K+ chest X-rays with disease labels",
        link: "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        pricing: "Free",
      },
      {
        name: "MONAI Framework",
        type: "Framework",
        description: "Medical imaging AI framework by NVIDIA",
        link: "https://monai.io/",
        pricing: "Free",
      },
      {
        name: "Medical Segmentation Decathlon",
        type: "Dataset",
        description: "Multi-organ segmentation challenges",
        link: "http://medicaldecathlon.com/",
        pricing: "Free",
      },
    ],
    codeExample: `import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pydicom
from PIL import Image
import pandas as pd

class MedicalImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])

        # Load DICOM or regular image
        if img_path.endswith('.dcm'):
            dicom = pydicom.dcmread(img_path)
            image = dicom.pixel_array.astype(np.float32)
            # Normalize to 0-255 range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            image = Image.fromarray(image).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')

        # Get labels (multi-label classification)
        labels = self.data_frame.iloc[idx, 1:].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)

class MedicalCNN(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(MedicalCNN, self).__init__()

        # Use DenseNet as backbone (good for medical images)
        self.backbone = torchvision.models.densenet121(pretrained=pretrained)

        # Replace classifier for multi-label classification
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # For multi-label classification
        )

    def forward(self, x):
        return self.backbone(x)

# Medical-specific data augmentation
medical_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

class MedicalTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()  # For multi-label
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}')

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())

        predictions = np.vstack(predictions)
        targets = np.vstack(targets)

        # Calculate AUC for each class
        from sklearn.metrics import roc_auc_score
        auc_scores = []
        for i in range(targets.shape[1]):
            if len(np.unique(targets[:, i])) > 1:  # Only if both classes present
                auc = roc_auc_score(targets[:, i], predictions[:, i])
                auc_scores.append(auc)

        mean_auc = np.mean(auc_scores) if auc_scores else 0

        return total_loss / len(dataloader), mean_auc

# Training function
def train_medical_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    train_dataset = MedicalImageDataset('train.csv', 'train_images/', medical_transforms)
    val_dataset = MedicalImageDataset('val.csv', 'val_images/', medical_transforms)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model and trainer
    model = MedicalCNN(num_classes=14)
    trainer = MedicalTrainer(model, device)

    best_auc = 0
    for epoch in range(50):
        print(f'Epoch {epoch+1}/50')

        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_auc = trainer.validate(val_loader)

        trainer.scheduler.step(val_loss)

        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_medical_model.pth')
            print(f'New best model saved with AUC: {best_auc:.4f}')

# Run training
train_medical_model()`,
    icon: "🏥",
    tags: ["Medical AI", "Medical Imaging", "Healthcare", "Radiology"],
    theoreticalConcepts: {
      fundamentals: [
        "DICOM Standard - Digital imaging format for medical images with metadata",
        "Transfer Learning - Adapt natural image models to medical domain",
        "Class Imbalance - Handle rare diseases with weighted losses and sampling",
        "Multi-label Classification - Detect multiple conditions simultaneously",
        "Medical Image Preprocessing - Windowing, normalization, and enhancement techniques",
      ],
      keyTheory:
        "Medical image analysis applies computer vision to healthcare, requiring domain expertise for proper preprocessing, validation, and clinical integration. Models must handle class imbalance and provide interpretable results for medical professionals.",
      mathematicalFoundations:
        "Weighted BCE Loss: L = -Σᵢ wᵢ[yᵢlog(p̂ᵢ) + (1-yᵢ)log(1-p̂ᵢ)]. AUC-ROC for diagnostic performance evaluation. Dice coefficient for segmentation: 2|A∩B|/(|A|+|B|)",
      importantPapers: [
        "ChexNet (2017) - Radiologist-level pneumonia detection",
        "Dermatologist-level classification (2017) - Skin cancer detection",
        "Diabetic retinopathy detection (2016) - Google's eye disease screening",
        "Medical Image Computing and Computer Assisted Intervention (MICCAI) - Annual conference",
      ],
      businessImpact:
        "Improves diagnostic accuracy by 20%+, reduces radiologist workload, enables screening in underserved areas, and accelerates drug discovery through medical image biomarkers worth billions.",
    },
  },
  {
    id: 14,
    title: "Edge AI Deployment",
    category: "Machine Learning",
    description:
      "Deploy AI models on edge devices like mobile phones, IoT devices, and embedded systems.",
    difficulty: "Advanced",
    timeToComplete: "4-6 weeks",
    useCases: [
      "Mobile app AI features",
      "IoT sensor data processing",
      "Autonomous vehicle perception",
      "Smart camera applications",
      "Industrial automation",
    ],
    trainingApproach:
      "Optimize models for edge deployment using quantization, pruning, and mobile-optimized architectures. Use frameworks like TensorFlow Lite and ONNX.",
    keySteps: [
      "Select and train appropriate model architecture",
      "Apply model optimization techniques",
      "Convert to edge deployment format",
      "Implement on-device inference pipeline",
      "Optimize for latency and power consumption",
      "Test performance on target hardware",
    ],
    resources: [
      {
        name: "TensorFlow Lite",
        type: "Framework",
        description: "Deploy ML on mobile and embedded devices",
        link: "https://www.tensorflow.org/lite",
        pricing: "Free",
      },
      {
        name: "NVIDIA Jetson",
        type: "Framework",
        description: "Edge AI computing platform",
        link: "https://developer.nvidia.com/embedded/jetson",
        pricing: "$99+ hardware",
      },
      {
        name: "Core ML",
        type: "Framework",
        description: "Apple's on-device machine learning",
        link: "https://developer.apple.com/machine-learning/core-ml/",
        pricing: "Free",
      },
    ],
    codeExample: `import tensorflow as tf
import numpy as np
import cv2
from tensorflow.lite.python import interpreter as tflite_interpreter

class EdgeAIOptimizer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def quantize_model(self, representative_dataset=None):
        """Quantize model to reduce size and improve inference speed"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Enable optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Integer quantization for further compression
        if representative_dataset:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        return tflite_model

    def prune_model(self, target_sparsity=0.5):
        """Prune model weights to reduce parameters"""
        import tensorflow_model_optimization as tfmot

        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

        # Define pruning parameters
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=0,
                end_step=1000
            )
        }

        # Apply pruning
        model_for_pruning = prune_low_magnitude(self.model, **pruning_params)
        model_for_pruning.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model_for_pruning

class EdgeInferenceEngine:
    def __init__(self, tflite_model_path):
        self.interpreter = tflite_interpreter.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']

    def preprocess_input(self, image):
        """Preprocess input for edge inference"""
        # Resize to model input size
        input_size = (self.input_shape[1], self.input_shape[2])
        image = cv2.resize(image, input_size)

        # Normalize based on model requirements
        if self.input_dtype == np.uint8:
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32) / 255.0
            image = (image - 0.5) * 2.0  # Normalize to [-1, 1]

        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, input_data):
        """Run inference on edge device"""
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output, inference_time

    def benchmark_performance(self, test_images, num_runs=100):
        """Benchmark model performance on target hardware"""
        inference_times = []

        for i in range(num_runs):
            # Random test image
            test_image = test_images[i % len(test_images)]
            preprocessed = self.preprocess_input(test_image)

            _, inference_time = self.predict(preprocessed)
            inference_times.append(inference_time)

        avg_time = np.mean(inference_times)
        fps = 1.0 / avg_time

        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'min_time': np.min(inference_times),
            'max_time': np.max(inference_times),
            'std_time': np.std(inference_times)
        }

# Mobile deployment example
class MobileAIApp:
    def __init__(self, model_path):
        self.engine = EdgeInferenceEngine(model_path)
        self.class_names = self.load_class_names()

    def load_class_names(self):
        # Load your class names
        return ['class1', 'class2', 'class3']  # Replace with actual classes

    def process_camera_frame(self, frame):
        """Process single camera frame"""
        # Preprocess
        input_data = self.engine.preprocess_input(frame)

        # Predict
        predictions, inference_time = self.engine.predict(input_data)

        # Post-process
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]

        # Draw results on frame
        result_frame = self.draw_results(frame, predicted_class, confidence, inference_time)

        return result_frame

    def draw_results(self, frame, predicted_class, confidence, inference_time):
        """Draw prediction results on frame"""
        h, w = frame.shape[:2]

        # Class name and confidence
        class_name = self.class_names[predicted_class]
        text = f"{class_name}: {confidence:.2f}"

        # Performance info
        fps = 1.0 / inference_time
        perf_text = f"FPS: {fps:.1f}"

        # Draw text
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, perf_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return frame

# Complete edge deployment pipeline
def deploy_to_edge():
    # 1. Load and optimize model
    optimizer = EdgeAIOptimizer('trained_model.h5')

    # Create representative dataset for quantization
    def representative_dataset():
        for _ in range(100):
            yield [np.random.random((1, 224, 224, 3)).astype(np.float32)]

    # Quantize model
    tflite_model = optimizer.quantize_model(representative_dataset)

    # Save optimized model
    with open('optimized_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model optimized and saved for edge deployment!")

    # 2. Test performance
    engine = EdgeInferenceEngine('optimized_model.tflite')

    # Create dummy test data
    test_images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]

    # Benchmark
    performance = engine.benchmark_performance(test_images)
    print(f"Performance: {performance}")

    return engine

# Run deployment
edge_engine = deploy_to_edge()`,
    icon: "📱",
    tags: ["Edge AI", "Mobile ML", "TensorFlow Lite", "Model Optimization"],
    theoreticalConcepts: {
      fundamentals: [
        "Model Quantization - Reduce precision from FP32 to INT8 for faster inference",
        "Neural Network Pruning - Remove unnecessary weights to reduce model size",
        "Knowledge Distillation - Train smaller student models from larger teachers",
        "Mobile-Optimized Architectures - MobileNets, EfficientNets designed for edge devices",
        "Hardware Acceleration - Leverage NPUs, GPUs, and specialized chips for AI inference",
      ],
      keyTheory:
        "Edge AI deployment optimizes models for resource-constrained devices through compression techniques while maintaining accuracy. The trade-off between model size, speed, and accuracy must be carefully balanced for each deployment scenario.",
      mathematicalFoundations:
        "Quantization: x_q = round(x/s) + z where s=scale, z=zero_point. Model compression ratio = original_size/compressed_size. Latency optimization: minimize E[inference_time] subject to accuracy constraints",
      importantPapers: [
        "MobileNets (2017) - Efficient CNNs for mobile vision",
        "Quantization and Training (2018) - Low-precision neural networks",
        "Neural Network Distillation (2015) - Knowledge transfer to smaller models",
        "EfficientNet (2019) - Scaling CNNs with compound coefficients",
      ],
      businessImpact:
        "Enables AI in billions of mobile devices, reduces cloud computing costs by 60%+, provides real-time responses with <100ms latency, and creates new edge AI markets worth hundreds of billions.",
    },
  },
];

export default function AIProjects() {
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [filterCategory, setFilterCategory] = useState<string>("All");
  const [filterDifficulty, setFilterDifficulty] = useState<string>("All");
  const [showCode, setShowCode] = useState<{ [key: number]: boolean }>({});

  const categories = [
    "All",
    "Computer Vision",
    "Natural Language Processing",
    "Machine Learning",
  ];
  const difficulties = ["All", "Beginner", "Intermediate", "Advanced"];

  const getFilteredProjects = () => {
    let filtered = [...projects];

    if (filterCategory !== "All") {
      filtered = filtered.filter(
        (project) => project.category === filterCategory,
      );
    }

    if (filterDifficulty !== "All") {
      filtered = filtered.filter(
        (project) => project.difficulty === filterDifficulty,
      );
    }

    return filtered;
  };

  const toggleCode = (projectId: number) => {
    setShowCode((prev) => ({
      ...prev,
      [projectId]: !prev[projectId],
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-900 via-indigo-900 to-cyan-900 relative overflow-hidden pt-20">
      {/* Animated Background Elements */}
      <div className="absolute inset-0">
        <div className="absolute top-10 left-20 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-60 right-10 w-80 h-80 bg-cyan-500/20 rounded-full blur-3xl animate-bounce"></div>
        <div className="absolute bottom-10 left-1/3 w-72 h-72 bg-purple-500/20 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-40 left-1/2 w-64 h-64 bg-emerald-500/20 rounded-full blur-3xl animate-bounce delay-700"></div>
        <div className="absolute bottom-40 right-1/3 w-88 h-88 bg-orange-500/20 rounded-full blur-3xl animate-pulse delay-500"></div>
      </div>

      <Navigation />

      <div className="container mx-auto px-6 py-12">
        {/* Header */}
        <div className="text-center mb-16 relative z-10">
          <motion.div
            className="inline-block p-1 rounded-full bg-gradient-to-r from-pink-400 via-purple-400 to-cyan-400 mb-8"
            initial={{ opacity: 0, scale: 0.5, rotateY: -180 }}
            animate={{ opacity: 1, scale: 1, rotateY: 0 }}
            transition={{ duration: 1.2, ease: "backOut" }}
          >
            <h1 className="text-6xl md:text-8xl font-black bg-gradient-to-r from-white via-purple-100 to-cyan-100 bg-clip-text text-transparent px-8 py-6">
              AI Projects Guide
            </h1>
          </motion.div>

          <motion.p
            className="text-xl text-gray-100 max-w-5xl mx-auto mb-8 leading-relaxed"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Your ultimate comprehensive guide to the most common AI projects 🚀
            Step-by-step training approaches, recommended APIs, pre-trained
            models, datasets, and production-ready code examples to launch your
            AI career! ✨
          </motion.p>

          <motion.div
            className="flex flex-wrap justify-center gap-4 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
              <span className="text-white font-bold">🎆 Trending Projects</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
              <span className="text-white font-bold">💻 Code Examples</span>
            </div>
            <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20">
              <span className="text-white font-bold">📊 Real Projects</span>
            </div>
          </motion.div>

          {/* Quick Stats */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-4 gap-6 max-w-6xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group"
              whileHover={{ y: -10, rotateY: 5, scale: 1.05 }}
              transition={{ duration: 0.3 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-pink-400 to-rose-500 bg-clip-text text-transparent">
                {projects.length}
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                AI Project Types
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group"
              whileHover={{ y: -10, rotateY: 5, scale: 1.05 }}
              transition={{ duration: 0.3 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-emerald-400 to-green-500 bg-clip-text text-transparent">
                40+
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                Use Cases Covered
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group"
              whileHover={{ y: -10, rotateY: 5, scale: 1.05 }}
              transition={{ duration: 0.3 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-purple-400 to-violet-500 bg-clip-text text-transparent">
                25+
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                APIs & Tools
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group"
              whileHover={{ y: -10, rotateY: 5, scale: 1.05 }}
              transition={{ duration: 0.3 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-orange-400 to-amber-500 bg-clip-text text-transparent">
                100%
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                Production Ready
              </div>
            </motion.div>
          </motion.div>
        </div>

        {/* Filters */}
        <div className="mb-12 space-y-6 relative z-10">
          {/* Clear All Filters Button */}
          {(filterCategory !== "All" || filterDifficulty !== "All") && (
            <motion.div
              className="text-center mb-6"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ duration: 0.3 }}
            >
              <motion.button
                onClick={() => {
                  setFilterCategory("All");
                  setFilterDifficulty("All");
                }}
                className="px-6 py-3 bg-gradient-to-r from-red-500 to-pink-600 text-white rounded-full font-bold hover:from-red-600 hover:to-pink-700 transition-all duration-300 shadow-xl border border-red-400/30"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
              >
                ✖️ Clear All Filters
              </motion.button>
            </motion.div>
          )}

          <div className="flex flex-wrap gap-4 justify-center">
            <span className="text-sm font-black text-white px-6 py-3 bg-white/10 backdrop-blur-md rounded-full border border-white/20">
              🎨 Filter by Category:
            </span>
            {categories.map((category) => (
              <motion.button
                key={category}
                onClick={() => setFilterCategory(category)}
                className={`px-6 py-3 rounded-full text-sm font-bold transition-all duration-300 ${
                  filterCategory === category
                    ? "bg-gradient-to-r from-pink-500 to-purple-600 text-white shadow-2xl scale-105 border border-pink-400/50"
                    : "bg-white/10 backdrop-blur-md text-white border border-white/20 hover:bg-white/20 hover:scale-105"
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {category}
              </motion.button>
            ))}
          </div>

          <div className="flex flex-wrap gap-4 justify-center">
            <span className="text-sm font-black text-white px-6 py-3 bg-white/10 backdrop-blur-md rounded-full border border-white/20">
              🏆 Filter by Difficulty:
            </span>
            {difficulties.map((difficulty) => {
              const difficultyEmojis = {
                All: "📊",
                Beginner: "🌱",
                Intermediate: "🔥",
                Advanced: "🚀",
              };
              return (
                <motion.button
                  key={difficulty}
                  onClick={() => setFilterDifficulty(difficulty)}
                  className={`px-6 py-3 rounded-full text-sm font-bold transition-all duration-300 ${
                    filterDifficulty === difficulty
                      ? "bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-2xl scale-105 border border-cyan-400/50"
                      : "bg-white/10 backdrop-blur-md text-white border border-white/20 hover:bg-white/20 hover:scale-105"
                  }`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {difficultyEmojis[difficulty]} {difficulty}
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Results Count */}
        <motion.div
          className="text-center mb-8 relative z-10"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="bg-white/10 backdrop-blur-md rounded-full px-6 py-3 border border-white/20 inline-block">
            <span className="text-white font-bold">
              🎯 Showing {getFilteredProjects().length} of {projects.length}{" "}
              projects
              {filterCategory !== "All" && ` in ${filterCategory}`}
              {filterDifficulty !== "All" && ` (${filterDifficulty} level)`}
            </span>
          </div>
        </motion.div>

        {/* Projects Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-12 relative z-10">
          {getFilteredProjects().length > 0 ? (
            getFilteredProjects().map((project, index) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 50, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{
                  duration: 0.6,
                  delay: index * 0.15,
                  ease: "backOut",
                }}
                className={`bg-white/10 backdrop-blur-xl rounded-3xl border overflow-hidden shadow-2xl group ${
                  showCode[project.id]
                    ? "border-cyan-400/50 shadow-cyan-500/20"
                    : "border-white/20"
                }`}
                whileHover={{
                  scale: 1.05,
                  rotateY: 3,
                  rotateX: 3,
                }}
              >
                <div className="p-8">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-6">
                    <div className="flex items-center gap-4">
                      <motion.div
                        className="text-5xl group-hover:scale-110 transition-transform duration-300 relative"
                        whileHover={{ rotate: 10, scale: 1.2 }}
                      >
                        {project.icon}
                        {showCode[project.id] && (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className="absolute -top-2 -right-2 w-5 h-5 bg-cyan-500 rounded-full flex items-center justify-center"
                          >
                            <span className="text-xs">💻</span>
                          </motion.div>
                        )}
                      </motion.div>
                      <div>
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="text-2xl font-black text-white group-hover:bg-gradient-to-r group-hover:from-pink-400 group-hover:to-purple-500 group-hover:bg-clip-text group-hover:text-transparent transition-all duration-300">
                            {project.title}
                          </h3>
                          {showCode[project.id] && (
                            <motion.span
                              initial={{ opacity: 0, scale: 0 }}
                              animate={{ opacity: 1, scale: 1 }}
                              className="text-xs bg-cyan-500/30 text-cyan-200 px-2 py-1 rounded-full font-bold border border-cyan-400/50"
                            >
                              CODE ACTIVE
                            </motion.span>
                          )}
                        </div>
                        <p className="text-sm text-gray-300 font-medium bg-white/10 rounded-full px-3 py-1 inline-block">
                          {project.category}
                        </p>
                      </div>
                    </div>
                    <div className="flex flex-col gap-3">
                      <motion.span
                        className={`px-4 py-2 rounded-full text-xs font-bold border-2 backdrop-blur-md ${
                          project.difficulty === "Beginner"
                            ? "bg-green-500/30 border-green-400/50 text-green-200"
                            : project.difficulty === "Intermediate"
                              ? "bg-orange-500/30 border-orange-400/50 text-orange-200"
                              : "bg-red-500/30 border-red-400/50 text-red-200"
                        }`}
                        whileHover={{ scale: 1.1 }}
                      >
                        {project.difficulty === "Beginner"
                          ? "🌱"
                          : project.difficulty === "Intermediate"
                            ? "🔥"
                            : "🚀"}{" "}
                        {project.difficulty}
                      </motion.span>
                      <span className="px-4 py-2 bg-blue-500/30 border-2 border-blue-400/50 text-blue-200 rounded-full text-xs font-bold backdrop-blur-md">
                        ⏱️ {project.timeToComplete}
                      </span>
                    </div>
                  </div>

                  {/* Description */}
                  <p className="text-gray-100 mb-6 text-lg leading-relaxed">
                    {project.description}
                  </p>

                  {/* Use Cases */}
                  <div className="mb-6">
                    <h4 className="font-bold text-cyan-300 mb-3 text-lg">
                      🎯 Common Use Cases:
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {project.useCases.slice(0, 3).map((useCase, idx) => (
                        <motion.span
                          key={idx}
                          className="text-xs bg-white/20 text-gray-200 px-3 py-2 rounded-full font-medium border border-white/30 backdrop-blur-md"
                          whileHover={{ scale: 1.05 }}
                        >
                          {useCase}
                        </motion.span>
                      ))}
                      {project.useCases.length > 3 && (
                        <span className="text-xs text-gray-300 font-medium bg-white/10 rounded-full px-3 py-2">
                          +{project.useCases.length - 3} more
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Training Approach */}
                  <div className="mb-6">
                    <h4 className="font-bold text-purple-300 mb-3 text-lg">
                      🧠 Training Approach:
                    </h4>
                    <p className="text-sm text-gray-200 leading-relaxed bg-white/10 backdrop-blur-md rounded-2xl p-4 border border-white/20">
                      {project.trainingApproach}
                    </p>
                  </div>

                  {/* Tags */}
                  <div className="mb-6">
                    <div className="flex flex-wrap gap-2">
                      {project.tags.map((tag, idx) => (
                        <motion.span
                          key={idx}
                          className="text-xs bg-gradient-to-r from-blue-500/30 to-purple-500/30 text-blue-200 px-3 py-2 rounded-full border border-blue-400/50 font-bold backdrop-blur-md"
                          whileHover={{ scale: 1.1, y: -2 }}
                        >
                          {tag}
                        </motion.span>
                      ))}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-4">
                    <motion.button
                      onClick={() => setSelectedProject(project)}
                      className="flex-1 bg-gradient-to-r from-pink-500 to-purple-600 text-white px-6 py-3 rounded-2xl font-bold text-sm hover:from-pink-600 hover:to-purple-700 transition-all duration-300 border border-pink-400/30 shadow-xl relative overflow-hidden group"
                      whileHover={{ scale: 1.02, y: -2 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <span className="relative z-10">🚀 Full Guide</span>
                      <motion.div
                        className="absolute inset-0 bg-gradient-to-r from-pink-600 to-purple-700"
                        initial={{ scaleX: 0, opacity: 0 }}
                        whileHover={{ scaleX: 1, opacity: 1 }}
                        style={{ transformOrigin: "left" }}
                        transition={{ duration: 0.3 }}
                      />
                    </motion.button>
                    <motion.button
                      onClick={() => toggleCode(project.id)}
                      className={`bg-white/10 backdrop-blur-md text-white px-6 py-3 rounded-2xl font-bold text-sm border border-white/20 hover:bg-white/20 transition-all duration-300 relative overflow-hidden group ${
                        showCode[project.id]
                          ? "ring-2 ring-cyan-400/50 bg-cyan-500/20"
                          : ""
                      }`}
                      whileHover={{ scale: 1.02, y: -2 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <span className="relative z-10">
                        {showCode[project.id] ? "🙈 Hide Code" : "💻 Show Code"}
                      </span>
                      <motion.div
                        className="absolute inset-0 bg-white/10"
                        initial={{ opacity: 0 }}
                        whileHover={{ opacity: 1 }}
                        transition={{ duration: 0.3 }}
                      />
                    </motion.button>
                  </div>

                  {/* Code Example */}
                  <AnimatePresence>
                    {showCode[project.id] && (
                      <motion.div
                        initial={{ opacity: 0, height: 0, y: -20 }}
                        animate={{ opacity: 1, height: "auto", y: 0 }}
                        exit={{ opacity: 0, height: 0, y: -20 }}
                        transition={{ duration: 0.5, ease: "easeOut" }}
                        className="mt-6"
                      >
                        <div className="bg-black/60 backdrop-blur-xl text-gray-100 p-6 rounded-2xl overflow-x-auto border border-white/20 shadow-2xl">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                            <span className="text-gray-400 text-sm font-mono ml-4">
                              {project.title.toLowerCase().replace(/\s+/g, "_")}
                              .py
                            </span>
                          </div>
                          <pre className="text-sm leading-relaxed">
                            <code className="language-python">
                              {project.codeExample}
                            </code>
                          </pre>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </motion.div>
            ))
          ) : (
            // No results state
            <motion.div
              className="col-span-full text-center py-20"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6 }}
            >
              <div className="bg-white/10 backdrop-blur-xl rounded-3xl p-12 border border-white/20 max-w-2xl mx-auto">
                <motion.div
                  className="text-8xl mb-6"
                  animate={{ rotate: [0, 10, -10, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  🔍
                </motion.div>
                <h3 className="text-3xl font-black text-white mb-4">
                  No Projects Found
                </h3>
                <p className="text-gray-200 text-lg mb-8">
                  No projects match your current filters. Try adjusting your
                  category or difficulty selection.
                </p>
                <div className="flex flex-wrap gap-4 justify-center">
                  <motion.button
                    onClick={() => setFilterCategory("All")}
                    className="px-6 py-3 bg-gradient-to-r from-pink-500 to-purple-600 text-white rounded-full font-bold hover:from-pink-600 hover:to-purple-700 transition-all duration-300"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Reset Category
                  </motion.button>
                  <motion.button
                    onClick={() => setFilterDifficulty("All")}
                    className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-full font-bold hover:from-cyan-600 hover:to-blue-700 transition-all duration-300"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Reset Difficulty
                  </motion.button>
                  <motion.button
                    onClick={() => {
                      setFilterCategory("All");
                      setFilterDifficulty("All");
                    }}
                    className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-teal-600 text-white rounded-full font-bold hover:from-emerald-600 hover:to-teal-700 transition-all duration-300"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Reset All Filters
                  </motion.button>
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Detailed Modal */}
        <AnimatePresence>
          {selectedProject && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50"
              onClick={() => setSelectedProject(null)}
            >
              <motion.div
                initial={{ scale: 0.8, opacity: 0, rotateY: -30 }}
                animate={{ scale: 1, opacity: 1, rotateY: 0 }}
                exit={{ scale: 0.8, opacity: 0, rotateY: 30 }}
                transition={{ type: "spring", bounce: 0.3, duration: 0.6 }}
                className="bg-gradient-to-br from-violet-900/95 via-indigo-900/95 to-cyan-900/95 backdrop-blur-xl rounded-3xl max-w-7xl w-full max-h-[90vh] overflow-y-auto border border-white/20 shadow-2xl"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="p-10">
                  {/* Header */}
                  <div className="flex justify-between items-start mb-10">
                    <div className="flex items-center gap-6">
                      <motion.div
                        className="text-8xl"
                        initial={{ scale: 0, rotate: -180 }}
                        animate={{ scale: 1, rotate: 0 }}
                        transition={{ delay: 0.2, type: "spring", bounce: 0.6 }}
                      >
                        {selectedProject.icon}
                      </motion.div>
                      <div>
                        <motion.h2
                          className="text-5xl font-black bg-gradient-to-r from-white via-pink-200 to-cyan-200 bg-clip-text text-transparent mb-4"
                          initial={{ opacity: 0, x: -50 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: 0.3 }}
                        >
                          {selectedProject.title}
                        </motion.h2>
                        <div className="flex flex-wrap items-center gap-4">
                          <span className="text-gray-200 bg-white/10 backdrop-blur-md rounded-full px-4 py-2 font-bold">
                            {selectedProject.category}
                          </span>
                          <motion.span
                            className={`px-4 py-2 rounded-full text-sm font-bold border-2 backdrop-blur-md ${
                              selectedProject.difficulty === "Beginner"
                                ? "bg-green-500/30 border-green-400/50 text-green-200"
                                : selectedProject.difficulty === "Intermediate"
                                  ? "bg-orange-500/30 border-orange-400/50 text-orange-200"
                                  : "bg-red-500/30 border-red-400/50 text-red-200"
                            }`}
                            initial={{ opacity: 0, scale: 0 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: 0.4 }}
                          >
                            {selectedProject.difficulty === "Beginner"
                              ? "🌱"
                              : selectedProject.difficulty === "Intermediate"
                                ? "🔥"
                                : "🚀"}{" "}
                            {selectedProject.difficulty}
                          </motion.span>
                          <span className="px-4 py-2 bg-blue-500/30 border-2 border-blue-400/50 text-blue-200 rounded-full text-sm font-bold backdrop-blur-md">
                            ⏱️ {selectedProject.timeToComplete}
                          </span>
                        </div>
                      </div>
                    </div>
                    <motion.button
                      onClick={() => setSelectedProject(null)}
                      className="text-white hover:text-red-400 text-4xl font-bold bg-white/10 backdrop-blur-md rounded-full w-14 h-14 flex items-center justify-center border border-white/20 hover:bg-red-500/20 transition-all duration-300"
                      whileHover={{ scale: 1.1, rotate: 90 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      ×
                    </motion.button>
                  </div>

                  {/* Content Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                    {/* Left Column */}
                    <div className="space-y-8">
                      {/* Description */}
                      <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.5 }}
                      >
                        <h3 className="text-2xl font-black text-white mb-4">
                          💬 Project Overview
                        </h3>
                        <p className="text-gray-200 text-lg leading-relaxed bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                          {selectedProject.description}
                        </p>
                      </motion.div>

                      {/* Use Cases */}
                      <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.6 }}
                      >
                        <h3 className="text-2xl font-black text-white mb-4">
                          🎯 Use Cases
                        </h3>
                        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                          <ul className="space-y-3">
                            {selectedProject.useCases.map((useCase, idx) => (
                              <motion.li
                                key={idx}
                                className="flex items-start gap-3"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.7 + idx * 0.1 }}
                              >
                                <span className="text-pink-400 mt-1 text-lg">
                                  •
                                </span>
                                <span className="text-gray-200">{useCase}</span>
                              </motion.li>
                            ))}
                          </ul>
                        </div>
                      </motion.div>

                      {/* Training Approach */}
                      <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.8 }}
                      >
                        <h3 className="text-2xl font-black text-white mb-4">
                          🧠 Training Approach
                        </h3>
                        <p className="text-gray-200 leading-relaxed bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                          {selectedProject.trainingApproach}
                        </p>
                      </motion.div>

                      {/* Theoretical Concepts */}
                      <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.85 }}
                      >
                        <h3 className="text-2xl font-black text-white mb-4">
                          🎓 Key Theory & Concepts
                        </h3>
                        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20 space-y-6">
                          {/* Core Theory */}
                          <div>
                            <h4 className="text-lg font-bold text-cyan-300 mb-3">
                              💡 Core Theory
                            </h4>
                            <p className="text-gray-200 leading-relaxed text-sm">
                              {selectedProject.theoreticalConcepts.keyTheory}
                            </p>
                          </div>

                          {/* Fundamentals */}
                          <div>
                            <h4 className="text-lg font-bold text-purple-300 mb-3">
                              🔬 Key Fundamentals
                            </h4>
                            <ul className="space-y-2">
                              {selectedProject.theoreticalConcepts.fundamentals.map(
                                (concept, idx) => (
                                  <li
                                    key={idx}
                                    className="flex items-start gap-2 text-sm text-gray-200"
                                  >
                                    <span className="text-pink-400 mt-1 flex-shrink-0">
                                      •
                                    </span>
                                    <span>{concept}</span>
                                  </li>
                                ),
                              )}
                            </ul>
                          </div>

                          {/* Mathematical Foundations */}
                          <div>
                            <h4 className="text-lg font-bold text-yellow-300 mb-3">
                              🧮 Mathematical Foundations
                            </h4>
                            <p className="text-gray-200 text-sm font-mono bg-black/40 p-4 rounded-xl">
                              {
                                selectedProject.theoreticalConcepts
                                  .mathematicalFoundations
                              }
                            </p>
                          </div>

                          {/* Business Impact */}
                          <div>
                            <h4 className="text-lg font-bold text-green-300 mb-3">
                              💼 Business Impact
                            </h4>
                            <p className="text-gray-200 leading-relaxed text-sm">
                              {
                                selectedProject.theoreticalConcepts
                                  .businessImpact
                              }
                            </p>
                          </div>

                          {/* Important Papers */}
                          <div>
                            <h4 className="text-lg font-bold text-orange-300 mb-3">
                              📚 Important Papers
                            </h4>
                            <ul className="space-y-2">
                              {selectedProject.theoreticalConcepts.importantPapers.map(
                                (paper, idx) => (
                                  <li
                                    key={idx}
                                    className="flex items-start gap-2 text-sm text-gray-200"
                                  >
                                    <span className="text-orange-400 mt-1 flex-shrink-0">
                                      📄
                                    </span>
                                    <span>{paper}</span>
                                  </li>
                                ),
                              )}
                            </ul>
                          </div>
                        </div>
                      </motion.div>

                      {/* Key Steps */}
                      <motion.div
                        initial={{ opacity: 0, x: -30 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.9 }}
                      >
                        <h3 className="text-2xl font-black text-white mb-4">
                          🚀 Implementation Steps
                        </h3>
                        <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 border border-white/20">
                          <ol className="space-y-4">
                            {selectedProject.keySteps.map((step, idx) => (
                              <motion.li
                                key={idx}
                                className="flex items-start gap-4"
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 1.0 + idx * 0.1 }}
                              >
                                <span className="w-8 h-8 bg-gradient-to-r from-pink-500 to-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold border border-pink-400/50">
                                  {idx + 1}
                                </span>
                                <span className="text-gray-200 leading-relaxed">
                                  {step}
                                </span>
                              </motion.li>
                            ))}
                          </ol>
                        </div>
                      </motion.div>
                    </div>

                    {/* Right Column */}
                    <div className="space-y-8">
                      {/* Resources */}
                      <motion.div
                        initial={{ opacity: 0, x: 30 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.5 }}
                      >
                        <h3 className="text-2xl font-black text-white mb-6">
                          🛠️ Recommended Resources
                        </h3>
                        <div className="space-y-6">
                          {selectedProject.resources.map((resource, idx) => (
                            <motion.div
                              key={idx}
                              className="bg-white/10 backdrop-blur-xl p-6 rounded-2xl border border-white/20 hover:bg-white/15 transition-all duration-300 group"
                              initial={{ opacity: 0, y: 20 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: 0.6 + idx * 0.1 }}
                              whileHover={{ scale: 1.02, y: -5 }}
                            >
                              <div className="flex justify-between items-start mb-4">
                                <h4 className="font-black text-white text-lg group-hover:text-cyan-300 transition-colors">
                                  {resource.name}
                                </h4>
                                <span
                                  className={`px-3 py-2 rounded-full text-xs font-bold border backdrop-blur-md ${
                                    resource.type === "API"
                                      ? "bg-blue-500/30 border-blue-400/50 text-blue-200"
                                      : resource.type === "Dataset"
                                        ? "bg-green-500/30 border-green-400/50 text-green-200"
                                        : resource.type === "Model"
                                          ? "bg-purple-500/30 border-purple-400/50 text-purple-200"
                                          : resource.type === "Framework"
                                            ? "bg-orange-500/30 border-orange-400/50 text-orange-200"
                                            : resource.type === "Company"
                                              ? "bg-pink-500/30 border-pink-400/50 text-pink-200"
                                              : resource.type === "Research Lab"
                                                ? "bg-cyan-500/30 border-cyan-400/50 text-cyan-200"
                                                : resource.type === "Paper"
                                                  ? "bg-yellow-500/30 border-yellow-400/50 text-yellow-200"
                                                  : "bg-gray-500/30 border-gray-400/50 text-gray-200"
                                  }`}
                                >
                                  {resource.type}
                                </span>
                              </div>
                              <p className="text-gray-200 text-sm mb-4 leading-relaxed">
                                {resource.description}
                              </p>
                              <div className="flex justify-between items-center pt-4 border-t border-white/20">
                                <span className="text-green-400 font-bold text-lg">
                                  {resource.pricing}
                                </span>
                                <motion.a
                                  href={resource.link}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-cyan-300 hover:text-cyan-100 font-bold transition-colors text-sm"
                                  whileHover={{ scale: 1.1 }}
                                >
                                  Access Resource →
                                </motion.a>
                              </div>
                            </motion.div>
                          ))}
                        </div>
                      </motion.div>

                      {/* Code Example */}
                      <motion.div
                        initial={{ opacity: 0, x: 30 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.9 }}
                      >
                        <h3 className="text-2xl font-black text-white mb-6">
                          💻 Code Example
                        </h3>
                        <div className="bg-black/60 backdrop-blur-xl text-gray-100 p-6 rounded-2xl overflow-x-auto border border-white/20 shadow-2xl">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                            <span className="text-gray-400 text-sm font-mono ml-4">
                              {selectedProject.title
                                .toLowerCase()
                                .replace(/\s+/g, "_")}
                              .py
                            </span>
                          </div>
                          <pre className="text-sm leading-relaxed">
                            <code className="language-python">
                              {selectedProject.codeExample}
                            </code>
                          </pre>
                        </div>
                      </motion.div>
                    </div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Bottom Navigation */}
        <div className="text-center relative z-10">
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Link
              to="/"
              className="inline-block bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-bold px-8 py-4 rounded-full hover:from-cyan-600 hover:to-blue-700 transition-all duration-300 shadow-2xl border border-white/20 backdrop-blur-md"
            >
              ← Back to Portfolio
            </Link>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
