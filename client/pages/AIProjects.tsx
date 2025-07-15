import React, { useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import Navigation from "../components/Navigation";

interface Resource {
  name: string;
  type: "API" | "Dataset" | "Model" | "Framework";
  description: string;
  link: string;
  pricing: string;
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
      },
      {
        name: "ImageNet Dataset",
        type: "Dataset",
        description: "1.2M labeled images across 1000 categories",
        link: "https://www.image-net.org/",
        pricing: "Free",
      },
      {
        name: "PyTorch Vision Models",
        type: "Model",
        description: "Pre-trained models: ResNet, VGG, EfficientNet",
        link: "https://pytorch.org/vision/stable/models.html",
        pricing: "Free",
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
      },
      {
        name: "Hugging Face Transformers",
        type: "Framework",
        description: "Open-source library for transformer models",
        link: "https://huggingface.co/transformers/",
        pricing: "Free",
      },
      {
        name: "PersonaChat Dataset",
        type: "Dataset",
        description: "Conversations with personality traits",
        link: "https://huggingface.co/datasets/persona_chat",
        pricing: "Free",
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
      },
      {
        name: "MovieLens Dataset",
        type: "Dataset",
        description: "Movie ratings from 100K+ users",
        link: "https://grouplens.org/datasets/movielens/",
        pricing: "Free",
      },
      {
        name: "Surprise Library",
        type: "Framework",
        description: "Python scikit for recommender systems",
        link: "https://surprise.readthedocs.io/",
        pricing: "Free",
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
      },
      {
        name: "COCO Dataset",
        type: "Dataset",
        description: "330K images with 80 object categories",
        link: "https://cocodataset.org/",
        pricing: "Free",
      },
      {
        name: "Detectron2",
        type: "Framework",
        description: "Facebook's detection and segmentation platform",
        link: "https://detectron2.readthedocs.io/",
        pricing: "Free",
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
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group hover:scale-105 transition-all duration-300"
              whileHover={{ y: -10, rotateY: 5 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-pink-400 to-rose-500 bg-clip-text text-transparent">
                {projects.length}
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                AI Project Types
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group hover:scale-105 transition-all duration-300"
              whileHover={{ y: -10, rotateY: 5 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-emerald-400 to-green-500 bg-clip-text text-transparent">
                40+
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                Use Cases Covered
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group hover:scale-105 transition-all duration-300"
              whileHover={{ y: -10, rotateY: 5 }}
            >
              <div className="text-5xl font-black bg-gradient-to-r from-purple-400 to-violet-500 bg-clip-text text-transparent">
                25+
              </div>
              <div className="text-sm text-gray-200 font-bold mt-2">
                APIs & Tools
              </div>
            </motion.div>
            <motion.div
              className="bg-white/10 backdrop-blur-xl p-8 rounded-3xl border border-white/20 shadow-2xl group hover:scale-105 transition-all duration-300"
              whileHover={{ y: -10, rotateY: 5 }}
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

        {/* Projects Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 mb-12 relative z-10">
          {getFilteredProjects().map((project, index) => (
            <motion.div
              key={project.id}
              initial={{ opacity: 0, y: 50, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{
                duration: 0.6,
                delay: index * 0.15,
                ease: "backOut",
              }}
              className={`bg-white/10 backdrop-blur-xl rounded-3xl border overflow-hidden hover:scale-105 transition-all duration-500 shadow-2xl hover:shadow-purple-500/25 group ${
                showCode[project.id]
                  ? "border-cyan-400/50 shadow-cyan-500/20"
                  : "border-white/20"
              }`}
              whileHover={{
                scale: 1.02,
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
          ))}
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
                                          : "bg-orange-500/30 border-orange-400/50 text-orange-200"
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
