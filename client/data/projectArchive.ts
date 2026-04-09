export interface ProjectResource {
  name: string;
  type: "Official docs" | "Framework" | "Dataset" | "Paper" | "Papers with Code" | "API";
  url: string;
  note: string;
}

export interface ProjectPaper {
  title: string;
  url: string;
}

export interface Project {
  id: number;
  title: string;
  category: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  timeToComplete: string;
  summary: string;
  buildNow: string;
  useCases: string[];
  recommendedStack: string[];
  buildSteps: string[];
  resources: ProjectResource[];
  keyPapers: ProjectPaper[];
  codeExample: string;
  icon: string;
  tags: string[];
}

export const projects: Project[] = [
  {
    id: 1,
    title: "Image Classification",
    category: "Computer Vision",
    difficulty: "Beginner",
    timeToComplete: "1-2 weeks",
    summary: "Train or fine-tune a vision model to classify images into a closed label set.",
    buildNow: "Use transfer learning with modern vision backbones such as ConvNeXt, ViT, or DINOv2 embeddings rather than training CNNs from scratch.",
    useCases: [
      "Visual quality control",
      "Medical triage support",
      "Retail catalog tagging",
      "Content moderation",
    ],
    recommendedStack: ["PyTorch", "timm", "Weights & Biases", "Hugging Face Datasets"],
    buildSteps: [
      "Start with a benchmark-ready dataset split and clear label taxonomy.",
      "Fine-tune a pretrained backbone instead of building a network from zero.",
      "Track confusion matrices, per-class recall, and calibration quality.",
      "Export an ONNX or TorchScript artifact for deployment testing.",
    ],
    resources: [
      {
        name: "Papers with Code: Image Classification",
        type: "Papers with Code",
        url: "https://paperswithcode.com/task/image-classification",
        note: "Current benchmark hub and model landscape.",
      },
      {
        name: "timm",
        type: "Framework",
        url: "https://huggingface.co/docs/timm/index",
        note: "Modern pretrained vision backbones in one library.",
      },
      {
        name: "ImageNet",
        type: "Dataset",
        url: "https://www.image-net.org/",
        note: "Canonical large-scale image-classification dataset and taxonomy.",
      },
      {
        name: "PyTorch vision models",
        type: "Official docs",
        url: "https://pytorch.org/vision/stable/models.html",
        note: "Official docs for pretrained production-ready baselines.",
      },
    ],
    keyPapers: [
      { title: "ResNet", url: "https://arxiv.org/abs/1512.03385" },
      { title: "EfficientNet", url: "https://arxiv.org/abs/1905.11946" },
      { title: "Vision Transformer", url: "https://arxiv.org/abs/2010.11929" },
    ],
    codeExample: `from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

dataset = load_dataset("beans")
checkpoint = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(checkpoint)
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(dataset["train"].features["labels"].names),
    ignore_mismatched_sizes=True,
)

args = TrainingArguments("vit-beans", per_device_train_batch_size=16, num_train_epochs=3)
trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"])
trainer.train()`,
    icon: "📸",
    tags: ["ViT", "Transfer Learning", "PyTorch", "Classification"],
  },
  {
    id: 2,
    title: "Chatbot Development",
    category: "Natural Language Processing",
    difficulty: "Intermediate",
    timeToComplete: "2-4 weeks",
    summary: "Build a grounded conversational assistant for a specific workflow rather than a generic chatbot.",
    buildNow: "Use retrieval, tool calling, and evals before considering fine-tuning; most teams win by improving grounding and orchestration first.",
    useCases: [
      "Internal help desk",
      "Customer support",
      "Knowledge assistants",
      "Sales enablement",
    ],
    recommendedStack: ["OpenAI or Anthropic API", "LangGraph", "vector DB", "guardrails and evals"],
    buildSteps: [
      "Define the exact user jobs and escalation boundaries.",
      "Add retrieval over trusted documents with permissions-aware sources.",
      "Design tool calls for actions such as ticket creation or CRM lookup.",
      "Evaluate hallucination rate, refusal quality, and fallback behavior.",
    ],
    resources: [
      {
        name: "OpenAI text-generation guide",
        type: "Official docs",
        url: "https://platform.openai.com/docs/guides/text",
        note: "Current chat and response patterns for production assistants.",
      },
      {
        name: "Anthropic tool use",
        type: "Official docs",
        url: "https://docs.anthropic.com/en/docs/build-with-claude/tool-use",
        note: "Reliable tool-calling reference for Claude-based assistants.",
      },
      {
        name: "LangGraph",
        type: "Framework",
        url: "https://langchain-ai.github.io/langgraph/",
        note: "Stateful orchestration for multi-step conversational agents.",
      },
      {
        name: "RAG overview",
        type: "Paper",
        url: "https://arxiv.org/abs/2005.11401",
        note: "Foundational paper for retrieval-augmented generation.",
      },
    ],
    keyPapers: [
      { title: "Retrieval-Augmented Generation", url: "https://arxiv.org/abs/2005.11401" },
      { title: "Toolformer", url: "https://arxiv.org/abs/2302.04761" },
      { title: "ReAct", url: "https://arxiv.org/abs/2210.03629" },
    ],
    codeExample: `from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-4.1",
    input="Answer using the help-center knowledge base and cite the article slug.",
    tools=[{"type": "file_search"}],
)
print(response.output_text)`,
    icon: "💬",
    tags: ["RAG", "Tool Use", "Evals", "Agents"],
  },
  {
    id: 3,
    title: "Recommendation System",
    category: "Machine Learning",
    difficulty: "Intermediate",
    timeToComplete: "3-5 weeks",
    summary: "Rank items for each user with a retrieval-and-reranking pipeline instead of a single static scoring model.",
    buildNow: "Modern recommendation stacks mix embeddings, candidate retrieval, reranking, and online feedback rather than only matrix factorization.",
    useCases: [
      "Product recommendations",
      "Content feeds",
      "Creator discovery",
      "Learning-path personalization",
    ],
    recommendedStack: ["PyTorch", "implicit or TorchRec", "feature store", "bandit or online eval layer"],
    buildSteps: [
      "Define the target action: click, watch time, purchase, or retention.",
      "Build candidate generation with embeddings or collaborative filtering.",
      "Add a reranker with richer user, item, and session features.",
      "Measure offline ranking metrics and online business lift separately.",
    ],
    resources: [
      {
        name: "TorchRec",
        type: "Framework",
        url: "https://pytorch.org/torchrec/",
        note: "Meta-backed recommendation framework for large-scale ranking systems.",
      },
      {
        name: "RecBole",
        type: "Framework",
        url: "https://recbole.io/",
        note: "Open-source experimentation framework for recommendation research.",
      },
      {
        name: "MovieLens",
        type: "Dataset",
        url: "https://grouplens.org/datasets/movielens/",
        note: "Classic public dataset for starting offline recommendation experiments.",
      },
      {
        name: "Two-Tower recommendation systems",
        type: "Paper",
        url: "https://arxiv.org/abs/2003.11084",
        note: "Useful retrieval architecture pattern for modern recsys pipelines.",
      },
    ],
    keyPapers: [
      { title: "Wide & Deep Learning", url: "https://arxiv.org/abs/1606.07792" },
      { title: "DeepFM", url: "https://arxiv.org/abs/1703.04247" },
      { title: "Two-Tower Models", url: "https://arxiv.org/abs/2003.11084" },
    ],
    codeExample: `import torch
import torch.nn.functional as F

user_emb = torch.nn.Embedding(num_users, 128)
item_emb = torch.nn.Embedding(num_items, 128)

def score(user_ids, item_ids):
    return F.cosine_similarity(user_emb(user_ids), item_emb(item_ids))
`,
    icon: "🎯",
    tags: ["Ranking", "Embeddings", "Retrieval", "Personalization"],
  },
  {
    id: 4,
    title: "Object Detection",
    category: "Computer Vision",
    difficulty: "Intermediate",
    timeToComplete: "2-4 weeks",
    summary: "Detect and localize objects in images or video with modern detection architectures.",
    buildNow: "Start with YOLOv10/11-class tooling or DETR-style models depending on whether latency or flexibility matters more.",
    useCases: [
      "Warehouse automation",
      "Traffic analytics",
      "Retail shelf monitoring",
      "Manufacturing inspection",
    ],
    recommendedStack: ["Ultralytics", "PyTorch", "Roboflow", "FiftyOne"],
    buildSteps: [
      "Collect high-quality bounding boxes with clear class definitions.",
      "Benchmark a fast one-stage model before trying heavier detectors.",
      "Measure mAP along with false positives in your real deployment scenario.",
      "Test on video streams and edge hardware early if latency matters.",
    ],
    resources: [
      {
        name: "Ultralytics docs",
        type: "Official docs",
        url: "https://docs.ultralytics.com/",
        note: "Practical end-to-end docs for current YOLO-based workflows.",
      },
      {
        name: "Papers with Code: Object Detection",
        type: "Papers with Code",
        url: "https://paperswithcode.com/task/object-detection",
        note: "Benchmarks and current detector families.",
      },
      {
        name: "COCO",
        type: "Dataset",
        url: "https://cocodataset.org/#home",
        note: "Standard dataset for detection baselines and evaluation.",
      },
      {
        name: "DETR",
        type: "Paper",
        url: "https://arxiv.org/abs/2005.12872",
        note: "Transformer-based detection line that changed the design space.",
      },
    ],
    keyPapers: [
      { title: "YOLOv1", url: "https://arxiv.org/abs/1506.02640" },
      { title: "DETR", url: "https://arxiv.org/abs/2005.12872" },
      { title: "DINO-DETR", url: "https://arxiv.org/abs/2203.03605" },
    ],
    codeExample: `from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=20, imgsz=640)
results = model("warehouse.jpg")
print(results[0].boxes.xyxy)`,
    icon: "📦",
    tags: ["YOLO", "DETR", "Detection", "CV"],
  },
  {
    id: 5,
    title: "Sentiment Analysis",
    category: "Natural Language Processing",
    difficulty: "Beginner",
    timeToComplete: "1 week",
    summary: "Classify text polarity or opinion signals with encoder models and robust labeling practices.",
    buildNow: "Use compact transformer encoders and domain-specific evaluation rather than generic social-media baselines only.",
    useCases: [
      "Review monitoring",
      "Support-ticket triage",
      "Social listening",
      "Voice-of-customer analytics",
    ],
    recommendedStack: ["Transformers", "Weights & Biases", "label studio", "sentencepiece"],
    buildSteps: [
      "Define sentiment taxonomy clearly, especially for neutral and mixed cases.",
      "Use a domain-relevant training set instead of generic movie reviews alone.",
      "Track macro-F1 and per-class confusion, not just overall accuracy.",
      "Add a calibration pass before using outputs for automation.",
    ],
    resources: [
      {
        name: "Papers with Code: Sentiment Analysis",
        type: "Papers with Code",
        url: "https://paperswithcode.com/task/sentiment-analysis",
        note: "Task hub for datasets and benchmark models.",
      },
      {
        name: "Hugging Face text classification",
        type: "Official docs",
        url: "https://huggingface.co/docs/transformers/tasks/sequence_classification",
        note: "Modern fine-tuning guide for encoder-based classification.",
      },
      {
        name: "SST-2",
        type: "Dataset",
        url: "https://huggingface.co/datasets/stanfordnlp/sst2",
        note: "Standard benchmark dataset for sentiment classification.",
      },
      {
        name: "RoBERTa",
        type: "Paper",
        url: "https://arxiv.org/abs/1907.11692",
        note: "Strong encoder baseline for practical text classification.",
      },
    ],
    keyPapers: [
      { title: "BERT", url: "https://arxiv.org/abs/1810.04805" },
      { title: "RoBERTa", url: "https://arxiv.org/abs/1907.11692" },
      { title: "DeBERTa", url: "https://arxiv.org/abs/2006.03654" },
    ],
    codeExample: `from transformers import pipeline

classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
print(classifier("The product shipped fast and support resolved my issue in minutes."))`,
    icon: "😊",
    tags: ["NLP", "Classification", "Encoders", "Text Analytics"],
  },
  {
    id: 6,
    title: "Time Series Forecasting",
    category: "Machine Learning",
    difficulty: "Intermediate",
    timeToComplete: "2-4 weeks",
    summary: "Forecast future values from sequential business or sensor data with statistical and neural baselines.",
    buildNow: "Benchmark simple models first, then compare against modern forecasting architectures such as PatchTST or TimesFM-style approaches.",
    useCases: [
      "Demand forecasting",
      "Capacity planning",
      "Financial forecasting",
      "Predictive maintenance",
    ],
    recommendedStack: ["StatsForecast", "NeuralForecast", "pandas", "feature store"],
    buildSteps: [
      "Define forecast horizon, frequency, and acceptable error range.",
      "Build naive and seasonal baselines before deep models.",
      "Use backtesting across many rolling windows instead of one split.",
      "Separate probabilistic forecast evaluation from point-forecast evaluation.",
    ],
    resources: [
      {
        name: "Nixtla docs",
        type: "Official docs",
        url: "https://nixtlaverse.nixtla.io/",
        note: "Excellent modern stack for statistical and neural forecasting.",
      },
      {
        name: "Papers with Code: Time Series Forecasting",
        type: "Papers with Code",
        url: "https://paperswithcode.com/task/time-series-forecasting",
        note: "Task hub for current benchmark datasets and methods.",
      },
      {
        name: "M4 Dataset",
        type: "Dataset",
        url: "https://github.com/Mcompetitions/M4-methods",
        note: "Widely used forecasting benchmark collection.",
      },
      {
        name: "PatchTST",
        type: "Paper",
        url: "https://arxiv.org/abs/2211.14730",
        note: "Strong transformer baseline for multivariate forecasting.",
      },
    ],
    keyPapers: [
      { title: "N-BEATS", url: "https://arxiv.org/abs/1905.10437" },
      { title: "PatchTST", url: "https://arxiv.org/abs/2211.14730" },
      { title: "TimesFM", url: "https://arxiv.org/abs/2310.10688" },
    ],
    codeExample: `from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

sf = StatsForecast(models=[AutoARIMA(season_length=7)], freq="D")
preds = sf.forecast(df=train_df, h=14)
print(preds.head())`,
    icon: "📈",
    tags: ["Forecasting", "Time Series", "PatchTST", "Baselines"],
  },
  {
    id: 7,
    title: "Text Generation",
    category: "Natural Language Processing",
    difficulty: "Intermediate",
    timeToComplete: "1-3 weeks",
    summary: "Generate long-form text, summaries, or structured outputs using modern language-model APIs or open models.",
    buildNow: "Focus on structured outputs, evals, and grounded generation rather than free-form prompting alone.",
    useCases: [
      "Report drafting",
      "Marketing copy",
      "Knowledge summaries",
      "Structured extraction pipelines",
    ],
    recommendedStack: ["OpenAI Responses API", "Anthropic API", "JSON schema outputs", "eval harness"],
    buildSteps: [
      "Define output schema and success criteria before prompt design.",
      "Use few-shot examples and schema constraints for consistent outputs.",
      "Add retrieval if the text must reflect current or proprietary sources.",
      "Measure factuality and edit distance against human-approved targets.",
    ],
    resources: [
      {
        name: "OpenAI structured outputs",
        type: "Official docs",
        url: "https://platform.openai.com/docs/guides/structured-outputs",
        note: "Useful for predictable production-grade generation.",
      },
      {
        name: "Anthropic prompt engineering",
        type: "Official docs",
        url: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview",
        note: "Current guidance for strong prompt and output design.",
      },
      {
        name: "Transformers text generation",
        type: "Official docs",
        url: "https://huggingface.co/docs/transformers/en/main_classes/text_generation",
        note: "Open-model generation settings and inference patterns.",
      },
      {
        name: "Language modeling paper",
        type: "Paper",
        url: "https://arxiv.org/abs/2005.14165",
        note: "Canonical scaling-era reference for generative language models.",
      },
    ],
    keyPapers: [
      { title: "GPT-3", url: "https://arxiv.org/abs/2005.14165" },
      { title: "InstructGPT", url: "https://arxiv.org/abs/2203.02155" },
      { title: "Toolformer", url: "https://arxiv.org/abs/2302.04761" },
    ],
    codeExample: `from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-4.1-mini",
    input="Write a concise quarterly update in JSON with keys: summary, risks, next_steps.",
    text={"format": {"type": "json_schema", "name": "quarterly_update", "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "risks": {"type": "array", "items": {"type": "string"}},
            "next_steps": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["summary", "risks", "next_steps"],
        "additionalProperties": False
    }}}
)`,
    icon: "✍️",
    tags: ["LLMs", "Structured Output", "Generation", "Prompting"],
  },
  {
    id: 8,
    title: "Anomaly Detection",
    category: "Machine Learning",
    difficulty: "Intermediate",
    timeToComplete: "2-3 weeks",
    summary: "Detect rare, high-cost deviations in images, logs, or metrics with unsupervised or weakly supervised methods.",
    buildNow: "Choose the anomaly type first: tabular, time series, image, or graph anomalies each need a different baseline.",
    useCases: [
      "Fraud detection",
      "Quality assurance",
      "Security monitoring",
      "Machine-health alerts",
    ],
    recommendedStack: ["scikit-learn", "PyTorch", "FiftyOne", "prometheus or observability stack"],
    buildSteps: [
      "Define whether you are finding point anomalies, contextual anomalies, or collective anomalies.",
      "Start with isolation forest or simple reconstruction baselines before deep models.",
      "Use precision-at-k and alert fatigue measures, not just ROC curves.",
      "Design a human-review workflow because anomaly thresholds always drift.",
    ],
    resources: [
      {
        name: "scikit-learn anomaly detection",
        type: "Official docs",
        url: "https://scikit-learn.org/stable/modules/outlier_detection.html",
        note: "Reliable classical baselines for many anomaly tasks.",
      },
      {
        name: "MVTec AD",
        type: "Dataset",
        url: "https://www.mvtec.com/company/research/datasets/mvtec-ad",
        note: "Well-known industrial visual-anomaly benchmark.",
      },
      {
        name: "Papers with Code anomaly detection",
        type: "Papers with Code",
        url: "https://paperswithcode.com/task/multi-class-anomaly-detection",
        note: "Useful starting point for current benchmark families.",
      },
      {
        name: "Isolation Forest",
        type: "Paper",
        url: "https://www.researchgate.net/publication/224384174_Isolation_Forest",
        note: "Simple and still relevant anomaly baseline.",
      },
    ],
    keyPapers: [
      { title: "Isolation Forest", url: "https://www.researchgate.net/publication/224384174_Isolation_Forest" },
      { title: "AnoGAN", url: "https://arxiv.org/abs/1703.05921" },
      { title: "PatchCore", url: "https://arxiv.org/abs/2106.08265" },
    ],
    codeExample: `from sklearn.ensemble import IsolationForest

detector = IsolationForest(contamination=0.01, random_state=42)
detector.fit(train_features)
scores = detector.decision_function(eval_features)
flags = detector.predict(eval_features)`,
    icon: "🚨",
    tags: ["Outlier Detection", "MVTec", "Monitoring", "Isolation Forest"],
  },
  {
    id: 9,
    title: "Speech Recognition",
    category: "Speech & Audio",
    difficulty: "Intermediate",
    timeToComplete: "2-4 weeks",
    summary: "Transcribe speech to text with current encoder-decoder or self-supervised speech models.",
    buildNow: "Benchmark Whisper-class models first, then optimize for latency, domain language, and multilingual coverage.",
    useCases: [
      "Meeting transcription",
      "Call-center QA",
      "Voice notes",
      "Accessibility tooling",
    ],
    recommendedStack: ["Whisper", "PyTorch", "Hugging Face", "forced-alignment tools"],
    buildSteps: [
      "Choose between batch accuracy and streaming latency goals early.",
      "Normalize noisy audio and benchmark against diverse accents and environments.",
      "Measure WER with domain-specific vocabulary before deployment.",
      "Add speaker diarization and timestamping if the workflow needs searchable records.",
    ],
    resources: [
      {
        name: "OpenAI Whisper repo",
        type: "Official docs",
        url: "https://github.com/openai/whisper",
        note: "Reference implementation and model family for practical ASR baselines.",
      },
      {
        name: "Papers with Code: Speech Recognition",
        type: "Papers with Code",
        url: "https://paperswithcode.com/task/speech-recognition/latest",
        note: "Benchmark hub for current ASR systems.",
      },
      {
        name: "LibriSpeech",
        type: "Dataset",
        url: "https://www.openslr.org/12",
        note: "Core public benchmark for English ASR.",
      },
      {
        name: "wav2vec 2.0",
        type: "Paper",
        url: "https://arxiv.org/abs/2006.11477",
        note: "Foundational self-supervised speech pretraining paper.",
      },
    ],
    keyPapers: [
      { title: "wav2vec 2.0", url: "https://arxiv.org/abs/2006.11477" },
      { title: "Whisper", url: "https://cdn.openai.com/papers/whisper.pdf" },
      { title: "Conformer", url: "https://arxiv.org/abs/2005.08100" },
    ],
    codeExample: `import whisper

model = whisper.load_model("small")
result = model.transcribe("meeting.wav")
print(result["text"])`,
    icon: "🎙️",
    tags: ["ASR", "Whisper", "Speech", "Transcription"],
  },
  {
    id: 10,
    title: "Generative Adversarial Networks (GANs)",
    category: "Machine Learning",
    difficulty: "Advanced",
    timeToComplete: "3-6 weeks",
    summary: "Train a generator-discriminator pair for image synthesis, augmentation, or representation learning.",
    buildNow: "GANs are no longer the default for frontier image generation, but they remain useful for low-latency synthesis and domain-specific augmentation.",
    useCases: [
      "Synthetic data augmentation",
      "Style transfer",
      "Image-to-image translation",
      "Face anonymization",
    ],
    recommendedStack: ["PyTorch", "Lightning", "Weights & Biases", "custom training loop"],
    buildSteps: [
      "Start with a narrow domain and stable data distribution.",
      "Monitor generator and discriminator balance at every stage.",
      "Use FID or task-specific downstream metrics, not visuals alone.",
      "Keep a diffusion baseline handy so you know whether GAN tradeoffs are still worth it.",
    ],
    resources: [
      {
        name: "GAN paper",
        type: "Paper",
        url: "https://arxiv.org/abs/1406.2661",
        note: "Original adversarial-learning paper.",
      },
      {
        name: "StyleGAN2-ADA",
        type: "Paper",
        url: "https://arxiv.org/abs/2006.06676",
        note: "Modern high-quality image-generation reference.",
      },
      {
        name: "PyTorch DCGAN tutorial",
        type: "Official docs",
        url: "https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html",
        note: "Good starting point for practical GAN training loops.",
      },
      {
        name: "CelebA",
        type: "Dataset",
        url: "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        note: "Common dataset for image-generation experiments.",
      },
    ],
    keyPapers: [
      { title: "GANs", url: "https://arxiv.org/abs/1406.2661" },
      { title: "CycleGAN", url: "https://arxiv.org/abs/1703.10593" },
      { title: "StyleGAN2-ADA", url: "https://arxiv.org/abs/2006.06676" },
    ],
    codeExample: `real = next(iter(dataloader)).to(device)
noise = torch.randn(real.size(0), latent_dim, 1, 1, device=device)
fake = generator(noise)

loss_d = criterion(discriminator(real), torch.ones_like(...)) + criterion(discriminator(fake.detach()), torch.zeros_like(...))
loss_g = criterion(discriminator(fake), torch.ones_like(...))`,
    icon: "🎭",
    tags: ["GAN", "Generation", "StyleGAN", "Synthesis"],
  },
  {
    id: 11,
    title: "Reinforcement Learning Agent",
    category: "Machine Learning",
    difficulty: "Advanced",
    timeToComplete: "3-6 weeks",
    summary: "Train an agent that improves through interaction, reward, and policy updates in a simulated environment.",
    buildNow: "Use clean environments, strong offline logging, and stable-baselines before building custom RL stacks from scratch.",
    useCases: [
      "Game AI",
      "Resource allocation",
      "Robotics simulation",
      "Dynamic pricing research",
    ],
    recommendedStack: ["Gymnasium", "Stable-Baselines3", "Weights & Biases", "simulation env"],
    buildSteps: [
      "Choose an environment where reward design is explicit and measurable.",
      "Start with PPO or SAC before experimenting with custom algorithms.",
      "Log every trajectory artifact so training regressions are debuggable.",
      "Validate that the learned policy transfers to realistic constraints.",
    ],
    resources: [
      {
        name: "Gymnasium",
        type: "Framework",
        url: "https://gymnasium.farama.org/",
        note: "Standard environment API for RL experiments.",
      },
      {
        name: "Stable-Baselines3",
        type: "Framework",
        url: "https://stable-baselines3.readthedocs.io/en/master/",
        note: "Strong baseline implementations for PPO, SAC, DQN, and more.",
      },
      {
        name: "Spinning Up",
        type: "Official docs",
        url: "https://spinningup.openai.com/en/latest/",
        note: "Clear conceptual guide for practical RL.",
      },
      {
        name: "PPO",
        type: "Paper",
        url: "https://arxiv.org/abs/1707.06347",
        note: "Most common first serious policy-optimization baseline.",
      },
    ],
    keyPapers: [
      { title: "DQN", url: "https://www.nature.com/articles/nature14236" },
      { title: "PPO", url: "https://arxiv.org/abs/1707.06347" },
      { title: "SAC", url: "https://arxiv.org/abs/1801.01290" },
    ],
    codeExample: `from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)`,
    icon: "🕹️",
    tags: ["RL", "PPO", "Simulation", "Agents"],
  },
  {
    id: 12,
    title: "Neural Machine Translation",
    category: "Natural Language Processing",
    difficulty: "Advanced",
    timeToComplete: "3-5 weeks",
    summary: "Translate between languages with seq2seq or transformer-based multilingual models.",
    buildNow: "Start with pretrained translation models and evaluate on your actual domain vocabulary before attempting custom training.",
    useCases: [
      "Localization pipelines",
      "Support translation",
      "Cross-border commerce",
      "Knowledge-base translation",
    ],
    recommendedStack: ["Transformers", "sacreBLEU", "MarianMT or NLLB", "translation memory"],
    buildSteps: [
      "Pick language pairs and domain terminology constraints early.",
      "Benchmark pretrained multilingual models against a curated human set.",
      "Use BLEU or COMET, but also review terminology consistency manually.",
      "Add retrieval or glossary injection for sensitive domain translations.",
    ],
    resources: [
      {
        name: "Hugging Face translation task guide",
        type: "Official docs",
        url: "https://huggingface.co/docs/transformers/tasks/translation",
        note: "Modern practical guide for translation fine-tuning.",
      },
      {
        name: "NLLB team and models",
        type: "Official docs",
        url: "https://ai.meta.com/research/no-language-left-behind/",
        note: "Strong multilingual translation resource from Meta.",
      },
      {
        name: "WMT datasets",
        type: "Dataset",
        url: "https://www.statmt.org/wmt24/",
        note: "Benchmark ecosystem for machine translation.",
      },
      {
        name: "Transformer",
        type: "Paper",
        url: "https://arxiv.org/abs/1706.03762",
        note: "Architectural foundation for modern translation systems.",
      },
    ],
    keyPapers: [
      { title: "Bahdanau Attention", url: "https://arxiv.org/abs/1409.0473" },
      { title: "Transformer", url: "https://arxiv.org/abs/1706.03762" },
      { title: "No Language Left Behind", url: "https://arxiv.org/abs/2207.04672" },
    ],
    codeExample: `from transformers import pipeline

translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")
print(translator("This contract renews automatically unless cancelled within 30 days."))`,
    icon: "🌍",
    tags: ["Translation", "Transformers", "NLLB", "Multilingual"],
  },
  {
    id: 13,
    title: "Medical Image Analysis",
    category: "Computer Vision",
    difficulty: "Advanced",
    timeToComplete: "4-8 weeks",
    summary: "Build clinically useful image-analysis models for classification, segmentation, or triage support.",
    buildNow: "Use clinically relevant labels, calibration, and physician-reviewed error analysis before optimizing benchmark scores.",
    useCases: [
      "Radiology triage",
      "Pathology support",
      "Retinal screening",
      "Tumor segmentation",
    ],
    recommendedStack: ["MONAI", "PyTorch", "DICOM tooling", "medical-label workflow"],
    buildSteps: [
      "Work with a clinical partner to define the exact target decision.",
      "Use patient-level splits and leakage checks from the start.",
      "Measure sensitivity, specificity, and calibration in addition to AUROC.",
      "Design a decision-support workflow rather than fully automated diagnosis.",
    ],
    resources: [
      {
        name: "MONAI",
        type: "Official docs",
        url: "https://docs.monai.io/en/stable/",
        note: "Leading open-source framework for medical imaging AI.",
      },
      {
        name: "nnU-Net",
        type: "Paper",
        url: "https://www.nature.com/articles/s41592-020-01008-z",
        note: "Strong practical baseline for medical segmentation tasks.",
      },
      {
        name: "RSNA datasets",
        type: "Dataset",
        url: "https://www.rsna.org/rsnai/ai-image-challenge",
        note: "Useful public challenge datasets for radiology workflows.",
      },
      {
        name: "MedSAM",
        type: "Paper",
        url: "https://arxiv.org/abs/2304.12306",
        note: "Current segmentation direction built on foundation-model ideas.",
      },
    ],
    keyPapers: [
      { title: "U-Net", url: "https://arxiv.org/abs/1505.04597" },
      { title: "nnU-Net", url: "https://www.nature.com/articles/s41592-020-01008-z" },
      { title: "MedSAM", url: "https://arxiv.org/abs/2304.12306" },
    ],
    codeExample: `from monai.networks.nets import UNet

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
)`,
    icon: "🩺",
    tags: ["MONAI", "Medical Imaging", "Segmentation", "Clinical AI"],
  },
  {
    id: 14,
    title: "Edge AI Deployment",
    category: "Edge AI",
    difficulty: "Advanced",
    timeToComplete: "3-6 weeks",
    summary: "Optimize and ship models to phones, cameras, browsers, or embedded hardware with tight latency and size limits.",
    buildNow: "Treat compression, quantization, and hardware profiling as first-class design constraints, not final deployment chores.",
    useCases: [
      "Mobile assistants",
      "On-device vision",
      "Industrial sensors",
      "Privacy-sensitive inference",
    ],
    recommendedStack: ["ONNX Runtime", "TensorRT", "TensorFlow Lite", "profiling tools"],
    buildSteps: [
      "Pick the target hardware and latency budget before choosing a model family.",
      "Benchmark quantized and distilled variants early in the build.",
      "Measure end-to-end latency on target hardware, not only desktop proxies.",
      "Plan for model updates, rollout safety, and offline fallback behavior.",
    ],
    resources: [
      {
        name: "ONNX Runtime",
        type: "Official docs",
        url: "https://onnxruntime.ai/docs/",
        note: "Common inference runtime for portable deployment.",
      },
      {
        name: "TensorFlow Lite",
        type: "Official docs",
        url: "https://www.tensorflow.org/lite",
        note: "Mature stack for mobile and embedded deployments.",
      },
      {
        name: "TensorRT",
        type: "Official docs",
        url: "https://docs.nvidia.com/deeplearning/tensorrt/latest/",
        note: "High-performance inference optimization for NVIDIA hardware.",
      },
      {
        name: "MobileNets",
        type: "Paper",
        url: "https://arxiv.org/abs/1704.04861",
        note: "Canonical efficiency-first architecture family for edge vision.",
      },
    ],
    keyPapers: [
      { title: "MobileNets", url: "https://arxiv.org/abs/1704.04861" },
      { title: "Quantization and Training of Neural Networks", url: "https://arxiv.org/abs/1712.05877" },
      { title: "Distilling the Knowledge in a Neural Network", url: "https://arxiv.org/abs/1503.02531" },
    ],
    codeExample: `import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
outputs = session.run(None, {"input": input_tensor.numpy()})`,
    icon: "📱",
    tags: ["Edge AI", "ONNX", "Quantization", "Deployment"],
  },
];

export const projectCategories = [
  "All",
  ...Array.from(new Set(projects.map((project) => project.category))).sort(),
];
