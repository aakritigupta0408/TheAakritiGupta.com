import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  FaRobot,
  FaBrain,
  FaCode,
  FaCogs,
  FaNetworkWired,
  FaDatabase,
  FaSearch,
  FaChartLine,
  FaShieldAlt,
  FaUsers,
  FaTimes,
  FaPlay,
  FaLightbulb,
  FaGraduationCap,
  FaRocket,
} from "react-icons/fa";

interface AgentExample {
  id: string;
  title: string;
  category: string;
  difficulty: "Beginner" | "Intermediate" | "Advanced";
  icon: React.ReactNode;
  description: string;
  before: {
    title: string;
    content: string;
    issues: string[];
  };
  after: {
    title: string;
    content: string;
    improvements: string[];
  };
  tips: string[];
  useCase: string;
  resources: {
    codeExamples: { title: string; url: string; language: string }[];
    tutorials: { title: string; url: string; difficulty: string }[];
    videos: { title: string; url: string; creator: string; duration: string }[];
    documentation: { title: string; url: string; type: string }[];
  };
}

interface TrainingTechnique {
  id: string;
  name: string;
  category: string;
  description: string;
  icon: React.ReactNode;
  keyPoints: string[];
  example: string;
  benefits: string[];
  resources: {
    tutorials: { title: string; url: string; type: string }[];
    videos: { title: string; url: string; duration: string }[];
    papers: { title: string; url: string; authors: string }[];
  };
}

const aiAgentExamples: AgentExample[] = [
  {
    id: "customer-service",
    title: "Customer Service Agent",
    category: "Business Automation",
    difficulty: "Beginner",
    icon: <FaUsers className="text-blue-400" />,
    description:
      "Build an AI agent that handles customer inquiries with empathy and efficiency",
    before: {
      title: "Basic Chatbot Response",
      content: `Agent: Hello. How can I help you?
User: My order hasn't arrived yet and I'm frustrated.
Agent: Please provide your order number for assistance.`,
      issues: [
        "No empathy recognition",
        "Robotic responses",
        "Doesn't acknowledge emotions",
      ],
    },
    after: {
      title: "Empathetic AI Agent",
      content: `Agent: Hi there! I'm here to help you today. 😊
User: My order hasn't arrived yet and I'm frustrated.
Agent: I completely understand your frustration - waiting for an order can be really stressful. Let me personally look into this right away. Could you please share your order number so I can track exactly what happened and get this resolved for you?`,
      improvements: [
        "Acknowledges emotions",
        "Shows empathy and understanding",
        "Takes personal responsibility",
      ],
    },
    tips: [
      "Train on emotional intelligence datasets",
      "Include sentiment analysis capabilities",
      "Use warm, human-like language patterns",
    ],
    useCase: "E-commerce platforms, SaaS support, healthcare customer service",
    resources: {
      codeExamples: [
        {
          title: "OpenAI GPT-4 Customer Service Bot",
          url: "https://github.com/openai/openai-cookbook/blob/main/examples/How_to_build_a_customer_service_chatbot.ipynb",
          language: "Python",
        },
        {
          title: "Rasa Customer Service Framework",
          url: "https://github.com/RasaHQ/rasa",
          language: "Python",
        },
        {
          title: "Microsoft Bot Framework Customer Service",
          url: "https://github.com/microsoft/BotFramework-Samples",
          language: "C#/Node.js",
        },
      ],
      tutorials: [
        {
          title: "Building Empathetic AI Chatbots",
          url: "https://platform.openai.com/docs/guides/fine-tuning",
          difficulty: "Intermediate",
        },
        {
          title: "Customer Service Bot with LangChain",
          url: "https://python.langchain.com/docs/use_cases/chatbots",
          difficulty: "Beginner",
        },
        {
          title: "Implementing RLHF for Customer Service",
          url: "https://huggingface.co/blog/rlhf",
          difficulty: "Advanced",
        },
      ],
      videos: [
        {
          title: "Building AI Customer Service Agents from Scratch",
          url: "https://www.youtube.com/watch?v=5dWNiKOldh0",
          creator: "AI Jason",
          duration: "45 min",
        },
        {
          title: "Customer Service Chatbot with OpenAI API",
          url: "https://www.youtube.com/watch?v=dI2WkwqOhqQ",
          creator: "Tech With Tim",
          duration: "32 min",
        },
        {
          title: "Advanced Conversational AI Techniques",
          url: "https://www.youtube.com/watch?v=yj-wSRJwrrc",
          creator: "DeepLearning.AI",
          duration: "1 hr 15 min",
        },
      ],
      documentation: [
        {
          title: "OpenAI Assistant API Documentation",
          url: "https://platform.openai.com/docs/assistants/overview",
          type: "API Reference",
        },
        {
          title: "Anthropic Claude for Customer Service",
          url: "https://docs.anthropic.com/claude/docs/intro-to-claude",
          type: "Guide",
        },
        {
          title: "Azure Bot Service Documentation",
          url: "https://docs.microsoft.com/en-us/azure/bot-service/",
          type: "Platform Guide",
        },
      ],
    },
  },
  {
    id: "research-assistant",
    title: "Research & Analysis Agent",
    category: "Knowledge Work",
    difficulty: "Advanced",
    icon: <FaSearch className="text-green-400" />,
    description:
      "Create an AI agent that conducts thorough research and provides analytical insights",
    before: {
      title: "Simple Search Results",
      content: `User: Research the impact of AI on healthcare
Agent: AI is being used in healthcare for diagnosis, treatment, and drug discovery. Here are some examples: medical imaging, predictive analytics, robotic surgery.`,
      issues: [
        "Surface-level information",
        "No structured analysis",
        "Missing citations and sources",
      ],
    },
    after: {
      title: "Comprehensive Research Agent",
      content: `User: Research the impact of AI on healthcare
Agent: I'll conduct a comprehensive analysis of AI's impact on healthcare across multiple dimensions:

📊 **MARKET IMPACT**
- Market size: $45B (2022) → projected $148B (2029)
- Key growth drivers: aging population, labor shortages

🔬 **TECHNOLOGICAL BREAKTHROUGHS**
1. Diagnostic AI: 95% accuracy in radiology (Nature Medicine, 2023)
2. Drug Discovery: 70% reduction in development time (McKinsey, 2023)
3. Personalized Medicine: 40% improvement in treatment outcomes

⚠️ **CHALLENGES & CONSIDERATIONS**
- Regulatory approval processes
- Data privacy concerns (HIPAA compliance)
- Integration with existing workflows

📚 **Sources**: Nature Medicine, McKinsey Global Institute, FDA Guidelines`,
      improvements: [
        "Structured, multi-dimensional analysis",
        "Quantitative data with sources",
        "Balanced view including challenges",
      ],
    },
    tips: [
      "Integrate multiple data sources and APIs",
      "Train on academic papers and research methodologies",
      "Include fact-checking and source verification",
    ],
    useCase:
      "Investment firms, consultancies, academic institutions, policy makers",
    resources: {
      codeExamples: [
        {
          title: "LangChain Research Assistant Agent",
          url: "https://github.com/langchain-ai/langchain/tree/master/cookbook/agents",
          language: "Python",
        },
        {
          title: "AutoGPT Research Implementation",
          url: "https://github.com/Significant-Gravitas/AutoGPT",
          language: "Python",
        },
        {
          title: "CrewAI Multi-Agent Research Team",
          url: "https://github.com/joaomdmoura/crewAI",
          language: "Python",
        },
      ],
      tutorials: [
        {
          title: "Building Research Agents with Tool Use",
          url: "https://python.langchain.com/docs/modules/agents/tools/",
          difficulty: "Advanced",
        },
        {
          title: "Multi-Agent Research Systems",
          url: "https://github.com/microsoft/autogen",
          difficulty: "Advanced",
        },
        {
          title: "RAG for Research Applications",
          url: "https://python.langchain.com/docs/use_cases/question_answering",
          difficulty: "Intermediate",
        },
      ],
      videos: [
        {
          title: "Building an AI Research Assistant with LangChain",
          url: "https://www.youtube.com/watch?v=ZzgUqFtxgXI",
          creator: "LangChain",
          duration: "52 min",
        },
        {
          title: "Multi-Agent Research Systems Tutorial",
          url: "https://www.youtube.com/watch?v=BYnTzaGo06I",
          creator: "AI Explained",
          duration: "38 min",
        },
        {
          title: "Advanced RAG Techniques for Research",
          url: "https://www.youtube.com/watch?v=wd7TZ4w1mSw",
          creator: "Weights & Biases",
          duration: "1 hr 8 min",
        },
      ],
      documentation: [
        {
          title: "Semantic Scholar API for Research",
          url: "https://api.semanticscholar.org/",
          type: "API Reference",
        },
        {
          title: "Perplexity API Documentation",
          url: "https://docs.perplexity.ai/",
          type: "API Reference",
        },
        {
          title: "Arxiv API Integration Guide",
          url: "https://arxiv.org/help/api/user-manual",
          type: "API Guide",
        },
      ],
    },
  },
  {
    id: "code-reviewer",
    title: "Code Review Agent",
    category: "Software Development",
    difficulty: "Advanced",
    icon: <FaCode className="text-purple-400" />,
    description:
      "Develop an AI agent that provides comprehensive code reviews with security and performance insights",
    before: {
      title: "Basic Syntax Check",
      content: `function processUser(data) {
  return data.name + " " + data.email;
}

Agent: The function looks fine. No syntax errors detected.`,
      issues: [
        "No security analysis",
        "Missing performance considerations",
        "No architectural feedback",
      ],
    },
    after: {
      title: "Comprehensive Code Review Agent",
      content: `function processUser(data) {
  return data.name + " " + data.email;
}

🔍 **CODE REVIEW ANALYSIS**

🚨 **SECURITY ISSUES**
- XSS Risk: Unescaped user input concatenation
- Data Validation: No input sanitization

⚡ **PERFORMANCE CONCERNS**
- String concatenation inefficiency
- Missing null/undefined checks

🏗️ **ARCHITECTURAL IMPROVEMENTS**
\`\`\`javascript
function processUser(data) {
  // Input validation
  if (!data?.name || !data?.email) {
    throw new Error('Invalid user data');
  }
  
  // Sanitize and format safely
  const sanitizedName = escapeHtml(data.name);
  const sanitizedEmail = escapeHtml(data.email);
  
  return \`\${sanitizedName} \${sanitizedEmail}\`;
}
\`\`\`

📋 **RECOMMENDATIONS**
1. Add TypeScript interfaces
2. Implement proper error handling
3. Consider using template literals
4. Add unit tests`,
      improvements: [
        "Security vulnerability detection",
        "Performance optimization suggestions",
        "Improved code with explanations",
      ],
    },
    tips: [
      "Train on security vulnerability databases",
      "Include static analysis tools integration",
      "Learn from open-source code review patterns",
    ],
    useCase: "Development teams, code quality automation, CI/CD pipelines",
    resources: {
      codeExamples: [
        {
          title: "GitHub Copilot Code Review Implementation",
          url: "https://github.com/github/copilot-docs",
          language: "Various",
        },
        {
          title: "SonarQube AI Code Analysis",
          url: "https://github.com/SonarSource/sonarqube",
          language: "Java",
        },
        {
          title: "DeepCode AI Code Review",
          url: "https://github.com/DeepCodeAI/cli",
          language: "Python",
        },
      ],
      tutorials: [
        {
          title: "Building AI Code Review Systems",
          url: "https://docs.github.com/en/codespaces/developing-in-codespaces/using-github-copilot-in-codespaces",
          difficulty: "Advanced",
        },
        {
          title: "Static Analysis with AI",
          url: "https://semgrep.dev/docs/",
          difficulty: "Intermediate",
        },
      ],
      videos: [
        {
          title: "AI-Powered Code Review Best Practices",
          url: "https://www.youtube.com/watch?v=8c5NQbJNQkE",
          creator: "GitHub",
          duration: "28 min",
        },
        {
          title: "Building Code Analysis Tools with AI",
          url: "https://www.youtube.com/watch?v=v8dDj28x2nE",
          creator: "Code With Me",
          duration: "42 min",
        },
      ],
      documentation: [
        {
          title: "GitHub Copilot API",
          url: "https://docs.github.com/en/copilot",
          type: "Platform Guide",
        },
        {
          title: "OpenAI Codex Documentation",
          url: "https://platform.openai.com/docs/guides/code",
          type: "API Reference",
        },
      ],
    },
  },
  {
    id: "data-analyst",
    title: "Data Analysis Agent",
    category: "Analytics",
    difficulty: "Intermediate",
    icon: <FaChartLine className="text-orange-400" />,
    description:
      "Build an AI agent that performs automated data analysis and generates insights",
    before: {
      title: "Basic Data Summary",
      content: `Dataset: Sales data (1000 rows)
Agent: The dataset contains sales information. Average sales: $1,234. Total revenue: $1,234,000.`,
      issues: [
        "No trend analysis",
        "Missing visualizations",
        "No actionable insights",
      ],
    },
    after: {
      title: "Intelligent Data Analysis Agent",
      content: `📊 **SALES PERFORMANCE ANALYSIS**

�� **KEY FINDINGS**
- Revenue Trend: 15% growth over last quarter
- Peak Performance: Fridays show 23% higher sales
- Geographic Insights: West Coast markets outperforming by 31%

📈 **STATISTICAL ANALYSIS**
- Correlation: Marketing spend vs Sales (r=0.78)
- Seasonality: 40% revenue spike in Q4
- Customer Segments: Premium customers (20%) drive 60% revenue

💡 **ACTIONABLE RECOMMENDATIONS**
1. Increase Friday marketing campaigns (+23% ROI potential)
2. Expand West Coast operations
3. Focus premium customer retention programs
4. Prepare Q4 inventory 40% above baseline

📊 **VISUALIZATIONS GENERATED**
- Revenue trend chart
- Geographic heatmap
- Customer segment breakdown`,
      improvements: [
        "Statistical analysis and correlations",
        "Actionable business recommendations",
        "Visual insights generation",
      ],
    },
    tips: [
      "Integrate statistical analysis libraries",
      "Train on business intelligence best practices",
      "Include data visualization generation capabilities",
    ],
    useCase: "Business intelligence, marketing analytics, financial analysis",
    resources: {
      codeExamples: [
        {
          title: "AutoML Data Analysis Pipeline",
          url: "https://github.com/microsoft/FLAML",
          language: "Python",
        },
        {
          title: "PandasAI Data Analysis Agent",
          url: "https://github.com/gventuri/pandas-ai",
          language: "Python",
        },
      ],
      tutorials: [
        {
          title: "Building AI Data Analysts",
          url: "https://docs.datarobot.com/en/docs/",
          difficulty: "Intermediate",
        },
      ],
      videos: [
        {
          title: "AI Data Analysis with Python",
          url: "https://www.youtube.com/watch?v=vmEHCJofslg",
          creator: "freeCodeCamp",
          duration: "2 hr 15 min",
        },
      ],
      documentation: [
        {
          title: "Tableau AI Analytics",
          url: "https://help.tableau.com/current/pro/desktop/en-us/ask_data.htm",
          type: "Platform Guide",
        },
      ],
    },
  },
  {
    id: "security-monitor",
    title: "Cybersecurity Monitoring Agent",
    category: "Security",
    difficulty: "Advanced",
    icon: <FaShieldAlt className="text-red-400" />,
    description:
      "Create an AI agent that monitors systems for security threats and responds autonomously",
    before: {
      title: "Basic Alert System",
      content: `Alert: Unusual login detected
Time: 2:30 AM
Location: Unknown
Agent: Security alert logged. Please review.`,
      issues: [
        "No threat assessment",
        "Manual intervention required",
        "Limited context analysis",
      ],
    },
    after: {
      title: "Intelligent Security Agent",
      content: `🚨 **SECURITY INCIDENT DETECTED**

⚠️ **THREAT ANALYSIS**
- Severity: HIGH (8.5/10)
- Attack Vector: Credential stuffing attempt
- Source: TOR exit node (Russia)
- Pattern: 847 failed logins in 3 minutes

🔍 **CONTEXTUAL INTELLIGENCE**
- Target Account: admin@company.com (high-privilege)
- Historical Pattern: Similar attack 3 months ago
- Geolocation Anomaly: 6,000+ miles from usual location

🛡️ **AUTOMATED RESPONSE ACTIONS**
✅ Account temporarily locked
✅ IP address blocked across all systems
✅ Security team notified (SMS sent)
✅ Backup admin accounts secured
✅ Forensic data collection initiated

📋 **RECOMMENDED NEXT STEPS**
1. Enable MFA for all admin accounts
2. Review and update password policies
3. Consider geo-blocking for admin access`,
      improvements: [
        "Real-time threat assessment",
        "Automated protective actions",
        "Comprehensive incident documentation",
      ],
    },
    tips: [
      "Train on cybersecurity threat databases",
      "Integrate with SIEM and security tools",
      "Include automated response capabilities",
    ],
    useCase: "Enterprise security, SOC automation, incident response",
    resources: {
      codeExamples: [
        {
          title: "Security Monitoring with AI",
          url: "https://github.com/OTRF/Security-Datasets",
          language: "Python",
        },
        {
          title: "Splunk SOAR Security Automation",
          url: "https://github.com/splunk/security_content",
          language: "Python",
        },
      ],
      tutorials: [
        {
          title: "AI-Powered SOC Operations",
          url: "https://docs.splunk.com/Documentation/SOAR",
          difficulty: "Advanced",
        },
      ],
      videos: [
        {
          title: "Building AI Security Systems",
          url: "https://www.youtube.com/watch?v=P4nGZ8zxFCI",
          creator: "SANS Institute",
          duration: "1 hr 22 min",
        },
      ],
      documentation: [
        {
          title: "IBM QRadar AI Security",
          url: "https://www.ibm.com/docs/en/qsip",
          type: "Platform Guide",
        },
      ],
    },
  },
  {
    id: "workflow-optimizer",
    title: "Workflow Optimization Agent",
    category: "Process Automation",
    difficulty: "Intermediate",
    icon: <FaCogs className="text-cyan-400" />,
    description:
      "Design an AI agent that analyzes and optimizes business workflows automatically",
    before: {
      title: "Manual Process Documentation",
      content: `Current Process: Order fulfillment
Steps: 
1. Receive order
2. Check inventory
3. Process payment
4. Ship order
5. Send confirmation

Agent: Process documented. 5 steps identified.`,
      issues: [
        "No efficiency analysis",
        "Missing bottleneck identification",
        "No optimization recommendations",
      ],
    },
    after: {
      title: "Intelligent Workflow Optimization Agent",
      content: `🔄 **WORKFLOW ANALYSIS: Order Fulfillment**

⏱️ **PERFORMANCE METRICS**
- Average completion time: 4.2 hours
- Bottleneck identified: Payment processing (2.1 hours)
- Success rate: 94% (6% failures in inventory check)

🎯 **OPTIMIZATION OPPORTUNITIES**
1. **Parallel Processing**: Run inventory check + payment simultaneously
   - Time savings: 1.8 hours (-43%)
   
2. **Predictive Inventory**: Pre-allocate popular items
   - Failure reduction: 6% → 1.5%
   
3. **Payment Automation**: Implement instant verification
   - Processing time: 2.1h → 15 minutes

📊 **OPTIMIZED WORKFLOW**
\`\`\`
Parallel: [Inventory Check + Payment Processing] → 15 min
Sequential: [Auto-allocation → Shipping → Confirmation] → 45 min
Total Time: 4.2h → 1.0h (76% improvement)
\`\`\`

💰 **BUSINESS IMPACT**
- Capacity increase: 4x more orders per day
- Customer satisfaction: +23% (faster delivery)
- Cost reduction: 31% operational savings`,
      improvements: [
        "Quantitative bottleneck analysis",
        "Specific optimization recommendations",
        "Business impact projections",
      ],
    },
    tips: [
      "Train on process mining datasets",
      "Include business process modeling knowledge",
      "Learn efficiency optimization techniques",
    ],
    useCase:
      "Operations management, business process improvement, supply chain optimization",
    resources: {
      codeExamples: [
        {
          title: "Process Mining with AI",
          url: "https://github.com/pm4py/pm4py-core",
          language: "Python",
        },
        {
          title: "Workflow Automation Framework",
          url: "https://github.com/n8n-io/n8n",
          language: "TypeScript",
        },
      ],
      tutorials: [
        {
          title: "Business Process Optimization with AI",
          url: "https://docs.celonis.com/en/process-mining.html",
          difficulty: "Intermediate",
        },
      ],
      videos: [
        {
          title: "AI for Business Process Automation",
          url: "https://www.youtube.com/watch?v=qGhyKNl5Cs8",
          creator: "Process Street",
          duration: "34 min",
        },
      ],
      documentation: [
        {
          title: "Microsoft Power Automate AI",
          url: "https://docs.microsoft.com/en-us/power-automate/",
          type: "Platform Guide",
        },
      ],
    },
  },
];

const trainingTechniques: TrainingTechnique[] = [
  {
    id: "reinforcement-learning",
    name: "Reinforcement Learning from Human Feedback (RLHF)",
    category: "Training Method",
    description:
      "Train agents through reward-based learning using human preferences and feedback",
    icon: <FaBrain className="text-blue-400" />,
    keyPoints: [
      "RLHF uses human feedback to train reward models that guide AI behavior, making agents more helpful and aligned with human values.",
      "Human evaluators rank agent responses",
      "Reward model learns from preferences",
      "Agent optimizes for higher reward scores",
      "Iterative improvement through feedback loops",
    ],
    example:
      "Training a customer service agent where humans rate responses on helpfulness, empathy, and accuracy. The agent learns to maximize these human-preferred qualities.",
    benefits: [
      "Aligns with human values and preferences",
      "Reduces harmful or inappropriate outputs",
      "Improves response quality over time",
      "Adaptable to different use cases",
    ],
    resources: {
      tutorials: [
        {
          title: "Introduction to RLHF",
          url: "https://huggingface.co/blog/rlhf",
          type: "Comprehensive Guide",
        },
        {
          title: "OpenAI RLHF Research",
          url: "https://openai.com/research/learning-from-human-preferences",
          type: "Research Paper",
        },
        {
          title: "Implementing RLHF with TRL",
          url: "https://github.com/huggingface/trl",
          type: "Code Repository",
        },
      ],
      videos: [
        {
          title: "RLHF Explained: Training AI with Human Feedback",
          url: "https://www.youtube.com/watch?v=2MBJOuVq380",
          duration: "18 min",
        },
        {
          title: "ChatGPT Training Process Deep Dive",
          url: "https://www.youtube.com/watch?v=VPRSBzXzavo",
          duration: "45 min",
        },
        {
          title: "Building RLHF Systems from Scratch",
          url: "https://www.youtube.com/watch?v=TfBpKRj-7K4",
          duration: "1 hr 12 min",
        },
      ],
      papers: [
        {
          title:
            "Training language models to follow instructions with human feedback",
          url: "https://arxiv.org/abs/2203.02155",
          authors: "Ouyang et al. (OpenAI)",
        },
        {
          title: "Learning to summarize from human feedback",
          url: "https://arxiv.org/abs/2009.01325",
          authors: "Stiennon et al. (OpenAI)",
        },
        {
          title: "Deep reinforcement learning from human preferences",
          url: "https://arxiv.org/abs/1706.03741",
          authors: "Christiano et al.",
        },
      ],
    },
  },
  {
    id: "multi-agent-systems",
    name: "Multi-Agent Collaboration",
    category: "Architecture",
    description:
      "Deploy multiple specialized agents that work together to solve complex problems",
    icon: <FaNetworkWired className="text-green-400" />,
    keyPoints: [
      "Multi-agent systems create teams of specialized AI agents that communicate and collaborate to solve complex problems beyond single agent capabilities.",
      "Specialized agents for different tasks",
      "Communication protocols between agents",
      "Consensus mechanisms for decisions",
      "Hierarchical or peer-to-peer organization",
    ],
    example:
      "A research team with agents for data collection, analysis, fact-checking, and report generation working together on market research.",
    benefits: [
      "Handles complex multi-step problems",
      "Reduces single points of failure",
      "Leverages specialized expertise",
      "Scalable problem-solving approach",
    ],
    resources: {
      tutorials: [
        {
          title: "Microsoft AutoGen Multi-Agent Framework",
          url: "https://github.com/microsoft/autogen",
          type: "Framework",
        },
        {
          title: "CrewAI Multi-Agent Orchestration",
          url: "https://docs.crewai.com/",
          type: "Documentation",
        },
        {
          title: "LangGraph Multi-Agent Workflows",
          url: "https://python.langchain.com/docs/langgraph",
          type: "Framework Guide",
        },
      ],
      videos: [
        {
          title: "Multi-Agent Systems with AutoGen",
          url: "https://www.youtube.com/watch?v=BYnTzaGo06I",
          duration: "35 min",
        },
        {
          title: "Building Agent Teams with CrewAI",
          url: "https://www.youtube.com/watch?v=VnBNJJd_JsE",
          duration: "28 min",
        },
        {
          title: "Advanced Multi-Agent Coordination",
          url: "https://www.youtube.com/watch?v=awUu83HVOh8",
          duration: "52 min",
        },
      ],
      papers: [
        {
          title: "Multi-Agent Reinforcement Learning: A Selective Overview",
          url: "https://arxiv.org/abs/1911.10635",
          authors: "Zhang et al.",
        },
        {
          title: "Communicative Multi-Agent Reinforcement Learning",
          url: "https://arxiv.org/abs/2003.06419",
          authors: "Singh et al.",
        },
      ],
    },
  },
  {
    id: "tool-integration",
    name: "Tool Use & API Integration",
    category: "Capability Enhancement",
    description:
      "Enable agents to interact with external tools, APIs, and services",
    icon: <FaCogs className="text-purple-400" />,
    keyPoints: [
      "Tool integration transforms static AI into dynamic agents that can browse the web, run code, send emails, and interact with any API or database.",
      "Function calling capabilities",
      "API authentication and security",
      "Error handling and retry logic",
      "Tool selection and routing",
    ],
    example:
      "A data analysis agent that can query databases, call statistical APIs, generate charts, and send reports via email automatically.",
    benefits: [
      "Extends agent capabilities beyond text",
      "Enables real-world task completion",
      "Integrates with existing workflows",
      "Automates complex operations",
    ],
    resources: {
      tutorials: [
        {
          title: "LangChain Tools and Agents",
          url: "https://python.langchain.com/docs/modules/agents/tools/",
          type: "Framework Guide",
        },
        {
          title: "OpenAI Function Calling",
          url: "https://platform.openai.com/docs/guides/function-calling",
          type: "API Documentation",
        },
        {
          title: "Tool Use with Anthropic Claude",
          url: "https://docs.anthropic.com/claude/docs/tool-use",
          type: "API Guide",
        },
      ],
      videos: [
        {
          title: "Building AI Agents with Tool Access",
          url: "https://www.youtube.com/watch?v=ZzgUqFtxgXI",
          duration: "42 min",
        },
        {
          title: "Function Calling with OpenAI API",
          url: "https://www.youtube.com/watch?v=0lyCjjUIQOw",
          duration: "25 min",
        },
      ],
      papers: [
        {
          title:
            "Toolformer: Language Models Can Teach Themselves to Use Tools",
          url: "https://arxiv.org/abs/2302.04761",
          authors: "Schick et al. (Meta)",
        },
      ],
    },
  },
  {
    id: "memory-systems",
    name: "Persistent Memory & Context",
    category: "Cognitive Architecture",
    description:
      "Implement memory systems for long-term context and personalization",
    icon: <FaDatabase className="text-orange-400" />,
    keyPoints: [
      "Memory systems enable AI agents to remember past conversations, learn from experience, and maintain context across long interactions like humans do.",
      "Long-term memory storage",
      "Context window management",
      "User preference learning",
      "Knowledge graph integration",
    ],
    example:
      "A personal assistant that remembers your preferences, past conversations, project history, and adapts its communication style over time.",
    benefits: [
      "Maintains context across sessions",
      "Personalizes interactions",
      "Builds knowledge over time",
      "Provides consistent experience",
    ],
    resources: {
      tutorials: [
        {
          title: "Building Memory Systems for AI Agents",
          url: "https://python.langchain.com/docs/modules/memory/",
          type: "Framework Guide",
        },
        {
          title: "Vector Databases for AI Memory",
          url: "https://docs.pinecone.io/docs/overview",
          type: "Database Guide",
        },
        {
          title: "Knowledge Graphs for AI Systems",
          url: "https://neo4j.com/developer/graph-data-science/",
          type: "Platform Guide",
        },
      ],
      videos: [
        {
          title: "Persistent Memory in AI Agents",
          url: "https://www.youtube.com/watch?v=VhwPbBw3L5w",
          duration: "38 min",
        },
        {
          title: "Vector Databases Explained",
          url: "https://www.youtube.com/watch?v=dN0lsF2cvm4",
          duration: "22 min",
        },
      ],
      papers: [
        {
          title: "MemGPT: Towards LLMs as Operating Systems",
          url: "https://arxiv.org/abs/2310.08560",
          authors: "Packer et al.",
        },
      ],
    },
  },
  {
    id: "safety-alignment",
    name: "Safety & Alignment Training",
    category: "Ethics & Safety",
    description:
      "Ensure agents behave safely and align with human values and intentions",
    icon: <FaShieldAlt className="text-red-400" />,
    keyPoints: [
      "Safety training ensures AI agents follow ethical guidelines, avoid harmful outputs, and remain helpful even when prompted to behave badly.",
      "Constitutional AI principles",
      "Red team testing and adversarial training",
      "Bias detection and mitigation",
      "Ethical guideline implementation",
    ],
    example:
      "Training an AI agent to refuse harmful requests, provide balanced information, and escalate sensitive situations to human oversight.",
    benefits: [
      "Prevents harmful outputs",
      "Builds user trust",
      "Ensures ethical behavior",
      "Complies with regulations",
    ],
    resources: {
      tutorials: [
        {
          title: "Constitutional AI by Anthropic",
          url: "https://www.anthropic.com/news/constitutional-ai-harmlessness-from-ai-feedback",
          type: "Research Article",
        },
        {
          title: "AI Safety Guidelines by Partnership on AI",
          url: "https://www.partnershiponai.org/",
          type: "Industry Standards",
        },
        {
          title: "Red Teaming AI Systems",
          url: "https://arxiv.org/abs/2202.03286",
          type: "Research Paper",
        },
      ],
      videos: [
        {
          title: "AI Safety and Alignment Explained",
          url: "https://www.youtube.com/watch?v=hEUO6pjwFOo",
          duration: "35 min",
        },
        {
          title: "Constitutional AI Deep Dive",
          url: "https://www.youtube.com/watch?v=VnBNJJd_JsE",
          duration: "48 min",
        },
      ],
      papers: [
        {
          title: "Constitutional AI: Harmlessness from AI Feedback",
          url: "https://arxiv.org/abs/2212.08073",
          authors: "Bai et al. (Anthropic)",
        },
        {
          title: "Red Teaming Language Models with Language Models",
          url: "https://arxiv.org/abs/2202.03286",
          authors: "Perez et al.",
        },
      ],
    },
  },
  {
    id: "continuous-learning",
    name: "Continuous Learning & Adaptation",
    category: "Learning Strategy",
    description:
      "Enable agents to learn and improve from ongoing interactions and new data",
    icon: <FaRocket className="text-cyan-400" />,
    keyPoints: [
      "Continuous learning allows AI agents to improve their performance automatically over time by learning from new data and user interactions.",
      "Online learning capabilities",
      "Performance monitoring and metrics",
      "Automated retraining pipelines",
      "A/B testing for improvements",
    ],
    example:
      "A recommendation agent that continuously learns from user interactions, feedback, and new product data to improve suggestions.",
    benefits: [
      "Adapts to changing environments",
      "Improves performance over time",
      "Handles new scenarios automatically",
      "Maintains relevance and accuracy",
    ],
    resources: {
      tutorials: [
        {
          title: "Online Learning for AI Systems",
          url: "https://scikit-learn.org/stable/modules/scaling_strategies.html",
          type: "Framework Guide",
        },
        {
          title: "MLOps for Continuous Learning",
          url: "https://ml-ops.org/",
          type: "Best Practices",
        },
        {
          title: "Model Monitoring and Retraining",
          url: "https://docs.wandb.ai/guides/models",
          type: "Platform Guide",
        },
      ],
      videos: [
        {
          title: "Continuous Learning in Production AI",
          url: "https://www.youtube.com/watch?v=Ksv9GxNKFT4",
          duration: "44 min",
        },
        {
          title: "MLOps: Continuous Integration for ML",
          url: "https://www.youtube.com/watch?v=9BgIDqAzfuA",
          duration: "1 hr 5 min",
        },
      ],
      papers: [
        {
          title: "Continual Learning in Neural Networks",
          url: "https://arxiv.org/abs/1802.07569",
          authors: "Parisi et al.",
        },
      ],
    },
  },
];

const AIAgentTraining: React.FC = () => {
  const [activeTab, setActiveTab] = useState<
    "examples" | "techniques" | "playground"
  >("examples");
  const [selectedExample, setSelectedExample] = useState<AgentExample | null>(
    null,
  );
  const [selectedTechnique, setSelectedTechnique] =
    useState<TrainingTechnique | null>(null);
  const [playgroundInput, setPlaygroundInput] = useState("");
  const [playgroundOutput, setPlaygroundOutput] = useState("");

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "Beginner":
        return "text-green-400";
      case "Intermediate":
        return "text-yellow-400";
      case "Advanced":
        return "text-red-400";
      default:
        return "text-gray-400";
    }
  };

  const simulateAgentAnalysis = () => {
    if (!playgroundInput.trim()) return;

    setTimeout(() => {
      setPlaygroundOutput(`🤖 **AI AGENT TRAINING ANALYSIS**

📋 **AGENT SPECIFICATION REVIEW**
Input: "${playgroundInput}"

🎯 **RECOMMENDED ARCHITECTURE**
- **Agent Type**: Multi-modal task specialist
- **Memory System**: Vector database with conversation history
- **Tool Integration**: 3-4 specialized APIs recommended
- **Safety Layer**: Constitutional AI with human oversight

⚙️ **TRAINING APPROACH**
1. **Foundation**: Start with general task understanding
2. **Specialization**: Fine-tune on domain-specific data
3. **RLHF**: Human feedback on 1000+ interactions
4. **Evaluation**: A/B testing against success metrics

🚀 **IMPLEMENTATION ROADMAP**
Week 1-2: Data collection and preprocessing
Week 3-4: Initial model training and testing
Week 5-6: RLHF integration and safety testing
Week 7-8: Production deployment and monitoring

📊 **SUCCESS METRICS**
- Task completion rate: Target 95%+
- User satisfaction: Target 4.5/5 stars
- Response time: Target <2 seconds
- Safety compliance: 100% requirement`);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Header */}
      <div className="relative overflow-hidden py-20 px-6">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-cyan-600/20" />
        <div className="relative max-w-6xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8"
          >
            <FaRobot className="text-6xl text-blue-400 mx-auto mb-6" />
            <h1 className="text-5xl md:text-7xl font-bold mb-6">
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                AI Agent Training
              </span>
            </h1>
            <p className="text-xl text-slate-300 max-w-3xl mx-auto leading-relaxed">
              Master the art of building, training, and deploying intelligent AI
              agents. Learn advanced techniques, best practices, and hands-on
              implementation strategies.
            </p>
          </motion.div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 mb-6 sm:mb-8">
        <div className="flex justify-center">
          <div className="backdrop-blur-xl bg-white/10 border border-white/20 rounded-2xl p-1 sm:p-2 overflow-x-auto">
            {[
              {
                id: "examples",
                label: "Training Examples",
                icon: <FaGraduationCap />,
              },
              {
                id: "techniques",
                label: "Advanced Techniques",
                icon: <FaBrain />,
              },
              { id: "playground", label: "Agent Builder", icon: <FaRocket /> },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`px-3 sm:px-4 lg:px-6 py-2 sm:py-3 rounded-xl font-medium transition-all duration-300 flex items-center gap-1 sm:gap-2 text-sm sm:text-base whitespace-nowrap ${
                  activeTab === tab.id
                    ? "bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg"
                    : "text-slate-300 hover:text-white hover:bg-white/10"
                }`}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-6 pb-20">
        <AnimatePresence mode="wait">
          {activeTab === "examples" && (
            <motion.div
              key="examples"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
                {aiAgentExamples.map((example, index) => (
                  <motion.div
                    key={example.id}
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="backdrop-blur-xl bg-white/10 border border-white/20 rounded-2xl p-4 sm:p-6 hover:bg-white/15 transition-all duration-300 cursor-pointer group"
                    onClick={() => setSelectedExample(example)}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <div className="text-2xl">{example.icon}</div>
                      <div>
                        <h3 className="text-xl font-bold text-white group-hover:text-blue-300 transition-colors">
                          {example.title}
                        </h3>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-sm text-slate-400">
                            {example.category}
                          </span>
                          <span
                            className={`text-sm ${getDifficultyColor(example.difficulty)}`}
                          >
                            {example.difficulty}
                          </span>
                        </div>
                      </div>
                    </div>
                    <p className="text-slate-300 text-sm leading-relaxed">
                      {example.description}
                    </p>
                    <div className="mt-4 flex items-center text-blue-400 text-sm font-medium">
                      View Training Example <FaPlay className="ml-2 text-xs" />
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {activeTab === "techniques" && (
            <motion.div
              key="techniques"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {trainingTechniques.map((technique, index) => (
                  <motion.div
                    key={technique.id}
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: index * 0.1 }}
                    className="backdrop-blur-xl bg-white/10 border border-white/20 rounded-2xl p-4 sm:p-6 hover:bg-white/15 transition-all duration-300 cursor-pointer group"
                    onClick={() => setSelectedTechnique(technique)}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <div className="text-2xl">{technique.icon}</div>
                      <div>
                        <h3 className="text-xl font-bold text-white group-hover:text-purple-300 transition-colors">
                          {technique.name}
                        </h3>
                        <span className="text-sm text-slate-400">
                          {technique.category}
                        </span>
                      </div>
                    </div>
                    <p className="text-slate-300 text-sm leading-relaxed mb-4">
                      {technique.description}
                    </p>
                    <div className="flex items-center text-purple-400 text-sm font-medium">
                      Learn Technique <FaLightbulb className="ml-2 text-xs" />
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {activeTab === "playground" && (
            <motion.div
              key="playground"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
            >
              <div className="backdrop-blur-xl bg-white/10 border border-white/20 rounded-2xl p-8">
                <div className="text-center mb-8">
                  <FaRocket className="text-4xl text-cyan-400 mx-auto mb-4" />
                  <h2 className="text-3xl font-bold text-white mb-2">
                    AI Agent Builder
                  </h2>
                  <p className="text-slate-300">
                    Describe your agent requirements and get a customized
                    training strategy
                  </p>
                </div>

                <div className="space-y-6">
                  <div>
                    <label className="block text-white font-medium mb-3">
                      Describe Your AI Agent Requirements:
                    </label>
                    <textarea
                      value={playgroundInput}
                      onChange={(e) => setPlaygroundInput(e.target.value)}
                      placeholder="Example: I want to build an AI agent that helps software developers write better code by analyzing their GitHub repositories, suggesting improvements, and automatically generating documentation..."
                      className="w-full h-32 bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-slate-400 resize-none focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
                    />
                  </div>

                  <button
                    onClick={simulateAgentAnalysis}
                    disabled={!playgroundInput.trim()}
                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-500 text-white py-3 px-6 rounded-xl font-medium hover:from-cyan-600 hover:to-blue-600 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    <FaRocket />
                    Generate Training Strategy
                  </button>

                  {playgroundOutput && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5 }}
                      className="bg-white/5 border border-white/20 rounded-xl p-6"
                    >
                      <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                        <FaLightbulb className="text-yellow-400" />
                        Training Strategy Analysis
                      </h3>
                      <div className="text-slate-300 whitespace-pre-line leading-relaxed">
                        {playgroundOutput}
                      </div>
                    </motion.div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Example Modal */}
      <AnimatePresence>
        {selectedExample && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedExample(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-slate-900 border border-white/20 rounded-2xl p-8 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="text-2xl">{selectedExample.icon}</div>
                  <div>
                    <h2 className="text-2xl font-bold text-white">
                      {selectedExample.title}
                    </h2>
                    <div className="flex items-center gap-2">
                      <span className="text-slate-400">
                        {selectedExample.category}
                      </span>
                      <span
                        className={getDifficultyColor(
                          selectedExample.difficulty,
                        )}
                      >
                        {selectedExample.difficulty}
                      </span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedExample(null)}
                  className="text-slate-400 hover:text-white transition-colors"
                >
                  <FaTimes className="text-xl" />
                </button>
              </div>

              <p className="text-slate-300 mb-8 leading-relaxed">
                {selectedExample.description}
              </p>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                {/* Before */}
                <div className="backdrop-blur-xl bg-red-500/10 border border-red-500/30 rounded-xl p-6">
                  <h3 className="text-lg font-bold text-red-400 mb-4">
                    ❌ {selectedExample.before.title}
                  </h3>
                  <div className="bg-black/20 rounded-lg p-4 mb-4">
                    <pre className="text-slate-300 text-sm whitespace-pre-wrap">
                      {selectedExample.before.content}
                    </pre>
                  </div>
                  <div className="space-y-2">
                    <h4 className="text-red-300 font-medium">Issues:</h4>
                    {selectedExample.before.issues.map((issue, index) => (
                      <div
                        key={index}
                        className="text-slate-400 text-sm flex items-center gap-2"
                      >
                        <span>•</span> {issue}
                      </div>
                    ))}
                  </div>
                </div>

                {/* After */}
                <div className="backdrop-blur-xl bg-green-500/10 border border-green-500/30 rounded-xl p-6">
                  <h3 className="text-lg font-bold text-green-400 mb-4">
                    ✅ {selectedExample.after.title}
                  </h3>
                  <div className="bg-black/20 rounded-lg p-4 mb-4">
                    <pre className="text-slate-300 text-sm whitespace-pre-wrap">
                      {selectedExample.after.content}
                    </pre>
                  </div>
                  <div className="space-y-2">
                    <h4 className="text-green-300 font-medium">
                      Improvements:
                    </h4>
                    {selectedExample.after.improvements.map(
                      (improvement, index) => (
                        <div
                          key={index}
                          className="text-slate-400 text-sm flex items-center gap-2"
                        >
                          <span>•</span> {improvement}
                        </div>
                      ),
                    )}
                  </div>
                </div>
              </div>

              {/* Tips and Use Case */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                <div className="backdrop-blur-xl bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
                  <h3 className="text-lg font-bold text-blue-400 mb-4 flex items-center gap-2">
                    <FaLightbulb /> Training Tips
                  </h3>
                  <div className="space-y-2">
                    {selectedExample.tips.map((tip, index) => (
                      <div
                        key={index}
                        className="text-slate-300 text-sm flex items-start gap-2"
                      >
                        <span className="text-blue-400 mt-1">•</span> {tip}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="backdrop-blur-xl bg-purple-500/10 border border-purple-500/30 rounded-xl p-6">
                  <h3 className="text-lg font-bold text-purple-400 mb-4 flex items-center gap-2">
                    <FaUsers /> Use Cases
                  </h3>
                  <p className="text-slate-300 text-sm leading-relaxed">
                    {selectedExample.useCase}
                  </p>
                </div>
              </div>

              {/* Resources Section */}
              <div className="backdrop-blur-xl bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30 rounded-xl p-6">
                <h3 className="text-xl font-bold text-cyan-400 mb-6 flex items-center gap-2">
                  <FaRocket /> Learning Resources
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Code Examples */}
                  <div>
                    <h4 className="text-lg font-bold text-green-400 mb-3 flex items-center gap-2">
                      <FaCode /> Code Examples
                    </h4>
                    <div className="space-y-2">
                      {selectedExample.resources.codeExamples.map(
                        (code, index) => (
                          <a
                            key={index}
                            href={code.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block p-3 bg-black/30 rounded-lg hover:bg-black/50 transition-all duration-300 border border-green-500/20 hover:border-green-500/40"
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-white font-medium text-sm">
                                  {code.title}
                                </div>
                                <div className="text-green-400 text-xs">
                                  {code.language}
                                </div>
                              </div>
                              <div className="text-green-400 hover:translate-x-1 transition-transform duration-300">
                                →
                              </div>
                            </div>
                          </a>
                        ),
                      )}
                    </div>
                  </div>

                  {/* Tutorials */}
                  <div>
                    <h4 className="text-lg font-bold text-blue-400 mb-3 flex items-center gap-2">
                      <FaGraduationCap /> Tutorials
                    </h4>
                    <div className="space-y-2">
                      {selectedExample.resources.tutorials.map(
                        (tutorial, index) => (
                          <a
                            key={index}
                            href={tutorial.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block p-3 bg-black/30 rounded-lg hover:bg-black/50 transition-all duration-300 border border-blue-500/20 hover:border-blue-500/40"
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-white font-medium text-sm">
                                  {tutorial.title}
                                </div>
                                <div
                                  className={`text-xs ${
                                    tutorial.difficulty === "Beginner"
                                      ? "text-green-400"
                                      : tutorial.difficulty === "Intermediate"
                                        ? "text-yellow-400"
                                        : "text-red-400"
                                  }`}
                                >
                                  {tutorial.difficulty}
                                </div>
                              </div>
                              <div className="text-blue-400 hover:translate-x-1 transition-transform duration-300">
                                →
                              </div>
                            </div>
                          </a>
                        ),
                      )}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                  {/* Videos */}
                  <div>
                    <h4 className="text-lg font-bold text-red-400 mb-3 flex items-center gap-2">
                      <FaPlay /> Video Tutorials
                    </h4>
                    <div className="space-y-2">
                      {selectedExample.resources.videos.map((video, index) => (
                        <a
                          key={index}
                          href={video.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="block p-3 bg-black/30 rounded-lg hover:bg-black/50 transition-all duration-300 border border-red-500/20 hover:border-red-500/40"
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <div className="text-white font-medium text-sm">
                                {video.title}
                              </div>
                              <div className="text-red-400 text-xs">
                                {video.creator} • {video.duration}
                              </div>
                            </div>
                            <div className="text-red-400 hover:translate-x-1 transition-transform duration-300">
                              ▶
                            </div>
                          </div>
                        </a>
                      ))}
                    </div>
                  </div>

                  {/* Documentation */}
                  <div>
                    <h4 className="text-lg font-bold text-purple-400 mb-3 flex items-center gap-2">
                      <FaDatabase /> Documentation
                    </h4>
                    <div className="space-y-2">
                      {selectedExample.resources.documentation.map(
                        (doc, index) => (
                          <a
                            key={index}
                            href={doc.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block p-3 bg-black/30 rounded-lg hover:bg-black/50 transition-all duration-300 border border-purple-500/20 hover:border-purple-500/40"
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-white font-medium text-sm">
                                  {doc.title}
                                </div>
                                <div className="text-purple-400 text-xs">
                                  {doc.type}
                                </div>
                              </div>
                              <div className="text-purple-400 hover:translate-x-1 transition-transform duration-300">
                                →
                              </div>
                            </div>
                          </a>
                        ),
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Technique Modal */}
      <AnimatePresence>
        {selectedTechnique && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedTechnique(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-slate-900 border border-white/20 rounded-2xl p-8 max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="text-2xl">{selectedTechnique.icon}</div>
                  <div>
                    <h2 className="text-2xl font-bold text-white">
                      {selectedTechnique.name}
                    </h2>
                    <span className="text-slate-400">
                      {selectedTechnique.category}
                    </span>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedTechnique(null)}
                  className="text-slate-400 hover:text-white transition-colors"
                >
                  <FaTimes className="text-xl" />
                </button>
              </div>

              <p className="text-slate-300 mb-8 leading-relaxed">
                {selectedTechnique.description}
              </p>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                <div className="backdrop-blur-xl bg-blue-500/10 border border-blue-500/30 rounded-xl p-6">
                  <h3 className="text-lg font-bold text-blue-400 mb-4">
                    Key Points
                  </h3>
                  <div className="space-y-3">
                    {selectedTechnique.keyPoints.map((point, index) => (
                      <div
                        key={index}
                        className="text-slate-300 text-sm flex items-start gap-2"
                      >
                        <span className="text-blue-400 mt-1">•</span> {point}
                      </div>
                    ))}
                  </div>
                </div>

                <div className="backdrop-blur-xl bg-green-500/10 border border-green-500/30 rounded-xl p-6">
                  <h3 className="text-lg font-bold text-green-400 mb-4">
                    Benefits
                  </h3>
                  <div className="space-y-3">
                    {selectedTechnique.benefits.map((benefit, index) => (
                      <div
                        key={index}
                        className="text-slate-300 text-sm flex items-start gap-2"
                      >
                        <span className="text-green-400 mt-1">•</span> {benefit}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="backdrop-blur-xl bg-purple-500/10 border border-purple-500/30 rounded-xl p-6 mb-8">
                <h3 className="text-lg font-bold text-purple-400 mb-4">
                  Example Implementation
                </h3>
                <p className="text-slate-300 text-sm leading-relaxed">
                  {selectedTechnique.example}
                </p>
              </div>

              {/* Resources Section */}
              <div className="backdrop-blur-xl bg-gradient-to-r from-orange-500/10 to-red-500/10 border border-orange-500/30 rounded-xl p-6">
                <h3 className="text-xl font-bold text-orange-400 mb-6 flex items-center gap-2">
                  <FaBrain /> Deep Learning Resources
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Tutorials */}
                  <div>
                    <h4 className="text-lg font-bold text-blue-400 mb-3 flex items-center gap-2">
                      <FaGraduationCap /> Tutorials & Guides
                    </h4>
                    <div className="space-y-2">
                      {selectedTechnique.resources.tutorials.map(
                        (tutorial, index) => (
                          <a
                            key={index}
                            href={tutorial.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block p-3 bg-black/30 rounded-lg hover:bg-black/50 transition-all duration-300 border border-blue-500/20 hover:border-blue-500/40"
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-white font-medium text-sm">
                                  {tutorial.title}
                                </div>
                                <div className="text-blue-400 text-xs">
                                  {tutorial.type}
                                </div>
                              </div>
                              <div className="text-blue-400 hover:translate-x-1 transition-transform duration-300">
                                →
                              </div>
                            </div>
                          </a>
                        ),
                      )}
                    </div>
                  </div>

                  {/* Videos */}
                  <div>
                    <h4 className="text-lg font-bold text-red-400 mb-3 flex items-center gap-2">
                      <FaPlay /> Video Content
                    </h4>
                    <div className="space-y-2">
                      {selectedTechnique.resources.videos.map(
                        (video, index) => (
                          <a
                            key={index}
                            href={video.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block p-3 bg-black/30 rounded-lg hover:bg-black/50 transition-all duration-300 border border-red-500/20 hover:border-red-500/40"
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-white font-medium text-sm">
                                  {video.title}
                                </div>
                                <div className="text-red-400 text-xs">
                                  {video.duration}
                                </div>
                              </div>
                              <div className="text-red-400 hover:translate-x-1 transition-transform duration-300">
                                ▶
                              </div>
                            </div>
                          </a>
                        ),
                      )}
                    </div>
                  </div>

                  {/* Research Papers */}
                  <div>
                    <h4 className="text-lg font-bold text-green-400 mb-3 flex items-center gap-2">
                      <FaSearch /> Research Papers
                    </h4>
                    <div className="space-y-2">
                      {selectedTechnique.resources.papers.map(
                        (paper, index) => (
                          <a
                            key={index}
                            href={paper.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="block p-3 bg-black/30 rounded-lg hover:bg-black/50 transition-all duration-300 border border-green-500/20 hover:border-green-500/40"
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="text-white font-medium text-sm">
                                  {paper.title}
                                </div>
                                <div className="text-green-400 text-xs">
                                  {paper.authors}
                                </div>
                              </div>
                              <div className="text-green-400 hover:translate-x-1 transition-transform duration-300">
                                →
                              </div>
                            </div>
                          </a>
                        ),
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AIAgentTraining;
