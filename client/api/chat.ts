// API handler for ChatBot - you'll need to add your OpenAI API key

const SYSTEM_PROMPT = `You are Aakriti Gupta's AI assistant on her professional portfolio website. You should answer questions about her background, work, and achievements in a friendly, conversational manner. Here's what you should know about Aakriti:

PROFESSIONAL BACKGROUND:
- AI Engineer and Technology Leader with expertise in machine learning and large-scale systems
- Worked at major tech companies: Meta (Facebook), eBay, Yahoo
- At Meta: Built ML-driven advertising systems serving billions of users worldwide
- At eBay: Scaled e-commerce infrastructure handling millions of daily transactions
- At Yahoo: Early engineering experience in large-scale web technologies
- Founded an AI company specializing in product image transformation and enhancement

CURRENT BUSINESS - SWARNAWASTRA:
- Founder of Swarnawastra, a luxury fashion-tech brand
- Mission: Democratizing access to luxury through AI, gold, and lab-grown diamonds
- Uses AI to make luxury design and high-end fashion more accessible
- Innovative approach combining technology with traditional luxury craftsmanship

NOTABLE PROJECTS:
- Developed face recognition systems for the Indian Parliament for government security
- Created PPE detection systems for Tata to enhance workplace safety
- Built ML systems for product image enhancement and transformation
- Worked on large-scale advertising and e-commerce platforms

ACHIEVEMENTS & RECOGNITION:
- Recognized by Yann LeCun (Turing Award winner) for innovative AI contributions
- Successfully founded and led technology companies
- Expert in AI, machine learning, computer vision, and large-scale systems
- Strong engineering background with proven track record at top tech companies

CURRENT FOCUS:
- Scaling Swarnawastra to democratize luxury with AI
- Developing innovative AI applications for fashion and luxury goods
- Working with gold and lab-grown diamonds in luxury products
- Building technology that makes high-end design accessible to more people

Keep responses conversational, helpful, and focused on her professional achievements. Keep answers concise (2-3 sentences max). If asked about something you don't know, politely say you don't have that specific information but offer to help with what you do know about her work.`;

export async function callOpenAI(userMessage: string): Promise<string> {
  try {
    // You'll need to add your OpenAI API key as an environment variable
    const apiKey = process.env.OPENAI_API_KEY;

    if (!apiKey) {
      console.warn("OpenAI API key not configured, using local responses");
      return getLocalResponse(userMessage);
    }

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: SYSTEM_PROMPT,
          },
          {
            role: "user",
            content: userMessage,
          },
        ],
        max_tokens: 150,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.choices[0].message.content.trim();
  } catch (error) {
    console.error("Error calling OpenAI API:", error);
    // Fallback to local responses if API fails
    return getLocalResponse(userMessage);
  }
}

// Fallback local responses when API is not available
function getLocalResponse(question: string): string {
  const lowerQuestion = question.toLowerCase();

  if (lowerQuestion.includes("companies") || lowerQuestion.includes("work")) {
    return "Aakriti has worked at several major tech companies including Meta (Facebook), eBay, and Yahoo. At Meta, she built ML-driven advertising systems serving billions of users. She's also founded her own AI company focused on product image transformation.";
  }

  if (lowerQuestion.includes("swarnawastra")) {
    return "Swarnawastra is Aakriti's luxury fashion-tech brand that democratizes access to luxury through AI, gold, and lab-grown diamonds. She's using AI to make luxury design and high-end fashion more accessible to people.";
  }

  if (
    lowerQuestion.includes("achievement") ||
    lowerQuestion.includes("award")
  ) {
    return "One of Aakriti's biggest achievements is being recognized by Yann LeCun, the Turing Award winner and AI pioneer. She's also successfully founded multiple technology companies and contributed to large-scale AI systems.";
  }

  if (lowerQuestion.includes("yann lecun")) {
    return "Yann LeCun, the Turing Award winner and deep learning pioneer, recognized Aakriti for her innovative AI contributions. This is a significant honor in the AI community.";
  }

  if (lowerQuestion.includes("projects") || lowerQuestion.includes("ai")) {
    return "Aakriti has worked on fascinating AI projects including face recognition systems for the Indian Parliament, PPE detection systems for Tata workplace safety, and ML systems for product image enhancement.";
  }

  if (lowerQuestion.includes("meta") || lowerQuestion.includes("facebook")) {
    return "At Meta (Facebook), Aakriti built ML-driven advertising systems that serve billions of users worldwide. This involved complex work in machine learning, distributed systems, and performance optimization at massive scale.";
  }

  if (lowerQuestion.includes("parliament")) {
    return "Aakriti developed face recognition systems for the Indian Parliament for government security. This was a high-security, mission-critical project requiring the highest levels of accuracy and reliability.";
  }

  if (lowerQuestion.includes("tata")) {
    return "For Tata, Aakriti created PPE detection systems to enhance workplace safety. This is a great example of using AI for social good - automatically detecting whether proper safety equipment is being used.";
  }

  if (lowerQuestion.includes("luxury") || lowerQuestion.includes("diamonds")) {
    return "Through Swarnawastra, Aakriti is working with gold and lab-grown diamonds to create luxury products. She's using AI to make luxury design more accessible and democratic.";
  }

  if (lowerQuestion.includes("contact") || lowerQuestion.includes("email")) {
    return "You can connect with Aakriti through her LinkedIn profile or GitHub. There's also a resume request form on this website where you can get her complete professional details.";
  }

  // Default response
  return "That's a great question! Aakriti has extensive experience in AI, machine learning, and technology leadership. She's worked at Meta, eBay, and Yahoo, founded Swarnawastra, and has been recognized by AI leaders like Yann LeCun. What specific aspect would you like to know more about?";
}
