# ChatBot Setup Instructions

## 🤖 Virtual Assistant ChatBot

I've added a professional virtual assistant chatbot to your website that can answer questions about your background, work, and achievements.

## ✨ Features Implemented

### 🎯 Knowledge Base

The chatbot knows about:

- ✅ Professional background (AI, ML systems at Meta, Yahoo, eBay)
- ✅ Swarnawastra luxury fashion-tech brand
- ✅ Projects: face recognition for Indian Parliament, safety systems for Tata
- ✅ Awards and recognition from Yann LeCun
- ✅ Current business: democratizing luxury with AI, gold, lab-grown diamonds

### 🔧 Technical Features

- ✅ Floating chat bubble (bottom right)
- ✅ Responsive chat window
- ✅ Mobile-friendly design
- ✅ Professional dark/light theme integration
- ✅ Suggested questions for easy starting
- ✅ Typing indicators and smooth animations
- ✅ Local fallback responses (currently active)

## 🚀 To Enable OpenAI API Integration

### Step 1: Get OpenAI API Key

1. Go to [OpenAI API](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Generate a new API key
4. Copy the key (starts with `sk-`)

### Step 2: Add Environment Variable

Add to your `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

### Step 3: Create API Endpoint

Create `pages/api/chat.ts` (or `app/api/chat/route.ts` for App Router):

```typescript
import { callOpenAI } from "@/api/chat";

export async function POST(request: Request) {
  try {
    const { message } = await request.json();
    const response = await callOpenAI(message);

    return new Response(JSON.stringify({ response }), {
      headers: { "Content-Type": "application/json" },
    });
  } catch (error) {
    return new Response(
      JSON.stringify({ error: "Failed to process message" }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}
```

### Step 4: Enable API in ChatBot

In `client/components/ChatBot.tsx`, uncomment the API call section (lines ~85-95) and remove the local fallback.

## 💡 Current Status

**Currently Running**: Local intelligent responses (no API key required)

- The chatbot is fully functional with smart local responses
- Covers all major questions about your background
- Professional and conversational tone
- Ready to use immediately

**After API Setup**: Enhanced OpenAI-powered responses

- More natural conversation flow
- Better handling of edge cases
- Ability to answer more complex questions

## 📱 User Experience

### Chat Bubble Location

- Fixed position: bottom right corner
- Blue gradient design matching your site theme
- Smooth animations and hover effects

### Suggested Questions

When users first open the chat:

- "What companies have you worked for?"
- "Tell me about Swarnawastra"
- "What is your biggest achievement?"

### Response Examples

- **Companies**: "Aakriti has worked at several major tech companies including Meta (Facebook), eBay, and Yahoo..."
- **Swarnawastra**: "Swarnawastra is Aakriti's luxury fashion-tech brand that democratizes access to luxury through AI..."
- **Achievements**: "One of Aakriti's biggest achievements is being recognized by Yann LeCun, the Turing Award winner..."

## 🔧 Customization Options

### Modify Responses

Edit the responses in `client/components/ChatBot.tsx` in the `simulateAIResponse` function.

### Change Appearance

- Colors: Update the gradient classes in the floating button
- Size: Modify the chat window dimensions (currently 96w x 500h)
- Position: Change the `fixed bottom-6 right-6` classes

### Add More Knowledge

Update the `SYSTEM_PROMPT` in `client/api/chat.ts` to include additional information about your work.

## 🚀 The chatbot is now live on your website!

Users can click the chat bubble in the bottom right corner to start asking questions about your professional background and achievements.
