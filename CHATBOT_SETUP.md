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
- ✅ Server-side `/api/chat` endpoint with local fallback responses

## 🚀 To Enable OpenAI API Integration

### Step 1: Get OpenAI API Key

1. Go to [OpenAI API](https://platform.openai.com/api-keys)
2. Create an account or sign in
3. Generate a new API key
4. Copy the key (starts with `sk-`)

### Step 2: Add Environment Variable

Add to your `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### Step 3: Start the Existing API Endpoint

This repo already includes the chat endpoint:

```text
POST /api/chat
```

It is wired through:

- `client/components/ChatBot.tsx`
- `client/api/chat.ts`
- `server/routes/chat.ts`
- `shared/chat.ts`

### Step 4: Run the App

```bash
npm run dev
```

## 💡 Current Status

**Currently Running**: Server-backed chat with local intelligent fallback

- The chatbot is fully functional without an API key
- If `OPENAI_API_KEY` is set, `/api/chat` calls OpenAI from the server
- If the API key is missing or the request fails, the chatbot falls back locally
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

Edit the fallback responses in `shared/chat.ts`.

### Change Appearance

- Colors: Update the gradient classes in the floating button
- Size: Modify the chat window dimensions (currently 96w x 500h)
- Position: Change the `fixed bottom-6 right-6` classes

### Add More Knowledge

Update the system prompt and fallback responses in `shared/chat.ts`.

## 🚀 The chatbot is now live on your website!

Users can click the chat bubble in the bottom right corner to start asking questions about your professional background and achievements.
