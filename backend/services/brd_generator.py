import os
import asyncio
from dotenv import load_dotenv
from ollama import AsyncClient
from typing import List, Dict, AsyncGenerator
import re
from datetime import datetime

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:instruct")

class BRDGenerator:
    def __init__(self):
        try:
            self.client = AsyncClient(host="http://localhost:11434")
            print("✓ BRD Generator initialized")
        except Exception as e:
            print(f"✗ BRD Generator error: {e}")
            self.client = None

    BRD_GENERATION_PROMPT = """You are a Senior Business Analyst creating a comprehensive Business Requirements Document (BRD).

INSTRUCTIONS:
1. Generate a complete, professional BRD based ONLY on the conversation provided
2. Use the exact section headers provided below
3. Be specific, actionable, and comprehensive
4. Include concrete details from the conversation
5. Format with proper markdown

BRD STRUCTURE (use these exact headers):
# Business Requirements Document

## 1. Executive Summary
[Provide a high-level overview of the project based on discussion]

## 2. Project Overview
[Describe the business context, background, and scope]

## 3. Business Objectives
[List specific, measurable business goals]

## 4. Stakeholder Analysis
[Identify key stakeholders, users, and their needs]

## 5. Functional Requirements
[List specific system functionalities and features required]

## 6. Non-Functional Requirements
[Specify performance, security, usability requirements]

## 7. Constraints and Assumptions
[Document technical and business constraints, key assumptions]

## 8. Timeline and Milestones
[Outline project timeline with key milestones]

## 9. Success Criteria
[Define measurable success metrics and KPIs]

## 10. Risks and Mitigations
[Identify potential risks and mitigation strategies]

Now, here is the conversation you need to base the BRD on:

{conversation_summary}

Generate the complete Business Requirements Document:"""

    def _extract_conversation_summary(self, messages: List[Dict]) -> str:
        """Extract key information from conversation for BRD generation"""
        conversation_summary = "CONVERSATION SUMMARY:\n\n"
        
        # Extract all user messages (business requirements)
        user_messages = []
        for msg in messages:
            if msg["role"] == "user":
                user_messages.append(msg["content"])
        
        # Extract all analyst questions for context
        analyst_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                analyst_messages.append(msg["content"])
        
        # Start with business idea
        if user_messages:
            conversation_summary += f"Business Idea: {user_messages[0]}\n\n"
        
        # Extract key Q&A pairs
        conversation_summary += "Key Requirements Gathered:\n"
        qa_pairs = []
        
        # Pair questions with answers
        for i in range(len(messages)):
            if i < len(messages) - 1:
                if messages[i]["role"] == "assistant" and messages[i+1]["role"] == "user":
                    q = messages[i]["content"][:100] + ("..." if len(messages[i]["content"]) > 100 else "")
                    a = messages[i+1]["content"][:150] + ("..." if len(messages[i+1]["content"]) > 150 else "")
                    qa_pairs.append(f"Q: {q}\nA: {a}\n")
        
        # Add up to 10 most relevant Q&A pairs
        for qa in qa_pairs[-10:]:
            conversation_summary += qa + "\n"
        
        return conversation_summary

    async def generate_brd_stream(self, messages: List[Dict], retriever=None) -> AsyncGenerator[str, None]:
        """Generate BRD as a stream"""
        
        if self.client is None:
            yield "Error: BRD generation service unavailable"
            return
        
        try:
            # Build conversation summary
            conversation_summary = self._extract_conversation_summary(messages)
            
            # Create system prompt with conversation context
            system_prompt = self.BRD_GENERATION_PROMPT.format(conversation_summary=conversation_summary)
            
            # Add metadata
            metadata = f"""**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Based on:** Requirements gathering session ({len([m for m in messages if m['role'] == 'user'])} user responses)
**Status:** Draft

"""
            
            # Generate BRD
            response = await self.client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the complete Business Requirements Document."}
                ],
                options={
                    "temperature": 0.2,  # Low temperature for consistent output
                    "num_predict": 4000,  # Large token limit for comprehensive BRD
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            )
            
            full_brd = response['message']['content']
            
            # Add metadata to the beginning
            full_brd = metadata + full_brd
            
            # Stream BRD in chunks (words for faster display)
            words = full_brd.split()
            chunk_size = 8  # 8 words at a time for smooth streaming
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if i + chunk_size < len(words):
                    chunk += ' '
                yield chunk
                await asyncio.sleep(0.005)  # Fast streaming
            
        except Exception as e:
            yield f"Error generating BRD: {str(e)}"