import logging
import logging.handlers
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from openai import AsyncOpenAI, AsyncAzureOpenAI
from a2a.utils.telemetry import SpanKind
from config import (
    API_KEY, 
    LLM_MODEL, 
    LLM_MAX_TOKENS, 
    LLM_TEMPERATURE,
    BASE_URL,
    AZURE_API_VERSION,
    LLM_PROVIDER
)
# from logging_config import get_# logger

# # logger = get_# logger('llm_client')

class MessageContent:
    """Base class for message content."""
    pass

class TextContent(MessageContent):
    """Text content for messages."""
    def __init__(self, text: str):
        self.text = text
    
    def to_dict(self) -> Dict[str, str]:
        return {"type": "text", "text": self.text}

class ImageContent(MessageContent):
    """Image content for messages."""
    def __init__(self, image_url: str, image_type: str = "data:image/png;base64"):
        self.image_url = image_url
        self.image_type = image_type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {
                "url": f"{self.image_type},{self.image_url}"
            }
        }

class Message:
    """Represents a chat message with support for multimodal content."""
    def __init__(self, role: str, content: Union[str, List[MessageContent]]):
        self.role = role
        if isinstance(content, str):
            self.content = [TextContent(content)]
        else:
            self.content = content
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": [item.to_dict() for item in self.content]
        }
    
    @classmethod
    def text_message(cls, role: str, text: str) -> 'Message':
        """Create a text-only message."""
        return cls(role, text)
    
    @classmethod
    def image_message(cls, role: str, text: str, image_url: str, image_type: str = "data:image/png;base64") -> 'Message':
        """Create a message with both text and image."""
        content = [
            ImageContent(image_url, image_type),
            TextContent(text)
        ]
        return cls(role, content)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def create_chat_completion(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        temperature: float,
        model: str
    ) -> str:
        """Create a chat completion and return the response content."""
        pass
    
    @abstractmethod
    async def create_chat_completion_stream(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        temperature: float,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion and yield response chunks."""
        pass

class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider implementation."""
    
    def __init__(self, api_key: str, base_url: str, api_version: str):
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=base_url,
            api_version=api_version
        )

    async def create_chat_completion(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        temperature: float,
        model: str
    ) -> str:
        """Create a chat completion using Azure OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[msg.to_dict() for msg in messages],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Azure OpenAI API error: {e}")

    async def create_chat_completion_stream(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        temperature: float,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion using Azure OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[msg.to_dict() for msg in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"Azure OpenAI streaming API error: {e}")

class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def create_chat_completion(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        temperature: float,
        model: str
    ) -> str:
        """Create a chat completion using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[msg.to_dict() for msg in messages],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")

    async def create_chat_completion_stream(
        self, 
        messages: List[Message], 
        max_tokens: int, 
        temperature: float,
        model: str
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[msg.to_dict() for msg in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise Exception(f"OpenAI streaming API error: {e}")


class LLMClient:
    """Client for interacting with LLM providers."""
    
    def __init__(self):
        self.provider = self._create_provider()

    def _create_provider(self) -> LLMProvider:
        """Create the appropriate LLM provider based on configuration."""
        if LLM_PROVIDER == "azure":
            return AzureOpenAIProvider(API_KEY, BASE_URL, AZURE_API_VERSION)
        elif LLM_PROVIDER == "openai":
            return OpenAIProvider(API_KEY)

    async def chat_completion(
        self, 
        messages: List[Message], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> str:
        """Create a chat completion."""
        max_tokens = max_tokens or LLM_MAX_TOKENS
        temperature = temperature or LLM_TEMPERATURE
        model = model or LLM_MODEL
        
        return await self.provider.create_chat_completion(
            messages, max_tokens, temperature, model
        )

    async def chat_completion_stream(
        self, 
        messages: List[Message], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion."""
        max_tokens = max_tokens or LLM_MAX_TOKENS
        temperature = temperature or LLM_TEMPERATURE
        model = model or LLM_MODEL
        
        async for chunk in self.provider.create_chat_completion_stream(
            messages, max_tokens, temperature, model
        ):
            yield chunk

    async def simple_chat(self, user_input: str, system_prompt: str) -> str:
        """Simple chat with system prompt and user input."""
        messages = [
            Message.text_message("system", system_prompt),
            Message.text_message("user", user_input)
        ]
        return await self.chat_completion(messages)

    async def simple_chat_stream(self, user_input: str, system_prompt: str) -> AsyncGenerator[str, None]:
        """Simple streaming chat with system prompt and user input."""
        messages = [
            Message.text_message("system", system_prompt),
            Message.text_message("user", user_input)
        ]
        async for chunk in self.chat_completion_stream(messages):
            yield chunk

    async def image_chat(self, user_text: str, image_url: str, system_prompt: str, image_type: str = "data:image/png;base64") -> str:
        """Chat with image input."""
        messages = [
            Message.text_message("system", system_prompt),
            Message.image_message("user", user_text, image_url, image_type)
        ]
        return await self.chat_completion(messages)

    def is_available(self) -> bool:
        """Check if the LLM client is available."""
        return self.provider is not None 