import os
from typing import List
from a2a.types import AgentCapabilities, AgentSkill, AgentCard

# Server Configuration
SERVER_HOST = os.getenv('AGENT_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('AGENT_PORT', '10001'))
AGENT_URL = os.getenv('AGENT_URL', f'http://localhost:{SERVER_PORT}/')

# LLM Configuration
API_KEY = os.getenv('API_KEY','282edc7433594a788ce28f3b0572dd2a')
BASE_URL = os.getenv('BASE_URL', 'https://gpt.yunstorm.com')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
LLM_MAX_TOKENS = int(os.getenv('LLM_MAX_TOKENS', '8000'))
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', '0.01'))

# LLM Provider Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'azure').lower()
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION', '2025-04-01-preview')

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Telemetry Configuration
TELEMETRY_ENABLED = int(os.getenv('TELEMETRY_ENABLED', '1'))
JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT", "http://localhost:4318/v1/traces")
TELEMETRY_SERVICE_NAME = os.getenv('TELEMETRY_SERVICE_NAME', 'simple-test-agent')
TELEMETRY_SERVICE_VERSION = os.getenv('TELEMETRY_SERVICE_VERSION', '1.0.0')

API_VERSION_STR = os.getenv("API_VERSION_STR", "/api/v1")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_BASE_URL = f"{API_BASE_URL.rstrip('/')}{API_VERSION_STR}"

# Agent Skills Configuration
def get_agent_skills() -> List[AgentSkill]:
    """Define all agent skills."""
    return [
        AgentSkill(
            id="openmanus.chat",
            name="Conversational Assistant",
            description=(
                "General-purpose conversational AI using LLMs, capable of answering questions, "
                "providing explanations, and maintaining multi-turn dialogue context."
            ),
            tags=["chat", "conversation", "llm", "text"],
            examples=[
                "Ask OpenManus to explain the transformer architecture",
                "Have a back-and-forth discussion on project requirements",
                "Request definitions, analogies, or summaries of complex topics",
                "Clarify ambiguous instructions in a step-by-step manner",
            ],
        ),

        AgentSkill(
            id="openmanus.code_exec",
            name="Code Execution",
            description=(
                "Run arbitrary Python code snippets in a secure sandbox, return stdout, plots, "
                "and errors, and integrate results back into the dialogue."
            ),
            tags=["code", "python", "execution", "sandbox"],
            examples=[
                "Execute a Python function to calculate Fibonacci numbers",
                "Generate and save a matplotlib plot, then retrieve it",
                "Debug a snippet with syntax or runtime errors",
                "Write and run a small script to process a CSV file",
            ],
        ),

        AgentSkill(
            id="openmanus.browser",
            name="Browser Automation",
            description=(
                "Control a headless browser (Playwright) to navigate web pages, perform searches, "
                "click elements, extract content, and take screenshots."
            ),
            tags=["browser", "playwright", "automation", "scraping"],
            examples=[
                "Visit a news site and scrape the top headlines",
                "Automate form filling on an internal dashboard",
                "Search a keyword on Baidu and extract the first 3 results",
                "Capture a screenshot of a dynamic chart on a web page",
            ],
        ),

        AgentSkill(
            id="openmanus.mcp",
            name="MCP Tool Integration",
            description=(
                "Dynamically connect to remote MCP servers over SSE or stdio, discover and invoke their "
                "tools as if they were local, and refresh tool schemas at runtime."
            ),
            tags=["mcp", "remote", "toolchain", "dynamic"],
            examples=[
                "Connect to an internal analytics MCP service and run `get_metrics`",
                "List available tools from a downstream agent via SSE",
                "Invoke a remote `generate_report` tool and integrate its output",
                "Handle schema changes of remote tools without restarting",
            ],
        ),

        AgentSkill(
            id="openmanus.text_edit",
            name="Text Editing & Replacement",
            description=(
                "Perform structured text edits such as find-and-replace, templating, and content transformations, "
                "leveraging a programmable editor interface."
            ),
            tags=["text", "editing", "replace", "templating"],
            examples=[
                "Replace all dates in a document with a new format",
                "Apply a custom template to generate boilerplate code",
                "Batch-rename variables in a codebase snippet",
                "Reformat markdown headings according to style guide",
            ],
        ),
        
        AgentSkill(
            id="openmanus.terminate",
            name="Task Termination & Cleanup",
            description=(
                "Gracefully terminate multi-step processes, clean up resources (browsers, MCP connections), "
                "and report final success or failure status."
            ),
            tags=["terminate", "cleanup", "lifecycle", "shutdown"],
            examples=[
                "Cancel a long-running browser scrape midway",
                "Disconnect from all MCP servers after job completion",
                "Handle exceptions by terminating the task with an error message",
                "Ensure sandbox resources are freed after execution",
            ],
        ),

    ]
'''
def get_extended_skills() -> List[AgentSkill]:
    """Define extended skills for authenticated users."""
    base_skills = get_agent_skills()
    extended_skills = [
        AgentSkill(
            id='streaming_responses',
            name='Streaming Responses',
            description='Enhanced conversational capabilities with real-time streaming responses for better user experience',
            tags=['streaming', 'real-time', 'conversation', 'enhanced'],
            examples=[
                'Get real-time streaming responses',
                'Experience faster response times',
                'Receive responses as they are generated',
            ],
        ),
        AgentSkill(
            id='advanced_task_features',
            name='Advanced Task Features',
            description='Enhanced task management with priority processing, detailed progress tracking, and advanced artifact generation',
            tags=['advanced', 'task', 'priority', 'detailed', 'enhanced'],
            examples=[
                'Access priority task processing',
                'Get detailed progress reports',
                'Generate enhanced artifacts',
                'Advanced task monitoring and control',
            ],
        ),
    ]
    return base_skills + extended_skills
'''
def get_public_agent_card() -> AgentCard:
    """Create the public agent card."""
    return AgentCard(
        name='OpenManus Agent',
        description='A versatile agent based on OpenManus, supporting local & MCP tools.',
        url=AGENT_URL,
        version='1.0.0',
        defaultInputModes=['text'],
        defaultOutputModes=['text'],
        capabilities=AgentCapabilities(streaming=False),
        skills=get_agent_skills(),
        supportsAuthenticatedExtendedCard=True,
    )
'''
def get_extended_agent_card() -> AgentCard:
    """Create the extended agent card for authenticated users."""
    public_card = get_public_agent_card()
    return public_card.model_copy(
        update={
            'name': 'Simple Test Agent - Extended Edition',
            'description': 'Enhanced version of the Simple Test Agent with additional capabilities for authenticated users, including advanced testing features and extended functionality.',
            'version': '1.1.0',
            'skills': get_extended_skills(),
        }
    ) 
'''