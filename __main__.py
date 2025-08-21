import uvicorn
import sys
import os
from datetime import datetime

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from agent_executor import OpenManusAgentExecutor
from config import (
    SERVER_HOST, 
    SERVER_PORT, 
    AGENT_URL, 
    API_KEY,
    LLM_MODEL,
    TELEMETRY_ENABLED,
    get_public_agent_card,
)
from logging_config import get_logger

logger = get_logger('OpenManus_agent')

def create_server():
    if not API_KEY:
        logger.error("API_KEY environment variable is required!")
        logger.error("Set API_KEY environment variable or create .env file")
        sys.exit(1)

    logger.info(f"Config: {SERVER_HOST}:{SERVER_PORT}, Model: {LLM_MODEL}")
    logger.info("Creating agent cards...")
    public_agent_card = get_public_agent_card()
    #extended_agent_card = get_extended_agent_card()
    logger.info("Initializing components...")
    request_handler = DefaultRequestHandler(
        agent_executor=OpenManusAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
        #extended_agent_card=extended_agent_card,
    )
    
    app = server.build()
    
    # if TELEMETRY_ENABLED:
    #     setup_telemetry_only()
    # else:
    #     logger.info("ℹ️  Telemetry disabled")
    
    return app

if __name__ == '__main__':
    logger.info("Starting OpenManus Agent...")
    try:
        logger.info(f"Starting server at {AGENT_URL}...")
        uvicorn.run(create_server(), host=SERVER_HOST, port=SERVER_PORT, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start agent: {e}", exc_info=True)
        sys.exit(1) 
