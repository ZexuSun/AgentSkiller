"""
Step 2: Entity Extraction

Extract entities from domain topics using LLM.

Input: domain_topics.json
Output: entities.json
"""

import logging
from pathlib import Path

from ..models.state import WorkflowState
from ..models.entities import EntityInfo, AttributeInfo
from ..config.settings import get_settings
from ..prompts import ENTITY_EXTRACTION_PROMPT
from .base import step_handler, save_json, load_json, get_client

logger = logging.getLogger(__name__)


def fix_entity_id_attribute(entity_name: str, attributes: dict) -> dict:
    """确保 entity 有正确命名的 id 字段。
    
    id 字段格式应为: {entity_name.lower()}_id
    例如: InvoiceDocument -> invoicedocument_id
    """
    expected_id = f"{entity_name.lower()}_id"
    
    if expected_id in attributes:
        return attributes  # 已经正确
    
    # 查找可能的错误命名 id 字段并重命名
    for attr_name in list(attributes.keys()):
        if attr_name.endswith("_id"):
            attributes[expected_id] = attributes.pop(attr_name)
            logger.debug(f"Renamed '{attr_name}' to '{expected_id}' for entity '{entity_name}'")
            break
    
    return attributes


@step_handler("s02_entity_extraction", auto_retry=True)
def entity_extraction_step(state: WorkflowState) -> WorkflowState:
    """
    Extract entities from domain topics.
    
    Process:
    1. Load domain topics
    2. Generate entities for all domains in a single LLM call
    3. Validate entity structure
    
    Output:
    - entities.json: Dict of entity name -> EntityInfo
    """
    settings = get_settings()
    outputs_dir = settings.paths.outputs_dir
    
    # Load domain topics
    topics_path = Path(state.domain_topics_path)
    topics_data = load_json(topics_path)
    domains = topics_data.get("domains", [])
    
    logger.info(f"Extracting entities from {len(domains)} domains")
    
    # Check if already processed
    output_path = outputs_dir / "entities.json"
    if output_path.exists():
        logger.info("Entities already extracted, loading existing file")
        state.entities_path = str(output_path)
        return state
    
    # Format all domains into the prompt
    domains_text = "\n".join(f"- {domain}" for domain in domains)
    prompt = ENTITY_EXTRACTION_PROMPT.format(domains=domains_text)
    
    # Single LLM call to extract all entities
    client = get_client()
    try:
        response = client.chat(query=prompt, model_type="textual")
        entities = response.parse_json()
    except Exception as e:
        logger.error(f"Failed to extract entities: {e}")
        raise
    
    # Post-processing: Filter out "Session" entity
    if "Session" in entities:
        del entities["Session"]
        logger.info("Removed 'Session' entity")
    
    # Post-processing: Fix xx_id attribute naming
    for entity_name, entity_data in entities.items():
        if "attributes" in entity_data:
            entity_data["attributes"] = fix_entity_id_attribute(
                entity_name, entity_data["attributes"]
            )
    
    # Save results
    save_json(entities, output_path)
    state.entities_path = str(output_path)
    
    logger.info(f"Entity extraction complete: {len(entities)} entities from {len(domains)} domains")
    return state
