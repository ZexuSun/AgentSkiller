"""
Blueprint models for MCP server definition.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class FunctionParameter(BaseModel):
    """Parameter definition for a blueprint function."""
    
    description: str
    type: str  # e.g., "string", "int", "float", "bool", "list", "dict"
    range: str = ""  # Value constraints
    required: bool = True
    default: Optional[Any] = None


class FunctionDefinition(BaseModel):
    """
    Function definition within an MCP blueprint.
    
    Represents a tool/function that the MCP server exposes.
    """
    
    name: str
    description: str
    legal_accessor: List[str] = Field(default_factory=list)  # Who can call this function
    parameters: Dict[str, FunctionParameter] = Field(default_factory=dict)
    returns: Optional[str] = None  # Return type description
    
    def get_required_params(self) -> List[str]:
        """Get list of required parameter names."""
        return [
            name for name, param in self.parameters.items() 
            if param.required
        ]
    
    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for name, param in self.parameters.items():
            prop = {
                "type": self._map_type(param.type),
                "description": param.description,
            }
            if param.range:
                prop["description"] += f" (range: {param.range})"
            properties[name] = prop
            
            if param.required:
                required.append(name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    
    @staticmethod
    def _map_type(type_str: str) -> str:
        """Map our types to JSON Schema types."""
        mapping = {
            "string": "string",
            "str": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "object": "object",
        }
        return mapping.get(type_str.lower(), "string")


class RelationshipAttribute(BaseModel):
    """Attribute within a relationship definition."""
    
    type: str
    value_from_entity: str = "Random"  # Which entity provides this value
    range: str = ""


class RelationshipDefinition(BaseModel):
    """
    Relationship between entities in a blueprint.
    
    Represents a many-to-many or one-to-many relationship table.
    """
    
    name: str
    description: str
    attributes: Dict[str, RelationshipAttribute] = Field(default_factory=dict)
    source_entity: Optional[str] = None
    target_entity: Optional[str] = None


class MCPBlueprint(BaseModel):
    """
    Complete MCP Server blueprint definition.
    
    Defines:
    - Core entity (the person/user type this server serves)
    - Peripheral entities (related entities, max 3)
    - Relationships between entities
    - Functions/tools exposed by the server
    """
    
    MCP_server_name: str
    description: str
    core_entity: str
    peripheral_entities: List[str] = Field(default_factory=list)
    relationships: List[RelationshipDefinition] = Field(default_factory=list)
    functions: List[FunctionDefinition] = Field(default_factory=list)
    
    # Metadata
    domain: Optional[str] = None  # Source domain topic
    version: str = "1.0"
    
    @field_validator('peripheral_entities')
    @classmethod
    def validate_peripheral_limit(cls, v):
        if len(v) > 3:
            raise ValueError("MCP server can have at most 3 peripheral entities")
        return v
    
    @field_validator('functions')
    @classmethod
    def validate_min_functions(cls, v):
        if len(v) < 10:
            raise ValueError("MCP server must have at least 10 functions")
        return v
    
    def get_all_entities(self) -> List[str]:
        """Get all entities (core + peripheral)."""
        return [self.core_entity] + self.peripheral_entities
    
    def get_function_names(self) -> List[str]:
        """Get all function names."""
        return [f.name for f in self.functions]
    
    def get_function(self, name: str) -> Optional[FunctionDefinition]:
        """Get function by name."""
        for f in self.functions:
            if f.name == name:
                return f
        return None
    
    def to_tool_list(self) -> List[Dict[str, Any]]:
        """Convert functions to OpenAI tool format."""
        return [
            {"type": "function", "function": f.to_openai_function()}
            for f in self.functions
        ]


class BlueprintFixupResult(BaseModel):
    """Result of blueprint fixup process."""
    
    server_name: str
    original_entities: List[str]
    fixed_entities: List[str]
    entity_mappings: Dict[str, str]  # old_name -> new_name
    tool_list_path: Optional[str] = None
    success: bool = True
    errors: List[str] = Field(default_factory=list)
