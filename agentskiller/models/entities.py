"""
Entity models for domain and entity extraction steps.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class AttributeInfo(BaseModel):
    """Information about an entity attribute."""
    
    description: str
    type: str  # e.g., "string", "int", "float", "datetime", "enum"
    range: str = ""  # Value range or constraints, e.g., "1-100", "enum:active,inactive"
    
    def get_example_value(self) -> Any:
        """Generate an example value based on type and range."""
        if self.type == "int":
            return 42
        elif self.type == "float":
            return 3.14
        elif self.type == "bool":
            return True
        elif self.type == "datetime":
            return "2025-01-23T15:00:00"
        elif "enum:" in self.range:
            values = self.range.replace("enum:", "").split(",")
            return values[0].strip() if values else "value"
        else:
            return "example_value"


class EntityInfo(BaseModel):
    """
    Full entity information including attributes.
    
    An entity represents a domain concept like "Patient", "Flight", "Order".
    """
    
    attributes: Dict[str, AttributeInfo]
    description: str
    is_person: bool = False  # True if entity represents a person/user type
    
    def get_id_field(self) -> Optional[str]:
        """Get the primary ID field for this entity."""
        for attr_name in self.attributes:
            if attr_name.lower().endswith("_id") or attr_name.lower() == "id":
                return attr_name
        return None
    
    def get_attribute_names(self) -> List[str]:
        """Get all attribute names."""
        return list(self.attributes.keys())


class EntitiesOutput(BaseModel):
    """Output from entity generation step."""
    entities: Dict[str, EntityInfo]


class EntityGraphNode(BaseModel):
    """Node in the entity relationship graph."""
    
    id: str  # Entity name
    is_person: bool = False
    attributes: List[str] = Field(default_factory=list)
    description: str = ""


class EntityGraphEdge(BaseModel):
    """Edge in the entity relationship graph."""
    
    source: str  # Source entity name
    target: str  # Target entity name
    relationship: str  # Relationship type/name
    description: str = ""


class EntityGraph(BaseModel):
    """
    NetworkX-compatible entity relationship graph.
    
    Used for:
    - Finding entity neighbors for blueprint generation
    - Traversing entity dependencies
    """
    
    nodes: List[EntityGraphNode]
    links: List[EntityGraphEdge]  # "links" for NetworkX JSON compatibility
    
    def get_neighbors(self, entity_name: str) -> List[str]:
        """Get all entities connected to the given entity."""
        neighbors = set()
        for edge in self.links:
            if edge.source == entity_name:
                neighbors.add(edge.target)
            elif edge.target == entity_name:
                neighbors.add(edge.source)
        return list(neighbors)
    
    def get_person_entities(self) -> List[str]:
        """Get all entities marked as person types."""
        return [node.id for node in self.nodes if node.is_person]
    
    def to_networkx_dict(self) -> Dict[str, Any]:
        """Convert to NetworkX JSON format."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "is_person": node.is_person,
                    "attributes": node.attributes,
                    "description": node.description,
                }
                for node in self.nodes
            ],
            "links": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relationship": edge.relationship,
                    "description": edge.description,
                }
                for edge in self.links
            ],
        }
    
    @classmethod
    def from_networkx_dict(cls, data: Dict[str, Any]) -> "EntityGraph":
        """Create from NetworkX JSON format."""
        return cls(
            nodes=[EntityGraphNode(**n) for n in data.get("nodes", [])],
            links=[EntityGraphEdge(**e) for e in data.get("links", [])],
        )
