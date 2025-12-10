# High-level KG representations, and utils functions

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import re
from typing import Dict, Any, Optional
import unicodedata

# Monkey-patch for serializing KGEntity/KGRelation to JSON
def _default(self, obj):
    return getattr(obj.__class__, "to_dict", _default.default)(obj)

_default.default = json.JSONEncoder().default
json.JSONEncoder.default = _default

# Useful constant definition
PROP_NAME = "name"
PROP_DESCRIPTION = "_description"
PROP_PARAGRAPH = "_paragraph"
PROP_CREATED = "_created_at"
PROP_MODIFIED = "_modified_at"
PROP_REFERENCE = "_ref"
PROP_EMBEDDING = "_embedding"
PROP_EXCLUSIVE = "_exclusive"
RESERVED_KEYS = {PROP_NAME, PROP_DESCRIPTION, PROP_PARAGRAPH, PROP_CREATED, PROP_MODIFIED,
                 PROP_REFERENCE, PROP_EMBEDDING, PROP_EXCLUSIVE}

TYPE_EMBEDDABLE = "_Embeddable"
TYPE_RELATIONSCHEMA = "_RelationSchema"
RESERVED_TYPES = {TYPE_EMBEDDABLE, TYPE_RELATIONSCHEMA}


@dataclass
class KGEntity:
    """Representation of an entity in the Knowledge Graph."""
    id: str
    type: str
    name: str
    description: Optional[str] = None
    paragraph: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert KGEntity to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "paragraph": self.paragraph,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "properties": self.properties,
            "ref": self.ref
        }

    def to_json(self) -> str:
        """Convert KGEntity to a JSON string."""
        return json.dumps(self.to_dict(), indent=4)

    def equals(self, other: "KGEntity", ignore_fields: Optional[set] = None) -> bool:
        """
        Compare this KGEntity with another for logical equality, excluding volatile or transient fields.

        Args:
            other (KGEntity): The other entity to compare with.
            ignore_fields (set, optional): Property keys to ignore during comparison (e.g., {'_embedding', '_timestamp'}).

        Returns:
            bool: True if the two entities are logically equivalent.
        """
        if not isinstance(other, KGEntity):
            return False

        ignore_fields = ignore_fields or {PROP_EMBEDDING}

        def cleaned_props(props: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in props.items() if k not in ignore_fields}

        return (
            self.type == other.type and
            self.name == other.name and
            self.description == other.description and
            self.ref == other.ref and
            cleaned_props(self.properties) == cleaned_props(other.properties)
        )


@dataclass
class KGRelation:
    """Representation of a relation in the Knowledge Graph."""
    id: str
    name: str
    source: KGEntity
    target: KGEntity
    direction: str = "forward"
    description: Optional[str] = None
    paragraph: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert KGRelation to a JSON-serializable dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source.to_dict(),  # Convert KGEntity to dictionary
            "target": self.target.to_dict(),  # Convert KGEntity to dictionary
            "direction": self.direction,
            "description": self.description,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "properties": self.properties,
            "ref": self.ref
        }

    def to_json(self) -> str:
        """Convert KGRelation to a JSON string."""
        return json.dumps(self.to_dict(), indent=4)
    
    def equals(self, other: "KGRelation", ignore_fields: Optional[set] = None) -> bool:
        """
        Compare this KGRelation with another for logical equality, ignoring transient fields.

        Args:
            other (KGRelation): The other relation to compare with.
            ignore_fields (set, optional): Set of property keys to ignore during comparison.

        Returns:
            bool: True if the two relations are logically equivalent.
        """
        if not isinstance(other, KGRelation):
            return False

        ignore_fields = ignore_fields or {PROP_EMBEDDING}

        def cleaned_props(props: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in props.items() if k not in ignore_fields}

        return (
            self.name == other.name and
            self.direction == other.direction and
            self.description == other.description and
            self.confidence == other.confidence and
            self.ref == other.ref and
            self.source.equals(other.source, ignore_fields=ignore_fields) and
            self.target.equals(other.target, ignore_fields=ignore_fields) and
            cleaned_props(self.properties) == cleaned_props(other.properties)
        )

@dataclass
class RelevantEntity():
    entity: KGEntity
    score: float

@dataclass
class RelevantRelation():
    relation: KGRelation
    score: float

@dataclass
class CandidateEntity:
    extracted: KGEntity
    aligned: Optional[KGEntity] = None
    merged: Optional[KGEntity] = None
    final: Optional[KGEntity] = None


@dataclass
class CandidateRelation:
    extracted: KGRelation
    aligned: Optional[KGRelation] = None
    merged: Optional[KGRelation] = None
    final: Optional[KGRelation] = None


# Define a decay factor for time-based confidence reduction
DECAY_FACTOR = 0.001  # Adjust for faster/slower decay

def compute_decay_weight(count, last_seen, current_time, decay_factor=0.01):
    """Compute unnormalized temporal confidence weight."""
    last_seen = datetime.fromisoformat(last_seen) if last_seen else datetime.now(timezone.utc)
    time_diff = (current_time - last_seen).total_seconds() / (60 * 60 * 24)  # in days
    return count * math.exp(-decay_factor * time_diff)

def entity_to_text(entity: KGEntity, 
                   current_time: datetime = None,
                   include_id: bool = False,
                   include_des: bool = True,
                   include_par: bool = False,
                   include_prop: bool = True,
                  ) -> str:
    """
    Convert a KGEntity object into a readable text format.

    Args:
        entity (KGEntity): The KGEntity object.
        current_time (datetime): The current timestamp.
        include_id (bool): Whether to include the entity ID in the output.
        include_des (bool): Whether to include the entity description.
        include_prop (bool): Whether to include entity properties.

    Returns:
        str: A human-readable string describing the entity.
    """
    if entity is None:
        return ""

    if current_time is None:
        current_time = datetime.now(timezone.utc)

    description_str = f', desc: "{entity.description}"' if include_des and entity.description else ""
    paragraph_str = f', paragraph: "{entity.paragraph}"' if include_par and entity.paragraph else ""

    properties_str = ""
    if include_prop:
        formatted_properties = []

        for prop, values in entity.properties.items():
            if prop in RESERVED_KEYS:
                continue  # Skip reserved keys

            if isinstance(values, dict):
                # Step 1: Compute unnormalized decay weights
                decay_weights = {
                    val: compute_decay_weight(info["count"], info["last_seen"], current_time)
                    for val, info in values.items()
                }
    
                # Step 2: Normalize the scores
                total_weight = sum(decay_weights.values())
                if total_weight == 0:
                    continue  # skip property with no valid data
    
                confidence_values = [
                    (val, values[val]['context'], round(weight / total_weight, 4))
                    for val, weight in decay_weights.items()
                ]
    
                # Sort by confidence
                confidence_values.sort(key=lambda x: -x[2])
    
                # Format
                props = []
                for val, context, conf in confidence_values:
                    info = [f"{int(round(100 * conf, 0))}%"]
                    if context and context != "None": info.append(f"ctx:{context}")                    
                    props.append(f"{val} ({", ".join(info)})")
                formatted = ("[" + ", ".join(props) + "]") if len(confidence_values) > 1 else (f"{confidence_values[0][0]}")
                formatted_properties.append(f"{prop}: {formatted}")
            else:
                formatted_properties.append(f"{prop}: {values}")
            
        properties_str = f", props: {{{', '.join(formatted_properties)}}}" if formatted_properties else ""

    if include_id:
        return f"({entity.type}: {entity.name} (ID: {entity.id}){description_str}{paragraph_str}{properties_str})"
    else:
        return f"({entity.type}: {entity.name}{description_str}{paragraph_str}{properties_str})"

def relation_to_text(relation: KGRelation,
                     current_time: datetime = None,
                     include_id: bool = False,
                     include_des: bool = True,
                     include_par: bool = False,
                     include_prop: bool = True,
                     include_src_des: bool = True,
                     include_src_prop: bool = True,
                     include_dst_des: bool = True,
                     include_dst_prop: bool = True,
                     property_key_only: bool = False) -> str:
    """
    Convert a KGRelation object into a readable text format.

    Args:
        relation (KGRelation): The KGRelation object.
        include_id (bool): Whether to include the relation ID in the output.

    Returns:
        str: A human-readable string describing the relation.
    """
    if relation is None:
        return ""

    if current_time is None:
        current_time = datetime.now(timezone.utc)

    description_str = f', desc: "{relation.description}"' if include_des and relation.description else ""
    paragraph_str = f', paragraph: "{relation.paragraph}"' if include_par and relation.paragraph else ""

    properties_str = ""
    if include_prop:
        formatted_properties = []
        
        for prop, values in relation.properties.items():
            if prop in RESERVED_KEYS:
                continue  # Skip reserved keys

            if isinstance(values, dict):
                # Step 1: Compute unnormalized decay weights
                decay_weights = {
                    val: compute_decay_weight(info["count"], info["last_seen"], current_time)
                    for val, info in values.items()
                }
    
                # Step 2: Normalize the scores
                total_weight = sum(decay_weights.values())
                if total_weight == 0:
                    continue  # skip property with no valid data
    
                confidence_values = [
                    (val, values[val]['context'], round(weight / total_weight, 4))
                    for val, weight in decay_weights.items()
                ]
    
                # Sort by confidence
                confidence_values.sort(key=lambda x: -x[2])
    
                # Format
                props = []
                for val, context, conf in confidence_values:
                    info = [f"{int(round(100 * conf, 0))}%"]
                    if context and context != "None": info.append(f"ctx:{context}")                    
                    props.append(f"{val} ({", ".join(info)})")
                formatted = ("[" + ", ".join(props) + "]") if len(confidence_values) > 1 else (f"{confidence_values[0][0]}")
                formatted_properties.append(f"{prop}: {formatted}")
            else:
                formatted_properties.append(f"{prop}: {values}")

        properties_str = f", props: {{{', '.join(formatted_properties)}}}" if formatted_properties else ""

    source_text = entity_to_text(relation.source, include_id=include_id,
                                 include_des=include_src_des, include_par=include_par, include_prop=include_src_prop)
    target_text = entity_to_text(relation.target, include_id=include_id,
                                 include_des=include_dst_des, include_par=include_par, include_prop=include_dst_prop)

    if relation.direction == 'forward':
        left_arrow, right_arrow = "-", "->"
    else:
        left_arrow, right_arrow = "<-", "-"
        
    if include_id:
        return f"{source_text}{left_arrow}[{relation.name} (ID: {relation.id}){description_str}{paragraph_str}{properties_str}]{right_arrow}{target_text}"
    else:
        return f"{source_text}{left_arrow}[{relation.name}{description_str}{paragraph_str}{properties_str}]{right_arrow}{target_text}"

def entity_schema_to_text(entity_schema: str) -> str:
    return normalize_entity_type(entity_schema)

def relation_schema_to_text(relation_schema: tuple) -> str:
    return f"({normalize_entity_type(relation_schema[0])})-[{normalize_relation(relation_schema[1])}]->({normalize_entity_type(relation_schema[2])})"

def timestamp_to_text(timestamp: datetime, 
                      isDate: bool = False) -> str:
    """Convert a datetime object to ISO 8601 string format.
    
    Args:
        timestamp (datetime): A datetime object.
        isDate (bool): If True, return only the date part (YYYY-MM-DD). Otherwise include full time.

    Returns:
        str: ISO 8601 formatted date/time string.
    """
    if isDate:
        return timestamp.date().isoformat()
    return timestamp.isoformat()

# def update_ref(old: str, new: str) -> str:
#     """Update a reference JSON string to include the new reference.
#     """
#     if not old: return new
#     if not new: return old
#     return json.dumps({**json.loads(old), **json.loads(new)})

ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:.'-_,& ")
def normalize_string(text: str, 
                     delim: str = " ", 
                     strip: str = None,
                     allowed = ALLOWED) -> str:
    """General helper function to normalize string and replace illegal character with specified delimiter."""
    if not text: text = ""
    # Remove accents
    # text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Remove accents from Latin characters
    text = ''.join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)  # removes accent marks
    )

    # Allow common name punctuations like ':', '-', '.', "'", ',' and replace all others with space
    # You can customize the allowed set as needed
    text = ''.join(c if c in allowed or ord(c) > 127 else ' ' for c in text)

    # Collapse multiple spaces
    return re.sub(r"\s+", delim, text).strip(strip)

def normalize_entity_type(entity):
    """Convert entity type to Neo4j-compatible format."""
    return normalize_string(entity, 
                            delim="_", 
                            strip="_", 
                            allowed=set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                           ).title()  # Convert to lowercase for consistency

def normalize_entity(entity: str) -> str:
    """Normalize entity names while preserving meaningful punctuation."""
    return normalize_string(entity, delim=" ").upper()

def normalize_relation(relation):
    """Convert relation name to Neo4j-compatible format."""
    return normalize_string(relation,
                            delim="_",
                            strip="_",
                            allowed=set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                           ).upper()  # Convert to all uppercase for consistency

def normalize_key(key):
    """Convert property keys to Neo4j-compatible format."""
    return normalize_string(key, 
                            delim="_",
                            allowed=set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
                           ).lower()  # Convert to all uppercase for consistency).lower()  # Convert to lowercase for consistency

def normalize_value(value):
    """Convert property values to string format."""
    if value is None:
        return "None"
    return value if isinstance(value, str) else json.dumps(value)
