"""
Template hierarchy for commitment learning.

This module implements the template algebra from the GCL theoretical framework:
- CommitmentTemplate: Parameterized commitment patterns (Definition 8)
- Template composition operators (Definition 11)
- Template induction from experience
- Analogical transfer (Theorem 5)
"""

from gcl.templates.template import (
    CommitmentTemplate,
    TemplateMetadata,
    ContextRegionSpec,
    TriggerSchema,
    ActionSchema,
    VerificationSchema,
)
from gcl.templates.composition import (
    TemplateComposer,
    CompositionType,
)
from gcl.templates.induction import (
    TemplateInducer,
    InductionConfig,
    TemplateLibrary,
)

__all__ = [
    # Template core
    "CommitmentTemplate",
    "TemplateMetadata",
    "ContextRegionSpec",
    "TriggerSchema",
    "ActionSchema",
    "VerificationSchema",
    # Composition
    "TemplateComposer",
    "CompositionType",
    # Induction
    "TemplateInducer",
    "InductionConfig",
    "TemplateLibrary",
]
